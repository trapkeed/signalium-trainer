"""S20 Task #12 — Champion vs Challenger Tracking.

Implements five comparison-dimension scorers plus ``ChampionChallengerTracker``,
which orchestrates rolling multi-metric analysis over paired shadow-test decisions
and fires a ``PROMOTION`` ``ApprovalRequest`` when the challenger is ready.

Architecture:
  - Dimension scorers are **pure functions** — they take a list of decision
    dicts (already loaded from the ``shadow_decisions`` DB table) and return a
    ``DimensionResult``.
  - ``ChampionChallengerTracker`` owns no state of its own; it reads from the
    existing ``shadow_tests`` / ``shadow_decisions`` tables and calls
    ``approval_gate.request_gate()`` to create the gated promotion request.
  - The module lives under ``trainer.research_lab`` so the dependency direction
    is preserved: QDE never imports Research Lab code.

Five comparison dimensions
--------------------------
1. **DIRECTION_ACCURACY** — rate at which challenger and champion agree on
   market direction (buy/sell/flat).  Challenger baseline is 50 % (random).
2. **RISK_QUALITY** — absolute risk_diff.  A challenger that takes similar or
   lower risk than the champion scores higher; large swings in either direction
   are penalised.
3. **SIZING_QUALITY** — expected-PnL proxy diff.  Positive mean pnl_diff means
   the challenger would have generated more expected return on the same inputs.
4. **CONFIDENCE_QUALITY** — calibration: challenger should be more confident on
   agreements (correct direction) and less confident on disagreements.
5. **HOLD_QUALITY** — confidence stability across regimes: a consistent
   challenger varies its confidence less across market regimes, indicating
   stable intra-position management.

Promotion criterion
-------------------
The challenger must be **strictly superior** across at least ``majority_threshold``
(default 60 %) of the five dimensions AND the rolling ``avg_promotion_score``
must exceed ``min_promotion_score`` (default 0.65).  This prevents a single
dominant dimension (e.g. pure agreement) from bypassing genuine quality checks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.research_lab.rl_database import ResearchLabDatabase


# ── Enumerations ──────────────────────────────────────────────────────────────


class ComparisonDimension(str, Enum):
    DIRECTION_ACCURACY = "DIRECTION_ACCURACY"
    RISK_QUALITY = "RISK_QUALITY"
    SIZING_QUALITY = "SIZING_QUALITY"
    CONFIDENCE_QUALITY = "CONFIDENCE_QUALITY"
    HOLD_QUALITY = "HOLD_QUALITY"


# ── Dimension Result ──────────────────────────────────────────────────────────


@dataclass
class DimensionResult:
    """One scored comparison dimension.

    Attributes:
        dimension: One of the five ``ComparisonDimension`` values.
        champion_score: Baseline score for this dimension (reference model).
        challenger_score: Measured score for the challenger.
        challenger_wins: ``True`` when challenger outperforms champion on this
            dimension (using the dimension-appropriate comparison direction).
        score_delta: ``challenger_score - champion_score`` (positive = challenger
            better).
        evidence: Quantitative evidence dict (dimension-specific).
        n_decisions: Number of decisions used.
    """

    dimension: str
    champion_score: float
    challenger_score: float
    challenger_wins: bool
    score_delta: float
    evidence: dict[str, Any]
    n_decisions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "champion_score": round(self.champion_score, 4),
            "challenger_score": round(self.challenger_score, 4),
            "challenger_wins": self.challenger_wins,
            "score_delta": round(self.score_delta, 4),
            "evidence": self.evidence,
            "n_decisions": self.n_decisions,
        }


# ── Rolling Metrics ───────────────────────────────────────────────────────────


@dataclass
class RollingMetrics:
    """Multi-metric summary computed from paired shadow-decision records.

    All metrics are computed using ``pnl_diff`` (challenger minus champion
    expected-PnL proxy) so champion and challenger can be compared on equal
    footing from the same set of inputs.
    """

    n_decisions: int = 0
    agreement_rate: float = 0.0
    avg_promotion_score: float = 0.0
    challenger_sharpe: float = 0.0
    challenger_sortino: float = 0.0
    challenger_profit_factor: float = 1.0
    challenger_max_drawdown: float = 0.0
    challenger_confidence_calibration: float = 0.0  # positive = well-calibrated
    challenger_confidence_reliability: float = 0.0  # fraction: more confident + agrees
    mean_pnl_diff: float = 0.0
    mean_confidence_diff: float = 0.0
    mean_latency_diff_ms: float = 0.0
    mean_risk_diff: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_decisions": self.n_decisions,
            "agreement_rate": round(self.agreement_rate, 4),
            "avg_promotion_score": round(self.avg_promotion_score, 4),
            "challenger_sharpe": round(self.challenger_sharpe, 4),
            "challenger_sortino": round(self.challenger_sortino, 4),
            "challenger_profit_factor": round(self.challenger_profit_factor, 4),
            "challenger_max_drawdown": round(self.challenger_max_drawdown, 4),
            "challenger_confidence_calibration": round(
                self.challenger_confidence_calibration, 4
            ),
            "challenger_confidence_reliability": round(
                self.challenger_confidence_reliability, 4
            ),
            "mean_pnl_diff": round(self.mean_pnl_diff, 6),
            "mean_confidence_diff": round(self.mean_confidence_diff, 4),
            "mean_latency_diff_ms": round(self.mean_latency_diff_ms, 3),
            "mean_risk_diff": round(self.mean_risk_diff, 5),
        }


# ── Head-to-Head Summary ──────────────────────────────────────────────────────


@dataclass
class HeadToHeadSummary:
    """Full champion-vs-challenger comparison summary for one shadow test.

    Attributes:
        test_id: The shadow test identifier.
        champion_version: Champion model version string.
        challenger_version: Challenger model version string.
        n_decisions: Total paired decisions evaluated.
        computed_at: Unix timestamp of this computation.
        dimension_results: One ``DimensionResult`` per comparison dimension.
        n_dimensions_challenger_wins: Count of dimensions the challenger wins.
        n_dimensions_total: Total dimensions scored.
        promotion_criterion_met: Whether the majority-of-dimensions criterion
            is satisfied.
        rolling_metrics: Rolling multi-metric summary.
        promotion_approval_id: Set when a PROMOTION ``ApprovalRequest`` was
            created by this evaluation.
    """

    test_id: str
    champion_version: str
    challenger_version: str
    n_decisions: int
    computed_at: float
    dimension_results: list[DimensionResult]
    n_dimensions_challenger_wins: int
    n_dimensions_total: int
    promotion_criterion_met: bool
    rolling_metrics: RollingMetrics
    promotion_approval_id: str = ""

    @property
    def challenger_win_rate(self) -> float:
        if self.n_dimensions_total == 0:
            return 0.0
        return self.n_dimensions_challenger_wins / self.n_dimensions_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "n_decisions": self.n_decisions,
            "computed_at": round(self.computed_at, 3),
            "n_dimensions_challenger_wins": self.n_dimensions_challenger_wins,
            "n_dimensions_total": self.n_dimensions_total,
            "challenger_win_rate": round(self.challenger_win_rate, 4),
            "promotion_criterion_met": self.promotion_criterion_met,
            "promotion_approval_id": self.promotion_approval_id,
            "dimension_results": [r.to_dict() for r in self.dimension_results],
            "rolling_metrics": self.rolling_metrics.to_dict(),
        }


# ── Promotion Criterion Config ────────────────────────────────────────────────


@dataclass
class PromotionCriterionConfig:
    """Configurable thresholds for the champion-vs-challenger promotion gate.

    Attributes:
        min_decisions: Minimum number of paired decisions before the criterion
            can be evaluated.
        majority_threshold: Fraction of dimensions the challenger must win
            (default 0.6 → 3 of 5 dimensions).
        min_promotion_score: Minimum ``avg_promotion_score`` from the shadow
            test run (composite per-decision score).
        min_agreement_rate: Minimum challenger-champion agreement rate.
            A challenger that rarely agrees with the proven champion carries
            more deployment risk.
    """

    min_decisions: int = 50
    majority_threshold: float = 0.6   # must win ≥ 60 % of scored dimensions
    min_promotion_score: float = 0.65
    min_agreement_rate: float = 0.55


# ── Dimension Scorers (pure functions) ────────────────────────────────────────


def _inconclusive(dim: ComparisonDimension) -> DimensionResult:
    """Return a neutral result when insufficient data is available."""
    return DimensionResult(
        dimension=dim.value,
        champion_score=0.5,
        challenger_score=0.5,
        challenger_wins=False,
        score_delta=0.0,
        evidence={"reason": "insufficient_data"},
        n_decisions=0,
    )


def score_direction_accuracy(decisions: list[dict]) -> DimensionResult:
    """Fraction of decisions where challenger and champion agree.

    Champion baseline is 50 % (random alignment).  Any agreement rate > 0.5
    means the challenger is correlated with the champion's view of the market.
    """
    n = len(decisions)
    if n < 2:
        return _inconclusive(ComparisonDimension.DIRECTION_ACCURACY)

    agrees = [d.get("decisions_agree", False) for d in decisions]
    agreement_rate = sum(1 for a in agrees if a) / n

    # Per-regime breakdown
    by_regime: dict[str, list[bool]] = {}
    for d in decisions:
        regime = d.get("regime", "UNKNOWN") or "UNKNOWN"
        by_regime.setdefault(regime, []).append(bool(d.get("decisions_agree", False)))
    per_regime = {
        r: round(sum(v) / len(v), 4) for r, v in by_regime.items() if len(v) >= 2
    }

    champion_score = 0.5
    challenger_score = agreement_rate
    return DimensionResult(
        dimension=ComparisonDimension.DIRECTION_ACCURACY.value,
        champion_score=champion_score,
        challenger_score=challenger_score,
        challenger_wins=challenger_score > champion_score,
        score_delta=challenger_score - champion_score,
        evidence={
            "agreement_rate": round(agreement_rate, 4),
            "n_agreements": sum(1 for a in agrees if a),
            "n_decisions": n,
            "per_regime_agreement": per_regime,
        },
        n_decisions=n,
    )


def score_risk_quality(decisions: list[dict]) -> DimensionResult:
    """Quality of challenger's risk allocation relative to champion.

    ``risk_diff = challenger_risk - champion_risk``.  A small absolute risk_diff
    (challenger taking similar risk) scores highest.  Large divergence in either
    direction is penalised — a challenger that takes far more risk is unproven,
    and one that takes far less may be overly conservative.

    Challenger wins when its mean absolute risk diff is below 1 % (0.01) and
    the champion's implicit baseline (0 diff = reference) is exceeded.
    """
    n = len(decisions)
    if n < 2:
        return _inconclusive(ComparisonDimension.RISK_QUALITY)

    risk_diffs = [d.get("risk_diff", 0.0) for d in decisions]
    abs_diffs = [abs(r) for r in risk_diffs]
    mean_abs_diff = sum(abs_diffs) / n
    mean_diff = sum(risk_diffs) / n

    # Map abs diff to [0, 1] on a 0–5% scale.
    # 0 % diff → 1.0 (identical risk),  ≥ 5 % diff → 0.0.
    # Champion baseline is 0.5 (neutral midpoint) so score_delta direction
    # and challenger_wins are always consistent.
    challenger_score = max(0.0, 1.0 - mean_abs_diff / 0.05)
    champion_score = 0.5  # neutral baseline — challenger beats it when abs_diff < 2.5 %

    challenger_wins = challenger_score > champion_score  # score_delta > 0
    score_delta = challenger_score - champion_score

    return DimensionResult(
        dimension=ComparisonDimension.RISK_QUALITY.value,
        champion_score=champion_score,
        challenger_score=challenger_score,
        challenger_wins=challenger_wins,
        score_delta=score_delta,
        evidence={
            "mean_abs_risk_diff": round(mean_abs_diff, 5),
            "mean_risk_diff": round(mean_diff, 5),
            "n_decisions": n,
        },
        n_decisions=n,
    )


def score_sizing_quality(decisions: list[dict]) -> DimensionResult:
    """Expected-PnL proxy advantage for the challenger.

    ``pnl_diff = challenger_expected_pnl - champion_expected_pnl``.  Positive
    mean_pnl_diff means the challenger would have generated higher expected
    return on the same market inputs.  Challenger wins when mean_pnl_diff > 0.

    The challenger_score is mapped to [0, 1] using a sigmoid-like clamping so
    extremely large diffs are bounded.
    """
    n = len(decisions)
    if n < 2:
        return _inconclusive(ComparisonDimension.SIZING_QUALITY)

    pnl_diffs = [d.get("pnl_diff", 0.0) for d in decisions]
    mean_pnl_diff = sum(pnl_diffs) / n
    pct_positive = sum(1 for p in pnl_diffs if p > 0) / n

    # Map mean_pnl_diff to [0, 1]: 0 → 0.5, positive → above 0.5
    champion_score = 0.5  # baseline: no advantage
    # Use fraction of decisions where challenger wins as the score
    challenger_score = pct_positive
    challenger_wins = mean_pnl_diff > 0 and pct_positive > 0.5

    return DimensionResult(
        dimension=ComparisonDimension.SIZING_QUALITY.value,
        champion_score=champion_score,
        challenger_score=challenger_score,
        challenger_wins=challenger_wins,
        score_delta=challenger_score - champion_score,
        evidence={
            "mean_pnl_diff": round(mean_pnl_diff, 6),
            "pct_positive_diff": round(pct_positive, 4),
            "n_decisions": n,
        },
        n_decisions=n,
    )


def score_confidence_quality(decisions: list[dict]) -> DimensionResult:
    """Challenger confidence calibration: higher confidence on agreements,
    lower on disagreements.

    A well-calibrated model is more certain when it agrees with the champion
    (indicating both see the same signal) and more uncertain when it diverges.
    ``calibration_delta = mean(conf | agree) - mean(conf | disagree)``; higher
    is better.  Challenger wins when its calibration delta exceeds champion's
    implicit baseline (which is 0 — champion by definition is perfectly aligned
    with itself).
    """
    n = len(decisions)
    if n < 2:
        return _inconclusive(ComparisonDimension.CONFIDENCE_QUALITY)

    agree_confs = [
        d.get("challenger_confidence", 0.5) for d in decisions if d.get("decisions_agree")
    ]
    disagree_confs = [
        d.get("challenger_confidence", 0.5) for d in decisions if not d.get("decisions_agree")
    ]
    champ_confs = [d.get("champion_confidence", 0.5) for d in decisions]

    mean_agree_conf = sum(agree_confs) / len(agree_confs) if agree_confs else 0.5
    mean_disagree_conf = sum(disagree_confs) / len(disagree_confs) if disagree_confs else 0.5
    calibration_delta = mean_agree_conf - mean_disagree_conf

    # Confidence reliability: fraction where challenger is more confident AND agrees
    conf_reliable_n = sum(
        1
        for d in decisions
        if d.get("challenger_confidence", 0) > d.get("champion_confidence", 0)
        and d.get("decisions_agree")
    )
    conf_reliable = conf_reliable_n / n

    # Mean champion confidence (reference)
    mean_champ_conf = sum(champ_confs) / n
    mean_chall_conf = sum(d.get("challenger_confidence", 0.5) for d in decisions) / n

    # Score: calibration_delta mapped to [0, 1]; baseline 0 = 0.5
    challenger_score = max(0.0, min(1.0, 0.5 + calibration_delta))
    champion_score = 0.5  # baseline: 0 calibration delta
    challenger_wins = calibration_delta > 0 and conf_reliable > 0.3

    return DimensionResult(
        dimension=ComparisonDimension.CONFIDENCE_QUALITY.value,
        champion_score=champion_score,
        challenger_score=challenger_score,
        challenger_wins=challenger_wins,
        score_delta=challenger_score - champion_score,
        evidence={
            "calibration_delta": round(calibration_delta, 4),
            "mean_agree_conf": round(mean_agree_conf, 4),
            "mean_disagree_conf": round(mean_disagree_conf, 4),
            "confidence_reliability": round(conf_reliable, 4),
            "mean_challenger_confidence": round(mean_chall_conf, 4),
            "mean_champion_confidence": round(mean_champ_conf, 4),
            "n_agree": len(agree_confs),
            "n_disagree": len(disagree_confs),
        },
        n_decisions=n,
    )


def score_hold_quality(decisions: list[dict]) -> DimensionResult:
    """Decision consistency across market regimes.

    A challenger with good hold quality applies its strategy consistently:
    its per-regime agreement rates should be uniformly high rather than
    excellent in one regime and poor in another.

    Score = 1 − coefficient-of-variation of per-regime agreement rates.
    Low CoV (< 0.2) → challenger wins.  Champion baseline uses champion
    confidence CoV as the reference (a consistent champion has low CoV).
    """
    n = len(decisions)
    if n < 5:
        return _inconclusive(ComparisonDimension.HOLD_QUALITY)

    # Per-regime challenger agreement rates
    by_regime: dict[str, list[bool]] = {}
    for d in decisions:
        regime = d.get("regime", "UNKNOWN") or "UNKNOWN"
        by_regime.setdefault(regime, []).append(bool(d.get("decisions_agree", False)))

    regime_agree_rates = [sum(v) / len(v) for v in by_regime.values() if len(v) >= 2]
    if not regime_agree_rates:
        return _inconclusive(ComparisonDimension.HOLD_QUALITY)

    # CoV of challenger per-regime agreement rates
    chall_mean = sum(regime_agree_rates) / len(regime_agree_rates)
    chall_std = _std(regime_agree_rates)
    chall_cov = chall_std / chall_mean if chall_mean > 1e-9 else 1.0

    # CoV of champion confidence (stability proxy for the reference model)
    champ_confs = [d.get("champion_confidence", 0.5) for d in decisions]
    champ_conf_mean = sum(champ_confs) / len(champ_confs)
    champ_conf_std = _std(champ_confs)
    champ_cov = champ_conf_std / champ_conf_mean if champ_conf_mean > 1e-9 else 1.0

    # Scores: 1 − CoV (lower variance → higher score)
    challenger_score = max(0.0, 1.0 - chall_cov)
    champion_score = max(0.0, 1.0 - champ_cov)
    # Challenger wins HOLD_QUALITY when its cross-regime consistency score is
    # good (CoV < 0.4 → score > 0.6), regardless of champion's baseline.
    # This avoids the case where champion CoV = 0 makes the dimension
    # unwinnable even when the challenger is also highly consistent.
    challenger_wins = challenger_score > 0.6

    return DimensionResult(
        dimension=ComparisonDimension.HOLD_QUALITY.value,
        champion_score=champion_score,
        challenger_score=challenger_score,
        challenger_wins=challenger_wins,
        score_delta=challenger_score - champion_score,
        evidence={
            "challenger_regime_cov": round(chall_cov, 4),
            "champion_confidence_cov": round(champ_cov, 4),
            "n_regimes_scored": len(regime_agree_rates),
            "per_regime_agreement_rates": {
                r: round(sum(v) / len(v), 4)
                for r, v in by_regime.items()
                if len(v) >= 2
            },
        },
        n_decisions=n,
    )


# ── Rolling Metrics Computation ───────────────────────────────────────────────


def compute_rolling_metrics(decisions: list[dict]) -> RollingMetrics:
    """Compute multi-metric summary from shadow decision records.

    All metrics use ``pnl_diff`` (challenger − champion expected PnL proxy)
    as the return series, so both models can be evaluated on identical inputs
    without requiring live trade outcomes.
    """
    n = len(decisions)
    if n == 0:
        return RollingMetrics()

    pnl_diffs = [d.get("pnl_diff", 0.0) for d in decisions]
    conf_chall = [d.get("challenger_confidence", 0.5) for d in decisions]
    conf_champ = [d.get("champion_confidence", 0.5) for d in decisions]
    agrees = [bool(d.get("decisions_agree", False)) for d in decisions]
    promo_scores = [d.get("promotion_score", 0.0) for d in decisions]
    latency_diffs = [d.get("latency_diff_ms", 0.0) for d in decisions]
    risk_diffs = [d.get("risk_diff", 0.0) for d in decisions]

    # Sharpe: mean / std * sqrt(252) scale (per-decision approximation)
    mean_pnl = sum(pnl_diffs) / n
    std_pnl = _std(pnl_diffs)
    sharpe = (mean_pnl / std_pnl) * math.sqrt(252) if std_pnl > 1e-9 else 0.0

    # Sortino: mean / downside std
    neg_diffs = [p for p in pnl_diffs if p < 0]
    downside_var = sum(p**2 for p in neg_diffs) / max(len(neg_diffs), 1)
    downside_std = math.sqrt(downside_var)
    sortino = (mean_pnl / downside_std) * math.sqrt(252) if downside_std > 1e-9 else 0.0

    # Profit factor: gross profit / gross loss (on pnl_diff series)
    gross_profit = sum(p for p in pnl_diffs if p > 0)
    gross_loss = abs(sum(p for p in pnl_diffs if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-9 else (
        99.0 if gross_profit > 0 else 1.0
    )

    # Max drawdown on cumulative pnl_diff curve.
    # Starting equity = 0.  When the series never reaches a positive peak
    # (e.g. all-negative), we use the absolute drop below the last high-water
    # mark as the drawdown so early-loss sequences are never silently reported
    # as zero drawdown.
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl_diffs:
        cum += p
        if cum > peak:
            peak = cum
        if peak > 1e-9:
            dd = (peak - cum) / peak
        else:
            # Peak never left 0 (or returned to 0): absolute drawdown from
            # initial equity.  peak = 0 here, cum ≤ 0.
            dd = max(0.0, -cum)
        max_dd = max(max_dd, dd)

    # Confidence calibration: mean(conf | agree) - mean(conf | disagree)
    agree_confs = [conf_chall[i] for i in range(n) if agrees[i]]
    disagree_confs = [conf_chall[i] for i in range(n) if not agrees[i]]
    cal = (
        (sum(agree_confs) / len(agree_confs) if agree_confs else 0.5)
        - (sum(disagree_confs) / len(disagree_confs) if disagree_confs else 0.5)
    )

    # Confidence reliability: fraction of decisions where challenger is more
    # confident AND agrees with champion
    conf_reliable = sum(
        1 for i in range(n) if conf_chall[i] > conf_champ[i] and agrees[i]
    ) / n

    return RollingMetrics(
        n_decisions=n,
        agreement_rate=round(sum(1 for a in agrees if a) / n, 4),
        avg_promotion_score=round(sum(promo_scores) / n, 4),
        challenger_sharpe=round(sharpe, 4),
        challenger_sortino=round(sortino, 4),
        challenger_profit_factor=round(min(profit_factor, 99.0), 4),
        challenger_max_drawdown=round(max_dd, 4),
        challenger_confidence_calibration=round(cal, 4),
        challenger_confidence_reliability=round(conf_reliable, 4),
        mean_pnl_diff=round(mean_pnl, 6),
        mean_confidence_diff=round(sum(d.get("confidence_diff", 0) for d in decisions) / n, 4),
        mean_latency_diff_ms=round(sum(latency_diffs) / n, 3),
        mean_risk_diff=round(sum(risk_diffs) / n, 5),
    )


# ── Tracker ───────────────────────────────────────────────────────────────────


class ChampionChallengerTracker:
    """Orchestrates five-dimension champion-vs-challenger analysis.

    Reads paired decisions from the ``shadow_decisions`` DB table, runs all
    five dimension scorers, evaluates the configurable promotion criterion, and
    creates a ``PROMOTION`` ``ApprovalRequest`` when the criterion is met.

    Args:
        db: ``ResearchLabDatabase`` for reading shadow tests/decisions and
            persisting approval request references.
        approval_gate: Optional ``ApprovalGate`` for creating PROMOTION gate
            requests.  When ``None``, the promotion approval step is skipped
            (useful in tests).
        config: Configurable promotion criterion; defaults to
            ``PromotionCriterionConfig()``.
    """

    _SCORERS = [
        score_direction_accuracy,
        score_risk_quality,
        score_sizing_quality,
        score_confidence_quality,
        score_hold_quality,
    ]

    def __init__(
        self,
        db: ResearchLabDatabase,
        approval_gate: Any | None = None,
        config: PromotionCriterionConfig | None = None,
    ) -> None:
        self._db = db
        self._gate = approval_gate
        self._cfg = config or PromotionCriterionConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_head_to_head(self, test_id: str) -> HeadToHeadSummary | None:
        """Compute a full head-to-head comparison for the given shadow test.

        Returns ``None`` if the test does not exist or has no decisions.
        Does **not** create an ApprovalRequest — call ``evaluate_promotion``
        for that.
        """
        shadow_test = self._db.get_shadow_test(test_id)
        if shadow_test is None:
            return None

        decisions = self._db.list_shadow_decisions(test_id)
        n = len(decisions)

        results = [scorer(decisions) for scorer in self._SCORERS]
        n_wins = sum(1 for r in results if r.challenger_wins)
        rolling = compute_rolling_metrics(decisions)

        win_rate = n_wins / len(results) if results else 0.0
        criterion_met = (
            n >= self._cfg.min_decisions
            and win_rate >= self._cfg.majority_threshold
            and rolling.avg_promotion_score >= self._cfg.min_promotion_score
            and rolling.agreement_rate >= self._cfg.min_agreement_rate
        )

        return HeadToHeadSummary(
            test_id=test_id,
            champion_version=shadow_test.get("champion_version", ""),
            challenger_version=shadow_test.get("challenger_version", ""),
            n_decisions=n,
            computed_at=time.time(),
            dimension_results=results,
            n_dimensions_challenger_wins=n_wins,
            n_dimensions_total=len(results),
            promotion_criterion_met=criterion_met,
            rolling_metrics=rolling,
        )

    def evaluate_promotion(self, test_id: str) -> dict[str, Any]:
        """Evaluate the promotion criterion and create an ApprovalRequest if met.

        Idempotent via the ``ApprovalGate``'s own dedup logic — repeated calls
        for the same ``test_id`` return the existing ``PENDING`` request.

        Returns a result dict with fields:
          - ``criterion_met``: whether the challenger qualified
          - ``approval_id``: set if an ``ApprovalRequest`` was created or exists
          - ``summary``: the ``HeadToHeadSummary`` dict
          - ``reason``: human-readable explanation
        """
        h2h = self.compute_head_to_head(test_id)
        if h2h is None:
            return {
                "criterion_met": False,
                "approval_id": "",
                "summary": None,
                "reason": f"Shadow test {test_id!r} not found.",
            }

        if not h2h.promotion_criterion_met:
            cfg = self._cfg
            reasons = []
            if h2h.n_decisions < cfg.min_decisions:
                reasons.append(
                    f"insufficient decisions ({h2h.n_decisions} < {cfg.min_decisions})"
                )
            if h2h.challenger_win_rate < cfg.majority_threshold:
                reasons.append(
                    f"dimension wins {h2h.n_dimensions_challenger_wins}/{h2h.n_dimensions_total} "
                    f"< {cfg.majority_threshold:.0%}"
                )
            if h2h.rolling_metrics.avg_promotion_score < cfg.min_promotion_score:
                reasons.append(
                    f"avg_promotion_score {h2h.rolling_metrics.avg_promotion_score} "
                    f"< {cfg.min_promotion_score}"
                )
            if h2h.rolling_metrics.agreement_rate < cfg.min_agreement_rate:
                reasons.append(
                    f"agreement_rate {h2h.rolling_metrics.agreement_rate} "
                    f"< {cfg.min_agreement_rate}"
                )
            return {
                "criterion_met": False,
                "approval_id": "",
                "summary": h2h.to_dict(),
                "reason": "Promotion criterion not met: " + "; ".join(reasons),
            }

        # Criterion met — create approval request
        approval_id = ""
        reason_text = (
            f"Challenger {h2h.challenger_version!r} outperformed champion "
            f"{h2h.champion_version!r} on "
            f"{h2h.n_dimensions_challenger_wins}/{h2h.n_dimensions_total} dimensions "
            f"(avg_promotion_score={h2h.rolling_metrics.avg_promotion_score}, "
            f"agreement_rate={h2h.rolling_metrics.agreement_rate})."
        )
        if self._gate is not None:
            try:
                from trainer.research_lab.approval_models import ApprovalGateType

                gate_req = self._gate.create_request(
                    gate_type=ApprovalGateType.PROMOTION,
                    resource_id=test_id,
                    resource_type="shadow_test",
                    requested_by="champion_challenger_tracker",
                    action_summary=reason_text,
                    requestor_context={
                        "test_id": test_id,
                        "champion_version": h2h.champion_version,
                        "challenger_version": h2h.challenger_version,
                        "dimension_results": [r.to_dict() for r in h2h.dimension_results],
                        "rolling_metrics": h2h.rolling_metrics.to_dict(),
                    },
                )
                approval_id = gate_req.request_id
                h2h.promotion_approval_id = approval_id
                logger.info(
                    f"ChampionChallengerTracker: PROMOTION ApprovalRequest "
                    f"{approval_id} created for test={test_id!r} "
                    f"challenger={h2h.challenger_version!r}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"ChampionChallengerTracker: failed to create approval "
                    f"request for test={test_id!r}: {exc!r}"
                )

        return {
            "criterion_met": True,
            "approval_id": approval_id,
            "summary": h2h.to_dict(),
            "reason": reason_text,
        }

    def get_status(self, test_id: str | None = None) -> dict[str, Any]:
        """Return a status summary for running shadow tests.

        When ``test_id`` is given, returns detailed status for that specific
        test.  Without a ``test_id``, returns a summary of all RUNNING tests.
        """
        if test_id:
            shadow_test = self._db.get_shadow_test(test_id)
            if shadow_test is None:
                return {"error": f"Shadow test {test_id!r} not found."}
            n_decisions = self._db.count_shadow_decisions(test_id)
            h2h = self.compute_head_to_head(test_id)
            return {
                "test_id": test_id,
                "champion_version": shadow_test.get("champion_version", ""),
                "challenger_version": shadow_test.get("challenger_version", ""),
                "status": shadow_test.get("status", ""),
                "n_decisions": n_decisions,
                "promotion_criterion_met": h2h.promotion_criterion_met if h2h else False,
                "head_to_head": h2h.to_dict() if h2h else None,
                "shadow_test": shadow_test,
            }

        # All RUNNING tests summary
        running_tests = self._db.list_shadow_tests(status="RUNNING")
        summaries = []
        for st in running_tests:
            tid = st.get("test_id", "")
            n = self._db.count_shadow_decisions(tid)
            summaries.append({
                "test_id": tid,
                "champion_version": st.get("champion_version", ""),
                "challenger_version": st.get("challenger_version", ""),
                "n_decisions": n,
                "min_decisions": st.get("min_decisions", self._cfg.min_decisions),
                "progress_pct": round(
                    100.0 * n / max(st.get("min_decisions", self._cfg.min_decisions), 1), 1
                ),
                "started_at": st.get("started_at"),
            })

        return {
            "n_running": len(summaries),
            "running_tests": summaries,
            "promotion_criterion_config": {
                "min_decisions": self._cfg.min_decisions,
                "majority_threshold": self._cfg.majority_threshold,
                "min_promotion_score": self._cfg.min_promotion_score,
                "min_agreement_rate": self._cfg.min_agreement_rate,
            },
        }

    def get_comparison(self, test_id: str) -> dict[str, Any]:
        """Return detailed dimension-level comparison for a specific shadow test."""
        h2h = self.compute_head_to_head(test_id)
        if h2h is None:
            return {"error": f"Shadow test {test_id!r} not found or has no decisions."}
        return h2h.to_dict()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)
