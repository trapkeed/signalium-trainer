"""S20 — Per-dimension evaluators for the Retraining Recommendation Engine.

Five evaluators, each reading from the telemetry warehouse and producing an
EvaluationResult. They are pure functions over telemetry data — no side effects,
no DB writes.

Dimensions:
  1. SharpeDegradationEvaluator  — rolling Sharpe vs baseline Sharpe
  2. ConfidenceDriftEvaluator    — mean confidence shift + calibration proxy
  3. ReasoningDriftEvaluator     — output-head distribution shift
  4. RegimeUnderperformEvaluator — per-regime Sharpe breakdown
  5. PortfolioInefficiencyEval   — concentration + scale-in success rate
"""

from __future__ import annotations

import math
from typing import Any

from trainer.research_lab.recommendation_models import (
    EvaluationDimension,
    EvaluationResult,
    RecommendationConfig,
    RecommendationUrgency,
    RecommendedScope,
    _urgency_from_score,
)
from trainer.research_lab.telemetry_models import TelemetryQuery
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

# ── Helpers ────────────────────────────────────────────────────────────────────


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    m = _mean(returns)
    s = _std(returns)
    if s < 1e-9:
        # Zero variance: use sign of mean as a sentinel
        return 1.0 if m > 0 else (-1.0 if m < 0 else 0.0)
    return m / s


def _tv_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Total-variation distance between two distributions over the same keys."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / len(keys)


def _split_windows(
    data: list,
    baseline_window: int,
    rolling_window: int,
    min_samples: int,
) -> tuple[list, list] | tuple[None, None]:
    """Split `data` (oldest→newest) into disjoint (baseline, recent) halves.

    Strategy:
    - If len ≥ baseline_window + rolling_window: use first baseline_window rows
      as baseline and last rolling_window rows as recent (guaranteed disjoint).
    - Elif len ≥ 2 * min_samples: split 67% / 33% (oldest portion = baseline).
    - Else: insufficient data — return (None, None).

    This prevents the overlap that occurs when callers naively slice
    [:baseline_window] and [-rolling_window:] on a short dataset.
    """
    n = len(data)
    full = baseline_window + rolling_window
    if n >= full:
        return data[:baseline_window], data[-rolling_window:]
    if n >= 2 * min_samples:
        split = (2 * n) // 3  # older 67% = baseline, newer 33% = recent
        split = max(split, min_samples)
        split = min(split, n - min_samples)
        return data[:split], data[split:]
    return None, None


# ── 1. Sharpe Degradation ─────────────────────────────────────────────────────


class SharpeDegradationEvaluator:
    """Detects sustained Sharpe-ratio degradation by comparing a rolling
    recent window against a baseline window."""

    def __init__(self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig) -> None:
        self._wh = warehouse
        self._cfg = cfg

    def evaluate(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> EvaluationResult:
        q = TelemetryQuery(model_version=model_version, since=since, until=until,
                           limit=self._cfg.baseline_window + self._cfg.rolling_window)
        trades = self._wh.query_trades(q)
        # oldest → newest
        trades = sorted(trades, key=lambda t: t.get("timestamp", 0))

        dim = EvaluationDimension.SHARPE_DEGRADATION
        _not_triggered = EvaluationResult(
            dimension=dim, triggered=False, score=0.0, evidence={},
            recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=RecommendationUrgency.LOW,
            summary="Insufficient trade data",
        )
        if len(trades) < self._cfg.min_trades_for_evaluation:
            return _not_triggered

        pnl_pcts = [t.get("pnl_pct", 0.0) for t in trades]
        baseline_pnls, recent_pnls = _split_windows(
            pnl_pcts,
            self._cfg.baseline_window,
            self._cfg.rolling_window,
            self._cfg.min_trades_for_evaluation,
        )
        if baseline_pnls is None:
            return _not_triggered

        baseline_sharpe = _sharpe(baseline_pnls)
        recent_sharpe = _sharpe(recent_pnls)

        # Normalised degradation relative to baseline magnitude
        denom = max(abs(baseline_sharpe), 0.1)
        degradation = (baseline_sharpe - recent_sharpe) / denom  # positive = worse
        score = max(0.0, degradation / self._cfg.sharpe_degradation_threshold)
        triggered = degradation > self._cfg.sharpe_degradation_threshold

        urgency = _urgency_from_score(score) if triggered else RecommendationUrgency.LOW
        scope = RecommendedScope.FULL_RETRAIN if urgency == RecommendationUrgency.CRITICAL \
            else RecommendedScope.FINE_TUNE

        evidence = {
            "baseline_sharpe": round(baseline_sharpe, 4),
            "recent_sharpe": round(recent_sharpe, 4),
            "degradation_fraction": round(degradation, 4),
            "n_baseline_trades": len(baseline_pnls),
            "n_recent_trades": len(recent_pnls),
            "baseline_mean_pnl_pct": round(_mean(baseline_pnls), 4),
            "recent_mean_pnl_pct": round(_mean(recent_pnls), 4),
        }
        summary = (
            f"Sharpe degraded from {baseline_sharpe:.3f} → {recent_sharpe:.3f} "
            f"({degradation * 100:.1f}% drop)"
        )
        return EvaluationResult(
            dimension=dim, triggered=triggered, score=round(score, 4),
            evidence=evidence, recommended_scope=scope, urgency=urgency, summary=summary,
        )


# ── 2. Confidence Drift ───────────────────────────────────────────────────────


class ConfidenceDriftEvaluator:
    """Detects calibration shift by comparing mean confidence in the rolling
    window against the baseline window, and (where available) computing the
    confidence-outcome correlation."""

    def __init__(self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig) -> None:
        self._wh = warehouse
        self._cfg = cfg

    def evaluate(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> EvaluationResult:
        total = self._cfg.baseline_window + self._cfg.rolling_window
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=total)
        decisions = sorted(self._wh.query_decisions(q), key=lambda d: d.get("timestamp", 0))

        dim = EvaluationDimension.CONFIDENCE_DRIFT
        _not_triggered = EvaluationResult(
            dimension=dim, triggered=False, score=0.0, evidence={},
            recommended_scope=RecommendedScope.CALIBRATION_ONLY,
            urgency=RecommendationUrgency.LOW,
            summary="Insufficient decision data",
        )
        if len(decisions) < self._cfg.min_trades_for_evaluation:
            return _not_triggered

        confs = [d.get("confidence", 0.5) for d in decisions]
        baseline_confs, recent_confs = _split_windows(
            confs,
            self._cfg.baseline_window,
            self._cfg.rolling_window,
            self._cfg.min_trades_for_evaluation,
        )
        if baseline_confs is None:
            return _not_triggered

        baseline_conf = _mean(baseline_confs)
        recent_conf = _mean(recent_confs)
        drift = abs(recent_conf - baseline_conf)

        score = drift / self._cfg.confidence_drift_threshold
        triggered = drift > self._cfg.confidence_drift_threshold

        # Confidence-outcome calibration proxy: are high-confidence recent trades winning?
        trades_q = TelemetryQuery(model_version=model_version, since=since, until=until,
                                  limit=self._cfg.rolling_window)
        trades = self._wh.query_trades(trades_q)
        conf_corr: float | None = None
        if len(trades) >= 5:
            t_confs = [t.get("confidence_at_entry", 0.5) for t in trades]
            t_wins = [1.0 if t.get("pnl", 0) > 0 else 0.0 for t in trades]
            # Pearson correlation proxy
            m_c = _mean(t_confs)
            m_w = _mean(t_wins)
            num = sum((c - m_c) * (w - m_w) for c, w in zip(t_confs, t_wins))
            den = math.sqrt(
                sum((c - m_c) ** 2 for c in t_confs)
                * sum((w - m_w) ** 2 for w in t_wins)
            )
            conf_corr = round(num / den, 4) if den > 1e-9 else 0.0

        urgency = _urgency_from_score(score) if triggered else RecommendationUrgency.LOW
        evidence = {
            "baseline_mean_confidence": round(baseline_conf, 4),
            "recent_mean_confidence": round(recent_conf, 4),
            "drift": round(drift, 4),
            "threshold": self._cfg.confidence_drift_threshold,
            "confidence_outcome_correlation": conf_corr,
            "n_baseline_decisions": len(baseline_confs),
            "n_recent_decisions": len(recent_confs),
        }
        summary = (
            f"Confidence drifted from {baseline_conf:.3f} → {recent_conf:.3f} "
            f"(Δ={drift:.3f}, threshold={self._cfg.confidence_drift_threshold})"
        )
        return EvaluationResult(
            dimension=dim, triggered=triggered, score=round(score, 4),
            evidence=evidence, recommended_scope=RecommendedScope.CALIBRATION_ONLY,
            urgency=urgency, summary=summary,
        )


# ── 3. Reasoning Drift ────────────────────────────────────────────────────────


class ReasoningDriftEvaluator:
    """Detects output-head distribution shift by comparing per-head mean values
    in the baseline window vs. the rolling window."""

    def __init__(self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig) -> None:
        self._wh = warehouse
        self._cfg = cfg

    def evaluate(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> EvaluationResult:
        total = self._cfg.baseline_window + self._cfg.rolling_window
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=total)
        records = sorted(self._wh.query_reasoning(q), key=lambda r: r.get("timestamp", 0))

        dim = EvaluationDimension.REASONING_DRIFT
        _not_triggered = EvaluationResult(
            dimension=dim, triggered=False, score=0.0, evidence={},
            recommended_scope=RecommendedScope.HEAD_RETRAIN,
            urgency=RecommendationUrgency.LOW,
            summary="Insufficient reasoning data",
        )
        if len(records) < self._cfg.min_trades_for_evaluation:
            return _not_triggered

        baseline_recs, recent_recs = _split_windows(
            records,
            self._cfg.baseline_window,
            self._cfg.rolling_window,
            self._cfg.min_trades_for_evaluation,
        )
        if baseline_recs is None:
            return _not_triggered

        # Aggregate per-head means in each window
        def _mean_heads(recs: list[dict]) -> dict[str, float]:
            acc: dict[str, list[float]] = {}
            for r in recs:
                heads = r.get("output_heads", {})
                for k, v in heads.items():
                    if isinstance(v, (int, float)):
                        acc.setdefault(k, []).append(float(v))
            return {k: _mean(vs) for k, vs in acc.items()}

        baseline_heads = _mean_heads(baseline_recs)
        recent_heads = _mean_heads(recent_recs)

        # Per-head drift (abs diff); only for heads present in both windows
        shared_heads = set(baseline_heads) & set(recent_heads)
        if not shared_heads:
            return _not_triggered

        head_drifts = {
            k: abs(recent_heads[k] - baseline_heads[k]) for k in shared_heads
        }
        avg_drift = _mean(list(head_drifts.values()))
        score = avg_drift / self._cfg.reasoning_drift_threshold
        triggered = avg_drift > self._cfg.reasoning_drift_threshold

        urgency = _urgency_from_score(score) if triggered else RecommendationUrgency.LOW
        top_drifters = sorted(head_drifts.items(), key=lambda x: x[1], reverse=True)[:5]
        evidence = {
            "avg_head_drift": round(avg_drift, 4),
            "threshold": self._cfg.reasoning_drift_threshold,
            "n_heads_compared": len(shared_heads),
            "top_drifting_heads": {k: round(v, 4) for k, v in top_drifters},
            "n_baseline_records": len(baseline_recs),
            "n_recent_records": len(recent_recs),
        }
        summary = (
            f"Output-head distribution drift={avg_drift:.4f} "
            f"(threshold={self._cfg.reasoning_drift_threshold}) "
            f"across {len(shared_heads)} heads"
        )
        return EvaluationResult(
            dimension=dim, triggered=triggered, score=round(score, 4),
            evidence=evidence, recommended_scope=RecommendedScope.HEAD_RETRAIN,
            urgency=urgency, summary=summary,
        )


# ── 4. Regime Underperformance ────────────────────────────────────────────────


class RegimeUnderperformEvaluator:
    """Computes per-regime Sharpe from trade telemetry and triggers when any
    regime falls below the configured Sharpe threshold."""

    def __init__(self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig) -> None:
        self._wh = warehouse
        self._cfg = cfg

    def evaluate(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> EvaluationResult:
        q = TelemetryQuery(model_version=model_version, since=since, until=until,
                           limit=self._cfg.baseline_window + self._cfg.rolling_window)
        trades = self._wh.query_trades(q)

        dim = EvaluationDimension.REGIME_UNDERPERFORMANCE
        _not_triggered = EvaluationResult(
            dimension=dim, triggered=False, score=0.0, evidence={},
            recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=RecommendationUrgency.LOW,
            summary="Insufficient trade data for regime analysis",
        )
        if len(trades) < self._cfg.min_trades_for_evaluation:
            return _not_triggered

        # Group by regime
        regime_returns: dict[str, list[float]] = {}
        for t in trades:
            regime = t.get("regime", "unknown") or "unknown"
            pnl_pct = t.get("pnl_pct", 0.0)
            regime_returns.setdefault(regime, []).append(pnl_pct)

        per_regime_sharpe: dict[str, float] = {}
        for regime, returns in regime_returns.items():
            if len(returns) >= 2:
                per_regime_sharpe[regime] = round(_sharpe(returns), 4)

        underperforming = {
            r: s for r, s in per_regime_sharpe.items()
            if s < self._cfg.regime_sharpe_threshold
        }
        triggered = bool(underperforming)

        if triggered:
            # Score = how far below threshold the worst regime is (normalised to 0.1)
            worst_sharpe = min(underperforming.values())
            excess = self._cfg.regime_sharpe_threshold - worst_sharpe
            score = excess / 0.1  # 0.1 Sharpe unit per score point
        else:
            score = 0.0

        urgency = _urgency_from_score(score) if triggered else RecommendationUrgency.LOW
        evidence = {
            "per_regime_sharpe": per_regime_sharpe,
            "underperforming_regimes": underperforming,
            "regime_sharpe_threshold": self._cfg.regime_sharpe_threshold,
            "n_regimes": len(per_regime_sharpe),
            "n_trades_total": len(trades),
        }
        summary = (
            f"{len(underperforming)} of {len(per_regime_sharpe)} regimes below "
            f"Sharpe threshold ({self._cfg.regime_sharpe_threshold}): "
            f"{list(underperforming.keys())}"
        )
        return EvaluationResult(
            dimension=dim, triggered=triggered, score=round(score, 4),
            evidence=evidence, recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=urgency, summary=summary,
        )


# ── 5. Portfolio Inefficiency ─────────────────────────────────────────────────


class PortfolioInefficiencyEvaluator:
    """Detects portfolio inefficiency via symbol concentration (Herfindahl-style
    index) and scale-in success rate from position telemetry."""

    def __init__(self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig) -> None:
        self._wh = warehouse
        self._cfg = cfg

    def evaluate(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> EvaluationResult:
        total = self._cfg.baseline_window + self._cfg.rolling_window
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=total)
        positions = self._wh.query_positions(q)
        trades = self._wh.query_trades(q)

        dim = EvaluationDimension.PORTFOLIO_INEFFICIENCY
        _not_triggered = EvaluationResult(
            dimension=dim, triggered=False, score=0.0, evidence={},
            recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=RecommendationUrgency.LOW,
            summary="Insufficient portfolio data",
        )
        if not positions and not trades:
            return _not_triggered

        # ── Symbol concentration ──────────────────────────────────────────────
        symbol_counts: dict[str, int] = {}
        for t in trades:
            sym = t.get("symbol", "unknown")
            symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

        n_total = sum(symbol_counts.values())
        herfindahl = 0.0
        dominant_symbol = ""
        dominant_pct = 0.0
        if n_total > 0:
            shares = {s: c / n_total for s, c in symbol_counts.items()}
            herfindahl = sum(v ** 2 for v in shares.values())
            dominant_symbol = max(shares, key=lambda s: shares[s])
            dominant_pct = round(shares[dominant_symbol], 4)

        # ── Scale-in success rate ─────────────────────────────────────────────
        scale_ins = [p for p in positions if p.get("action") == "SCALE_IN"]
        scale_in_wins = [p for p in scale_ins if p.get("cumulative_pnl", 0) > 0]
        scale_in_rate = len(scale_in_wins) / len(scale_ins) if scale_ins else 1.0

        # ── Trigger logic ─────────────────────────────────────────────────────
        # HHI above threshold OR scale-in success rate below 40%
        hhi_score = herfindahl / self._cfg.portfolio_concentration_threshold
        scale_in_score = max(0.0, (0.4 - scale_in_rate) / 0.4)
        score = max(hhi_score, scale_in_score)
        triggered = (
            herfindahl > self._cfg.portfolio_concentration_threshold
            or scale_in_rate < 0.4
        )

        urgency = _urgency_from_score(score) if triggered else RecommendationUrgency.LOW
        evidence = {
            "herfindahl_index": round(herfindahl, 4),
            "concentration_threshold": self._cfg.portfolio_concentration_threshold,
            "dominant_symbol": dominant_symbol,
            "dominant_symbol_pct": dominant_pct,
            "n_symbols": len(symbol_counts),
            "scale_in_success_rate": round(scale_in_rate, 4),
            "n_scale_ins": len(scale_ins),
            "n_trades": n_total,
        }
        summary = (
            f"HHI={herfindahl:.3f} (threshold={self._cfg.portfolio_concentration_threshold}), "
            f"scale-in success rate={scale_in_rate:.2%}"
        )
        return EvaluationResult(
            dimension=dim, triggered=triggered, score=round(score, 4),
            evidence=evidence, recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=urgency, summary=summary,
        )
