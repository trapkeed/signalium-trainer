"""S20 — FeatureDiscoveryPipeline: discovers candidate input features from telemetry.

Computes six candidate feature signals from the TelemetryWarehouse, scores
evidence strength, and persists DiscoveredFeature records.  All computation
is done on already-ingested telemetry — no live market data access required.

Discovered features are in CANDIDATE status and require admin acceptance
before they affect the training pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.research_lab.discovery_models import DiscoveryQuery, DiscoverySummary
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.rl_models import DiscoveredFeature
from trainer.research_lab.telemetry_models import TelemetryQuery
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

# ── Signal computation helpers ─────────────────────────────────────────────────


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient; returns 0 if undefined."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    """Slope of the OLS regression line; returns 0 if undefined."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = sum((x - mx) ** 2 for x in xs)
    return num / denom if denom != 0 else 0.0


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _evidence_strength(sample_count: int, signal_magnitude: float, min_samples: int) -> float:
    """Combine sample count and signal magnitude into a [0, 1] evidence score."""
    count_factor = _clamp(sample_count / max(1, min_samples * 3))
    mag_factor = _clamp(abs(signal_magnitude))
    return round(_clamp(0.5 * count_factor + 0.5 * mag_factor), 4)


def _importance(evidence: float, signal_magnitude: float) -> float:
    """Map evidence + magnitude onto an importance score [0, 1]."""
    return round(_clamp(evidence * 0.6 + _clamp(abs(signal_magnitude)) * 0.4), 4)


# ── Feature definitions ────────────────────────────────────────────────────────


@dataclass
class _FeatureCandidate:
    name: str
    category: str
    formula_description: str
    suggested_qde_role: str
    importance_score: float
    evidence_strength: float
    sample_count: int
    notes: str


class FeatureDiscoveryPipeline:
    """Computes six candidate feature signals from telemetry and persists them.

    Features:
        1. confidence_slope            — trend in QDE confidence over time per symbol
        2. session_decision_density    — decisions per unit time (urgency proxy)
        3. loss_run_length             — max consecutive losing trades per symbol
        4. entry_confidence_outcome_correlation — correlation between confidence and PnL
        5. reversal_rate_by_session    — quick-loss fraction as a proxy for session fatigue
        6. scale_in_success_rate       — SCALE_IN events resulting in positive cumulative PnL
    """

    def __init__(self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase) -> None:
        self._wh = warehouse
        self._db = db

    # ── Public API ─────────────────────────────────────────────────────────────

    def discover(self, query: DiscoveryQuery, summary: DiscoverySummary) -> None:
        """Run all feature detectors and persist qualifying candidates."""
        tq = TelemetryQuery(
            model_version=query.model_version,
            symbol=query.symbol,
            since=query.since,
            until=query.until,
            limit=query.limit_per_stream,
        )
        decisions = self._wh.query_decisions(tq)
        trades = self._wh.query_trades(tq)
        positions = self._wh.query_positions(tq)

        candidates: list[_FeatureCandidate] = []
        detectors = [
            self._confidence_slope,
            self._session_decision_density,
            self._loss_run_length,
            self._entry_confidence_outcome_correlation,
            self._reversal_rate,
            self._scale_in_success_rate,
        ]
        for detector in detectors:
            try:
                result = detector(decisions, trades, positions, query)
                if result is not None and result.sample_count >= query.min_feature_samples:
                    candidates.append(result)
            except Exception as exc:
                msg = f"FeatureDiscoveryPipeline.{detector.__name__}: {exc}"
                logger.warning(msg)
                summary.errors.append(msg)

        # Persist (skip if a feature with the same name already exists)
        existing_names = {f.get("name") for f in self._db.list_features()}
        for cand in candidates:
            if cand.name in existing_names:
                continue
            feat = DiscoveredFeature.new(
                name=cand.name,
                category=cand.category,
                importance_score=cand.importance_score,
                notes=cand.notes,
                formula_description=cand.formula_description,
                evidence_strength=cand.evidence_strength,
                suggested_qde_role=cand.suggested_qde_role,
                sample_count=cand.sample_count,
            )
            self._db.upsert_feature(feat.to_dict())
            summary.feature_ids.append(feat.feature_id)
            summary.features_discovered += 1
            existing_names.add(cand.name)
            logger.info(
                f"FeatureDiscovery: persisted '{feat.name}' "
                f"importance={feat.importance_score:.3f} evidence={feat.evidence_strength:.3f}"
            )

    # ── Detectors ──────────────────────────────────────────────────────────────

    def _confidence_slope(
        self,
        decisions: list[dict],
        trades: list[dict],
        positions: list[dict],
        query: DiscoveryQuery,
    ) -> _FeatureCandidate | None:
        """Linear slope of QDE confidence values per symbol over time."""
        if not decisions:
            return None
        sorted_dec = sorted(decisions, key=lambda d: d.get("timestamp", 0.0))
        ts = [d.get("timestamp", 0.0) for d in sorted_dec]
        conf = [d.get("confidence", 0.5) for d in sorted_dec]
        slope = _linear_slope(ts, conf)
        n = len(ts)
        ev = _evidence_strength(n, slope * 3600, min_samples=10)  # slope per hour
        return _FeatureCandidate(
            name="confidence_slope_per_hour",
            category="telemetry_signal",
            formula_description=(
                "OLS slope of QDE confidence values over time, normalised to per-hour rate. "
                "Formula: Δconfidence / Δtime (hours) via linear regression over all decisions."
            ),
            suggested_qde_role="confidence_calibration_input",
            importance_score=_importance(ev, slope * 3600),
            evidence_strength=ev,
            sample_count=n,
            notes=(
                f"Slope={slope * 3600:.4f}/hr over {n} decisions. "
                "Negative slope signals eroding model conviction over a session."
            ),
        )

    def _session_decision_density(
        self,
        decisions: list[dict],
        trades: list[dict],
        positions: list[dict],
        query: DiscoveryQuery,
    ) -> _FeatureCandidate | None:
        """Decisions per hour — a proxy for market urgency / session intensity."""
        if len(decisions) < 2:
            return None
        ts = [d.get("timestamp", 0.0) for d in decisions]
        span_h = (max(ts) - min(ts)) / 3600.0
        if span_h < 0.001:
            return None
        density = len(decisions) / span_h
        ev = _evidence_strength(len(decisions), density / 100.0, min_samples=10)
        return _FeatureCandidate(
            name="session_decision_density_per_hour",
            category="telemetry_signal",
            formula_description=(
                "Count of QDE decisions divided by elapsed session time in hours. "
                "Formula: count(decisions) / (max_ts - min_ts) in hours."
            ),
            suggested_qde_role="session_activity_context",
            importance_score=_importance(ev, density / 100.0),
            evidence_strength=ev,
            sample_count=len(decisions),
            notes=(
                f"Density={density:.1f} decisions/hr across "
                f"{span_h:.1f}h of session data. "
                "High density correlates with volatile market conditions."
            ),
        )

    def _loss_run_length(
        self,
        decisions: list[dict],
        trades: list[dict],
        positions: list[dict],
        query: DiscoveryQuery,
    ) -> _FeatureCandidate | None:
        """Maximum consecutive losing trade run per symbol."""
        if not trades:
            return None
        sorted_trades = sorted(trades, key=lambda t: t.get("timestamp", 0.0))
        max_run = 0
        cur_run = 0
        for t in sorted_trades:
            if t.get("pnl", 0.0) < 0:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        if max_run == 0:
            return None
        n = len(sorted_trades)
        ev = _evidence_strength(n, max_run / 10.0, min_samples=5)
        return _FeatureCandidate(
            name="max_consecutive_loss_run",
            category="telemetry_signal",
            formula_description=(
                "Maximum number of consecutive losing trades (pnl < 0) "
                "in the observed trade history. "
                "Formula: max streak length of pnl < 0 trades sorted by timestamp."
            ),
            suggested_qde_role="adversity_detection_input",
            importance_score=_importance(ev, max_run / 10.0),
            evidence_strength=ev,
            sample_count=n,
            notes=(
                f"Observed max run={max_run} consecutive losses "
                f"over {n} total trades. "
                "Long loss runs signal regime mismatch or miscalibration."
            ),
        )

    def _entry_confidence_outcome_correlation(
        self,
        decisions: list[dict],
        trades: list[dict],
        positions: list[dict],
        query: DiscoveryQuery,
    ) -> _FeatureCandidate | None:
        """Pearson correlation between entry confidence and realised PnL %."""
        if len(trades) < 5:
            return None
        conf_vals = [t.get("confidence_at_entry", 0.5) for t in trades]
        pnl_vals = [t.get("pnl_pct", 0.0) for t in trades]
        r = _pearson(conf_vals, pnl_vals)
        ev = _evidence_strength(len(trades), abs(r), min_samples=10)
        return _FeatureCandidate(
            name="entry_confidence_pnl_correlation",
            category="calibration_signal",
            formula_description=(
                "Pearson correlation between confidence_at_entry and realised pnl_pct "
                "across all observed trades. "
                "Formula: pearson(confidence_at_entry, pnl_pct) over trade history."
            ),
            suggested_qde_role="confidence_calibration_target",
            importance_score=_importance(ev, abs(r)),
            evidence_strength=ev,
            sample_count=len(trades),
            notes=(
                f"r={r:.3f} over {len(trades)} trades. "
                "Values near 0 indicate miscalibrated confidence; "
                "values near 1 indicate well-calibrated confidence."
            ),
        )

    def _reversal_rate(
        self,
        decisions: list[dict],
        trades: list[dict],
        positions: list[dict],
        query: DiscoveryQuery,
    ) -> _FeatureCandidate | None:
        """Fraction of trades that are quick losses (hold < 5 min, pnl < 0)."""
        if not trades:
            return None
        quick_loss_threshold_s = 300.0
        quick_losses = sum(
            1
            for t in trades
            if t.get("pnl", 0.0) < 0 and t.get("hold_duration_s", 99999) < quick_loss_threshold_s
        )
        rate = quick_losses / len(trades)
        ev = _evidence_strength(len(trades), rate, min_samples=5)
        return _FeatureCandidate(
            name="quick_loss_reversal_rate",
            category="session_signal",
            formula_description=(
                "Fraction of all trades that are closed within 300s at a loss. "
                "Formula: count(hold_s < 300 AND pnl < 0) / count(trades)."
            ),
            suggested_qde_role="session_fatigue_indicator",
            importance_score=_importance(ev, rate),
            evidence_strength=ev,
            sample_count=len(trades),
            notes=(
                f"Rate={rate:.3f} ({quick_losses}/{len(trades)}) quick-loss trades. "
                "High rates signal the model is fighting the short-term trend."
            ),
        )

    def _scale_in_success_rate(
        self,
        decisions: list[dict],
        trades: list[dict],
        positions: list[dict],
        query: DiscoveryQuery,
    ) -> _FeatureCandidate | None:
        """Fraction of SCALE_IN events where cumulative_pnl > 0."""
        scale_ins = [p for p in positions if str(p.get("action", "")).upper() == "SCALE_IN"]
        if not scale_ins:
            return None
        successful = sum(1 for p in scale_ins if p.get("cumulative_pnl", 0.0) > 0)
        rate = successful / len(scale_ins)
        ev = _evidence_strength(len(scale_ins), abs(rate - 0.5) * 2, min_samples=5)
        return _FeatureCandidate(
            name="scale_in_success_rate",
            category="position_signal",
            formula_description=(
                "Fraction of SCALE_IN position events where cumulative_pnl > 0 at the "
                "time of the event. "
                "Formula: count(SCALE_IN AND cumulative_pnl > 0) / count(SCALE_IN)."
            ),
            suggested_qde_role="scale_in_timing_signal",
            importance_score=_importance(ev, abs(rate - 0.5) * 2),
            evidence_strength=ev,
            sample_count=len(scale_ins),
            notes=(
                f"Success rate={rate:.3f} ({successful}/{len(scale_ins)}) SCALE_IN events. "
                "Rates below 0.5 indicate systematic scale-in timing errors."
            ),
        )
