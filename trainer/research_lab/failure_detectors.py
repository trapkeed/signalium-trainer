"""S20 — Failure detectors.

One detector class per InvestigationFailureKind. Each detector:
  - Accepts a TelemetryWarehouse and InvestigationConfig.
  - Implements detect(query) → list[FailureReport].
  - Is pure/side-effect-free (reads only; no DB writes).

The FailureInvestigationEngine in failure_investigation_engine.py orchestrates all
detectors and persists the results.
"""

from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.research_lab.investigation_models import (
    EvidenceItem,
    FailureReport,
    InvestigationConfig,
    InvestigationFailureKind,
    InvestigationQuery,
)
from trainer.research_lab.telemetry_models import TelemetryQuery
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

# ── Base ──────────────────────────────────────────────────────────────────────


class BaseDetector(ABC):
    kind: InvestigationFailureKind

    def __init__(self, warehouse: TelemetryWarehouse, config: InvestigationConfig) -> None:
        self._wh = warehouse
        self._cfg = config

    @abstractmethod
    def detect(self, query: InvestigationQuery) -> list[FailureReport]:
        """Run detection and return zero or more FailureReport objects."""
        ...

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tq(self, query: InvestigationQuery) -> TelemetryQuery:
        """Convert InvestigationQuery → TelemetryQuery for warehouse reads."""
        return TelemetryQuery(
            model_version=query.model_version,
            symbol=query.symbol,
            since=query.since,
            until=query.until,
            limit=query.limit_per_stream,
        )

    @staticmethod
    def _clamp_severity(raw: float) -> float:
        return min(1.0, max(0.0, raw))

    @staticmethod
    def _symbols(events: list[dict[str, Any]]) -> list[str]:
        return sorted(set(e.get("symbol", "") for e in events))

    @staticmethod
    def _time_range(events: list[dict[str, Any]]) -> tuple[float, float]:
        if not events:
            now = time.time()
            return now, now
        ts = [e.get("timestamp", 0.0) for e in events]
        return min(ts), max(ts)

    @staticmethod
    def _mv(events: list[dict[str, Any]], query: InvestigationQuery) -> str:
        if query.model_version:
            return query.model_version
        versions = [e.get("model_version", "") for e in events if e.get("model_version")]
        return versions[0] if versions else ""


# ── Detector 1: HIGH_CONFIDENCE_LOSS ─────────────────────────────────────────


class HighConfidenceLossDetector(BaseDetector):
    """Finds trades where confidence_at_entry was high but PnL was negative.

    Groups by symbol so each symbol with a pattern gets its own report.
    One aggregated report per qualifying symbol.
    """

    kind = InvestigationFailureKind.HIGH_CONFIDENCE_LOSS

    def detect(self, query: InvestigationQuery) -> list[FailureReport]:
        trades = self._wh.query_trades(self._tq(query))
        high_conf_losses: dict[str, list[dict[str, Any]]] = {}

        for t in trades:
            conf = t.get("confidence_at_entry", 0.0) or 0.0
            pnl = t.get("pnl", 0.0) or 0.0
            if conf >= self._cfg.high_confidence_threshold and pnl < 0:
                sym = t.get("symbol", "UNKNOWN")
                high_conf_losses.setdefault(sym, []).append(t)

        reports: list[FailureReport] = []
        for sym, bad_trades in high_conf_losses.items():
            if len(bad_trades) < self._cfg.min_high_confidence_losses:
                continue

            total_loss = sum(t.get("pnl", 0.0) or 0.0 for t in bad_trades)
            avg_conf = statistics.mean(t.get("confidence_at_entry", 0.0) or 0.0 for t in bad_trades)
            # severity: blend of loss magnitude (capped at 10% account loss proxy)
            # and count relative to min threshold
            count_factor = min(1.0, len(bad_trades) / (self._cfg.min_high_confidence_losses * 5))
            severity = self._clamp_severity(
                0.5 * count_factor + 0.5 * min(1.0, abs(total_loss) / 1000.0)
            )

            evidence = [
                EvidenceItem(
                    event_id=t.get("event_id", ""),
                    symbol=sym,
                    timestamp=t.get("timestamp", 0.0),
                    metric_value=t.get("pnl", 0.0) or 0.0,
                    metric_label="pnl",
                    raw=t,
                )
                for t in sorted(bad_trades, key=lambda x: x.get("pnl", 0.0))[:20]
            ]

            t_start, t_end = self._time_range(bad_trades)
            summary = (
                f"{sym}: {len(bad_trades)} high-confidence losses "
                f"(avg confidence {avg_conf:.0%}, total PnL {total_loss:.2f}). "
                f"Model was overconfident on {sym} — consider confidence calibration."
            )
            reports.append(
                FailureReport.new(
                    failure_kind=self.kind,
                    model_version=self._mv(bad_trades, query),
                    affected_symbols=[sym],
                    time_range_start=t_start,
                    time_range_end=t_end,
                    evidence=evidence,
                    severity_score=severity,
                    summary=summary,
                    metadata={
                        "count": len(bad_trades),
                        "total_loss": round(total_loss, 4),
                        "avg_confidence": round(avg_conf, 4),
                        "threshold_used": self._cfg.high_confidence_threshold,
                    },
                )
            )

        return reports


# ── Detector 2: UNEXPECTED_REVERSAL ──────────────────────────────────────────


class UnexpectedReversalDetector(BaseDetector):
    """Finds trades that closed at a loss within a very short hold window.

    A 'fast reversal' indicates the model entered on wrong direction, or
    poor entry timing (e.g. during spread widening or news spike).
    """

    kind = InvestigationFailureKind.UNEXPECTED_REVERSAL

    def detect(self, query: InvestigationQuery) -> list[FailureReport]:
        trades = self._wh.query_trades(self._tq(query))
        reversals: dict[str, list[dict[str, Any]]] = {}

        for t in trades:
            hold = t.get("hold_duration_s", 0.0) or 0.0
            pnl = t.get("pnl", 0.0) or 0.0
            if hold <= self._cfg.reversal_max_hold_s and pnl < 0:
                sym = t.get("symbol", "UNKNOWN")
                reversals.setdefault(sym, []).append(t)

        reports: list[FailureReport] = []
        for sym, rev_trades in reversals.items():
            if len(rev_trades) < self._cfg.min_reversals:
                continue

            total_loss = sum(t.get("pnl", 0.0) or 0.0 for t in rev_trades)
            avg_hold = statistics.mean(t.get("hold_duration_s", 0.0) or 0.0 for t in rev_trades)
            count_factor = min(1.0, len(rev_trades) / (self._cfg.min_reversals * 5))
            severity = self._clamp_severity(
                0.6 * count_factor + 0.4 * min(1.0, abs(total_loss) / 500.0)
            )

            evidence = [
                EvidenceItem(
                    event_id=t.get("event_id", ""),
                    symbol=sym,
                    timestamp=t.get("timestamp", 0.0),
                    metric_value=t.get("hold_duration_s", 0.0) or 0.0,
                    metric_label="hold_duration_s",
                    raw=t,
                )
                for t in sorted(rev_trades, key=lambda x: x.get("hold_duration_s", 0.0))[:20]
            ]

            t_start, t_end = self._time_range(rev_trades)
            summary = (
                f"{sym}: {len(rev_trades)} trades reversed within "
                f"{self._cfg.reversal_max_hold_s:.0f}s (avg hold {avg_hold:.0f}s, "
                f"total PnL {total_loss:.2f}). "
                f"Entry timing or direction quality may be degraded on {sym}."
            )
            reports.append(
                FailureReport.new(
                    failure_kind=self.kind,
                    model_version=self._mv(rev_trades, query),
                    affected_symbols=[sym],
                    time_range_start=t_start,
                    time_range_end=t_end,
                    evidence=evidence,
                    severity_score=severity,
                    summary=summary,
                    metadata={
                        "count": len(rev_trades),
                        "total_loss": round(total_loss, 4),
                        "avg_hold_s": round(avg_hold, 1),
                        "max_hold_threshold_s": self._cfg.reversal_max_hold_s,
                    },
                )
            )

        return reports


# ── Detector 3: LATE_EXIT ─────────────────────────────────────────────────────


class LateExitDetector(BaseDetector):
    """Detects trades where counterfactual analysis shows exiting earlier was better.

    Uses tel_counterfactuals: cases where actual_pnl < counterfactual_pnl
    and the pnl_delta is meaningful (> late_exit_min_missed_pnl).

    Interpretation: the model let profits run too long and gave back gains.
    """

    kind = InvestigationFailureKind.LATE_EXIT

    def detect(self, query: InvestigationQuery) -> list[FailureReport]:
        cfs = self._wh.query_counterfactuals(self._tq(query))
        late: dict[str, list[dict[str, Any]]] = {}

        for cf in cfs:
            actual = cf.get("actual_pnl", 0.0) or 0.0
            cfactual = cf.get("counterfactual_pnl", 0.0) or 0.0
            delta = cf.get("pnl_delta", cfactual - actual)
            if delta > self._cfg.late_exit_min_missed_pnl and cfactual > actual:
                sym = cf.get("symbol", "UNKNOWN")
                late.setdefault(sym, []).append(cf)

        reports: list[FailureReport] = []
        for sym, events in late.items():
            if len(events) < self._cfg.min_counterfactual_events:
                continue

            total_missed = sum(
                (
                    e.get("pnl_delta")
                    or (e.get("counterfactual_pnl", 0.0) - e.get("actual_pnl", 0.0))
                )
                for e in events
            )
            count_factor = min(1.0, len(events) / (self._cfg.min_counterfactual_events * 5))
            severity = self._clamp_severity(
                0.5 * count_factor + 0.5 * min(1.0, total_missed / 500.0)
            )

            evidence = [
                EvidenceItem(
                    event_id=e.get("event_id", ""),
                    symbol=sym,
                    timestamp=e.get("timestamp", 0.0),
                    metric_value=(
                        e.get("pnl_delta")
                        or (e.get("counterfactual_pnl", 0.0) - e.get("actual_pnl", 0.0))
                    ),
                    metric_label="pnl_delta (missed by exiting late)",
                    raw=e,
                )
                for e in sorted(
                    events,
                    key=lambda x: (x.get("pnl_delta") or 0.0),
                    reverse=True,
                )[:20]
            ]

            t_start, t_end = self._time_range(events)
            summary = (
                f"{sym}: {len(events)} counterfactuals show late exit cost "
                f"{total_missed:.2f} PnL. Model held positions past their optimal exit — "
                f"review TP distance and trailing stop calibration for {sym}."
            )
            reports.append(
                FailureReport.new(
                    failure_kind=self.kind,
                    model_version=self._mv(events, query),
                    affected_symbols=[sym],
                    time_range_start=t_start,
                    time_range_end=t_end,
                    evidence=evidence,
                    severity_score=severity,
                    summary=summary,
                    metadata={"count": len(events), "total_missed_pnl": round(total_missed, 4)},
                )
            )

        return reports


# ── Detector 4: PREMATURE_EXIT ────────────────────────────────────────────────


class PrematureExitDetector(BaseDetector):
    """Detects trades that closed before capturing available profit.

    Uses tel_counterfactuals: cases where counterfactual_pnl >= actual_pnl * ratio.
    Interpretation: model cut winners too early.
    """

    kind = InvestigationFailureKind.PREMATURE_EXIT

    def detect(self, query: InvestigationQuery) -> list[FailureReport]:
        cfs = self._wh.query_counterfactuals(self._tq(query))
        premature: dict[str, list[dict[str, Any]]] = {}

        for cf in cfs:
            actual = cf.get("actual_pnl", 0.0) or 0.0
            cfactual = cf.get("counterfactual_pnl", 0.0) or 0.0
            if actual > 0 and cfactual >= actual * self._cfg.premature_exit_ratio:
                sym = cf.get("symbol", "UNKNOWN")
                premature.setdefault(sym, []).append(cf)

        reports: list[FailureReport] = []
        for sym, events in premature.items():
            if len(events) < self._cfg.min_counterfactual_events:
                continue

            total_left_behind = sum(
                max(
                    0.0,
                    (e.get("counterfactual_pnl", 0.0) or 0.0) - (e.get("actual_pnl", 0.0) or 0.0),
                )
                for e in events
            )
            count_factor = min(1.0, len(events) / (self._cfg.min_counterfactual_events * 5))
            severity = self._clamp_severity(
                0.5 * count_factor + 0.5 * min(1.0, total_left_behind / 500.0)
            )

            evidence = [
                EvidenceItem(
                    event_id=e.get("event_id", ""),
                    symbol=sym,
                    timestamp=e.get("timestamp", 0.0),
                    metric_value=(e.get("counterfactual_pnl", 0.0) or 0.0)
                    / max(0.01, e.get("actual_pnl", 0.01) or 0.01),
                    metric_label="counterfactual/actual ratio",
                    raw=e,
                )
                for e in sorted(
                    events,
                    key=lambda x: (x.get("counterfactual_pnl", 0.0) or 0.0)
                    / max(0.01, x.get("actual_pnl", 0.01) or 0.01),
                    reverse=True,
                )[:20]
            ]

            t_start, t_end = self._time_range(events)
            summary = (
                f"{sym}: {len(events)} trades exited prematurely, leaving "
                f"{total_left_behind:.2f} PnL uncaptured (ratio threshold "
                f"{self._cfg.premature_exit_ratio:.1f}x). "
                f"Review TP targets and exit signals for {sym}."
            )
            reports.append(
                FailureReport.new(
                    failure_kind=self.kind,
                    model_version=self._mv(events, query),
                    affected_symbols=[sym],
                    time_range_start=t_start,
                    time_range_end=t_end,
                    evidence=evidence,
                    severity_score=severity,
                    summary=summary,
                    metadata={
                        "count": len(events),
                        "total_left_behind": round(total_left_behind, 4),
                        "ratio_threshold": self._cfg.premature_exit_ratio,
                    },
                )
            )

        return reports


# ── Detector 5: FAILED_SCALE_IN ───────────────────────────────────────────────


class FailedScaleInDetector(BaseDetector):
    """Detects scale-in position events that added to already-losing positions.

    A SCALE_IN event where cumulative_pnl at the time of scale-in is negative
    indicates the model averaged into a loser, a common trap.
    """

    kind = InvestigationFailureKind.FAILED_SCALE_IN

    def detect(self, query: InvestigationQuery) -> list[FailureReport]:
        positions = self._wh.query_positions(self._tq(query))
        failed: dict[str, list[dict[str, Any]]] = {}

        for pos in positions:
            if pos.get("action") != "SCALE_IN":
                continue
            cum_pnl = pos.get("cumulative_pnl", 0.0) or 0.0
            if cum_pnl < 0:
                sym = pos.get("symbol", "UNKNOWN")
                failed.setdefault(sym, []).append(pos)

        reports: list[FailureReport] = []
        for sym, events in failed.items():
            if len(events) < self._cfg.min_failed_scale_ins:
                continue

            worst_pnl = min(e.get("cumulative_pnl", 0.0) or 0.0 for e in events)
            count_factor = min(1.0, len(events) / (self._cfg.min_failed_scale_ins * 4))
            severity = self._clamp_severity(
                0.6 * count_factor + 0.4 * min(1.0, abs(worst_pnl) / 500.0)
            )

            evidence = [
                EvidenceItem(
                    event_id=e.get("event_id", ""),
                    symbol=sym,
                    timestamp=e.get("timestamp", 0.0),
                    metric_value=e.get("cumulative_pnl", 0.0) or 0.0,
                    metric_label="cumulative_pnl_at_scale_in",
                    raw=e,
                )
                for e in sorted(events, key=lambda x: x.get("cumulative_pnl", 0.0) or 0.0)[:20]
            ]

            t_start, t_end = self._time_range(events)
            summary = (
                f"{sym}: {len(events)} SCALE_IN events added to losing positions "
                f"(worst cumulative PnL at scale-in: {worst_pnl:.2f}). "
                f"Scale-in rules may be triggering on momentum in the wrong direction."
            )
            reports.append(
                FailureReport.new(
                    failure_kind=self.kind,
                    model_version=self._mv(events, query),
                    affected_symbols=[sym],
                    time_range_start=t_start,
                    time_range_end=t_end,
                    evidence=evidence,
                    severity_score=severity,
                    summary=summary,
                    metadata={"count": len(events), "worst_pnl": round(worst_pnl, 4)},
                )
            )

        return reports


# ── Detector 6: PORTFOLIO_CONCENTRATION ───────────────────────────────────────


class PortfolioConcentrationDetector(BaseDetector):
    """Flags when decisions cluster excessively on a single symbol in a time window.

    Reads tel_decisions and groups by symbol within rolling windows of
    config.concentration_window_s seconds. If any symbol dominates
    >= concentration_threshold of all decisions, a report is emitted.
    """

    kind = InvestigationFailureKind.PORTFOLIO_CONCENTRATION

    def detect(self, query: InvestigationQuery) -> list[FailureReport]:
        decisions = self._wh.query_decisions(self._tq(query))
        if not decisions:
            return []

        # Sort by timestamp then partition into non-overlapping fixed-width buckets.
        # Non-overlapping buckets give one clear report per episode (no sliding-window
        # over-reporting); the engine's interval-merge dedup handles any residual overlap
        # if the user runs investigate() multiple times over overlapping periods.
        decisions_sorted = sorted(decisions, key=lambda d: d.get("timestamp", 0.0))
        window = self._cfg.concentration_window_s

        t_first = decisions_sorted[0].get("timestamp", 0.0)

        # Group decisions into fixed-width buckets anchored to the first event timestamp
        buckets: dict[int, list[dict[str, Any]]] = {}
        for d in decisions_sorted:
            ts = d.get("timestamp", 0.0)
            bucket_idx = int((ts - t_first) / window)
            buckets.setdefault(bucket_idx, []).append(d)

        reports: list[FailureReport] = []
        for bucket_idx in sorted(buckets):
            bucket = buckets[bucket_idx]
            if len(bucket) < self._cfg.min_decisions_for_concentration:
                continue

            # Count per symbol in this bucket
            sym_counts: dict[str, int] = {}
            for d in bucket:
                s = d.get("symbol", "UNKNOWN")
                sym_counts[s] = sym_counts.get(s, 0) + 1

            total = len(bucket)
            t_start = bucket[0].get("timestamp", 0.0)
            t_end = bucket[-1].get("timestamp", 0.0)

            for sym, count in sym_counts.items():
                fraction = count / total
                if fraction < self._cfg.concentration_threshold:
                    continue

                count_factor = min(1.0, fraction)
                severity = self._clamp_severity(count_factor * 0.9)

                evidence = [
                    EvidenceItem(
                        event_id=d.get("event_id", ""),
                        symbol=sym,
                        timestamp=d.get("timestamp", 0.0),
                        metric_value=fraction,
                        metric_label="symbol_fraction_in_window",
                        raw=d,
                    )
                    for d in bucket
                    if d.get("symbol") == sym
                ][:20]

                mv = self._mv(bucket, query)
                summary = (
                    f"{sym}: {count}/{total} decisions ({fraction:.0%}) in a "
                    f"{window / 60:.0f}-minute window — portfolio is over-concentrated. "
                    f"Diversification guardrails or correlation filters may need tightening."
                )
                reports.append(
                    FailureReport.new(
                        failure_kind=self.kind,
                        model_version=mv,
                        affected_symbols=[sym],
                        time_range_start=t_start,
                        time_range_end=t_end,
                        evidence=evidence,
                        severity_score=severity,
                        summary=summary,
                        metadata={
                            "dominant_symbol": sym,
                            "symbol_count": count,
                            "total_decisions_in_window": total,
                            "concentration_fraction": round(fraction, 4),
                            "window_s": window,
                            "bucket_idx": bucket_idx,
                        },
                    )
                )

        return reports


# ── Registry of all detectors ─────────────────────────────────────────────────

ALL_DETECTOR_CLASSES: list[type[BaseDetector]] = [
    HighConfidenceLossDetector,
    UnexpectedReversalDetector,
    LateExitDetector,
    PrematureExitDetector,
    FailedScaleInDetector,
    PortfolioConcentrationDetector,
]
