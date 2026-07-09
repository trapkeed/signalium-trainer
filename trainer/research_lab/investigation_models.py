"""S20 — Failure Investigation models.

Separate from rl_models.py (which holds researcher-recorded FailureEvent) because
these are *automatically detected* patterns from telemetry, not manually recorded events.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Failure taxonomy ──────────────────────────────────────────────────────────


class InvestigationFailureKind(str, Enum):
    """The six categories of QDE failure the engine can detect automatically."""

    HIGH_CONFIDENCE_LOSS = "HIGH_CONFIDENCE_LOSS"
    """Trades entered above a confidence threshold that result in negative PnL."""

    UNEXPECTED_REVERSAL = "UNEXPECTED_REVERSAL"
    """Trades that closed at a loss within a very short hold window — the model
    was immediately wrong on direction."""

    LATE_EXIT = "LATE_EXIT"
    """Counterfactual evidence shows the model would have captured more PnL by
    exiting earlier than it did."""

    PREMATURE_EXIT = "PREMATURE_EXIT"
    """Counterfactual evidence shows the model left significant PnL on the table
    by exiting too early."""

    FAILED_SCALE_IN = "FAILED_SCALE_IN"
    """Scale-in position actions that worsened an already-losing position."""

    PORTFOLIO_CONCENTRATION = "PORTFOLIO_CONCENTRATION"
    """Excessive concentration of decisions on a single symbol within a time window,
    exposing the portfolio to concentrated directional risk."""


# ── Evidence items ────────────────────────────────────────────────────────────


@dataclass
class EvidenceItem:
    """A single supporting data point for a FailureReport.

    Contains whatever scalar fields are relevant to prove the detected pattern;
    the rest of the raw telemetry event is in `raw`.
    """

    event_id: str
    symbol: str
    timestamp: float
    metric_value: float          # The key scalar being flagged (e.g. pnl, pnl_delta)
    metric_label: str            # Human-readable label for metric_value
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "symbol": self.symbol,
            "timestamp": round(self.timestamp, 3),
            "metric_value": round(self.metric_value, 6),
            "metric_label": self.metric_label,
        }


# ── Failure report ────────────────────────────────────────────────────────────


@dataclass
class FailureReport:
    """Automatically detected failure pattern in QDE telemetry.

    A single report covers one failure kind across one or more symbols
    within a given time window for a given model version.
    """

    report_id: str
    failure_kind: InvestigationFailureKind
    model_version: str
    affected_symbols: list[str]
    time_range_start: float
    time_range_end: float
    evidence: list[EvidenceItem]
    severity_score: float         # 0.0 – 1.0; higher = more urgent
    summary: str                  # Natural-language explanation
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        failure_kind: InvestigationFailureKind,
        model_version: str,
        affected_symbols: list[str],
        time_range_start: float,
        time_range_end: float,
        evidence: list[EvidenceItem],
        severity_score: float,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> FailureReport:
        return cls(
            report_id=f"fr-{str(uuid.uuid4())[:12]}",
            failure_kind=failure_kind,
            model_version=model_version,
            affected_symbols=affected_symbols,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            evidence=evidence,
            severity_score=min(1.0, max(0.0, severity_score)),
            summary=summary,
            created_at=time.time(),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "failure_kind": self.failure_kind.value,
            "model_version": self.model_version,
            "affected_symbols": self.affected_symbols,
            "time_range_start": round(self.time_range_start, 3),
            "time_range_end": round(self.time_range_end, 3),
            "evidence_count": len(self.evidence),
            "evidence": [e.to_dict() for e in self.evidence],
            "severity_score": round(self.severity_score, 4),
            "summary": self.summary,
            "created_at": round(self.created_at, 3),
            "metadata": self.metadata,
        }


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class InvestigationConfig:
    """Thresholds used by the failure detectors.

    All defaults are conservative starting points; tune them once enough
    production telemetry has been ingested.
    """

    # HIGH_CONFIDENCE_LOSS
    high_confidence_threshold: float = 0.85
    """Trades with confidence_at_entry >= this are 'high confidence'."""
    min_high_confidence_losses: int = 3
    """Minimum number of high-confidence losers to trigger a report."""

    # UNEXPECTED_REVERSAL
    reversal_max_hold_s: float = 300.0
    """Trades closed within this many seconds of entry are 'fast reversals'."""
    min_reversals: int = 3
    """Minimum number of fast reversals to trigger a report."""

    # LATE_EXIT / PREMATURE_EXIT (via counterfactuals)
    late_exit_min_missed_pnl: float = 0.0
    """pnl_delta (counterfactual_pnl - actual_pnl) must exceed this to count."""
    premature_exit_ratio: float = 1.5
    """counterfactual_pnl must be >= actual_pnl * ratio to flag premature exit."""
    min_counterfactual_events: int = 2
    """Minimum events required to fire a counterfactual report."""

    # FAILED_SCALE_IN
    min_failed_scale_ins: int = 2
    """Minimum number of bad scale-in events to trigger a report."""

    # PORTFOLIO_CONCENTRATION
    concentration_window_s: float = 3600.0
    """Rolling window (seconds) to measure decision concentration."""
    concentration_threshold: float = 0.40
    """A symbol accounting for >= this fraction of decisions flags concentration."""
    min_decisions_for_concentration: int = 10
    """Minimum total decisions in window before checking concentration."""


# ── Query parameters ──────────────────────────────────────────────────────────


@dataclass
class InvestigationQuery:
    """Filters passed to FailureInvestigationEngine.investigate()."""

    model_version: str | None = None
    symbol: str | None = None
    since: float | None = None       # Unix timestamp
    until: float | None = None       # Unix timestamp; defaults to now
    failure_kinds: list[InvestigationFailureKind] | None = None
    """If set, run only the listed detectors."""
    limit_per_stream: int = 2000
    """Max telemetry rows fetched per stream per detector call."""
