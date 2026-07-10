"""QDE Brain — surprise engine.

Computes **surprise**: the magnitude of prediction error between what a
decision expected (`DecisionSnapshot` forecast fields, captured at decision
time) and what actually happened (`DecisionOutcome`, recorded once the trade
closes). This is a genuinely new concept — no other module in `trainer/`
computes a prediction-error / surprise signal; `SelfEvaluationEngine` and
`OutcomeTracker` report aggregate win-rate/PnL statistics across many
decisions, but never score how *unexpected* a single outcome was relative to
what was predicted for it.

Every input is read from `DecisionRecord`/`DecisionSnapshot`/`DecisionOutcome`
already produced elsewhere — this module never re-derives forecasts or
outcomes, it only compares them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainer.memory.decision_models import DecisionRecord

DEFAULT_HIGH_THRESHOLD = 0.6
DEFAULT_LOW_THRESHOLD = 0.3


class SurpriseLevel(str, Enum):
    EXPECTED = "EXPECTED"  # outcome tracked the forecast closely
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SHOCKING = "SHOCKING"  # outcome contradicted the forecast on multiple axes


@dataclass(frozen=True)
class SurpriseSignals:
    """Raw 0-1 contributors to the composite surprise score."""

    direction_surprise: float = 0.0  # forecast direction wrong given confidence
    magnitude_surprise: float = 0.0  # realized PnL far from expected move
    timing_surprise: float = 0.0  # holding time far from expected holding time
    outcome_type_surprise: float = 0.0  # e.g. UNEXPECTED_MARKET_EVENT, MISSED_OPPORTUNITY

    def to_dict(self) -> dict:
        return {
            "direction_surprise": round(self.direction_surprise, 4),
            "magnitude_surprise": round(self.magnitude_surprise, 4),
            "timing_surprise": round(self.timing_surprise, 4),
            "outcome_type_surprise": round(self.outcome_type_surprise, 4),
        }

    def max_signal(self) -> float:
        return max(
            self.direction_surprise,
            self.magnitude_surprise,
            self.timing_surprise,
            self.outcome_type_surprise,
        )


@dataclass
class SurpriseAssessment:
    """Surprise reading for one completed decision."""

    decision_id: str
    instrument: str
    magnitude: float  # composite 0-1
    level: SurpriseLevel
    signals: SurpriseSignals = field(default_factory=SurpriseSignals)
    explanation: str = ""
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "instrument": self.instrument,
            "magnitude": round(self.magnitude, 4),
            "level": self.level.value,
            "signals": self.signals.to_dict(),
            "explanation": self.explanation,
            "computed_at": round(self.computed_at, 3),
        }


_OUTCOME_TYPE_SURPRISE = {
    "UNEXPECTED_MARKET_EVENT": 1.0,
    "MISSED_OPPORTUNITY": 0.6,
    "PARTIAL_SUCCESS": 0.3,
}


def classify_level(
    magnitude: float,
    high: float = DEFAULT_HIGH_THRESHOLD,
    low: float = DEFAULT_LOW_THRESHOLD,
) -> SurpriseLevel:
    if magnitude >= high:
        return SurpriseLevel.SHOCKING if magnitude >= 0.85 else SurpriseLevel.HIGH
    if magnitude >= low:
        return SurpriseLevel.MODERATE
    return SurpriseLevel.EXPECTED


def compute_surprise(record: "DecisionRecord") -> SurpriseAssessment | None:
    """Compare a completed decision's forecast against its realized outcome.

    Returns `None` if the record has no outcome yet (nothing to compare).
    """
    outcome = record.outcome
    if outcome is None:
        return None

    snapshot = record.snapshot

    direction_surprise = 0.0
    if outcome.actual_direction_correct is False:
        # The higher the forecast confidence, the more surprising a wrong call.
        direction_surprise = max(0.0, snapshot.forecast_confidence)
    elif outcome.actual_direction_correct is None:
        direction_surprise = 0.0

    magnitude_surprise = 0.0
    expected_move = abs(snapshot.expected_move)
    if expected_move > 1e-9:
        actual_move = abs(outcome.actual_pnl)
        deviation = abs(actual_move - expected_move) / expected_move
        magnitude_surprise = min(1.0, deviation)
    elif abs(outcome.actual_pnl) > 1e-9:
        # Expected essentially no move but something happened.
        magnitude_surprise = 1.0

    timing_surprise = 0.0
    expected_holding = max(1e-9, snapshot.expected_holding_h)
    timing_deviation = abs(outcome.holding_time_h - expected_holding) / expected_holding
    timing_surprise = min(1.0, timing_deviation)

    outcome_type_surprise = _OUTCOME_TYPE_SURPRISE.get(outcome.outcome_type.value, 0.0)

    signals = SurpriseSignals(
        direction_surprise=direction_surprise,
        magnitude_surprise=magnitude_surprise,
        timing_surprise=timing_surprise,
        outcome_type_surprise=outcome_type_surprise,
    )

    # Weighted composite — direction being wrong dominates, since it's the
    # clearest signal the QDE's read of the setup was flat-out incorrect.
    magnitude = (
        0.45 * signals.direction_surprise
        + 0.25 * signals.magnitude_surprise
        + 0.10 * signals.timing_surprise
        + 0.20 * signals.outcome_type_surprise
    )
    magnitude = max(0.0, min(1.0, magnitude))

    level = classify_level(magnitude)

    parts = []
    if direction_surprise > 0:
        parts.append(f"forecast direction wrong at {snapshot.forecast_confidence:.0%} confidence")
    if magnitude_surprise >= 0.4:
        parts.append("realized move deviated sharply from the expected move")
    if timing_surprise >= 0.4:
        parts.append("holding time deviated sharply from expectation")
    if outcome_type_surprise > 0:
        parts.append(f"outcome type was {outcome.outcome_type.value}")
    explanation = (
        f"{record.instrument}: " + "; ".join(parts)
        if parts
        else f"{record.instrument}: outcome tracked the forecast closely"
    )

    return SurpriseAssessment(
        decision_id=record.decision_id,
        instrument=record.instrument,
        magnitude=magnitude,
        level=level,
        signals=signals,
        explanation=explanation,
    )
