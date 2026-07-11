"""C16 Counterfactual Consciousness — type definitions.

Goal: The QDE should continuously ask "What if I am wrong?"

This module defines all types for the counterfactual layer.  Nothing here
duplicates existing infrastructure:

    * ``Thesis`` and ``BeliefDirection`` are imported, not redefined.
    * ``ThesisEventType`` in ``thesis/thesis_types.py`` tracks single-thesis
      state transitions; the types here track *cross-thesis* events (flips
      between primary and secondary) — a categorically different concept.
    * ``TradeReflectionEngine`` does post-trade reflection; ``FlipLesson``
      does *pre-outcome* analysis of which ignored warnings preceded a flip.

Primary vs secondary thesis
-----------------------------
The counterfactual layer does not replace ``ThesisManager``'s
``bull_thesis`` / ``bear_thesis``.  Those are the evidence-weighted two
sides of every debate.  C16 adds a layer above: which side the QDE is
currently *committed* to (``primary_thesis``) and which side it is
continuously monitoring as a challenger (``secondary_thesis``).  A thesis
switch is when the challenger overtakes the primary with enough margin to
justify reorientation.

Five scenario types
--------------------
    CONTINUATION   — primary direction continues at the current pace.
    REVERSAL       — market reverses the primary direction.
    SIDEWAYS       — no clear directional resolution (consolidation).
    ACCELERATION   — primary direction but faster / more extreme.
    EXHAUSTION     — primary direction winding down toward inflection.

Three stored event types
-------------------------
    ThesisFlipRecord      — primary → secondary switch with trigger.
    ThesisReversalRecord  — quick flip-back (oscillation detection).
    UncertaintySpikeRecord — rapid jump in composite uncertainty.

FlipLesson
-----------
When a flip occurs, the engine analyses which items were present in the
primary thesis's ``conflicting_evidence`` *before* the flip happened.
Those items are the "ignored warnings".  This closes the loop on
assumption #5 (learn from incorrect assumptions).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from trainer.qde.brain.brain_types import BeliefDirection

if TYPE_CHECKING:
    pass  # avoid circular imports


# ── Enums ─────────────────────────────────────────────────────────────────────


class ScenarioType(str, Enum):
    """The kind of counterfactual market scenario."""

    CONTINUATION  = "CONTINUATION"   # primary direction continues
    REVERSAL      = "REVERSAL"        # primary direction reverses
    SIDEWAYS      = "SIDEWAYS"        # no clear direction, consolidation
    ACCELERATION  = "ACCELERATION"   # primary direction, faster/more extreme
    EXHAUSTION    = "EXHAUSTION"      # primary direction winding down


class FlipTrigger(str, Enum):
    """What caused a thesis flip / switch."""

    EVIDENCE_SHIFT       = "EVIDENCE_SHIFT"        # conflicting evidence piled up
    UNCERTAINTY_SPIKE    = "UNCERTAINTY_SPIKE"     # uncertainty jumped sharply
    CONFIDENCE_COLLAPSE  = "CONFIDENCE_COLLAPSE"   # primary confidence fell below floor
    SECONDARY_DOMINANCE  = "SECONDARY_DOMINANCE"   # secondary consistently stronger
    FORCED               = "FORCED"                # explicit caller override


# ── Thresholds ────────────────────────────────────────────────────────────────

# Margin (secondary_p - primary_p) required to recommend a switch
SWITCH_MARGIN: float = 0.15

# Primary confidence floor — below this the engine recommends switching
PRIMARY_CONFIDENCE_FLOOR: float = 0.35

# Uncertainty spike threshold (absolute delta in composite uncertainty)
UNCERTAINTY_SPIKE_DELTA: float = 0.20

# Reversal detection: a flip-back within this many seconds is a reversal
REVERSAL_WINDOW_SECONDS: float = 300.0   # 5 minutes

# Max items stored per event type
MAX_FLIP_RECORDS:       int = 100
MAX_REVERSAL_RECORDS:   int = 50
MAX_SPIKE_RECORDS:      int = 100
MAX_PROBABILITY_POINTS: int = 500

# Scenario generation weights (prior probability by type before evidence)
_SCENARIO_BASE_WEIGHTS: dict[ScenarioType, float] = {
    ScenarioType.CONTINUATION:  0.40,
    ScenarioType.REVERSAL:      0.25,
    ScenarioType.SIDEWAYS:      0.15,
    ScenarioType.ACCELERATION:  0.12,
    ScenarioType.EXHAUSTION:    0.08,
}


# ── Probability evolution ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbabilityPoint:
    """One timestamp sample in the probability evolution timeline.

    ``primary_p`` and ``secondary_p`` are derived from the respective
    ``Thesis.strength()`` scores at sampling time.  They are not
    independent estimates — they reflect the debate-engine evidence
    quality — but tracking them over time reveals directional drift,
    conviction reversals, and uncertainty spikes.
    """

    timestamp:    float
    primary_p:    float   # Thesis.strength() of the primary thesis
    secondary_p:  float   # Thesis.strength() of the secondary thesis
    uncertainty:  float   # composite uncertainty (0-1) at this moment

    @property
    def spread(self) -> float:
        """primary_p − secondary_p; positive = primary leads."""
        return round(self.primary_p - self.secondary_p, 4)

    def to_dict(self) -> dict:
        return {
            "timestamp":   round(self.timestamp, 3),
            "primary_p":   round(self.primary_p, 4),
            "secondary_p": round(self.secondary_p, 4),
            "uncertainty": round(self.uncertainty, 4),
            "spread":      round(self.spread, 4),
        }


# ── Counterfactual scenarios ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CounterfactualScenario:
    """One generated "what if" scenario.

    Produced by ``generate_alternatives()``.  Each scenario represents a
    plausible alternative market path that the QDE should monitor
    continuously — even (especially) while it is committed to a different
    primary thesis.

    ``source_evidence`` links this scenario back to which conflicting
    evidence items in the primary thesis seeded it, so that when
    ``evaluate_alternative()`` sees matching new evidence it can weight
    this scenario higher.
    """

    scenario_id:          str
    scenario_type:        ScenarioType
    direction:            BeliefDirection     # implied market direction if true
    trigger_conditions:   list[str]           # market signals confirming this scenario
    probability_weight:   float               # prior P(this scenario unfolds), 0-1
    expected_observations: list[str]          # signals to watch for as it develops
    invalidation_signals: list[str]           # signals that kill this scenario
    confidence_if_correct: float              # how certain to be if scenario triggers
    source_evidence:      list[str]           # seeded from these conflicting items
    created_at:           float

    def to_dict(self) -> dict:
        return {
            "scenario_id":           self.scenario_id,
            "scenario_type":         self.scenario_type.value,
            "direction":             self.direction.value,
            "trigger_conditions":    self.trigger_conditions,
            "probability_weight":    round(self.probability_weight, 4),
            "expected_observations": self.expected_observations,
            "invalidation_signals":  self.invalidation_signals,
            "confidence_if_correct": round(self.confidence_if_correct, 4),
            "source_evidence":       self.source_evidence,
            "created_at":            round(self.created_at, 3),
        }


# ── Flip lesson ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FlipLesson:
    """Derived insight when a thesis flip occurs.

    ``ignored_warnings`` lists the conflicting-evidence items that were
    present in the primary thesis *before* the flip — warnings that were
    available but not acted on.  ``time_to_flip`` measures how long those
    warnings had been present before the flip was finally triggered.
    ``could_have_switched_earlier`` is True when meaningful warnings were
    present for at least ``REVERSAL_WINDOW_SECONDS`` before the flip —
    indicating the QDE was slower than it should have been.
    """

    ignored_warnings:              list[str]   # conflicting evidence present before flip
    key_trigger:                   str         # the final signal that caused the flip
    time_to_flip:                  float       # seconds from earliest warning to flip
    could_have_switched_earlier:   bool        # True when warnings ≥ REVERSAL_WINDOW_SECONDS old

    def to_dict(self) -> dict:
        return {
            "ignored_warnings":            self.ignored_warnings,
            "key_trigger":                 self.key_trigger,
            "time_to_flip":                round(self.time_to_flip, 1),
            "could_have_switched_earlier": self.could_have_switched_earlier,
        }


# ── Event records ─────────────────────────────────────────────────────────────


@dataclass
class ThesisFlipRecord:
    """Recorded when the primary and secondary theses switch roles.

    ``primary_p_before`` / ``secondary_p_before`` capture the strength
    scores just before the switch so the evolution trajectory can be
    reconstructed.  ``evidence_snapshot`` is the slice of conflicting
    evidence that was present in the old primary thesis at flip time.
    """

    flip_id:               str
    previous_primary_side: BeliefDirection
    new_primary_side:      BeliefDirection
    trigger:               FlipTrigger
    trigger_detail:        str
    primary_p_before:      float
    secondary_p_before:    float
    evidence_snapshot:     list[str]
    flip_lesson:           FlipLesson | None
    flipped_at:            float

    def to_dict(self) -> dict:
        return {
            "flip_id":               self.flip_id,
            "previous_primary_side": self.previous_primary_side.value,
            "new_primary_side":      self.new_primary_side.value,
            "trigger":               self.trigger.value,
            "trigger_detail":        self.trigger_detail,
            "primary_p_before":      round(self.primary_p_before, 4),
            "secondary_p_before":    round(self.secondary_p_before, 4),
            "evidence_snapshot":     self.evidence_snapshot,
            "flip_lesson":           self.flip_lesson.to_dict() if self.flip_lesson else None,
            "flipped_at":            round(self.flipped_at, 3),
        }


@dataclass(frozen=True)
class ThesisReversalRecord:
    """Recorded when the QDE flips back to the previous primary thesis
    within ``REVERSAL_WINDOW_SECONDS``.

    A reversal signals that the flip may have been premature — the
    intermediate thesis did not hold long enough to constitute a genuine
    reorientation.  High reversal frequency is itself a behavioral
    pathology (oscillation / flip-flopping).
    """

    reversal_id:        str
    returning_to_side:  BeliefDirection   # the side we returned to
    intermediate_side:  BeliefDirection   # the side we briefly flipped to
    seconds_elapsed:    float             # how long the intermediate thesis lasted
    flip_id:            str               # the flip_id of the flip that is now reversed
    reversed_at:        float

    def to_dict(self) -> dict:
        return {
            "reversal_id":       self.reversal_id,
            "returning_to_side": self.returning_to_side.value,
            "intermediate_side": self.intermediate_side.value,
            "seconds_elapsed":   round(self.seconds_elapsed, 1),
            "flip_id":           self.flip_id,
            "reversed_at":       round(self.reversed_at, 3),
        }


@dataclass(frozen=True)
class UncertaintySpikeRecord:
    """Recorded when the composite uncertainty jumps sharply.

    ``delta`` = ``uncertainty_after`` − ``uncertainty_before``.
    Only positive spikes (uncertainty increasing) are recorded here;
    decreasing uncertainty is a recovery, not a pathology.
    """

    spike_id:          str
    instrument:        str
    uncertainty_before: float
    uncertainty_after:  float
    delta:             float
    level_after:       str    # UncertaintyLevel.value string
    context:           str    # brief description of what drove the spike
    spiked_at:         float

    def to_dict(self) -> dict:
        return {
            "spike_id":          self.spike_id,
            "instrument":        self.instrument,
            "uncertainty_before": round(self.uncertainty_before, 4),
            "uncertainty_after":  round(self.uncertainty_after, 4),
            "delta":             round(self.delta, 4),
            "level_after":       self.level_after,
            "context":           self.context,
            "spiked_at":         round(self.spiked_at, 3),
        }


# ── Switch result ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThesisSwitchResult:
    """Returned by ``switch_thesis()``.

    Bundles the flip record, newly generated scenarios for the new
    primary thesis, and whether the switch was detected as a reversal.
    """

    flip_record:    ThesisFlipRecord
    new_scenarios:  list[CounterfactualScenario]
    is_reversal:    bool
    reversal_record: ThesisReversalRecord | None

    def to_dict(self) -> dict:
        return {
            "flip_record":    self.flip_record.to_dict(),
            "new_scenarios":  [s.to_dict() for s in self.new_scenarios],
            "is_reversal":    self.is_reversal,
            "reversal_record": self.reversal_record.to_dict() if self.reversal_record else None,
        }


# ── Per-instrument state ──────────────────────────────────────────────────────


@dataclass
class CounterfactualState:
    """Per-instrument mutable state for the counterfactual layer.

    Owned by ``CounterfactualEngine``.  All three public callables also
    accept ``_state=`` for standalone use.

    ``primary_side`` / ``secondary_side`` track which ``BeliefDirection``
    is currently primary vs secondary — the actual ``Thesis`` objects
    are passed in per-call and not stored here, since they are owned by
    ``ThesisManager`` / ``Opportunity``.

    ``primary_evidence_at_last_flip`` stores the conflicting-evidence
    list from the previous primary thesis at the time of the last flip,
    enabling ``FlipLesson`` generation on the next flip.
    """

    instrument:   str
    primary_side: BeliefDirection = BeliefDirection.BULLISH
    secondary_side: BeliefDirection = BeliefDirection.BEARISH

    # Probability evolution timeline
    probability_evolution: list[ProbabilityPoint] = field(default_factory=list)

    # Event stores
    flip_records:       list[ThesisFlipRecord]       = field(default_factory=list)
    reversal_records:   list[ThesisReversalRecord]   = field(default_factory=list)
    uncertainty_spikes: list[UncertaintySpikeRecord] = field(default_factory=list)

    # Currently active scenarios
    current_scenarios:  list[CounterfactualScenario] = field(default_factory=list)

    # For FlipLesson generation on next flip
    primary_evidence_at_last_flip: list[str] = field(default_factory=list)
    last_flip_at: float | None = None

    created_at:     float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    def last_probability_point(self) -> ProbabilityPoint | None:
        return self.probability_evolution[-1] if self.probability_evolution else None

    def flip_count(self) -> int:
        return len(self.flip_records)

    def reversal_rate(self) -> float:
        """Fraction of flips that were reversed within REVERSAL_WINDOW_SECONDS."""
        if not self.flip_records:
            return 0.0
        return round(len(self.reversal_records) / len(self.flip_records), 4)

    def to_snapshot(self) -> dict:
        return {
            "instrument":       self.instrument,
            "primary_side":     self.primary_side.value,
            "secondary_side":   self.secondary_side.value,
            "flip_count":       self.flip_count(),
            "reversal_count":   len(self.reversal_records),
            "spike_count":      len(self.uncertainty_spikes),
            "reversal_rate":    self.reversal_rate(),
            "last_probability": (
                self.last_probability_point().to_dict()
                if self.last_probability_point() else None
            ),
            "scenario_count":   len(self.current_scenarios),
            "created_at":       round(self.created_at, 3),
            "last_updated_at":  round(self.last_updated_at, 3),
        }


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Enums
    "ScenarioType",
    "FlipTrigger",
    # Thresholds
    "SWITCH_MARGIN",
    "PRIMARY_CONFIDENCE_FLOOR",
    "UNCERTAINTY_SPIKE_DELTA",
    "REVERSAL_WINDOW_SECONDS",
    "MAX_FLIP_RECORDS",
    "MAX_REVERSAL_RECORDS",
    "MAX_SPIKE_RECORDS",
    "MAX_PROBABILITY_POINTS",
    # Data types
    "ProbabilityPoint",
    "CounterfactualScenario",
    "FlipLesson",
    "ThesisFlipRecord",
    "ThesisReversalRecord",
    "UncertaintySpikeRecord",
    "ThesisSwitchResult",
    "CounterfactualState",
]
