"""C16 Counterfactual Consciousness.

The QDE continuously asks: "What if I am wrong?"

Primary / secondary thesis
---------------------------
The counterfactual layer maintains a committed primary thesis and a
challenger secondary thesis.  These are ABOVE the existing
``ThesisManager`` bull/bear layer — they track *which side the QDE is
committed to* rather than storing evidence.

Five scenario types
--------------------
    CONTINUATION   — primary direction continues at current pace.
    REVERSAL       — market reverses the primary direction.
    SIDEWAYS       — neither direction resolves (consolidation).
    ACCELERATION   — primary direction faster / more extreme.
    EXHAUSTION     — primary direction winding down toward inflection.

Three stored event types
-------------------------
    ThesisFlipRecord      — primary ↔ secondary switch.
    ThesisReversalRecord  — quick flip-back (oscillation detection).
    UncertaintySpikeRecord — rapid jump in composite uncertainty.

Public callables (spec)
-----------------------
    generate_alternatives(primary_thesis, secondary_thesis, uncertainty_score)
        → list[CounterfactualScenario]

    evaluate_alternative(scenario, new_evidence, current_uncertainty)
        → float  (updated probability weight)

    switch_thesis(state, trigger, new_primary_side, new_primary_thesis,
                  new_secondary_thesis, ...)
        → ThesisSwitchResult

Stateful wrapper
----------------
    CounterfactualEngine — holds CounterfactualState + all event histories.

Supporting helpers
------------------
    record_probability_point — append a ProbabilityPoint to the evolution.
    recommend_switch         — check if secondary has overtaken primary.

Integration notes
-----------------
``Thesis`` objects are imported from ``trainer.qde.brain.thesis_manager``
and passed in per-call; this package does not own or re-derive them.
``BeliefDirection`` is imported from ``trainer.qde.brain.brain_types``.
"""

from trainer.qde.brain.counterfactual.counterfactual_engine import (
    CounterfactualEngine,
    evaluate_alternative,
    generate_alternatives,
    recommend_switch,
    record_probability_point,
    switch_thesis,
)
from trainer.qde.brain.counterfactual.counterfactual_types import (
    MAX_FLIP_RECORDS,
    MAX_PROBABILITY_POINTS,
    MAX_REVERSAL_RECORDS,
    MAX_SPIKE_RECORDS,
    PRIMARY_CONFIDENCE_FLOOR,
    REVERSAL_WINDOW_SECONDS,
    SWITCH_MARGIN,
    UNCERTAINTY_SPIKE_DELTA,
    CounterfactualScenario,
    CounterfactualState,
    FlipLesson,
    FlipTrigger,
    ProbabilityPoint,
    ScenarioType,
    ThesisFlipRecord,
    ThesisReversalRecord,
    ThesisSwitchResult,
    UncertaintySpikeRecord,
)

__all__ = [
    # ── Enums ─────────────────────────────────────────────────────────────
    "ScenarioType",
    "FlipTrigger",
    # ── Thresholds ────────────────────────────────────────────────────────
    "SWITCH_MARGIN",
    "PRIMARY_CONFIDENCE_FLOOR",
    "UNCERTAINTY_SPIKE_DELTA",
    "REVERSAL_WINDOW_SECONDS",
    "MAX_FLIP_RECORDS",
    "MAX_REVERSAL_RECORDS",
    "MAX_SPIKE_RECORDS",
    "MAX_PROBABILITY_POINTS",
    # ── Data types ────────────────────────────────────────────────────────
    "ProbabilityPoint",
    "CounterfactualScenario",
    "FlipLesson",
    "ThesisFlipRecord",
    "ThesisReversalRecord",
    "UncertaintySpikeRecord",
    "ThesisSwitchResult",
    "CounterfactualState",
    # ── Public callables (spec) ───────────────────────────────────────────
    "generate_alternatives",
    "evaluate_alternative",
    "switch_thesis",
    # ── Supporting helpers ────────────────────────────────────────────────
    "record_probability_point",
    "recommend_switch",
    # ── Stateful wrapper ──────────────────────────────────────────────────
    "CounterfactualEngine",
]
