"""C15 Behavioral Control Layer.

Prevents pathological QDE behavior by tracking five behavioral signals
and maintaining a dynamic internal CognitiveState.

Cognitive states
----------------
    CAUTION     — elevated vigilance; raise entry thresholds, shrink attention.
    NORMAL      — baseline processing.
    AGGRESSIVE  — positive momentum; lean in but watch for euphoria.
    DEFENSIVE   — loss-mode; maximum restraint, refuse marginal setups.

Five tracked pathologies
------------------------
    1. Confidence inflation   — predicted confidence > actual outcome rate.
    2. Overconfidence         — high-confidence trades that systematically lose.
    3. Drawdown response      — aggressiveness failure to decline during losses.
    4. Aggressiveness drift   — slow baseline creep in risk-taking.
    5. Risk escalation        — accelerating increase in per-trade risk.

Public callables (spec)
-----------------------
    evaluate_behavior(signals, current_state)   → BehavioralAssessment
    adjust_state(current_state, assessment)     → CognitiveState
    behavior_report(engine, now)                → dict

Stateful wrapper
----------------
    BehavioralControlEngine — holds CognitiveState + bounded BehavioralRecord history.

Integration notes
-----------------
``CognitiveState`` is the *brain-level* cognitive mode.  It is distinct from
``AccountAction`` (``qde_account_types``) which is the account-level posture.

``evaluate_behavior()`` optionally accepts pre-computed ``MetacognitiveAssessment``
fields (overconfidence_score, calibration_error, caution_level) to avoid
recomputing them — pass them via ``BehavioralSignals`` fields of the same names.
"""

from trainer.qde.brain.behavioral.behavioral_engine import (
    BehavioralControlEngine,
    adjust_state,
    behavior_report,
    evaluate_behavior,
)
from trainer.qde.brain.behavioral.behavioral_types import (
    AGGRESSION_DRIFT_MILD,
    AGGRESSION_DRIFT_SEVERE,
    DRAWDOWN_CAUTION_THRESHOLD,
    DRAWDOWN_DEFENSIVE_THRESHOLD,
    EUPHORIA_RISK_THRESHOLD,
    INFLATION_MILD_THRESHOLD,
    INFLATION_SEVERE_THRESHOLD,
    LOSING_STREAK_CAUTION,
    LOSING_STREAK_DEFENSIVE,
    MAX_BEHAVIORAL_HISTORY,
    OVERCONFIDENCE_RATE_MILD,
    OVERCONFIDENCE_RATE_SEVERE,
    PATHOLOGY_CAUTION_THRESHOLD,
    PATHOLOGY_DEFENSIVE_THRESHOLD,
    RISK_ESCALATION_MILD,
    RISK_ESCALATION_SEVERE,
    WINNING_STREAK_AGGRESSIVE,
    WINNING_STREAK_EUPHORIA,
    BehavioralAssessment,
    BehavioralRecord,
    BehavioralSignals,
    CognitiveState,
    state_severity,
)

__all__ = [
    # ── Cognitive state ───────────────────────────────────────────────────
    "CognitiveState",
    "state_severity",
    # ── Thresholds ────────────────────────────────────────────────────────
    "INFLATION_MILD_THRESHOLD",
    "INFLATION_SEVERE_THRESHOLD",
    "OVERCONFIDENCE_RATE_MILD",
    "OVERCONFIDENCE_RATE_SEVERE",
    "AGGRESSION_DRIFT_MILD",
    "AGGRESSION_DRIFT_SEVERE",
    "RISK_ESCALATION_MILD",
    "RISK_ESCALATION_SEVERE",
    "DRAWDOWN_CAUTION_THRESHOLD",
    "DRAWDOWN_DEFENSIVE_THRESHOLD",
    "LOSING_STREAK_CAUTION",
    "LOSING_STREAK_DEFENSIVE",
    "WINNING_STREAK_AGGRESSIVE",
    "WINNING_STREAK_EUPHORIA",
    "PATHOLOGY_CAUTION_THRESHOLD",
    "PATHOLOGY_DEFENSIVE_THRESHOLD",
    "EUPHORIA_RISK_THRESHOLD",
    "MAX_BEHAVIORAL_HISTORY",
    # ── Data types ────────────────────────────────────────────────────────
    "BehavioralSignals",
    "BehavioralAssessment",
    "BehavioralRecord",
    # ── Public callables (spec) ───────────────────────────────────────────
    "evaluate_behavior",
    "adjust_state",
    "behavior_report",
    # ── Stateful wrapper ──────────────────────────────────────────────────
    "BehavioralControlEngine",
]
