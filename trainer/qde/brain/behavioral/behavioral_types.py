"""C15 Behavioral Control Layer — type definitions.

Internal cognitive states and behavioral signal types for the QDE.

Cognitive states
----------------
    CAUTION     — elevated vigilance; raise entry thresholds, shrink attention
                  to high-conviction signals only.
    NORMAL      — baseline processing; standard thresholds throughout.
    AGGRESSIVE  — positive momentum, healthy capital; lean in, but watch for
                  euphoria.
    DEFENSIVE   — loss-mode; maximum restraint, refuse marginal setups, reduce
                  all risk aggressiveness.

Relation to AccountAction
-------------------------
``CognitiveState`` models the *brain's internal operating mode* — how reasoning
thresholds, patience requirements, and attention allocation should be tuned.
``AccountAction`` (``qde_account_types``) models the *account-level posture* —
size, bar, and exposure limits.  They are read by different downstream consumers
and must not be conflated.

Five behavioral pathologies tracked
------------------------------------
1. Confidence inflation  — systematic overestimation of predicted probability
                           vs. realised outcomes.
2. Overconfidence        — high-confidence trades that systematically lose.
3. Drawdown response     — aggressiveness failure to decline during drawdown
                           (misaligned or reversed response).
4. Aggressiveness drift  — slow baseline creep in risk-taking over rolling window.
5. Risk escalation       — accelerating increase in per-trade risk (second
                           derivative of risk positive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ── Cognitive State ───────────────────────────────────────────────────────────


class CognitiveState(str, Enum):
    """The QDE brain's current internal operating mode.

    Downstream consumers (patience engine, attention engine, thesis aging)
    read this to tune their own thresholds without coupling to account-level
    posture.
    """

    CAUTION = "CAUTION"  # elevated vigilance, narrowed entry filter
    NORMAL = "NORMAL"  # baseline processing
    AGGRESSIVE = "AGGRESSIVE"  # lean in — requires clean pathology signals
    DEFENSIVE = "DEFENSIVE"  # maximum restraint, loss-mode


# Canonical transition ordering by severity (for comparison helpers)
_STATE_SEVERITY: dict[CognitiveState, int] = {
    CognitiveState.AGGRESSIVE: 0,
    CognitiveState.NORMAL: 1,
    CognitiveState.CAUTION: 2,
    CognitiveState.DEFENSIVE: 3,
}


def state_severity(state: CognitiveState) -> int:
    """Return an ordinal severity where higher = more defensive."""
    return _STATE_SEVERITY[state]


# ── Thresholds ────────────────────────────────────────────────────────────────

# Confidence inflation (signed mean of predicted_confidence - actual_outcome)
INFLATION_MILD_THRESHOLD: float = 0.10  # → CAUTION flag
INFLATION_SEVERE_THRESHOLD: float = 0.20  # → strong pathology signal

# Overconfidence rate (fraction of high-confidence trades that lost)
OVERCONFIDENCE_RATE_MILD: float = 0.30  # → flag
OVERCONFIDENCE_RATE_SEVERE: float = 0.50

# Aggressiveness drift (normalised slope of rolling aggressiveness)
AGGRESSION_DRIFT_MILD: float = 0.15  # → CAUTION
AGGRESSION_DRIFT_SEVERE: float = 0.30  # → strong signal

# Risk escalation (normalised second-derivative of per-trade risk)
RISK_ESCALATION_MILD: float = 0.10
RISK_ESCALATION_SEVERE: float = 0.25

# Drawdown thresholds (fraction 0-1) — mirror qde_account_types soft limits
DRAWDOWN_CAUTION_THRESHOLD: float = 0.05  # 5% drawdown → CAUTION
DRAWDOWN_DEFENSIVE_THRESHOLD: float = 0.08  # 8% drawdown → DEFENSIVE

# Consecutive streaks
LOSING_STREAK_CAUTION: int = 2  # 2 losses → CAUTION
LOSING_STREAK_DEFENSIVE: int = 3  # 3 losses → DEFENSIVE (mirrors qde_account_types)
WINNING_STREAK_AGGRESSIVE: int = 4  # 4 wins   → allow AGGRESSIVE
WINNING_STREAK_EUPHORIA: int = 6  # 6 wins   → euphoria guard (force CAUTION)

# Pathology composite → state transitions
PATHOLOGY_CAUTION_THRESHOLD: float = 0.25  # composite ≥ this → CAUTION
PATHOLOGY_DEFENSIVE_THRESHOLD: float = 0.55  # composite ≥ this → DEFENSIVE

# Euphoria risk (after strong winning streak)
EUPHORIA_RISK_THRESHOLD: float = 0.55  # → downgrade AGGRESSIVE → CAUTION

# Max history retained in BehavioralRecord list
MAX_BEHAVIORAL_HISTORY: int = 200


# ── Input Signals ─────────────────────────────────────────────────────────────


@dataclass
class BehavioralSignals:
    """Inputs for ``evaluate_behavior()``.

    All list-valued fields may be empty (the engine degrades gracefully when
    data is sparse).  Scalar fields default to their "neutral" value so callers
    can populate only the signals they have.

    Recommendation: populate at a minimum ``recent_trade_confidences``,
    ``recent_trade_outcomes``, ``current_drawdown_fraction``, and
    ``consecutive_losses``.  The remaining signals improve accuracy but are
    not required.
    """

    # ── Confidence inflation ──────────────────────────────────────────────
    # Predicted confidence per recent closed trade (same order as outcomes).
    recent_trade_confidences: list[float] = field(default_factory=list)
    # Realised outcome per recent closed trade: 1.0 = win, 0.0 = loss,
    # 0.5 = scratch / break-even.
    recent_trade_outcomes: list[float] = field(default_factory=list)

    # ── Aggressiveness ────────────────────────────────────────────────────
    # Aggressiveness score (0-1) recorded per recent trade or tick.
    recent_aggressiveness_scores: list[float] = field(default_factory=list)

    # ── Risk escalation ───────────────────────────────────────────────────
    # Normalised risk per trade (0-1), same chronological order.
    recent_risk_per_trade: list[float] = field(default_factory=list)

    # ── Drawdown ──────────────────────────────────────────────────────────
    # Current equity drawdown as a fraction (0.0 = at high-water-mark,
    # 0.15 = 15% below high-water-mark).
    current_drawdown_fraction: float = 0.0
    # Peak aggressiveness observed *during* the current drawdown episode.
    drawdown_peak_aggressiveness: float = 0.0

    # ── Streaks ───────────────────────────────────────────────────────────
    consecutive_losses: int = 0
    consecutive_wins: int = 0

    # ── Session context ───────────────────────────────────────────────────
    # Normalised P&L for current session (+1 = excellent, -1 = severe loss).
    current_session_pnl: float = 0.0

    # ── Optional pre-computed metacognition signals ───────────────────────
    # Pass ``MetacognitionEngine.assess()`` values directly to avoid
    # recomputing them inside the behavioral engine.
    overconfidence_score: float = 0.0  # from MetacognitiveAssessment
    metacognition_caution_level: float = 0.0  # from MetacognitiveAssessment
    calibration_error: float = 0.0  # from MetacognitiveAssessment

    def __post_init__(self) -> None:
        self.current_drawdown_fraction = max(0.0, self.current_drawdown_fraction)
        self.consecutive_losses = max(0, self.consecutive_losses)
        self.consecutive_wins = max(0, self.consecutive_wins)


# ── Assessment Output ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BehavioralAssessment:
    """Stateless output of ``evaluate_behavior()``.

    All component scores are 0-1 except ``aggressiveness_drift`` (signed:
    positive = drifting more aggressive; negative = drifting more cautious).
    """

    # ── Five tracked pathologies ──────────────────────────────────────────
    confidence_inflation: float  # 0-1: how inflated predictions are
    overconfidence_level: float  # 0-1: rate of high-conf + loss combos
    drawdown_response_quality: float  # 0-1: 1 = ideal reduction; 0 = ignored/escalated
    aggressiveness_drift: float  # -1..+1: positive = drifting aggressive
    risk_escalation_rate: float  # 0-1: second-derivative of risk trend

    # ── Composite ─────────────────────────────────────────────────────────
    pathology_score: float  # 0-1: weighted composite of all pathologies
    euphoria_risk: float  # 0-1: post-win overconfidence risk
    loss_spiral_risk: float  # 0-1: post-loss revenge-trading risk

    # ── Recommendation ────────────────────────────────────────────────────
    recommended_state: CognitiveState
    reasoning: str  # compact log string

    def to_dict(self) -> dict:
        return {
            "confidence_inflation": round(self.confidence_inflation, 4),
            "overconfidence_level": round(self.overconfidence_level, 4),
            "drawdown_response_quality": round(self.drawdown_response_quality, 4),
            "aggressiveness_drift": round(self.aggressiveness_drift, 4),
            "risk_escalation_rate": round(self.risk_escalation_rate, 4),
            "pathology_score": round(self.pathology_score, 4),
            "euphoria_risk": round(self.euphoria_risk, 4),
            "loss_spiral_risk": round(self.loss_spiral_risk, 4),
            "recommended_state": self.recommended_state.value,
            "reasoning": self.reasoning,
        }


# ── Behavioral Record (History Entry) ────────────────────────────────────────


@dataclass
class BehavioralRecord:
    """One entry in the behavioral state history.

    Appended to ``BehavioralControlEngine._history`` on every call to
    ``adjust_state()`` that records a state (even when the state does not
    change — the snapshot itself is valuable for trend analysis).
    """

    record_id: str
    previous_state: CognitiveState
    new_state: CognitiveState
    assessment: BehavioralAssessment
    session_label: str
    recorded_at: float
    trigger: str  # brief description of what drove the transition

    @property
    def is_transition(self) -> bool:
        return self.previous_state != self.new_state

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "assessment": self.assessment.to_dict(),
            "session_label": self.session_label,
            "recorded_at": round(self.recorded_at, 3),
            "trigger": self.trigger,
            "is_transition": self.is_transition,
        }


# ── Public API list ───────────────────────────────────────────────────────────

__all__ = [
    "CognitiveState",
    "state_severity",
    # Thresholds
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
    # Data types
    "BehavioralSignals",
    "BehavioralAssessment",
    "BehavioralRecord",
]
