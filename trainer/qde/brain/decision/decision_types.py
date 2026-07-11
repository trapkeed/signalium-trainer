"""QDE Brain — cognitive decision types (Phase C6).

Defines the vocabulary of the Brain's *cognitive* decision layer — distinct
from the execution vocabulary in ``qde_inference.QDEDecision``:

  ``QDEDecision`` (qde_inference.py)
      Execution-facing: ENTER/WAIT/REJECT, position size lots, stop/tp,
      management actions.  ``ExecutionPlanner`` consumes this.

  ``CognitiveDecision`` (this module)
      Brain-facing: BUY/SELL/WAIT/MONITOR/IGNORE, six signal heads
      (direction, confidence, risk_signal, entry_urgency, hold_horizon,
      size_signal), uncertainty score, calibrated confidence, reasoning.
      ``DecisionAPI`` produces this from accumulated cognition.

WAIT and MONITOR are **first-class decisions** with distinct meanings:

  ``IGNORE``   — market is incomprehensible (EXTREME uncertainty) or no
                 cognitive basis exists for engaging this symbol at all.

  ``WAIT``     — conditions understood but unfavourable for action right
                 now (HIGH uncertainty, setup forming, overconfidence
                 flagged, risk limit reached).

  ``MONITOR``  — a developing setup exists, debate is ongoing, conviction
                 is building — continue tracking without committing.

  ``BUY``      — debate is conclusive BULLISH, uncertainty is acceptable,
                 conviction is above threshold.

  ``SELL``     — debate is conclusive BEARISH, uncertainty is acceptable,
                 conviction is above threshold.

Design constraints
------------------
* This module imports NOTHING from ``qde_inference``.
* No ``QDEDecision`` reference anywhere in the C6 package.
* All dataclasses follow the ``to_dict()`` / ``from_dict()`` round-trip
  pattern consistent with C3–C5.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.conviction_engine import ConvictionProfile
from trainer.qde.brain.uncertainty_engine import UncertaintyLevel

if TYPE_CHECKING:
    from trainer.qde.brain.brain_models import Hypothesis, MarketNarrative
    from trainer.qde.brain.conviction_engine import ConvictionScore
    from trainer.qde.brain.debate.debate_store import DebateRecord
    from trainer.qde.brain.uncertainty_engine import UncertaintyScore


# -------------------------------------------------------------------
# CognitiveAction
# -------------------------------------------------------------------


class CognitiveAction(str, Enum):
    """The five cognitive actions the Brain can recommend.

    The action is NOT an execution instruction — it is a recommendation
    to the inference layer about what kind of engagement the current
    cognitive state supports.
    """

    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    MONITOR = "MONITOR"
    IGNORE = "IGNORE"

    def is_actionable(self) -> bool:
        """True only for BUY and SELL — the two directions that suggest
        the brain is ready to act."""
        return self in (CognitiveAction.BUY, CognitiveAction.SELL)

    def is_passive(self) -> bool:
        """True for WAIT/MONITOR/IGNORE — no execution implied."""
        return not self.is_actionable()


# -------------------------------------------------------------------
# DecisionInput
# -------------------------------------------------------------------


@dataclass
class DecisionInput:
    """All cognitive context the ``DecisionEngine`` consumes.

    Built by ``DecisionAPI.make_decision()`` from ``BrainState`` + optional
    supplementary data.  Callers don't construct this directly.

    ``portfolio_open_positions``
        Count of live positions across the portfolio (any instrument).
        Used to detect concentration risk.

    ``portfolio_daily_pnl_pct``
        Current day's portfolio P&L as a fraction of equity.
        Negative → risk limiter kicks in.

    ``has_existing_position``
        Whether the instrument already has an open position.

    ``existing_position_direction``
        Direction of any existing position (None if none open).

    ``memory_win_rate``
        Historical win rate from episodic recall (None = no history).

    ``memory_recent_count``
        Number of recent episodes recalled.
    """

    instrument: str
    narrative: "MarketNarrative | None"
    active_hypotheses: "list[Hypothesis]"
    debate_record: "DebateRecord | None"
    uncertainty: "UncertaintyScore"
    conviction: "ConvictionScore | None"
    portfolio_open_positions: int = 0
    portfolio_daily_pnl_pct: float = 0.0
    has_existing_position: bool = False
    existing_position_direction: "BeliefDirection | None" = None
    memory_win_rate: float | None = None
    memory_recent_count: int = 0
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "has_narrative": self.narrative is not None,
            "narrative_tone": self.narrative.tone.value if self.narrative else "NEUTRAL",
            "market_regime": self.narrative.market_regime if self.narrative else "",
            "active_hypothesis_count": len(self.active_hypotheses),
            "has_debate": self.debate_record is not None,
            "debate_conclusive": (self.debate_record.conclusive if self.debate_record else False),
            "debate_quality": (self.debate_record.quality_grade if self.debate_record else "LOW"),
            "uncertainty_composite": round(self.uncertainty.composite, 4),
            "uncertainty_level": self.uncertainty.level.value,
            "conviction_score": round(self.conviction.conviction, 4) if self.conviction else 0.0,
            "conviction_profile": self.conviction.profile.value if self.conviction else "MIXED",
            "portfolio_open_positions": self.portfolio_open_positions,
            "portfolio_daily_pnl_pct": round(self.portfolio_daily_pnl_pct, 4),
            "has_existing_position": self.has_existing_position,
            "existing_position_direction": (
                self.existing_position_direction.value if self.existing_position_direction else None
            ),
            "memory_win_rate": (
                round(self.memory_win_rate, 4) if self.memory_win_rate is not None else None
            ),
            "memory_recent_count": self.memory_recent_count,
        }


# -------------------------------------------------------------------
# CognitiveDecision
# -------------------------------------------------------------------


@dataclass
class CognitiveDecision:
    """The Brain's cognitive verdict for one instrument at one point in time.

    Six signal heads — the spec-required outputs:

    ``direction``
        BULLISH, BEARISH, or NEUTRAL.

    ``confidence``
        Raw confidence (from dominant hypothesis or 0 if no hypothesis).

    ``calibrated_confidence``
        Confidence after applying ``CalibrationProfile.confidence_bias``.

    ``risk_signal``
        0-1: how much risk is appropriate given current uncertainty.
        ``1.0 - uncertainty.composite`` as the base; further penalised
        by portfolio risk limits.

    ``entry_urgency``
        0-1: how urgently the Brain thinks an entry should be considered
        (only meaningful for BUY/SELL; 0 for WAIT/MONITOR/IGNORE).

    ``hold_horizon``
        Estimated hold duration in seconds.  Derived from hypothesis
        persistence and maturity; 0 for non-actionable decisions.

    ``size_signal``
        0-1: relative position size suggested by conviction × risk
        headroom.  0 for WAIT/MONITOR/IGNORE.

    ``reasoning``
        One-paragraph natural-language explanation of the decision.

    ``overconfidence_flagged``
        Whether the C5 debate detected overconfidence during this cycle.

    ``ignored_evidence_flagged``
        Whether the C5 debate detected ignored contradicting evidence.
    """

    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    action: CognitiveAction = CognitiveAction.IGNORE
    direction: BeliefDirection = BeliefDirection.NEUTRAL
    confidence: float = 0.0
    calibrated_confidence: float = 0.0
    risk_signal: float = 0.0
    entry_urgency: float = 0.0
    hold_horizon: float = 0.0
    size_signal: float = 0.0
    uncertainty_score: float = 0.0
    uncertainty_level: str = UncertaintyLevel.LOW.value
    conviction_score: float = 0.0
    conviction_profile: str = ConvictionProfile.MIXED.value
    reasoning: str = ""
    primary_hypothesis_id: str | None = None
    debate_id: str | None = None
    overconfidence_flagged: bool = False
    ignored_evidence_flagged: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def is_actionable(self) -> bool:
        return self.action.is_actionable()

    @property
    def is_passive(self) -> bool:
        return self.action.is_passive()

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "instrument": self.instrument,
            "action": self.action.value,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "risk_signal": round(self.risk_signal, 4),
            "entry_urgency": round(self.entry_urgency, 4),
            "hold_horizon": round(self.hold_horizon, 1),
            "size_signal": round(self.size_signal, 4),
            "uncertainty_score": round(self.uncertainty_score, 4),
            "uncertainty_level": self.uncertainty_level,
            "conviction_score": round(self.conviction_score, 4),
            "conviction_profile": self.conviction_profile,
            "reasoning": self.reasoning,
            "primary_hypothesis_id": self.primary_hypothesis_id,
            "debate_id": self.debate_id,
            "overconfidence_flagged": self.overconfidence_flagged,
            "ignored_evidence_flagged": self.ignored_evidence_flagged,
            "created_at": round(self.created_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CognitiveDecision":
        return cls(
            decision_id=d.get("decision_id", str(uuid.uuid4())[:12]),
            instrument=d.get("instrument", ""),
            action=CognitiveAction(d.get("action", CognitiveAction.IGNORE.value)),
            direction=BeliefDirection(d.get("direction", BeliefDirection.NEUTRAL.value)),
            confidence=d.get("confidence", 0.0),
            calibrated_confidence=d.get("calibrated_confidence", 0.0),
            risk_signal=d.get("risk_signal", 0.0),
            entry_urgency=d.get("entry_urgency", 0.0),
            hold_horizon=d.get("hold_horizon", 0.0),
            size_signal=d.get("size_signal", 0.0),
            uncertainty_score=d.get("uncertainty_score", 0.0),
            uncertainty_level=d.get("uncertainty_level", UncertaintyLevel.LOW.value),
            conviction_score=d.get("conviction_score", 0.0),
            conviction_profile=d.get("conviction_profile", ConvictionProfile.MIXED.value),
            reasoning=d.get("reasoning", ""),
            primary_hypothesis_id=d.get("primary_hypothesis_id"),
            debate_id=d.get("debate_id"),
            overconfidence_flagged=d.get("overconfidence_flagged", False),
            ignored_evidence_flagged=d.get("ignored_evidence_flagged", False),
            created_at=d.get("created_at", 0.0),
        )


# -------------------------------------------------------------------
# DecisionReasoning
# -------------------------------------------------------------------


@dataclass
class DecisionReasoning:
    """Structured chain-of-thought explaining one ``CognitiveDecision``.

    Produced by ``DecisionAPI.explain_decision()``.  Every field is
    a human-readable string or a structured sub-explanation that a
    downstream layer (e.g. a dashboard, a logging pipeline, or a
    reflection engine) can display or store independently.

    ``chain``
        Ordered list of reasoning steps, each a short sentence.

    ``hypothesis_basis``
        Plain-language summary of which hypothesis drove the decision
        and why it was selected.

    ``debate_summary``
        Plain-language summary of the C5 debate result (quality, margin,
        overconfidence flag).

    ``uncertainty_basis``
        Why uncertainty is at the current level — lifted from
        ``UncertaintyScore.explanation``.

    ``memory_influence``
        How historical win-rate / experience influenced the decision.

    ``portfolio_constraint``
        Any portfolio-level constraint that affected risk/size signals.

    ``flags``
        List of active warning flags (overconfidence, ignored evidence, etc.).
    """

    decision_id: str = ""
    instrument: str = ""
    action: str = CognitiveAction.IGNORE.value
    chain: list[str] = field(default_factory=list)
    hypothesis_basis: str = ""
    debate_summary: str = ""
    uncertainty_basis: str = ""
    memory_influence: str = ""
    portfolio_constraint: str = ""
    flags: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def full_text(self) -> str:
        """Single concatenated plain-text explanation."""
        parts = [f"Decision: {self.action} on {self.instrument}."]
        parts.extend(self.chain)
        return " ".join(p.rstrip(".") + "." for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "instrument": self.instrument,
            "action": self.action,
            "chain": self.chain,
            "hypothesis_basis": self.hypothesis_basis,
            "debate_summary": self.debate_summary,
            "uncertainty_basis": self.uncertainty_basis,
            "memory_influence": self.memory_influence,
            "portfolio_constraint": self.portfolio_constraint,
            "flags": self.flags,
            "full_text": self.full_text(),
            "generated_at": round(self.generated_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionReasoning":
        return cls(
            decision_id=d.get("decision_id", ""),
            instrument=d.get("instrument", ""),
            action=d.get("action", CognitiveAction.IGNORE.value),
            chain=d.get("chain", []),
            hypothesis_basis=d.get("hypothesis_basis", ""),
            debate_summary=d.get("debate_summary", ""),
            uncertainty_basis=d.get("uncertainty_basis", ""),
            memory_influence=d.get("memory_influence", ""),
            portfolio_constraint=d.get("portfolio_constraint", ""),
            flags=d.get("flags", []),
            generated_at=d.get("generated_at", 0.0),
        )


# -------------------------------------------------------------------
# ConfidenceReport
# -------------------------------------------------------------------


@dataclass
class ConfidenceReport:
    """Full confidence accounting for one cognitive decision.

    Produced by ``DecisionAPI.confidence_report()``.

    ``raw_confidence``
        Hypothesis.confidence at the time of the decision (the model's
        direct probability estimate).

    ``calibration_bias``
        The ``CalibrationProfile.confidence_bias`` for this instrument
        — learned from historical over/under-statement.

    ``calibrated_confidence``
        ``raw_confidence + calibration_bias``, clamped to [0.05, 0.95].

    ``conviction_score``
        The structural conviction (reinforcement + evidence + maturity +
        persistence), separate from raw confidence.

    ``conviction_profile``
        Which of the four confidence × conviction quadrants this
        decision falls in.

    ``calibration_sample_count``
        How many past observations the calibration bias is based on.
        Low counts → less reliable bias.

    ``alignment_warning``
        True when profile is HIGH_CONFIDENCE_LOW_CONVICTION (high
        model confidence in a thinly-formed setup) — the most important
        calibration signal for the QDE to surface.

    ``historical_win_rate``
        Recent empirical win rate from episode recall (None if < 4
        episodes).

    ``accuracy_delta``
        ``raw_confidence - historical_win_rate`` when both are available:
        positive = over-confident historically; negative = under-confident.
    """

    decision_id: str = ""
    instrument: str = ""
    raw_confidence: float = 0.0
    calibration_bias: float = 0.0
    calibrated_confidence: float = 0.0
    conviction_score: float = 0.0
    conviction_profile: str = ConvictionProfile.MIXED.value
    calibration_sample_count: int = 0
    alignment_warning: bool = False
    historical_win_rate: float | None = None
    accuracy_delta: float | None = None
    breakdown: dict = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "instrument": self.instrument,
            "raw_confidence": round(self.raw_confidence, 4),
            "calibration_bias": round(self.calibration_bias, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "conviction_score": round(self.conviction_score, 4),
            "conviction_profile": self.conviction_profile,
            "calibration_sample_count": self.calibration_sample_count,
            "alignment_warning": self.alignment_warning,
            "historical_win_rate": (
                round(self.historical_win_rate, 4) if self.historical_win_rate is not None else None
            ),
            "accuracy_delta": (
                round(self.accuracy_delta, 4) if self.accuracy_delta is not None else None
            ),
            "breakdown": self.breakdown,
            "generated_at": round(self.generated_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConfidenceReport":
        return cls(
            decision_id=d.get("decision_id", ""),
            instrument=d.get("instrument", ""),
            raw_confidence=d.get("raw_confidence", 0.0),
            calibration_bias=d.get("calibration_bias", 0.0),
            calibrated_confidence=d.get("calibrated_confidence", 0.0),
            conviction_score=d.get("conviction_score", 0.0),
            conviction_profile=d.get("conviction_profile", ConvictionProfile.MIXED.value),
            calibration_sample_count=d.get("calibration_sample_count", 0),
            alignment_warning=d.get("alignment_warning", False),
            historical_win_rate=d.get("historical_win_rate"),
            accuracy_delta=d.get("accuracy_delta"),
            breakdown=d.get("breakdown", {}),
            generated_at=d.get("generated_at", 0.0),
        )


__all__ = [
    "CognitiveAction",
    "DecisionInput",
    "CognitiveDecision",
    "DecisionReasoning",
    "ConfidenceReport",
]
