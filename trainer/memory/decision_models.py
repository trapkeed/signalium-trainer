"""Phase 21.6 — Core decision data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trainer.memory.decision_types import (
    DecisionGrade,
    DecisionType,
    MarketRegime,
    OutcomeType,
    VolatilityRegime,
)

# ── DecisionSnapshot ─────────────────────────────────────────────────────────


@dataclass
class DecisionSnapshot:
    """Complete market and model context captured at decision time."""

    # Market context
    price: float = 0.0
    atr: float = 0.0
    spread: float = 0.0
    volatility: float = 0.01
    session: str = "LONDON"
    liquidity_estimate: float = 1.0  # 0-1
    market_regime: str = MarketRegime.UNKNOWN.value
    volatility_regime: str = VolatilityRegime.NORMAL.value
    timeframe: str = "H1"

    # Forecast
    forecast_direction: str = "BUY"
    forecast_confidence: float = 0.65
    expected_move: float = 0.0  # price units
    expected_holding_h: float = 24.0
    tp_probability: float = 0.50
    sl_probability: float = 0.30

    # Model
    model_version: str = ""
    champion_model: str = ""
    research_score: float = 0.0
    model_output: float = 0.0
    ensemble_output: float = 0.0

    # Portfolio state
    portfolio_exposure: float = 0.0  # fraction of account
    correlation_risk: float = 0.0  # 0-1

    # Detailed dicts (stored as JSON)
    risk_metrics: dict = field(default_factory=dict)
    feature_snapshot: dict = field(default_factory=dict)
    expert_weights: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "price": round(self.price, 5),
            "atr": round(self.atr, 5),
            "spread": round(self.spread, 5),
            "volatility": round(self.volatility, 4),
            "session": self.session,
            "liquidity_estimate": round(self.liquidity_estimate, 4),
            "market_regime": self.market_regime,
            "volatility_regime": self.volatility_regime,
            "timeframe": self.timeframe,
            "forecast_direction": self.forecast_direction,
            "forecast_confidence": round(self.forecast_confidence, 4),
            "expected_move": round(self.expected_move, 5),
            "expected_holding_h": round(self.expected_holding_h, 2),
            "tp_probability": round(self.tp_probability, 4),
            "sl_probability": round(self.sl_probability, 4),
            "model_version": self.model_version,
            "champion_model": self.champion_model,
            "research_score": round(self.research_score, 4),
            "model_output": round(self.model_output, 4),
            "ensemble_output": round(self.ensemble_output, 4),
            "portfolio_exposure": round(self.portfolio_exposure, 4),
            "correlation_risk": round(self.correlation_risk, 4),
            "risk_metrics": self.risk_metrics,
            "feature_snapshot": self.feature_snapshot,
            "expert_weights": self.expert_weights,
        }


# ── DecisionReasoning ─────────────────────────────────────────────────────────


@dataclass
class DecisionReasoning:
    """Structured decision reasoning — never free-form chain-of-thought."""

    primary_reasons: list[str] = field(default_factory=list)
    supporting_factors: list[str] = field(default_factory=list)
    contradicting_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    uncertainty_sources: list[str] = field(default_factory=list)
    alternatives_considered: list[str] = field(default_factory=list)
    final_rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "primary_reasons": self.primary_reasons,
            "supporting_factors": self.supporting_factors,
            "contradicting_factors": self.contradicting_factors,
            "risk_factors": self.risk_factors,
            "uncertainty_sources": self.uncertainty_sources,
            "alternatives_considered": self.alternatives_considered,
            "final_rationale": self.final_rationale,
        }


# ── DecisionOutcome ───────────────────────────────────────────────────────────


@dataclass
class DecisionOutcome:
    """Recorded outcome for a completed decision."""

    outcome_type: OutcomeType
    actual_direction_correct: bool | None = None
    actual_pnl: float = 0.0  # price units
    actual_pnl_pct: float = 0.0  # fraction of account
    hit_tp: bool = False
    hit_sl: bool = False
    holding_time_h: float = 0.0
    market_event: str = ""
    notes: str = ""
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "outcome_type": self.outcome_type.value,
            "actual_direction_correct": self.actual_direction_correct,
            "actual_pnl": round(self.actual_pnl, 5),
            "actual_pnl_pct": round(self.actual_pnl_pct, 4),
            "hit_tp": self.hit_tp,
            "hit_sl": self.hit_sl,
            "holding_time_h": round(self.holding_time_h, 2),
            "market_event": self.market_event,
            "notes": self.notes,
            "recorded_at": round(self.recorded_at, 3),
        }


# ── DecisionScore ─────────────────────────────────────────────────────────────


@dataclass
class DecisionScore:
    """Quality score for a decision (10 factors)."""

    overall: float
    grade: DecisionGrade
    sub_scores: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 4),
            "grade": self.grade.value,
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
            "notes": self.notes,
        }


# ── CounterfactualResult ──────────────────────────────────────────────────────


@dataclass
class CounterfactualResult:
    """One counterfactual scenario for a completed decision."""

    scenario: str  # e.g. "WAIT instead of BUY"
    alternative_action: str
    estimated_pnl: float = 0.0
    estimated_pnl_pct: float = 0.0
    probability: float = 0.50  # P(this outcome under alternative)
    better_than_actual: bool = False
    delta_pnl: float = 0.0  # alternative - actual
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "alternative_action": self.alternative_action,
            "estimated_pnl": round(self.estimated_pnl, 5),
            "estimated_pnl_pct": round(self.estimated_pnl_pct, 4),
            "probability": round(self.probability, 4),
            "better_than_actual": self.better_than_actual,
            "delta_pnl": round(self.delta_pnl, 5),
            "notes": self.notes,
        }


# ── DecisionRecord ────────────────────────────────────────────────────────────


@dataclass
class DecisionRecord:
    """Master record for a single Signalium decision."""

    decision_id: str
    decision_type: DecisionType
    instrument: str
    timestamp: float
    snapshot: DecisionSnapshot
    reasoning: DecisionReasoning
    outcome: DecisionOutcome | None = None
    score: DecisionScore | None = None
    counterfactuals: list[CounterfactualResult] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    lineage_id: str = ""
    trade_id: str = ""  # link to TradeLifecycle
    notes: str = ""

    @classmethod
    def create(
        cls,
        decision_type: DecisionType,
        instrument: str,
        snapshot: DecisionSnapshot,
        reasoning: DecisionReasoning,
        tags: list[str] | None = None,
        trade_id: str = "",
        lineage_id: str = "",
    ) -> "DecisionRecord":
        # Canonical-symbol boundary guard — memory must never store broker
        # symbols.  Raises SymbolTranslationError for non-canonical inputs
        # (e.g. "EURUSD.r").  Canonical inputs (e.g. "EURUSD") pass silently.
        from trainer.symbols import assert_canonical

        assert_canonical(instrument)
        return cls(
            decision_id=str(uuid.uuid4()),
            decision_type=decision_type,
            instrument=instrument,
            timestamp=time.time(),
            snapshot=snapshot,
            reasoning=reasoning,
            tags=tags or [],
            trade_id=trade_id,
            lineage_id=lineage_id,
        )

    @property
    def has_outcome(self) -> bool:
        return self.outcome is not None

    @property
    def has_score(self) -> bool:
        return self.score is not None

    @property
    def is_entry_decision(self) -> bool:
        return self.decision_type in (DecisionType.BUY, DecisionType.SELL)

    @property
    def grade(self) -> str | None:
        return self.score.grade.value if self.score else None

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "instrument": self.instrument,
            "timestamp": round(self.timestamp, 3),
            "has_outcome": self.has_outcome,
            "has_score": self.has_score,
            "grade": self.grade,
            "trade_id": self.trade_id,
            "lineage_id": self.lineage_id,
            "tags": self.tags,
            "notes": self.notes,
            "snapshot": self.snapshot.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "score": self.score.to_dict() if self.score else None,
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
        }

    def summary(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "instrument": self.instrument,
            "timestamp": round(self.timestamp, 3),
            "confidence": round(self.snapshot.forecast_confidence, 4),
            "regime": self.snapshot.market_regime,
            "session": self.snapshot.session,
            "has_outcome": self.has_outcome,
            "grade": self.grade,
            "trade_id": self.trade_id,
        }
