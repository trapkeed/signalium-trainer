"""Phase 21.5 — Core lifecycle data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trainer.lifecycle.lifecycle_states import LifecycleState

# ── MarketContext ─────────────────────────────────────────────────────────────


@dataclass
class MarketContext:
    """Snapshot of market conditions at a point in time."""

    price: float = 0.0
    atr: float = 0.0
    spread: float = 0.0
    volatility: float = 0.01
    momentum: float = 0.0  # -1 to +1
    news_risk: float = 0.0  # 0-1
    session: str = "LONDON"
    trend_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "price": round(self.price, 5),
            "atr": round(self.atr, 5),
            "spread": round(self.spread, 5),
            "volatility": round(self.volatility, 4),
            "momentum": round(self.momentum, 4),
            "news_risk": round(self.news_risk, 4),
            "session": self.session,
            "trend_valid": self.trend_valid,
        }


# ── PositionMetrics ───────────────────────────────────────────────────────────


@dataclass
class PositionMetrics:
    """Live metrics for an open position."""

    lots: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealised_pnl: float = 0.0  # price units
    unrealised_pct: float = 0.0  # fraction of account
    stop_price: float = 0.0
    target_price: float = 0.0
    time_in_trade_h: float = 0.0
    highest_price: float = 0.0  # MFE tracking
    lowest_price: float = 0.0  # MAE tracking

    def to_dict(self) -> dict:
        return {
            "lots": round(self.lots, 2),
            "entry_price": round(self.entry_price, 5),
            "current_price": round(self.current_price, 5),
            "unrealised_pnl": round(self.unrealised_pnl, 5),
            "unrealised_pct": round(self.unrealised_pct, 4),
            "stop_price": round(self.stop_price, 5),
            "target_price": round(self.target_price, 5),
            "time_in_trade_h": round(self.time_in_trade_h, 2),
            "highest_price": round(self.highest_price, 5),
            "lowest_price": round(self.lowest_price, 5),
        }


# ── StateRecord ───────────────────────────────────────────────────────────────


@dataclass
class StateRecord:
    """One entry in the trade's state history."""

    state: LifecycleState
    timestamp: float
    confidence: float  # 0-1
    reason: str
    market_context: MarketContext | None = None
    position_metrics: PositionMetrics | None = None
    duration_h: float = 0.0  # filled in when next state is entered

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "timestamp": round(self.timestamp, 3),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "duration_h": round(self.duration_h, 3),
            "market_context": self.market_context.to_dict() if self.market_context else None,
            "position_metrics": self.position_metrics.to_dict() if self.position_metrics else None,
        }


# ── LifecycleEvent ────────────────────────────────────────────────────────────


class EventType(str):
    SETUP_CONFIRMED = "SETUP_CONFIRMED"
    MOMENTUM_INCREASED = "MOMENTUM_INCREASED"
    MOMENTUM_DECREASED = "MOMENTUM_DECREASED"
    VOLATILITY_EXPANDED = "VOLATILITY_EXPANDED"
    VOLATILITY_CONTRACTED = "VOLATILITY_CONTRACTED"
    LIQUIDITY_DETERIORATED = "LIQUIDITY_DETERIORATED"
    LIQUIDITY_IMPROVED = "LIQUIDITY_IMPROVED"
    CONFIDENCE_DROPPED = "CONFIDENCE_DROPPED"
    CONFIDENCE_RAISED = "CONFIDENCE_RAISED"
    TARGET_APPROACHING = "TARGET_APPROACHING"
    RISK_INCREASED = "RISK_INCREASED"
    RISK_DECREASED = "RISK_DECREASED"
    TRAILING_ACTIVATED = "TRAILING_ACTIVATED"
    STATE_CHANGED = "STATE_CHANGED"
    NEWS_RISK_ELEVATED = "NEWS_RISK_ELEVATED"
    SPREAD_WIDENED = "SPREAD_WIDENED"
    TREND_INVALID = "TREND_INVALID"
    BREAKEVEN_TRIGGERED = "BREAKEVEN_TRIGGERED"
    PARTIAL_CLOSE_TRIGGERED = "PARTIAL_CLOSE_TRIGGERED"
    RUNNER_ACTIVATED = "RUNNER_ACTIVATED"


@dataclass
class LifecycleEvent:
    """A significant event in the trade's lifecycle."""

    event_id: str
    trade_id: str
    event_type: str
    state: LifecycleState
    timestamp: float
    description: str
    severity: str = "INFO"  # INFO | WARN | CRITICAL
    data: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        trade_id: str,
        event_type: str,
        state: LifecycleState,
        description: str,
        severity: str = "INFO",
        data: dict | None = None,
    ) -> LifecycleEvent:
        return cls(
            event_id=str(uuid.uuid4())[:8],
            trade_id=trade_id,
            event_type=event_type,
            state=state,
            timestamp=time.time(),
            description=description,
            severity=severity,
            data=data or {},
        )

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "trade_id": self.trade_id,
            "event_type": self.event_type,
            "state": self.state.value,
            "timestamp": round(self.timestamp, 3),
            "description": self.description,
            "severity": self.severity,
            "data": self.data,
        }


# ── TransitionResult ─────────────────────────────────────────────────────────


@dataclass
class TransitionResult:
    """Result of a state transition evaluation."""

    from_state: LifecycleState
    to_state: LifecycleState
    allowed: bool
    conditions_met: list[str] = field(default_factory=list)
    conditions_failed: list[str] = field(default_factory=list)
    confidence: float = 0.0
    expected_benefit: float = 0.0  # 0-1
    risk_rating: str = "MEDIUM"  # LOW | MEDIUM | HIGH | CRITICAL
    reason: str = ""
    timeout_hours: float = 0.0
    # True only when the transition is structurally invalid (not in the FSM
    # graph — a data-integrity invariant). False when `allowed=False` is due
    # to strategy-level context/confidence rules, which are advisory only —
    # QDE (or another decision layer) is the sole authority on whether to
    # proceed despite unmet advisory conditions.
    hard_block: bool = False

    def to_dict(self) -> dict:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "allowed": self.allowed,
            "hard_block": self.hard_block,
            "conditions_met": self.conditions_met,
            "conditions_failed": self.conditions_failed,
            "confidence": round(self.confidence, 4),
            "expected_benefit": round(self.expected_benefit, 4),
            "risk_rating": self.risk_rating,
            "reason": self.reason,
            "timeout_hours": self.timeout_hours,
        }


# ── LifecycleConfidence ───────────────────────────────────────────────────────


@dataclass
class LifecycleConfidence:
    """Confidence and forward projections for a lifecycle state."""

    state: LifecycleState
    confidence: float  # 0-1
    expected_duration_h: float
    expected_reward: float  # fraction
    expected_risk: float  # fraction
    transition_prob: float  # P(advance to next state)
    failure_prob: float  # P(stop/cancel/expire)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "expected_duration_h": round(self.expected_duration_h, 2),
            "expected_reward": round(self.expected_reward, 4),
            "expected_risk": round(self.expected_risk, 4),
            "transition_prob": round(self.transition_prob, 4),
            "failure_prob": round(self.failure_prob, 4),
            "notes": self.notes,
        }
