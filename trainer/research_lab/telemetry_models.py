"""S20 — Telemetry event models for the Signalium → Trainer ingestion pipeline.

Eight streams arrive from Signalium:
  decisions     — QDE decision records with context
  trades        — Completed trade outcomes
  positions     — Position lifecycle events (open/scale/close)
  failures      — Execution-level failures from the live system
  reasoning     — Raw model output heads and feature values
  execution     — Order execution records with slippage/timing
  counterfactuals — What-if retrospective analysis
  lifecycle     — QDE position/session lifecycle state transitions

All events share three mandatory fields: model_version, symbol, timestamp.
Nothing is discarded — rows are append-only and never deleted.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _event_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ── Enumerations ───────────────────────────────────────────────────────────────


class TelemetryStream(str, Enum):
    DECISIONS = "decisions"
    TRADES = "trades"
    POSITIONS = "positions"
    FAILURES = "failures"
    REASONING = "reasoning"
    EXECUTION = "execution"
    COUNTERFACTUALS = "counterfactuals"
    LIFECYCLE = "lifecycle"


class PositionAction(str, Enum):
    OPEN = "OPEN"
    SCALE_IN = "SCALE_IN"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSE = "CLOSE"
    STOP_HIT = "STOP_HIT"
    TP_HIT = "TP_HIT"
    LIQUIDATED = "LIQUIDATED"


class ExecutionStatus(str, Enum):
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


class TelemetryFailureType(str, Enum):
    EXECUTION_REJECTION = "EXECUTION_REJECTION"
    SLIPPAGE_EXCEEDED = "SLIPPAGE_EXCEEDED"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    INVALID_SIGNAL = "INVALID_SIGNAL"
    RISK_BLOCK = "RISK_BLOCK"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class LifecycleEventKind(str, Enum):
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    STOP_ADJUSTED = "STOP_ADJUSTED"
    TP_ADJUSTED = "TP_ADJUSTED"
    SIZE_SCALED = "SIZE_SCALED"
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_ENDED = "SESSION_ENDED"
    MONITORING_STARTED = "MONITORING_STARTED"
    MONITORING_STOPPED = "MONITORING_STOPPED"
    RISK_STATE_CHANGED = "RISK_STATE_CHANGED"


# ── Storage dataclasses (one per stream) ───────────────────────────────────────


@dataclass
class DecisionEvent:
    """A QDE decision record with all context at the moment of decision."""

    event_id: str
    model_version: str
    symbol: str
    timestamp: float

    # Decision content
    direction: str  # "LONG" | "SHORT" | "FLAT"
    confidence: float
    reasoning_summary: str

    # Sizing
    entry_price: float | None
    stop_distance: float | None
    take_profit_distance: float | None
    position_size: float | None

    # Context
    session: str
    market_conditions: dict[str, Any]
    regime: str

    # Traceability
    decision_id: str
    ingested_at: float

    @classmethod
    def new(
        cls,
        model_version: str,
        symbol: str,
        timestamp: float,
        direction: str,
        confidence: float,
        reasoning_summary: str = "",
        entry_price: float | None = None,
        stop_distance: float | None = None,
        take_profit_distance: float | None = None,
        position_size: float | None = None,
        session: str = "",
        market_conditions: dict[str, Any] | None = None,
        regime: str = "",
        decision_id: str = "",
        event_id: str = "",
    ) -> DecisionEvent:
        return cls(
            event_id=event_id or _event_id("dec"),
            model_version=model_version,
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
            entry_price=entry_price,
            stop_distance=stop_distance,
            take_profit_distance=take_profit_distance,
            position_size=position_size,
            session=session,
            market_conditions=market_conditions or {},
            regime=regime,
            decision_id=decision_id or _event_id("did"),
            ingested_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream": TelemetryStream.DECISIONS.value,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "direction": self.direction,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "entry_price": self.entry_price,
            "stop_distance": self.stop_distance,
            "take_profit_distance": self.take_profit_distance,
            "position_size": self.position_size,
            "session": self.session,
            "market_conditions": self.market_conditions,
            "regime": self.regime,
            "decision_id": self.decision_id,
            "ingested_at": self.ingested_at,
        }


@dataclass
class TradeEvent:
    """A completed trade outcome."""

    event_id: str
    model_version: str
    symbol: str
    timestamp: float

    # Trade outcome
    trade_id: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    hold_duration_s: float
    exit_reason: str

    # Context at entry
    confidence_at_entry: float
    market_conditions: dict[str, Any]
    session: str
    regime: str

    ingested_at: float

    @classmethod
    def new(
        cls,
        model_version: str,
        symbol: str,
        timestamp: float,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: float,
        pnl: float,
        pnl_pct: float,
        hold_duration_s: float,
        exit_reason: str = "",
        confidence_at_entry: float = 0.0,
        market_conditions: dict[str, Any] | None = None,
        session: str = "",
        regime: str = "",
        trade_id: str = "",
        event_id: str = "",
    ) -> TradeEvent:
        return cls(
            event_id=event_id or _event_id("trd"),
            model_version=model_version,
            symbol=symbol,
            timestamp=timestamp,
            trade_id=trade_id or _event_id("tid"),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_duration_s=hold_duration_s,
            exit_reason=exit_reason,
            confidence_at_entry=confidence_at_entry,
            market_conditions=market_conditions or {},
            session=session,
            regime=regime,
            ingested_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream": TelemetryStream.TRADES.value,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "trade_id": self.trade_id,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "hold_duration_s": self.hold_duration_s,
            "exit_reason": self.exit_reason,
            "confidence_at_entry": self.confidence_at_entry,
            "market_conditions": self.market_conditions,
            "session": self.session,
            "regime": self.regime,
            "ingested_at": self.ingested_at,
        }


@dataclass
class PositionEvent:
    """A position lifecycle event (open, scale, partial close, close, stop/TP hit)."""

    event_id: str
    model_version: str
    symbol: str
    timestamp: float

    position_id: str
    action: str  # PositionAction value
    size: float
    price: float
    confidence: float
    cumulative_pnl: float
    market_conditions: dict[str, Any]
    position_actions: list[dict[str, Any]]  # history of actions on this position

    ingested_at: float

    @classmethod
    def new(
        cls,
        model_version: str,
        symbol: str,
        timestamp: float,
        action: str,
        size: float,
        price: float,
        confidence: float = 0.0,
        cumulative_pnl: float = 0.0,
        market_conditions: dict[str, Any] | None = None,
        position_actions: list[dict[str, Any]] | None = None,
        position_id: str = "",
        event_id: str = "",
    ) -> PositionEvent:
        return cls(
            event_id=event_id or _event_id("pos"),
            model_version=model_version,
            symbol=symbol,
            timestamp=timestamp,
            position_id=position_id or _event_id("pid"),
            action=action,
            size=size,
            price=price,
            confidence=confidence,
            cumulative_pnl=cumulative_pnl,
            market_conditions=market_conditions or {},
            position_actions=position_actions or [],
            ingested_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream": TelemetryStream.POSITIONS.value,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "position_id": self.position_id,
            "action": self.action,
            "size": self.size,
            "price": self.price,
            "confidence": self.confidence,
            "cumulative_pnl": self.cumulative_pnl,
            "market_conditions": self.market_conditions,
            "position_actions": self.position_actions,
            "ingested_at": self.ingested_at,
        }


@dataclass
class TelemetryFailureEvent:
    """An execution-level failure reported by the live Signalium system.

    Distinct from research_lab FailureEvent (which is researcher-recorded).
    These arrive from the live system and feed the Failure Investigation Engine.
    """

    event_id: str
    model_version: str
    symbol: str
    timestamp: float

    failure_type: str  # TelemetryFailureType value
    severity: str
    details: str
    affected_order_id: str

    ingested_at: float

    @classmethod
    def new(
        cls,
        model_version: str,
        symbol: str,
        timestamp: float,
        failure_type: str,
        severity: str = "MEDIUM",
        details: str = "",
        affected_order_id: str = "",
        event_id: str = "",
    ) -> TelemetryFailureEvent:
        return cls(
            event_id=event_id or _event_id("tfail"),
            model_version=model_version,
            symbol=symbol,
            timestamp=timestamp,
            failure_type=failure_type,
            severity=severity,
            details=details,
            affected_order_id=affected_order_id,
            ingested_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream": TelemetryStream.FAILURES.value,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "failure_type": self.failure_type,
            "severity": self.severity,
            "details": self.details,
            "affected_order_id": self.affected_order_id,
            "ingested_at": self.ingested_at,
        }


@dataclass
class ReasoningEvent:
    """Raw model output heads and feature values for a given decision."""

    event_id: str
    model_version: str
    symbol: str
    timestamp: float

    decision_id: str
    output_heads: dict[str, Any]          # raw model output tensors/values
    feature_values: dict[str, float]      # input features at inference time
    confidence_breakdown: dict[str, float]  # per-head confidence components

    ingested_at: float

    @classmethod
    def new(
        cls,
        model_version: str,
        symbol: str,
        timestamp: float,
        decision_id: str,
        output_heads: dict[str, Any] | None = None,
        feature_values: dict[str, float] | None = None,
        confidence_breakdown: dict[str, float] | None = None,
        event_id: str = "",
    ) -> ReasoningEvent:
        return cls(
            event_id=event_id or _event_id("rsn"),
            model_version=model_version,
            symbol=symbol,
            timestamp=timestamp,
            decision_id=decision_id,
            output_heads=output_heads or {},
            feature_values=feature_values or {},
            confidence_breakdown=confidence_breakdown or {},
            ingested_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream": TelemetryStream.REASONING.value,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "decision_id": self.decision_id,
            "output_heads": self.output_heads,
            "feature_values": self.feature_values,
            "confidence_breakdown": self.confidence_breakdown,
            "ingested_at": self.ingested_at,
        }


@dataclass
class ExecutionEvent:
    """Order execution record with slippage and timing data."""

    event_id: str
    model_version: str
    symbol: str
    timestamp: float

    order_id: str
    order_type: str
    requested_price: float
    executed_price: float
    slippage: float       # executed_price - requested_price (signed)
    slippage_pct: float
    execution_time_ms: float
    venue: str
    status: str           # ExecutionStatus value
    size: float

    ingested_at: float

    @classmethod
    def new(
        cls,
        model_version: str,
        symbol: str,
        timestamp: float,
        order_type: str,
        requested_price: float,
        executed_price: float,
        size: float,
        execution_time_ms: float = 0.0,
        venue: str = "",
        status: str = "FILLED",
        order_id: str = "",
        event_id: str = "",
    ) -> ExecutionEvent:
        slippage = executed_price - requested_price
        slippage_pct = (slippage / requested_price * 100.0) if requested_price else 0.0
        return cls(
            event_id=event_id or _event_id("exec"),
            model_version=model_version,
            symbol=symbol,
            timestamp=timestamp,
            order_id=order_id or _event_id("oid"),
            order_type=order_type,
            requested_price=requested_price,
            executed_price=executed_price,
            slippage=slippage,
            slippage_pct=slippage_pct,
            execution_time_ms=execution_time_ms,
            venue=venue,
            status=status,
            size=size,
            ingested_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream": TelemetryStream.EXECUTION.value,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "order_id": self.order_id,
            "order_type": self.order_type,
            "requested_price": self.requested_price,
            "executed_price": self.executed_price,
            "slippage": self.slippage,
            "slippage_pct": self.slippage_pct,
            "execution_time_ms": self.execution_time_ms,
            "venue": self.venue,
            "status": self.status,
            "size": self.size,
            "ingested_at": self.ingested_at,
        }


@dataclass
class CounterfactualEvent:
    """Hindsight what-if analysis for a completed trade."""

    event_id: str
    model_version: str
    symbol: str
    timestamp: float

    trade_id: str
    what_if_description: str
    actual_pnl: float
    counterfactual_pnl: float
    pnl_delta: float           # counterfactual_pnl - actual_pnl
    actual_outcome: dict[str, Any]
    counterfactual_outcome: dict[str, Any]

    ingested_at: float

    @classmethod
    def new(
        cls,
        model_version: str,
        symbol: str,
        timestamp: float,
        trade_id: str,
        what_if_description: str,
        actual_pnl: float,
        counterfactual_pnl: float,
        actual_outcome: dict[str, Any] | None = None,
        counterfactual_outcome: dict[str, Any] | None = None,
        event_id: str = "",
    ) -> CounterfactualEvent:
        return cls(
            event_id=event_id or _event_id("cf"),
            model_version=model_version,
            symbol=symbol,
            timestamp=timestamp,
            trade_id=trade_id,
            what_if_description=what_if_description,
            actual_pnl=actual_pnl,
            counterfactual_pnl=counterfactual_pnl,
            pnl_delta=counterfactual_pnl - actual_pnl,
            actual_outcome=actual_outcome or {},
            counterfactual_outcome=counterfactual_outcome or {},
            ingested_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream": TelemetryStream.COUNTERFACTUALS.value,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "trade_id": self.trade_id,
            "what_if_description": self.what_if_description,
            "actual_pnl": self.actual_pnl,
            "counterfactual_pnl": self.counterfactual_pnl,
            "pnl_delta": self.pnl_delta,
            "actual_outcome": self.actual_outcome,
            "counterfactual_outcome": self.counterfactual_outcome,
            "ingested_at": self.ingested_at,
        }


@dataclass
class LifecycleEvent:
    """A QDE position or session lifecycle state transition."""

    event_id: str
    model_version: str
    symbol: str
    timestamp: float

    kind: str            # LifecycleEventKind value
    previous_state: str
    new_state: str
    trigger: str
    metadata: dict[str, Any]

    ingested_at: float

    @classmethod
    def new(
        cls,
        model_version: str,
        symbol: str,
        timestamp: float,
        kind: str,
        previous_state: str = "",
        new_state: str = "",
        trigger: str = "",
        metadata: dict[str, Any] | None = None,
        event_id: str = "",
    ) -> LifecycleEvent:
        return cls(
            event_id=event_id or _event_id("lce"),
            model_version=model_version,
            symbol=symbol,
            timestamp=timestamp,
            kind=kind,
            previous_state=previous_state,
            new_state=new_state,
            trigger=trigger,
            metadata=metadata or {},
            ingested_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream": TelemetryStream.LIFECYCLE.value,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "trigger": self.trigger,
            "metadata": self.metadata,
            "ingested_at": self.ingested_at,
        }


# ── Query filter helpers ───────────────────────────────────────────────────────


@dataclass
class TelemetryQuery:
    """Parameters for warehouse read queries."""

    model_version: str | None = None
    symbol: str | None = None
    since: float | None = None   # unix timestamp lower bound
    until: float | None = None   # unix timestamp upper bound
    limit: int = 1000
