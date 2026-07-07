"""Phase 21.5 — TradeLifecycle: the living trade object.

Every trade is a stateful entity that transitions through the lifecycle
from IDEA to CLOSED (or an alternative terminal state).

Does NOT execute trades — recommendation-only.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.ai_reasoning import AiReasoningEngine, ReasoningReport
from trainer.lifecycle.event_engine import EventEngine
from trainer.lifecycle.lifecycle_confidence import LifecycleConfidenceEngine
from trainer.lifecycle.lifecycle_models import (
    LifecycleConfidence,
    LifecycleEvent,
    MarketContext,
    PositionMetrics,
    StateRecord,
)
from trainer.lifecycle.lifecycle_states import (
    POSITION_OPEN_STATES,
    LifecycleState,
    can_transition,
    is_terminal,
)
from trainer.lifecycle.position_monitor import PositionManagementPlan, PositionMonitor
from trainer.lifecycle.state_machine import StateMachine
from trainer.lifecycle.transition_engine import TransitionEngine, TransitionRecommendation


@dataclass
class TradeLifecycle:
    """The complete living trade object.

    Created at IDEA state. Transitions through the state machine.
    Maintains full history, events, and all recommendations.

    Does NOT place trades.
    """

    trade_id: str
    instrument: str
    direction: str  # BUY | SELL
    created_at: float

    # Current state
    current_state: LifecycleState = LifecycleState.IDEA
    confidence: float = 0.50

    # History and events
    state_history: list[StateRecord] = field(default_factory=list)
    events: list[LifecycleEvent] = field(default_factory=list)

    # Latest context
    last_context: MarketContext | None = None
    last_metrics: PositionMetrics | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def new(
        cls,
        instrument: str,
        direction: str,
        confidence: float = 0.50,
        context: MarketContext | None = None,
        tags: list[str] | None = None,
    ) -> TradeLifecycle:
        trade_id = str(uuid.uuid4())[:12]
        tl = cls(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            created_at=time.time(),
            confidence=confidence,
            last_context=context,
        )
        tl.tags = tags or []
        # Record initial state
        tl.state_history.append(
            StateRecord(
                state=LifecycleState.IDEA,
                timestamp=tl.created_at,
                confidence=confidence,
                reason="Trade idea created",
                market_context=context,
            )
        )
        logger.info(f"TradeLifecycle created: {trade_id} {instrument} {direction}")
        return tl

    # ── State management ──────────────────────────────────────────────────────

    def transition_to(
        self,
        new_state: LifecycleState,
        reason: str,
        confidence: float | None = None,
        context: MarketContext | None = None,
        metrics: PositionMetrics | None = None,
        sm: StateMachine | None = None,
    ) -> bool:
        """Attempt a state transition. Returns True if successful."""

        if is_terminal(self.current_state):
            logger.debug(
                f"TradeLifecycle {self.trade_id}: already in terminal state "
                f"{self.current_state.value}"
            )
            return False

        if not can_transition(self.current_state, new_state):
            logger.warning(
                f"TradeLifecycle {self.trade_id}: invalid transition "
                f"{self.current_state.value} → {new_state.value}"
            )
            return False

        # Validate with state machine if provided
        if sm is not None:
            result = sm.validate_transition(
                self.current_state,
                new_state,
                context or self.last_context,
                confidence or self.confidence,
            )
            if not result.allowed:
                logger.debug(
                    f"TradeLifecycle {self.trade_id}: transition blocked: "
                    f"{result.conditions_failed}"
                )
                return False

        prev_state = self.current_state
        new_conf = confidence if confidence is not None else self.confidence
        now = time.time()

        # Close out the last state record duration
        if self.state_history:
            last = self.state_history[-1]
            last.duration_h = round((now - last.timestamp) / 3600.0, 3)

        # Update lifecycle
        self.current_state = new_state
        self.confidence = new_conf
        self.last_context = context or self.last_context
        self.last_metrics = metrics or self.last_metrics
        self.updated_at = now

        # Record state
        self.state_history.append(
            StateRecord(
                state=new_state,
                timestamp=now,
                confidence=new_conf,
                reason=reason,
                market_context=context or self.last_context,
                position_metrics=metrics or self.last_metrics,
            )
        )

        # Emit state-change event
        ev_engine = EventEngine()
        self.events.append(
            ev_engine.emit_state_change(self.trade_id, prev_state, new_state, reason, new_conf)
        )

        logger.info(
            f"TradeLifecycle {self.trade_id}: {prev_state.value} → {new_state.value} "
            f"(confidence={new_conf:.2f})"
        )
        return True

    # ── Recommendations ───────────────────────────────────────────────────────

    def recommend_transition(
        self,
        context: MarketContext | None = None,
        te: TransitionEngine | None = None,
    ) -> TransitionRecommendation:
        """Ask the transition engine for the best next state."""
        engine = te or TransitionEngine()
        return engine.evaluate(
            self.current_state,
            self.confidence,
            context or self.last_context,
        )

    def position_management(
        self,
        context: MarketContext | None = None,
        metrics: PositionMetrics | None = None,
        pm: PositionMonitor | None = None,
    ) -> PositionManagementPlan:
        """Get position management recommendation for current state."""
        monitor = pm or PositionMonitor()
        return monitor.evaluate(
            self.instrument,
            self.current_state,
            metrics or self.last_metrics,
            context or self.last_context,
            self.confidence,
        )

    def lifecycle_confidence(
        self,
        context: MarketContext | None = None,
        lce: LifecycleConfidenceEngine | None = None,
    ) -> LifecycleConfidence:
        """Get confidence assessment for current state."""
        engine = lce or LifecycleConfidenceEngine()
        return engine.compute(
            self.current_state,
            self.confidence,
            context or self.last_context,
        )

    def detect_events(
        self,
        context: MarketContext | None = None,
        metrics: PositionMetrics | None = None,
        prev_context: MarketContext | None = None,
        prev_conf: float = 0.65,
        ee: EventEngine | None = None,
    ) -> list[LifecycleEvent]:
        """Detect events and add to history."""
        engine = ee or EventEngine()
        new_events = engine.detect_events(
            self.trade_id,
            self.current_state,
            context or self.last_context,
            metrics or self.last_metrics,
            prev_context,
            prev_conf,
            self.confidence,
        )
        self.events.extend(new_events)
        return new_events

    def reasoning(
        self,
        context: MarketContext | None = None,
        metrics: PositionMetrics | None = None,
        prev_conf: float = 0.65,
        are: AiReasoningEngine | None = None,
        lce: LifecycleConfidenceEngine | None = None,
    ) -> ReasoningReport:
        """Generate institutional explanation for current state."""
        engine = are or AiReasoningEngine()
        conf_eng = lce or LifecycleConfidenceEngine()
        lc_conf = conf_eng.compute(
            self.current_state,
            self.confidence,
            context or self.last_context,
        )
        return engine.generate(
            self.trade_id,
            self.instrument,
            self.direction,
            self.current_state,
            self.confidence,
            context or self.last_context,
            metrics or self.last_metrics,
            lc_conf,
            prev_conf,
        )

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        return is_terminal(self.current_state)

    @property
    def is_position_open(self) -> bool:
        return self.current_state in POSITION_OPEN_STATES

    @property
    def age_hours(self) -> float:
        return round((time.time() - self.created_at) / 3600.0, 3)

    @property
    def time_in_current_state_hours(self) -> float:
        if not self.state_history:
            return 0.0
        return round((time.time() - self.state_history[-1].timestamp) / 3600.0, 3)

    @property
    def state_count(self) -> int:
        return len(self.state_history)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def critical_events(self) -> list[LifecycleEvent]:
        return [e for e in self.events if e.severity == "CRITICAL"]

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "instrument": self.instrument,
            "direction": self.direction,
            "current_state": self.current_state.value,
            "confidence": round(self.confidence, 4),
            "is_terminal": self.is_terminal,
            "is_position_open": self.is_position_open,
            "age_hours": self.age_hours,
            "time_in_state_h": self.time_in_current_state_hours,
            "state_count": self.state_count,
            "event_count": self.event_count,
            "critical_events": len(self.critical_events),
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "tags": self.tags,
            "state_history": [r.to_dict() for r in self.state_history],
            "events": [e.to_dict() for e in self.events],
        }

    def summary(self) -> dict:
        """Compact summary for dashboard display."""
        return {
            "trade_id": self.trade_id,
            "instrument": self.instrument,
            "direction": self.direction,
            "state": self.current_state.value,
            "confidence": round(self.confidence, 4),
            "age_hours": self.age_hours,
            "state_count": self.state_count,
            "event_count": self.event_count,
            "critical_events": len(self.critical_events),
            "is_terminal": self.is_terminal,
        }
