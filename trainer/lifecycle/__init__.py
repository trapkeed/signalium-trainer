"""Phase 21.5 — Trade Lifecycle Intelligence Engine.

Exports all public names from the lifecycle package.
"""

from trainer.lifecycle.ai_reasoning import AiReasoningEngine, ReasoningReport
from trainer.lifecycle.event_engine import EventEngine
from trainer.lifecycle.lifecycle_analytics import (
    LifecycleAnalytics,
    LifecycleAnalyticsReport,
    StateAnalytics,
)
from trainer.lifecycle.lifecycle_api import ApiResponse, LifecycleAPI
from trainer.lifecycle.lifecycle_confidence import LifecycleConfidenceEngine
from trainer.lifecycle.lifecycle_dashboard import LifecycleDashboard
from trainer.lifecycle.lifecycle_database import LifecycleDatabase
from trainer.lifecycle.lifecycle_models import (
    EventType,
    LifecycleConfidence,
    LifecycleEvent,
    MarketContext,
    PositionMetrics,
    StateRecord,
    TransitionResult,
)
from trainer.lifecycle.lifecycle_optimizer import (
    LifecycleOptimizer,
    OptimisationFinding,
    OptimisationReport,
)
from trainer.lifecycle.lifecycle_states import (
    CLOSING_STATES,
    POSITION_OPEN_STATES,
    PRE_ENTRY_STATES,
    STATE_EXPECTED_DURATION_HOURS,
    STATE_ORDER,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    LifecycleState,
    can_transition,
    is_position_open,
    is_terminal,
    valid_next_states,
)
from trainer.lifecycle.position_monitor import (
    ManagementRecommendation,
    PositionAction,
    PositionManagementPlan,
    PositionMonitor,
)
from trainer.lifecycle.state_machine import StateMachine, TransitionRule
from trainer.lifecycle.trade_lifecycle import TradeLifecycle
from trainer.lifecycle.transition_engine import TransitionEngine, TransitionRecommendation

__all__ = [
    # States
    "LifecycleState",
    "TERMINAL_STATES",
    "CLOSING_STATES",
    "POSITION_OPEN_STATES",
    "PRE_ENTRY_STATES",
    "VALID_TRANSITIONS",
    "STATE_ORDER",
    "STATE_EXPECTED_DURATION_HOURS",
    "is_terminal",
    "is_position_open",
    "valid_next_states",
    "can_transition",
    # Models
    "MarketContext",
    "PositionMetrics",
    "StateRecord",
    "LifecycleEvent",
    "EventType",
    "TransitionResult",
    "LifecycleConfidence",
    # State machine
    "StateMachine",
    "TransitionRule",
    # Transition engine
    "TransitionEngine",
    "TransitionRecommendation",
    # Position monitor
    "PositionMonitor",
    "PositionAction",
    "PositionManagementPlan",
    "ManagementRecommendation",
    # Confidence
    "LifecycleConfidenceEngine",
    # Events
    "EventEngine",
    # Reasoning
    "AiReasoningEngine",
    "ReasoningReport",
    # Core
    "TradeLifecycle",
    # Analytics
    "LifecycleAnalytics",
    "LifecycleAnalyticsReport",
    "StateAnalytics",
    # Optimizer
    "LifecycleOptimizer",
    "OptimisationReport",
    "OptimisationFinding",
    # Infrastructure
    "LifecycleDatabase",
    "LifecycleDashboard",
    "LifecycleAPI",
    "ApiResponse",
]
