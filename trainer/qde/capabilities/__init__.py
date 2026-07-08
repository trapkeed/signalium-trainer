"""Phase 28.3 — QDE Cognitive Capabilities.

Internal faculties of the Quant Decision Engine.

ARCHITECTURAL CONTRACT
----------------------
- Capabilities are READ-ONLY advisors.
- No capability produces a QDEDecision.
- No capability instructs a trade, modifies an order, or allocates capital.
- All capability outputs are consumed exclusively by QDEIntelligence.
- Only QDEIntelligence.reason() produces QDEDecision objects.
- Only QDEDecision objects leave the trainer.
"""

from trainer.qde.capabilities.cap_models import (
    BreakEvenSignal,
    DecisionType,
    EntrySignal,
    ExitSignal,
    MarketSnapshot,
    OpenTrade,
    PartialCloseSignal,
    QDEDecision,
    ScalingSignal,
)
from trainer.qde.capabilities.capital_preservation import CapitalPreservationCapability
from trainer.qde.capabilities.expected_value_reasoning import ExpectedValueReasoningCapability
from trainer.qde.capabilities.exposure_reasoning import ExposureReasoningCapability
from trainer.qde.capabilities.learning import LearningCapability
from trainer.qde.capabilities.market_memory import MarketMemoryCapability
from trainer.qde.capabilities.market_understanding import MarketUnderstandingCapability
from trainer.qde.capabilities.opportunity_discovery import OpportunityDiscoveryCapability
from trainer.qde.capabilities.position_management import PositionManagementCapability
from trainer.qde.capabilities.reflection import ReflectionCapability
from trainer.qde.capabilities.risk_reasoning import RiskReasoningCapability
from trainer.qde.capabilities.trade_management import TradeManagementCapability

__all__ = [
    # Shared models
    "MarketSnapshot",
    "OpenTrade",
    "QDEDecision",
    "DecisionType",
    "EntrySignal",
    "ExitSignal",
    "ScalingSignal",
    "PartialCloseSignal",
    "BreakEvenSignal",
    # Capabilities
    "MarketUnderstandingCapability",
    "OpportunityDiscoveryCapability",
    "ExpectedValueReasoningCapability",
    "RiskReasoningCapability",
    "ExposureReasoningCapability",
    "TradeManagementCapability",
    "PositionManagementCapability",
    "CapitalPreservationCapability",
    "MarketMemoryCapability",
    "ReflectionCapability",
    "LearningCapability",
]
