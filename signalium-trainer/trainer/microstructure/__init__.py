"""Phase 22 — Institutional Market Microstructure Intelligence Engine (MMIE)."""

from trainer.microstructure.micro_types import (
    MarketState,
    MomentumState,
    VolatilityState,
    LiquidityState,
    TrendStrength,
    DataGranularity,
    STATE_PRIORITY,
    EMBEDDING_DIMS,
    EMBEDDING_TOTAL,
    trend_strength_from_score,
)

from trainer.microstructure.micro_models import (
    OHLCBar,
    MicrostructureFeatures,
    MicrostructureScore,
    MarketStateRecord,
    MicrostructureEmbedding,
)

from trainer.microstructure.price_action      import PriceActionAnalyzer
from trainer.microstructure.volatility_micro  import VolatilityMicrostructure
from trainer.microstructure.momentum_micro    import MomentumMicrostructure
from trainer.microstructure.liquidity_proxy   import LiquidityProxy
from trainer.microstructure.market_efficiency import MarketEfficiency
from trainer.microstructure.micro_scorer      import MicrostructureScorer
from trainer.microstructure.state_classifier  import MarketStateClassifier
from trainer.microstructure.micro_embeddings  import MicrostructureEmbedder
from trainer.microstructure.tick_interface    import TickInterface, TickBar
from trainer.microstructure.order_flow_interface import (
    OrderFlowInterface,
    OrderFlowBar,
    OrderBookLevel,
)
from trainer.microstructure.micro_database   import MicrostructureDatabase
from trainer.microstructure.micro_dashboard  import MicrostructureDashboard
from trainer.microstructure.micro_api        import MicrostructureAPI

__all__ = [
    # Types
    "MarketState", "MomentumState", "VolatilityState", "LiquidityState",
    "TrendStrength", "DataGranularity", "STATE_PRIORITY",
    "EMBEDDING_DIMS", "EMBEDDING_TOTAL", "trend_strength_from_score",
    # Models
    "OHLCBar", "MicrostructureFeatures", "MicrostructureScore",
    "MarketStateRecord", "MicrostructureEmbedding",
    # Analyzers
    "PriceActionAnalyzer", "VolatilityMicrostructure", "MomentumMicrostructure",
    "LiquidityProxy", "MarketEfficiency",
    # Core engines
    "MicrostructureScorer", "MarketStateClassifier", "MicrostructureEmbedder",
    # Interfaces
    "TickInterface", "TickBar",
    "OrderFlowInterface", "OrderFlowBar", "OrderBookLevel",
    # Infrastructure
    "MicrostructureDatabase", "MicrostructureDashboard",
    # Unified API
    "MicrostructureAPI",
]
