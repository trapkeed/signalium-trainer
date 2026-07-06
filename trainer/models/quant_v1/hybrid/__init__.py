"""Hybrid Representation Learning package — Phase 14.9A.

Provides a modular multi-stream encoder that produces a shared latent
representation consumed by all existing prediction heads.  The architecture
is fully backward-compatible: when ``HybridEncoderConfig.enabled=False``
(the default), the model uses its original InputEncoder → SequenceEncoder
path with zero behavior change.

Streams:
  1. RawMarketStream         — OHLCV + Spread via LSTM/GRU
  2. MultiTimeframeStream    — per-timeframe independent encoders
  3. EngineeredFeatureStream — MLP over pre-computed features
  4. ContextEmbeddingStream  — learned categorical embeddings
  5. MarketMemoryStream      — Market Memory Engine state MLP
  6. DynamicFeatureGate      — context-conditioned multiplicative gating
  7. AdaptiveFusionLayer     — concat / weighted / attention fusion
"""

from trainer.models.quant_v1.hybrid.config import (
    SUPPORTED_TIMEFRAMES,
    AdaptiveFusionConfig,
    ContextEmbeddingStreamConfig,
    DynamicFeatureGateConfig,
    EmbeddingSpec,
    EngineeredFeatureStreamConfig,
    FusionStrategy,
    HybridEncoderConfig,
    MarketMemoryStreamConfig,
    MultiTimeframeStreamConfig,
    RawMarketStreamConfig,
    SequenceBackboneType,
    TimeframeEncoderConfig,
)
from trainer.models.quant_v1.hybrid.embeddings import ContextEmbeddingStream
from trainer.models.quant_v1.hybrid.encoder import HybridEncoder
from trainer.models.quant_v1.hybrid.fusion import AdaptiveFusionLayer
from trainer.models.quant_v1.hybrid.gate import DynamicFeatureGate
from trainer.models.quant_v1.hybrid.input import HybridInput
from trainer.models.quant_v1.hybrid.memory import MarketMemoryStream
from trainer.models.quant_v1.hybrid.streams import (
    EngineeredFeatureStream,
    MultiTimeframeStream,
    RawMarketStream,
)

__all__ = [
    # Config
    "HybridEncoderConfig",
    "RawMarketStreamConfig",
    "MultiTimeframeStreamConfig",
    "TimeframeEncoderConfig",
    "EngineeredFeatureStreamConfig",
    "ContextEmbeddingStreamConfig",
    "EmbeddingSpec",
    "MarketMemoryStreamConfig",
    "DynamicFeatureGateConfig",
    "AdaptiveFusionConfig",
    "FusionStrategy",
    "SequenceBackboneType",
    "SUPPORTED_TIMEFRAMES",
    # Input
    "HybridInput",
    # Modules
    "HybridEncoder",
    "RawMarketStream",
    "MultiTimeframeStream",
    "EngineeredFeatureStream",
    "ContextEmbeddingStream",
    "MarketMemoryStream",
    "DynamicFeatureGate",
    "AdaptiveFusionLayer",
]
