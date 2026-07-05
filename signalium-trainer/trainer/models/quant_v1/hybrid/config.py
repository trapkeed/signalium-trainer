"""Configuration dataclasses for Phase 14.9A Hybrid Representation Learning.

Every stream can be independently enabled or disabled.  When the entire
``HybridEncoderConfig`` is disabled (``enabled=False``), the model falls back
to the original InputEncoder → SequenceEncoder path with zero behavior change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FusionStrategy(str, Enum):
    """Strategy used by the AdaptiveFusionLayer to combine stream outputs."""

    CONCAT = "concat"
    """Project concatenated stream tensors to latent_dim."""

    WEIGHTED = "weighted"
    """Learnable per-stream softmax weights → weighted sum → projection."""

    ATTENTION = "attention"
    """Treat each stream as a token; cross-attention pool → projection."""


class SequenceBackboneType(str, Enum):
    LSTM = "lstm"
    GRU = "gru"


SUPPORTED_TIMEFRAMES: list[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

CONTEXT_CATEGORIES: list[str] = [
    "instrument",
    "timeframe",
    "session",
    "hour",
    "day",
    "week",
    "month",
    "regime",
]


# ---------------------------------------------------------------------------
# Stream configs
# ---------------------------------------------------------------------------


@dataclass
class RawMarketStreamConfig:
    """Configuration for the Raw Market Sequence stream.

    Encodes OHLC, Volume, Tick Volume, Spread, and optional Timestamp
    using a small recurrent backbone.

    Attributes:
        enabled: Enable this stream.
        input_features: List of feature names expected in the raw slice.
        input_dim: Number of raw market features per timestep.
        hidden_dim: RNN hidden size.
        num_layers: RNN depth.
        out_dim: Output vector dimension (projected after RNN).
        backbone_type: "lstm" or "gru".
        dropout: Dropout on the output.
        use_layer_norm: Normalise input features before encoding.
    """

    enabled: bool = True
    input_features: list[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "tick_volume", "spread",
    ])
    input_dim: int = 7
    hidden_dim: int = 128
    num_layers: int = 1
    out_dim: int = 128
    backbone_type: SequenceBackboneType = SequenceBackboneType.LSTM
    dropout: float = 0.2
    use_layer_norm: bool = True


@dataclass
class TimeframeEncoderConfig:
    """Per-timeframe encoder configuration.

    Attributes:
        enabled: Whether this individual timeframe is active.
        input_dim: Feature dimension at this timeframe.
        hidden_dim: Small RNN hidden size.
        out_dim: Output vector dimension.
    """

    enabled: bool = True
    input_dim: int = 7
    hidden_dim: int = 64
    out_dim: int = 64


@dataclass
class MultiTimeframeStreamConfig:
    """Configuration for the Multi-Timeframe stream.

    Each timeframe is processed by its own independent encoder.
    Timeframes whose config has ``enabled=False`` are skipped.
    The per-timeframe output vectors are pooled to a fixed dimension.

    Attributes:
        enabled: Enable the entire multi-timeframe stream.
        timeframes: Mapping from timeframe name → its encoder config.
        pooling: "mean" | "attention" for aggregating timeframe outputs.
        out_dim: Pooled output dimension (projected after pooling).
        dropout: Dropout after pooling.
    """

    enabled: bool = True
    timeframes: dict[str, TimeframeEncoderConfig] = field(
        default_factory=lambda: {
            tf: TimeframeEncoderConfig(enabled=True) for tf in SUPPORTED_TIMEFRAMES
        }
    )
    pooling: str = "mean"
    out_dim: int = 128
    dropout: float = 0.2

    @property
    def enabled_timeframes(self) -> list[str]:
        return [tf for tf, cfg in self.timeframes.items() if cfg.enabled]


@dataclass
class EngineeredFeatureStreamConfig:
    """Configuration for the Engineered Feature stream.

    Passes the full set of hand-crafted features (trend, momentum,
    volatility, liquidity, market structure, sessions, market memory)
    through an MLP with layer normalisation.

    Attributes:
        enabled: Enable this stream.
        input_dim: Total number of engineered features.
        hidden_dim: MLP intermediate dimension.
        out_dim: Output vector dimension.
        dropout: Dropout rate.
        use_layer_norm: Normalise each stream independently.
        num_layers: Depth of the feature MLP.
    """

    enabled: bool = True
    input_dim: int = 64
    hidden_dim: int = 128
    out_dim: int = 128
    dropout: float = 0.2
    use_layer_norm: bool = True
    num_layers: int = 2


@dataclass
class EmbeddingSpec:
    """Vocabulary and dimension for a single categorical context embedding.

    Attributes:
        vocab_size: Number of distinct values (e.g. 24 for hour-of-day).
        embed_dim: Learned embedding dimension.
        enabled: Whether this specific embedding is active.
    """

    vocab_size: int
    embed_dim: int = 16
    enabled: bool = True


@dataclass
class ContextEmbeddingStreamConfig:
    """Configuration for the Context Embedding stream.

    Creates a learned ``nn.Embedding`` for each active categorical and
    concatenates the results, projecting to ``out_dim``.

    Attributes:
        enabled: Enable this stream.
        embeddings: Mapping from category name → its EmbeddingSpec.
        out_dim: Final projected dimension.
        dropout: Dropout after projection.
    """

    enabled: bool = True
    embeddings: dict[str, EmbeddingSpec] = field(
        default_factory=lambda: {
            "instrument": EmbeddingSpec(vocab_size=64,  embed_dim=16),
            "timeframe":  EmbeddingSpec(vocab_size=8,   embed_dim=8),
            "session":    EmbeddingSpec(vocab_size=8,   embed_dim=8),
            "hour":       EmbeddingSpec(vocab_size=24,  embed_dim=8),
            "day":        EmbeddingSpec(vocab_size=7,   embed_dim=4),
            "week":       EmbeddingSpec(vocab_size=54,  embed_dim=8),
            "month":      EmbeddingSpec(vocab_size=12,  embed_dim=4),
            "regime":     EmbeddingSpec(vocab_size=8,   embed_dim=8),
        }
    )
    out_dim: int = 64
    dropout: float = 0.1

    @property
    def enabled_embeddings(self) -> dict[str, EmbeddingSpec]:
        return {k: v for k, v in self.embeddings.items() if v.enabled}

    @property
    def raw_concat_dim(self) -> int:
        return sum(v.embed_dim for v in self.embeddings.values() if v.enabled)


@dataclass
class MarketMemoryStreamConfig:
    """Configuration for the Market Memory stream.

    Encodes the memory state produced by the Market Memory Engine through
    a small MLP into a fixed-dimension vector.

    Attributes:
        enabled: Enable this stream.
        memory_dim: Dimension of the incoming memory vector.
        hidden_dim: MLP intermediate dimension.
        out_dim: Output vector dimension.
        dropout: Dropout rate.
    """

    enabled: bool = True
    memory_dim: int = 64
    hidden_dim: int = 64
    out_dim: int = 64
    dropout: float = 0.1


@dataclass
class DynamicFeatureGateConfig:
    """Configuration for the Dynamic Feature Gate module.

    The gate learns to amplify or suppress each engineered feature based on
    context from other streams.  Gate values are in [0.1, 2.0] — features
    are NEVER permanently disabled (minimum activation 0.1).

    Attributes:
        enabled: Enable feature gating.
        context_dim: Dimension of the context vector that drives the gate.
        hidden_dim: Gate network intermediate dimension.
    """

    enabled: bool = True
    context_dim: int = 128
    hidden_dim: int = 64


@dataclass
class AdaptiveFusionConfig:
    """Configuration for the Adaptive Fusion layer.

    Attributes:
        strategy: FusionStrategy — concat / weighted / attention.
        num_attention_heads: Heads used when strategy=attention.
        latent_dim: Output dimension — MUST match the backbone context_dim
                    so all existing prediction heads work without change.
        dropout: Dropout after fusion.
        track_contributions: If True, the fusion layer records per-stream
                             contribution statistics during training.
    """

    strategy: FusionStrategy = FusionStrategy.WEIGHTED
    num_attention_heads: int = 4
    latent_dim: int = 256
    dropout: float = 0.1
    track_contributions: bool = True


# ---------------------------------------------------------------------------
# Top-level hybrid config
# ---------------------------------------------------------------------------


@dataclass
class HybridEncoderConfig:
    """Top-level configuration for the Phase 14.9A Hybrid Encoder.

    When ``enabled=False`` the model uses the original InputEncoder →
    SequenceEncoder path unchanged (full backward compatibility).

    Attributes:
        enabled: Master switch for the entire hybrid architecture.
        raw_market: Configuration for stream 1 (Raw Market Sequence).
        multi_timeframe: Configuration for stream 2 (Multi-Timeframe).
        engineered_features: Configuration for stream 3 (Engineered Features).
        context_embeddings: Configuration for stream 4 (Context Embeddings).
        market_memory: Configuration for stream 5 (Market Memory).
        feature_gate: Configuration for the Dynamic Feature Gate.
        fusion: Configuration for the Adaptive Fusion layer.
    """

    enabled: bool = False
    raw_market: RawMarketStreamConfig = field(
        default_factory=RawMarketStreamConfig
    )
    multi_timeframe: MultiTimeframeStreamConfig = field(
        default_factory=MultiTimeframeStreamConfig
    )
    engineered_features: EngineeredFeatureStreamConfig = field(
        default_factory=EngineeredFeatureStreamConfig
    )
    context_embeddings: ContextEmbeddingStreamConfig = field(
        default_factory=ContextEmbeddingStreamConfig
    )
    market_memory: MarketMemoryStreamConfig = field(
        default_factory=MarketMemoryStreamConfig
    )
    feature_gate: DynamicFeatureGateConfig = field(
        default_factory=DynamicFeatureGateConfig
    )
    fusion: AdaptiveFusionConfig = field(
        default_factory=AdaptiveFusionConfig
    )
