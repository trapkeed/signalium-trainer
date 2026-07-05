"""QuantModelV1 hyper-parameter configuration — Phase 14.5 multi-task extension."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BackboneType(str, Enum):
    """Supported sequence encoder backbones."""

    LSTM = "lstm"
    GRU = "gru"


class LossType(str, Enum):
    """Supported loss functions."""

    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    WEIGHTED_CROSS_ENTROPY = "weighted_cross_entropy"


class BalancingStrategy(str, Enum):
    """Class imbalance mitigation strategies."""

    NONE = "none"
    CLASS_WEIGHTS = "class_weights"
    WEIGHTED_SAMPLER = "weighted_sampler"
    OVERSAMPLE = "oversample"
    UNDERSAMPLE = "undersample"


class RegressionUnit(str, Enum):
    """Output unit for regression heads."""

    PIPS = "pips"
    POINTS = "points"
    ATR = "atr"
    MINUTES = "minutes"
    BARS = "bars"
    CANDLES = "candles"


class RegressionLossType(str, Enum):
    """Loss type for regression heads."""

    MSE = "mse"
    HUBER = "huber"
    MAE = "mae"


@dataclass
class InputEncoderConfig:
    """Configuration for the InputEncoder."""

    input_dim: int = 128
    hidden_dim: int = 256
    use_layer_norm: bool = True
    use_projection: bool = True
    projection_dropout: float = 0.1


@dataclass
class BackboneConfig:
    """Configuration for the SequenceEncoder backbone."""

    type: BackboneType = BackboneType.LSTM
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    window_length: int = 60


@dataclass
class ClassificationHeadConfig:
    """Configuration for the ClassificationHead."""

    num_classes: int = 3
    hidden_dim: int = 128
    dropout: float = 0.3
    label_names: list[str] = field(default_factory=lambda: ["BUY", "SELL", "NO_TRADE"])


@dataclass
class ConfidenceHeadConfig:
    """Configuration for the ConfidenceHead."""

    hidden_dim: int = 64
    dropout: float = 0.2


@dataclass
class RegressionHeadConfig:
    """Generic configuration for regression output heads.

    Used by ExpectedMove, StopLoss, TakeProfit, and HoldingTime heads.
    """

    enabled: bool = True
    hidden_dim: int = 64
    dropout: float = 0.2
    unit: RegressionUnit = RegressionUnit.PIPS
    loss_type: RegressionLossType = RegressionLossType.HUBER
    huber_delta: float = 1.0


@dataclass
class TPProbabilityHeadConfig:
    """Configuration for the TP-before-SL probability head.

    Binary classification: probability that Take Profit is reached
    before Stop Loss.
    """

    enabled: bool = True
    hidden_dim: int = 64
    dropout: float = 0.2


@dataclass
class MultiTaskLossConfig:
    """Per-head loss weights for the combined multi-task loss.

    Setting a weight to 0.0 disables that head's contribution to the
    gradient (it still runs the forward pass if the head is enabled).
    """

    direction_weight: float = 1.0
    move_weight: float = 0.5
    sl_weight: float = 0.5
    tp_weight: float = 0.5
    holding_time_weight: float = 0.3
    confidence_weight: float = 0.2
    tp_probability_weight: float = 0.5


@dataclass
class LossConfig:
    """Loss function configuration."""

    type: LossType = LossType.CROSS_ENTROPY
    focal_gamma: float = 2.0
    focal_alpha: Optional[list[float]] = None
    label_smoothing: float = 0.0
    class_weights: Optional[list[float]] = None


@dataclass
class BalancingConfig:
    """Class imbalance handling configuration."""

    strategy: BalancingStrategy = BalancingStrategy.CLASS_WEIGHTS
    oversample_ratio: float = 1.0
    undersample_ratio: float = 1.0


@dataclass
class QuantV1Config:
    """Root configuration for QuantModelV1 — Phase 14.5 multi-task.

    All existing fields are preserved for backward compatibility.
    New multi-task heads are opt-in via their ``enabled`` flag.
    """

    encoder: InputEncoderConfig = field(default_factory=InputEncoderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    classification_head: ClassificationHeadConfig = field(
        default_factory=ClassificationHeadConfig
    )
    confidence_head: ConfidenceHeadConfig = field(default_factory=ConfidenceHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    balancing: BalancingConfig = field(default_factory=BalancingConfig)
    model_version: str = "1.0.0"
    architecture_name: str = "QuantModelV1"

    # ------------------------------------------------------------------
    # Phase 14.5 — Multi-task prediction heads (all disabled by default)
    # ------------------------------------------------------------------
    expected_move_head: RegressionHeadConfig = field(
        default_factory=lambda: RegressionHeadConfig(enabled=False)
    )
    stop_loss_head: RegressionHeadConfig = field(
        default_factory=lambda: RegressionHeadConfig(
            enabled=False, unit=RegressionUnit.PIPS
        )
    )
    take_profit_head: RegressionHeadConfig = field(
        default_factory=lambda: RegressionHeadConfig(
            enabled=False, unit=RegressionUnit.PIPS
        )
    )
    holding_time_head: RegressionHeadConfig = field(
        default_factory=lambda: RegressionHeadConfig(
            enabled=False, unit=RegressionUnit.MINUTES
        )
    )
    tp_probability_head: TPProbabilityHeadConfig = field(
        default_factory=lambda: TPProbabilityHeadConfig(enabled=False)
    )
    multi_task_loss: MultiTaskLossConfig = field(
        default_factory=MultiTaskLossConfig
    )
