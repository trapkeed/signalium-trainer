"""QuantModelV1 hyper-parameter configuration — Phase 14.9A Hybrid Architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trainer.models.quant_v1.hybrid.config import HybridEncoderConfig


def _default_hybrid_encoder_config() -> "HybridEncoderConfig":
    from trainer.models.quant_v1.hybrid.config import HybridEncoderConfig
    return HybridEncoderConfig(enabled=False)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


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
    TICKS = "ticks"
    PERCENT = "percent"
    MINUTES = "minutes"
    BARS = "bars"
    CANDLES = "candles"


class RegressionLossType(str, Enum):
    """Loss type for regression heads."""

    MSE = "mse"
    HUBER = "huber"
    MAE = "mae"
    GAUSSIAN_NLL = "gaussian_nll"
    QUANTILE = "quantile"


class MarketStabilityClass(str, Enum):
    """Market stability classification labels."""

    STABLE = "stable"
    MODERATELY_STABLE = "moderately_stable"
    UNSTABLE = "unstable"
    HIGHLY_UNSTABLE = "highly_unstable"


# ---------------------------------------------------------------------------
# Core head configs (Phase 14.5 — unchanged)
# ---------------------------------------------------------------------------


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
    """Generic configuration for scalar regression heads (Phase 14.5)."""

    enabled: bool = True
    hidden_dim: int = 64
    dropout: float = 0.2
    unit: RegressionUnit = RegressionUnit.PIPS
    loss_type: RegressionLossType = RegressionLossType.HUBER
    huber_delta: float = 1.0


@dataclass
class TPProbabilityHeadConfig:
    """Configuration for the TP-before-SL probability head."""

    enabled: bool = True
    hidden_dim: int = 64
    dropout: float = 0.2


# ---------------------------------------------------------------------------
# Phase 14.6 — Institutional head configs
# ---------------------------------------------------------------------------


@dataclass
class InstitutionalHeadConfig:
    """Generic configuration for institutional analytics heads.

    Used by all scalar and multi-output institutional heads that do not
    require configurable output units.
    """

    enabled: bool = False
    hidden_dim: int = 64
    dropout: float = 0.2


@dataclass
class ExcursionHeadConfig:
    """Configuration for MFE / MAE excursion regression heads.

    Supports configurable output units (pips, points, atr, ticks, percent).
    """

    enabled: bool = False
    hidden_dim: int = 64
    dropout: float = 0.2
    unit: RegressionUnit = RegressionUnit.PIPS
    loss_type: RegressionLossType = RegressionLossType.HUBER
    huber_delta: float = 1.0


@dataclass
class TradeSurvivalHeadConfig:
    """Configuration for the Trade Survival probability head.

    Outputs probabilities at multiple bar horizons.
    survival_bars: List of bar counts at which to predict survival probability.
    """

    enabled: bool = False
    hidden_dim: int = 64
    dropout: float = 0.2
    survival_bars: list[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class MarketStabilityHeadConfig:
    """Configuration for the Market Stability classification head.

    4-class output: Stable / Moderately Stable / Unstable / Highly Unstable.
    """

    enabled: bool = False
    hidden_dim: int = 64
    dropout: float = 0.2
    num_classes: int = 4
    label_names: list[str] = field(
        default_factory=lambda: [
            "stable", "moderately_stable", "unstable", "highly_unstable"
        ]
    )


@dataclass
class TradeDecayHeadConfig:
    """Configuration for the Trade Decay head.

    Outputs a decay probability and a multi-step decay curve.
    decay_steps: Number of future time steps to predict decay for.
    """

    enabled: bool = False
    hidden_dim: int = 64
    dropout: float = 0.2
    decay_steps: int = 5


@dataclass
class DistributionHeadConfig:
    """Configuration for Profit/Loss distribution heads.

    Profit distribution outputs: mean, median, variance, skewness, ci_lower, ci_upper.
    Loss distribution outputs: mean, median, variance, tail_risk.
    """

    enabled: bool = False
    hidden_dim: int = 64
    dropout: float = 0.2


@dataclass
class UncertaintyHeadConfig:
    """Configuration for the uncertainty estimation head.

    Outputs epistemic and aleatoric uncertainty estimates simultaneously.
    Both are strictly non-negative (Softplus activation).
    """

    enabled: bool = False
    hidden_dim: int = 64
    dropout: float = 0.2


# ---------------------------------------------------------------------------
# Loss configuration (Phase 14.5 + 14.6)
# ---------------------------------------------------------------------------


@dataclass
class MultiTaskLossConfig:
    """Per-head loss weights for the combined multi-task loss.

    Phase 14.5 weights (unchanged):
        direction_weight, move_weight, sl_weight, tp_weight,
        holding_time_weight, confidence_weight, tp_probability_weight

    Phase 14.6 weights (new):
        mfe_weight, mae_excursion_weight, trade_efficiency_weight,
        reward_risk_weight, trade_survival_weight, sl_resilience_weight,
        tp_achievability_weight, entry_quality_weight, exit_quality_weight,
        market_stability_weight, trade_decay_weight,
        profit_distribution_weight, loss_distribution_weight,
        position_size_quality_weight, uncertainty_weight
    """

    # Phase 14.5
    direction_weight: float = 1.0
    move_weight: float = 0.5
    sl_weight: float = 0.5
    tp_weight: float = 0.5
    holding_time_weight: float = 0.3
    confidence_weight: float = 0.2
    tp_probability_weight: float = 0.5

    # Phase 14.6
    mfe_weight: float = 0.3
    mae_excursion_weight: float = 0.3
    trade_efficiency_weight: float = 0.2
    reward_risk_weight: float = 0.3
    trade_survival_weight: float = 0.4
    sl_resilience_weight: float = 0.3
    tp_achievability_weight: float = 0.3
    entry_quality_weight: float = 0.2
    exit_quality_weight: float = 0.2
    market_stability_weight: float = 0.3
    trade_decay_weight: float = 0.2
    profit_distribution_weight: float = 0.2
    loss_distribution_weight: float = 0.2
    position_size_quality_weight: float = 0.2
    uncertainty_weight: float = 0.1


@dataclass
class LossConfig:
    """Loss function configuration for the direction head."""

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


# ---------------------------------------------------------------------------
# Root model config
# ---------------------------------------------------------------------------


@dataclass
class QuantV1Config:
    """Root configuration for QuantModelV1 — Phase 14.6 Institutional Analytics.

    All existing fields are preserved for backward compatibility.
    Phase 14.5 and 14.6 heads are opt-in via their ``enabled`` flag.
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
    # Phase 14.5 — Multi-task heads (all disabled by default)
    # ------------------------------------------------------------------
    expected_move_head: RegressionHeadConfig = field(
        default_factory=lambda: RegressionHeadConfig(enabled=False)
    )
    stop_loss_head: RegressionHeadConfig = field(
        default_factory=lambda: RegressionHeadConfig(enabled=False, unit=RegressionUnit.PIPS)
    )
    take_profit_head: RegressionHeadConfig = field(
        default_factory=lambda: RegressionHeadConfig(enabled=False, unit=RegressionUnit.PIPS)
    )
    holding_time_head: RegressionHeadConfig = field(
        default_factory=lambda: RegressionHeadConfig(enabled=False, unit=RegressionUnit.MINUTES)
    )
    tp_probability_head: TPProbabilityHeadConfig = field(
        default_factory=lambda: TPProbabilityHeadConfig(enabled=False)
    )

    # ------------------------------------------------------------------
    # Phase 14.6 — Institutional Analytics heads (all disabled by default)
    # ------------------------------------------------------------------
    mfe_head: ExcursionHeadConfig = field(
        default_factory=lambda: ExcursionHeadConfig(enabled=False)
    )
    mae_excursion_head: ExcursionHeadConfig = field(
        default_factory=lambda: ExcursionHeadConfig(enabled=False)
    )
    trade_efficiency_head: InstitutionalHeadConfig = field(
        default_factory=lambda: InstitutionalHeadConfig(enabled=False)
    )
    reward_risk_head: InstitutionalHeadConfig = field(
        default_factory=lambda: InstitutionalHeadConfig(enabled=False)
    )
    trade_survival_head: TradeSurvivalHeadConfig = field(
        default_factory=lambda: TradeSurvivalHeadConfig(enabled=False)
    )
    sl_resilience_head: InstitutionalHeadConfig = field(
        default_factory=lambda: InstitutionalHeadConfig(enabled=False)
    )
    tp_achievability_head: InstitutionalHeadConfig = field(
        default_factory=lambda: InstitutionalHeadConfig(enabled=False)
    )
    entry_quality_head: InstitutionalHeadConfig = field(
        default_factory=lambda: InstitutionalHeadConfig(enabled=False)
    )
    exit_quality_head: InstitutionalHeadConfig = field(
        default_factory=lambda: InstitutionalHeadConfig(enabled=False)
    )
    market_stability_head: MarketStabilityHeadConfig = field(
        default_factory=lambda: MarketStabilityHeadConfig(enabled=False)
    )
    trade_decay_head: TradeDecayHeadConfig = field(
        default_factory=lambda: TradeDecayHeadConfig(enabled=False)
    )
    profit_distribution_head: DistributionHeadConfig = field(
        default_factory=lambda: DistributionHeadConfig(enabled=False)
    )
    loss_distribution_head: DistributionHeadConfig = field(
        default_factory=lambda: DistributionHeadConfig(enabled=False)
    )
    position_size_quality_head: InstitutionalHeadConfig = field(
        default_factory=lambda: InstitutionalHeadConfig(enabled=False)
    )
    uncertainty_head: UncertaintyHeadConfig = field(
        default_factory=lambda: UncertaintyHeadConfig(enabled=False)
    )

    multi_task_loss: MultiTaskLossConfig = field(default_factory=MultiTaskLossConfig)

    # ------------------------------------------------------------------
    # Phase 14.9A — Hybrid Representation Learning (disabled by default)
    # ------------------------------------------------------------------
    hybrid_encoder: "HybridEncoderConfig" = field(  # type: ignore[type-arg]
        default_factory=lambda: _default_hybrid_encoder_config()
    )
