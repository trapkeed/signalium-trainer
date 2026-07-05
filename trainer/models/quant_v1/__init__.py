"""QuantModelV1 — Signalium's institutional quantitative trading model.

Phase 14.6: Extended with 15 Institutional Trade Analytics heads.
Phase 14.5: Extended with multi-task prediction heads.
"""

from trainer.models.quant_v1.config import (
    BackboneConfig,
    BackboneType,
    BalancingConfig,
    BalancingStrategy,
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    DistributionHeadConfig,
    ExcursionHeadConfig,
    InputEncoderConfig,
    InstitutionalHeadConfig,
    LossConfig,
    LossType,
    MarketStabilityClass,
    MarketStabilityHeadConfig,
    MultiTaskLossConfig,
    QuantV1Config,
    RegressionHeadConfig,
    RegressionLossType,
    RegressionUnit,
    TPProbabilityHeadConfig,
    TradeDecayHeadConfig,
    TradeSurvivalHeadConfig,
    UncertaintyHeadConfig,
)
from trainer.models.quant_v1.model import QuantModelV1
from trainer.models.quant_v1.output import MultiTaskQuantModelOutput, QuantModelOutput

__all__ = [
    "QuantModelV1",
    "QuantModelOutput",
    "MultiTaskQuantModelOutput",
    "QuantV1Config",
    # Enums
    "BackboneType",
    "BalancingStrategy",
    "LossType",
    "RegressionUnit",
    "RegressionLossType",
    "MarketStabilityClass",
    # Head configs — Phase 14.5
    "BackboneConfig",
    "ClassificationHeadConfig",
    "ConfidenceHeadConfig",
    "InputEncoderConfig",
    "LossConfig",
    "BalancingConfig",
    "MultiTaskLossConfig",
    "RegressionHeadConfig",
    "TPProbabilityHeadConfig",
    # Head configs — Phase 14.6
    "InstitutionalHeadConfig",
    "ExcursionHeadConfig",
    "TradeSurvivalHeadConfig",
    "MarketStabilityHeadConfig",
    "TradeDecayHeadConfig",
    "DistributionHeadConfig",
    "UncertaintyHeadConfig",
]
