"""QuantModelV1 — Signalium's first proprietary trading signal classifier.

Phase 14.5: Extended with multi-task prediction heads.
"""

from trainer.models.quant_v1.config import (
    BackboneType,
    BalancingConfig,
    BalancingStrategy,
    BackboneConfig,
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    InputEncoderConfig,
    LossConfig,
    LossType,
    MultiTaskLossConfig,
    QuantV1Config,
    RegressionHeadConfig,
    RegressionLossType,
    RegressionUnit,
    TPProbabilityHeadConfig,
)
from trainer.models.quant_v1.model import QuantModelV1
from trainer.models.quant_v1.output import MultiTaskQuantModelOutput, QuantModelOutput

__all__ = [
    "QuantModelV1",
    "QuantModelOutput",
    "MultiTaskQuantModelOutput",
    "QuantV1Config",
    "BackboneType",
    "BalancingStrategy",
    "LossType",
    "BackboneConfig",
    "ClassificationHeadConfig",
    "ConfidenceHeadConfig",
    "InputEncoderConfig",
    "LossConfig",
    "BalancingConfig",
    "MultiTaskLossConfig",
    "RegressionHeadConfig",
    "RegressionLossType",
    "RegressionUnit",
    "TPProbabilityHeadConfig",
]
