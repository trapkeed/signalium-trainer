"""QuantModelV1 — Signalium's first proprietary trading signal classifier."""

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
    QuantV1Config,
)
from trainer.models.quant_v1.model import QuantModelV1
from trainer.models.quant_v1.output import QuantModelOutput

__all__ = [
    "QuantModelV1",
    "QuantModelOutput",
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
]
