"""Configurable loss functions for Signalium Quant models."""

from trainer.losses.advanced import (
    CalibrationLoss,
    DistributionLoss,
    DynamicLossWeighting,
    GaussianNLLLoss,
    MultiStepBCELoss,
    QuantileLoss,
)
from trainer.losses.factory import build_loss
from trainer.losses.focal import FocalLoss
from trainer.losses.multi_task import MultiTaskLoss, MultiTaskLossOutput
from trainer.losses.weighted_ce import WeightedCrossEntropyLoss

__all__ = [
    "FocalLoss",
    "WeightedCrossEntropyLoss",
    "build_loss",
    "MultiTaskLoss",
    "MultiTaskLossOutput",
    "QuantileLoss",
    "GaussianNLLLoss",
    "DistributionLoss",
    "MultiStepBCELoss",
    "DynamicLossWeighting",
    "CalibrationLoss",
]
