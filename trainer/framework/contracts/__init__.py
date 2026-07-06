"""Standardised input/output contracts and model metadata for the Signalium framework."""

from trainer.framework.contracts.model_input import ModelInput
from trainer.framework.contracts.model_output import (
    CLASS_BUY,
    CLASS_LABELS,
    CLASS_NO_TRADE,
    CLASS_SELL,
    ModelOutput,
)
from trainer.framework.contracts.model_spec import FeatureSchema, ModelSpec, OutputSchema

__all__ = [
    "CLASS_BUY",
    "CLASS_LABELS",
    "CLASS_NO_TRADE",
    "CLASS_SELL",
    "FeatureSchema",
    "ModelInput",
    "ModelOutput",
    "ModelSpec",
    "OutputSchema",
]
