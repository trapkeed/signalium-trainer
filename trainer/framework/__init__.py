"""Signalium Model Framework.

A modular, architecture-agnostic framework for building, training, evaluating,
and deploying deep learning models for quantitative trading signal prediction.

Public API::

    from trainer.framework import ModelFactory, ModelInput, ModelOutput, ModelSpec
    from trainer.framework import MetricsEngine, InferenceAdapterFactory
    from trainer.framework.abstractions import (
        BaseFeatureEncoder,
        BaseSequenceEncoder,
        BaseClassificationHead,
        BaseConfidenceHead,
        BaseLoss,
        BaseMetrics,
        BaseFrameworkExporter,
        BaseInferenceAdapter,
    )
"""

from trainer.framework.abstractions import (
    BaseClassificationHead,
    BaseConfidenceHead,
    BaseFeatureEncoder,
    BaseFrameworkExporter,
    BaseInferenceAdapter,
    BaseLoss,
    BaseMetrics,
    BaseSequenceEncoder,
)
from trainer.framework.contracts import (
    CLASS_BUY,
    CLASS_LABELS,
    CLASS_NO_TRADE,
    CLASS_SELL,
    FeatureSchema,
    ModelInput,
    ModelOutput,
    ModelSpec,
    OutputSchema,
)
from trainer.framework.factory import ModelFactory
from trainer.framework.inference import InferenceAdapterFactory
from trainer.framework.metrics import MetricsEngine
from trainer.framework.traced_inference import TracedInferenceAdapter

__all__ = [
    # Abstractions
    "BaseFeatureEncoder",
    "BaseSequenceEncoder",
    "BaseClassificationHead",
    "BaseConfidenceHead",
    "BaseLoss",
    "BaseMetrics",
    "BaseFrameworkExporter",
    "BaseInferenceAdapter",
    # Contracts
    "ModelInput",
    "ModelOutput",
    "ModelSpec",
    "FeatureSchema",
    "OutputSchema",
    "CLASS_BUY",
    "CLASS_SELL",
    "CLASS_NO_TRADE",
    "CLASS_LABELS",
    # Runtime
    "ModelFactory",
    "MetricsEngine",
    "InferenceAdapterFactory",
    "TracedInferenceAdapter",
]
