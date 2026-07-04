"""Framework-level abstract base classes.

All concrete components (encoders, backbones, heads, losses, exporters,
inference adapters) implement these interfaces so they are interchangeable
without modifying the training pipeline.
"""

from trainer.framework.abstractions.encoder import BaseFeatureEncoder
from trainer.framework.abstractions.exporter import BaseFrameworkExporter
from trainer.framework.abstractions.head import BaseClassificationHead, BaseConfidenceHead
from trainer.framework.abstractions.inference import BaseInferenceAdapter
from trainer.framework.abstractions.loss import BaseLoss
from trainer.framework.abstractions.metrics import BaseMetrics
from trainer.framework.abstractions.sequence_encoder import BaseSequenceEncoder

__all__ = [
    "BaseFeatureEncoder",
    "BaseSequenceEncoder",
    "BaseClassificationHead",
    "BaseConfidenceHead",
    "BaseLoss",
    "BaseMetrics",
    "BaseFrameworkExporter",
    "BaseInferenceAdapter",
]
