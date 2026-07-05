"""Prediction traceability and metadata package for Signalium Quant.

Every inference produced by the model carries a ``PredictionMetadata``
object that fully identifies:

  • Which model produced the prediction (name, version, architecture)
  • Which experiment trained it (experiment ID, MLflow run ID)
  • Which data and configuration produced it (dataset version, schema versions)
  • The exact runtime environment (device, PyTorch version, git hash)
  • Reproducibility anchors (seed, config hashes, feature/label schema versions)

Prediction values are NEVER modified.  Metadata rides alongside them
in a ``TracedPrediction`` container.
"""

from trainer.traceability.metadata import (
    InferenceInfo,
    ModelInfo,
    PredictionMetadata,
    ReproducibilityInfo,
    SchemaVersions,
)
from trainer.traceability.context import InferenceContext
from trainer.traceability.hasher import hash_config, hash_schema, hash_string
from trainer.traceability.traced_output import TracedPrediction

__all__ = [
    "PredictionMetadata",
    "ModelInfo",
    "InferenceInfo",
    "ReproducibilityInfo",
    "SchemaVersions",
    "InferenceContext",
    "TracedPrediction",
    "hash_config",
    "hash_schema",
    "hash_string",
]
