"""TracedPrediction — bundles prediction values with their full metadata.

The prediction dict is NEVER modified.  All metadata travels in a
dedicated ``PredictionMetadata`` object alongside it.

Usage::

    result: TracedPrediction = traced_adapter.predict(x)

    # Existing consumers — no change required
    buy_prob = result.predictions["buy_prob"]
    confidence = result.predictions["confidence"]

    # Traceability consumers
    prediction_id = result.metadata.prediction_id
    duration_ms   = result.metadata.inference_duration_ms
    git_hash      = result.metadata.reproducibility.git_commit_hash
    model_version = result.metadata.model.model_version
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trainer.traceability.metadata import PredictionMetadata


@dataclass
class TracedPrediction:
    """A model prediction bundled with its full traceability metadata.

    Attributes:
        predictions: The raw prediction dict from the underlying adapter.
                     All values are exactly as produced by the model —
                     no modification whatsoever.
        metadata:    Complete ``PredictionMetadata`` for this inference call.

    Notes:
        - ``prediction_id`` is unique per ``TracedInferenceAdapter.predict()``
          call regardless of batch size.
        - ``predictions`` is a reference to the original dict; do not mutate it.
    """

    predictions: dict[str, Any]
    metadata: PredictionMetadata

    def to_dict(self) -> dict[str, Any]:
        """Return a merged dict for logging or serialisation.

        Prediction values are nested under key ``"predictions"``.
        Metadata is nested under key ``"metadata"``.

        The two namespaces never collide — prediction values are
        never polluted with metadata keys.
        """
        return {
            "predictions": self.predictions,
            "metadata": self.metadata.to_dict(),
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for DataFrame rows or CSV export.

        Prediction keys are prefixed with ``pred.``.
        Metadata keys follow the ``section.field`` convention from
        ``PredictionMetadata.to_flat_dict()``.
        """
        result: dict[str, Any] = {}
        for k, v in self.predictions.items():
            result[f"pred.{k}"] = v
        result.update(self.metadata.to_flat_dict())
        return result

    @property
    def prediction_id(self) -> str:
        """Unique ID for this inference call (UUID v4)."""
        return self.metadata.prediction_id

    @property
    def inference_duration_ms(self) -> float:
        """Wall-clock forward-pass duration in milliseconds."""
        return self.metadata.inference.inference_duration_ms

    @property
    def model_version(self) -> str:
        """Version of the model that produced this prediction."""
        return self.metadata.model.model_version
