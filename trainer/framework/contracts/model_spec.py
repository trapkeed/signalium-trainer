"""ModelSpec — rich metadata contract for the Model Registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureSchema:
    """Describes the expected feature tensor shape for a model.

    Attributes:
        feature_dim: Number of engineered features per timestep.
        seq_len: Expected sequence length (number of historical bars).
        feature_groups: Optional list of feature group names (e.g. ["ohlc", "ta", "vol"]).
        supported_feature_versions: Feature pipeline versions this model was tested with.
    """

    feature_dim: int
    seq_len: int
    feature_groups: list[str] = field(default_factory=list)
    supported_feature_versions: list[str] = field(default_factory=list)


@dataclass
class OutputSchema:
    """Describes the output shape and semantics of a model.

    Attributes:
        num_classes: Number of output classes.
        class_labels: Ordered list of class label strings.
        has_confidence: Whether the model produces a confidence score.
        auxiliary_outputs: Names of any additional auxiliary output tensors.
    """

    num_classes: int
    class_labels: list[str] = field(default_factory=lambda: ["BUY", "SELL", "NO_TRADE"])
    has_confidence: bool = True
    auxiliary_outputs: list[str] = field(default_factory=list)


@dataclass
class ModelSpec:
    """Rich metadata record stored alongside a model class in the Model Registry.

    The ModelSpec captures everything a downstream system needs to know about
    a model without instantiating it: its name, version, architecture family,
    input requirements, output semantics, and export capabilities.

    Attributes:
        name: Unique registry key (e.g. "quant_v1").
        version: Semantic version string (e.g. "1.0.0").
        architecture: Human-readable architecture description (e.g. "LSTM-2L-256H").
        description: One-line human description of the model.
        input_schema: FeatureSchema describing the expected input tensor.
        output_schema: OutputSchema describing the model's output structure.
        export_formats: List of supported export formats (e.g. ["torchscript", "onnx"]).
        tags: Arbitrary key-value metadata for search and filtering.
    """

    name: str
    version: str
    architecture: str
    description: str = ""
    input_schema: FeatureSchema | None = None
    output_schema: OutputSchema | None = None
    export_formats: list[str] = field(default_factory=lambda: ["torchscript", "onnx"])
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the spec to a plain dict (e.g. for manifest or logging)."""
        return {
            "name": self.name,
            "version": self.version,
            "architecture": self.architecture,
            "description": self.description,
            "input_schema": (
                {
                    "feature_dim": self.input_schema.feature_dim,
                    "seq_len": self.input_schema.seq_len,
                    "feature_groups": self.input_schema.feature_groups,
                    "supported_feature_versions": self.input_schema.supported_feature_versions,
                }
                if self.input_schema
                else None
            ),
            "output_schema": (
                {
                    "num_classes": self.output_schema.num_classes,
                    "class_labels": self.output_schema.class_labels,
                    "has_confidence": self.output_schema.has_confidence,
                    "auxiliary_outputs": self.output_schema.auxiliary_outputs,
                }
                if self.output_schema
                else None
            ),
            "export_formats": self.export_formats,
            "tags": self.tags,
        }
