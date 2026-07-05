"""Prediction metadata dataclasses for Signalium Quant traceability.

Every inference produced by the model carries a ``PredictionMetadata`` object
that fully identifies:

  • Which model produced the prediction
  • Which experiment, dataset, and configuration trained it
  • The runtime environment used for inference
  • Reproducibility anchors (seed, config hashes, schema versions)

Prediction values are NEVER modified — metadata rides alongside them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Sub-containers
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Static information about the loaded model artifact.

    Attributes:
        model_name: Human-readable model name (e.g. "signalium_quant_v1").
        model_version: Semantic version (e.g. "2.0.0").
        architecture: Architecture class name (e.g. "QuantModelV1-LSTM").
        num_parameters: Total trainable parameter count.
        input_shape: Expected input shape description.
        output_shape: Output shape description.
        enabled_heads: List of active prediction head names.
        framework: "pytorch" | "torchscript" | "onnx".
        precision: "fp32" | "fp16" | "int8".
        model_registry_id: Optional registry identifier.
        experiment_id: Optional experiment identifier.
        mlflow_run_id: Optional MLflow run ID.
        training_timestamp: UTC ISO-8601 timestamp of training completion.
        model_export_timestamp: UTC ISO-8601 timestamp of model export.
    """

    model_name: str = "signalium_quant_v1"
    model_version: str = "2.0.0"
    architecture: str = "QuantModelV1"
    num_parameters: int = 0
    input_shape: str = "(batch, seq_len, input_dim)"
    output_shape: str = "(batch, num_classes)"
    enabled_heads: list[str] = field(default_factory=list)
    framework: str = "pytorch"
    precision: str = "fp32"
    model_registry_id: Optional[str] = None
    experiment_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    training_timestamp: Optional[str] = None
    model_export_timestamp: Optional[str] = None
    # ── Phase 14.9C — Architecture metadata (all optional with safe defaults)
    phase: str = "14.6"
    backbone_type: str = "unknown"
    active_streams: list[str] = field(default_factory=list)
    fusion_strategy: Optional[str] = None
    encoder_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class InferenceInfo:
    """Runtime information captured at inference time.

    Attributes:
        inference_timestamp: UTC ISO-8601 timestamp of inference.
        inference_duration_ms: Wall-clock time for the forward pass (ms).
        batch_size: Number of samples in the inference batch.
        device: Device used ("cpu" | "cuda" | "mps").
        device_name: Full device name string (e.g. "NVIDIA A100 80GB").
        cuda_version: CUDA version string if on GPU, else None.
        pytorch_version: Installed PyTorch version string.
        python_version: Python runtime version string.
        memory_used_mb: GPU/RAM memory consumed during inference (MB).
        inference_engine_version: Version of the inference adapter.
    """

    inference_timestamp: str = field(default_factory=_utcnow)
    inference_duration_ms: float = 0.0
    batch_size: int = 1
    device: str = "cpu"
    device_name: str = "cpu"
    cuda_version: Optional[str] = None
    pytorch_version: str = "unknown"
    python_version: str = "unknown"
    memory_used_mb: float = 0.0
    inference_engine_version: str = "1.0.0"


@dataclass
class ReproducibilityInfo:
    """Information required to reproduce this prediction.

    Attributes:
        git_commit_hash: Short SHA of the git commit that produced the model.
        trainer_version: Signalium Trainer package version.
        framework_version: PyTorch version string (matches InferenceInfo).
        random_seed: Random seed used during training (if recorded).
        training_config_hash: SHA-256 prefix of the serialised training config.
        inference_config_hash: SHA-256 prefix of the inference config dict.
        feature_config_version: Feature engineering schema version.
        feature_schema_version: Labelled feature schema version.
        label_schema_version: Label generation schema version.
        normalization_version: Data normalisation pipeline version.
        scaling_version: Feature scaling pipeline version.
        configuration_version: Overall configuration version tag.
        dataset_version: Training dataset version identifier.
    """

    git_commit_hash: str = "unknown"
    trainer_version: str = "0.1.0"
    framework_version: str = "unknown"
    random_seed: Optional[int] = None
    training_config_hash: Optional[str] = None
    inference_config_hash: Optional[str] = None
    feature_config_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    label_schema_version: Optional[str] = None
    normalization_version: Optional[str] = None
    scaling_version: Optional[str] = None
    configuration_version: Optional[str] = None
    dataset_version: Optional[str] = None


@dataclass
class SchemaVersions:
    """Version identifiers for all data schemas used by this model.

    Attributes:
        output_schema_version: Version of the model output contract.
        prediction_schema_version: Version of the inference response schema.
        feature_schema_version: Feature engineering schema version.
        label_schema_version: Training label generation schema version.
        configuration_version: Overall configuration file version.
    """

    output_schema_version: str = "3.0.0"
    prediction_schema_version: str = "1.0.0"
    feature_schema_version: str = "1.0.0"
    label_schema_version: str = "1.0.0"
    configuration_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Root prediction metadata
# ---------------------------------------------------------------------------


@dataclass
class PredictionMetadata:
    """Complete metadata record accompanying every model inference.

    A new ``PredictionMetadata`` is created for each call to
    ``TracedInferenceAdapter.predict()``.  It does NOT change any
    prediction value — it travels alongside predictions in a dedicated
    ``TracedPrediction`` container.

    Attributes:
        prediction_id: UUID v4 — globally unique per inference call.
        prediction_timestamp: UTC ISO-8601 when the prediction was made.
        model: Static model information.
        inference: Runtime inference information.
        reproducibility: Reproducibility and version anchors.
        schemas: All schema version identifiers.
    """

    prediction_id: str = field(default_factory=_new_uuid)
    prediction_timestamp: str = field(default_factory=_utcnow)
    model: ModelInfo = field(default_factory=ModelInfo)
    inference: InferenceInfo = field(default_factory=InferenceInfo)
    reproducibility: ReproducibilityInfo = field(default_factory=ReproducibilityInfo)
    schemas: SchemaVersions = field(default_factory=SchemaVersions)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain nested dict (JSON-safe for manifests)."""
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a flat dict for logging / DataFrame rows."""
        d = self.to_dict()
        result: dict[str, Any] = {
            "prediction_id": d["prediction_id"],
            "prediction_timestamp": d["prediction_timestamp"],
        }
        for section in ("model", "inference", "reproducibility", "schemas"):
            for k, v in d[section].items():
                result[f"{section}.{k}"] = v
        return result

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def model_version(self) -> str:
        return self.model.model_version

    @property
    def inference_duration_ms(self) -> float:
        return self.inference.inference_duration_ms
