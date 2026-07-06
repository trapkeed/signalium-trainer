"""TraceabilityConfig — all configuration fields for the tracing layer.

Pass a ``TraceabilityConfig`` to ``TracedInferenceAdapter`` at construction
time.  Fields that are unknown at inference time (e.g. ``mlflow_run_id``)
can be left ``None`` — they will appear as ``null`` in metadata records.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TraceabilityConfig:
    """Configuration for the prediction traceability layer.

    Attributes:
        model_name: Human-readable model name.
        model_version: Semantic version of the model weights / architecture.
        architecture: Architecture class name.
        model_registry_id: Optional model registry identifier.
        experiment_id: Optional experiment / study identifier.
        mlflow_run_id: Optional MLflow run ID.
        training_timestamp: UTC ISO-8601 when training finished.
        model_export_timestamp: UTC ISO-8601 when the model was exported.

        dataset_version: Training dataset version identifier.
        feature_schema_version: Feature engineering schema version.
        label_schema_version: Training label generation schema version.
        output_schema_version: Model output contract version.
        prediction_schema_version: Inference response schema version.
        normalization_version: Data normalisation pipeline version.
        scaling_version: Feature scaling pipeline version.
        configuration_version: Overall configuration file version tag.

        random_seed: Random seed used during training.
        training_config_hash: SHA-256 prefix of the training configuration
            (generated automatically if a raw config object is provided).
        inference_config_hash: SHA-256 prefix of the inference configuration.
        feature_config_version: Feature config document version.

        trainer_version: Signalium Trainer library version.
        inference_engine_version: Inference adapter version string.
        framework: "pytorch" | "torchscript" | "onnx".
        precision: "fp32" | "fp16" | "int8".
    """

    # ── Model identity ────────────────────────────────────────────────────
    model_name: str = "signalium_quant_v1"
    model_version: str = "2.0.0"
    architecture: str = "QuantModelV1"
    model_registry_id: str | None = None
    experiment_id: str | None = None
    mlflow_run_id: str | None = None
    training_timestamp: str | None = None
    model_export_timestamp: str | None = None

    # ── Data and schema versions ──────────────────────────────────────────
    dataset_version: str | None = None
    feature_schema_version: str = "1.0.0"
    label_schema_version: str = "1.0.0"
    output_schema_version: str = "3.0.0"
    prediction_schema_version: str = "1.0.0"
    normalization_version: str | None = None
    scaling_version: str | None = None
    configuration_version: str | None = None

    # ── Reproducibility ───────────────────────────────────────────────────
    random_seed: int | None = None
    training_config_hash: str | None = None
    inference_config_hash: str | None = None
    feature_config_version: str | None = None

    # ── Runtime ──────────────────────────────────────────────────────────
    trainer_version: str = "0.1.0"
    inference_engine_version: str = "1.0.0"
    framework: str = "pytorch"
    precision: str = "fp32"
