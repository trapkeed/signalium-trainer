"""MetadataManifestBuilder — extends ManifestBuilder with full traceability.

Embeds ``PredictionMetadata`` into export manifests so that every
TorchScript and ONNX artifact is accompanied by a machine-readable
record of exactly which model, experiment, data, and configuration
produced it.

Usage::

    from trainer.exporters.metadata_manifest import MetadataManifestBuilder
    from trainer.traceability.config import TraceabilityConfig
    from trainer.traceability.metadata import PredictionMetadata

    builder = MetadataManifestBuilder.from_trace_config(
        trace_config=TraceabilityConfig(
            model_name="signalium_quant_v1",
            model_version="2.0.0",
            experiment_id="exp-001",
        ),
        framework="torchscript",
        architecture="QuantModelV1",
    )
    builder.add_file(Path("model.pt"))
    builder.write(output_dir=Path("exports/"))

The output manifest is a superset of the existing ``ManifestBuilder`` format
and is fully backward-compatible.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from trainer.exporters.manifest import ManifestBuilder
from trainer.traceability.config import TraceabilityConfig
from trainer.traceability.context import InferenceContext
from trainer.traceability.metadata import (
    ModelInfo,
    PredictionMetadata,
    ReproducibilityInfo,
    SchemaVersions,
)


class MetadataManifestBuilder(ManifestBuilder):
    """ManifestBuilder extended with full prediction traceability metadata.

    All methods from ``ManifestBuilder`` are preserved and work identically.
    The ``build()`` override injects the traceability section into the
    manifest dict.

    Args:
        model_id: Signalium model identifier.
        version: Semantic version string.
        framework: "pytorch" | "torchscript" | "onnx".
        architecture: Model class name.
        trace_config: Traceability configuration.
    """

    def __init__(
        self,
        model_id: str,
        version: str,
        framework: str,
        architecture: str,
        trace_config: Optional[TraceabilityConfig] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            version=version,
            framework=framework,
            architecture=architecture,
        )
        self._trace_config = trace_config or TraceabilityConfig(
            model_name=model_id,
            model_version=version,
            architecture=architecture,
            framework=framework,
        )
        self._enabled_heads: list[str] = []
        self._num_parameters: int = 0

    # ------------------------------------------------------------------
    # Extra setters
    # ------------------------------------------------------------------

    def set_enabled_heads(self, heads: list[str]) -> None:
        """Record which prediction heads were active at export time."""
        self._enabled_heads = list(heads)

    def set_num_parameters(self, count: int) -> None:
        """Record the total trainable parameter count."""
        self._num_parameters = count

    # ------------------------------------------------------------------
    # Build override
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Assemble the manifest dict with traceability section injected."""
        base = super().build()
        base["traceability"] = self._build_traceability_section()
        return base

    def _build_traceability_section(self) -> dict[str, Any]:
        cfg = self._trace_config
        try:
            from trainer.traceability.context import _git_commit, _python_version, _pytorch_version

            git_hash = _git_commit(short=True)
            pytorch_ver = _pytorch_version()
            python_ver = _python_version()
        except Exception:
            git_hash = "unknown"
            pytorch_ver = "unknown"
            python_ver = "unknown"

        model_info = ModelInfo(
            model_name=cfg.model_name,
            model_version=cfg.model_version,
            architecture=cfg.architecture,
            num_parameters=self._num_parameters,
            enabled_heads=self._enabled_heads,
            framework=cfg.framework,
            precision=cfg.precision,
            model_registry_id=cfg.model_registry_id,
            experiment_id=cfg.experiment_id,
            mlflow_run_id=cfg.mlflow_run_id,
            training_timestamp=cfg.training_timestamp,
            model_export_timestamp=cfg.model_export_timestamp,
        )
        repro_info = ReproducibilityInfo(
            git_commit_hash=git_hash,
            trainer_version=cfg.trainer_version,
            framework_version=pytorch_ver,
            random_seed=cfg.random_seed,
            training_config_hash=cfg.training_config_hash,
            inference_config_hash=cfg.inference_config_hash,
            feature_config_version=cfg.feature_config_version,
            feature_schema_version=cfg.feature_schema_version,
            label_schema_version=cfg.label_schema_version,
            normalization_version=cfg.normalization_version,
            scaling_version=cfg.scaling_version,
            configuration_version=cfg.configuration_version,
            dataset_version=cfg.dataset_version,
        )
        schemas = SchemaVersions(
            output_schema_version=cfg.output_schema_version,
            prediction_schema_version=cfg.prediction_schema_version,
            feature_schema_version=cfg.feature_schema_version,
            label_schema_version=cfg.label_schema_version,
            configuration_version=cfg.configuration_version or "1.0.0",
        )
        return {
            "model": asdict(model_info),
            "reproducibility": asdict(repro_info),
            "schemas": asdict(schemas),
            "manifest_generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "python_version": python_ver,
        }

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_config(
        cls,
        trace_config: TraceabilityConfig,
        framework: str,
        architecture: str,
    ) -> "MetadataManifestBuilder":
        """Create a builder directly from a ``TraceabilityConfig``."""
        return cls(
            model_id=trace_config.model_name,
            version=trace_config.model_version,
            framework=framework,
            architecture=architecture,
            trace_config=trace_config,
        )

    def write_metadata_sidecar(
        self,
        output_dir: Path,
        filename: str = "prediction_metadata.json",
    ) -> Path:
        """Write a standalone metadata sidecar file (no file checksums).

        This is the metadata-only document placed alongside exported
        model artifacts so tools that do not parse the full manifest
        can still access traceability information.

        Args:
            output_dir: Directory to write the sidecar into.
            filename: Output filename.

        Returns:
            Path to the written sidecar file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = output_dir / filename
        sidecar = {
            "document_type": "prediction_metadata_sidecar",
            "model_id": self._model_id,
            "version": self._version,
            **self._build_traceability_section(),
        }
        with sidecar_path.open("w") as fh:
            json.dump(sidecar, fh, indent=2, default=str)
        logger.info("Metadata sidecar written: {}", sidecar_path)
        return sidecar_path
