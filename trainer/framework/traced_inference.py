"""TracedInferenceAdapter — wraps any inference adapter with full traceability.

Usage::

    from trainer.traceability.config import TraceabilityConfig
    from trainer.framework.traced_inference import TracedInferenceAdapter
    from trainer.framework.inference import PyTorchInferenceAdapter

    # Build the inner adapter (existing, unchanged)
    inner = PyTorchInferenceAdapter(model=model, device=device)

    # Wrap it with traceability
    config = TraceabilityConfig(
        model_name="signalium_quant_v1",
        model_version="2.0.0",
        experiment_id="exp-001",
        dataset_version="dataset-v3",
    )
    adapter = TracedInferenceAdapter(inner, trace_config=config)

    # Use exactly as before — TracedPrediction has .predictions and .metadata
    result = adapter.predict(x)
    buy_prob = result.predictions["buy_prob"]       # unchanged
    pred_id  = result.metadata.prediction_id        # new
    duration = result.metadata.inference_duration_ms  # new

The existing ``PyTorchInferenceAdapter.predict()`` signature and return values
are NOT modified.  ``TracedInferenceAdapter`` is a pure decorator.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from trainer.traceability.config import TraceabilityConfig
from trainer.traceability.context import InferenceContext
from trainer.traceability.metadata import (
    InferenceInfo,
    ModelInfo,
    PredictionMetadata,
    ReproducibilityInfo,
    SchemaVersions,
)
from trainer.traceability.traced_output import TracedPrediction


class TracedInferenceAdapter:
    """Decorates any inference adapter with production-grade traceability.

    Every call to ``predict()`` returns a ``TracedPrediction`` that bundles:

    - The original prediction dict (values are **exactly unchanged**)
    - A ``PredictionMetadata`` object with all required fields

    Args:
        adapter: Any object with a ``predict(x) -> dict[str, Any]`` method.
        trace_config: Traceability configuration.
        device: Inference device.  Inferred from the adapter if ``None``.
    """

    def __init__(
        self,
        adapter: Any,
        trace_config: TraceabilityConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._adapter = adapter
        self._cfg = trace_config or TraceabilityConfig()

        # Resolve device — try common attribute names on the inner adapter
        if device is not None:
            _device = device
        elif hasattr(adapter, "device"):
            _device = adapter.device
        elif hasattr(adapter, "_device"):
            _device = adapter._device
        else:
            _device = torch.device("cpu")

        self._ctx = InferenceContext(
            device=_device,
            inference_engine_version=self._cfg.inference_engine_version,
        )

        # Snapshot model info (static — computed once)
        self._model_info = self._build_model_info()
        self._repro_info = self._build_repro_info()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor, **kwargs: Any) -> TracedPrediction:
        """Run inference and return predictions with full traceability metadata.

        Prediction values are **not changed**.  Metadata is returned in
        ``result.metadata``.

        Args:
            x: Input tensor to pass to the wrapped adapter.
            **kwargs: Additional keyword arguments forwarded to the adapter.

        Returns:
            A ``TracedPrediction`` containing the original prediction dict and
            a ``PredictionMetadata`` for this inference call.
        """
        t0 = time.perf_counter()
        predictions = self._adapter.predict(x, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0

        metadata = self._build_metadata(
            batch_size=x.shape[0] if hasattr(x, "shape") else 1,
            duration_ms=elapsed_ms,
        )
        return TracedPrediction(predictions=predictions, metadata=metadata)

    @property
    def adapter(self) -> Any:
        """The underlying (unwrapped) inference adapter."""
        return self._adapter

    @property
    def context(self) -> InferenceContext:
        """The static inference context (device, versions, git hash)."""
        return self._ctx

    # ------------------------------------------------------------------
    # Metadata construction
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        batch_size: int,
        duration_ms: float,
    ) -> PredictionMetadata:
        inference_info = InferenceInfo(
            inference_duration_ms=duration_ms,
            batch_size=batch_size,
            device=str(self._ctx.device),
            device_name=self._ctx.device_name,
            cuda_version=self._ctx.cuda_version,
            pytorch_version=self._ctx.pytorch_version,
            python_version=self._ctx.python_version,
            memory_used_mb=self._ctx.memory_used_mb(),
            inference_engine_version=self._cfg.inference_engine_version,
        )
        schemas = SchemaVersions(
            output_schema_version=self._cfg.output_schema_version,
            prediction_schema_version=self._cfg.prediction_schema_version,
            feature_schema_version=self._cfg.feature_schema_version,
            label_schema_version=self._cfg.label_schema_version,
            configuration_version=self._cfg.configuration_version or "1.0.0",
        )
        return PredictionMetadata(
            model=self._model_info,
            inference=inference_info,
            reproducibility=self._repro_info,
            schemas=schemas,
        )

    def _build_model_info(self) -> ModelInfo:
        """Resolve model information from the wrapped adapter (best-effort)."""
        num_params = 0
        enabled_heads: list[str] = []

        if hasattr(self._adapter, "_model") or hasattr(self._adapter, "model"):
            model = getattr(self._adapter, "_model", None) or getattr(self._adapter, "model", None)
            if model is not None:
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                if hasattr(model, "enabled_heads"):
                    enabled_heads = list(model.enabled_heads)

        return ModelInfo(
            model_name=self._cfg.model_name,
            model_version=self._cfg.model_version,
            architecture=self._cfg.architecture,
            num_parameters=num_params,
            enabled_heads=enabled_heads,
            framework=self._cfg.framework,
            precision=self._cfg.precision,
            model_registry_id=self._cfg.model_registry_id,
            experiment_id=self._cfg.experiment_id,
            mlflow_run_id=self._cfg.mlflow_run_id,
            training_timestamp=self._cfg.training_timestamp,
            model_export_timestamp=self._cfg.model_export_timestamp,
        )

    def _build_repro_info(self) -> ReproducibilityInfo:
        return ReproducibilityInfo(
            git_commit_hash=self._ctx.git_commit_hash,
            trainer_version=self._cfg.trainer_version,
            framework_version=self._ctx.pytorch_version,
            random_seed=self._cfg.random_seed,
            training_config_hash=self._cfg.training_config_hash,
            inference_config_hash=self._cfg.inference_config_hash,
            feature_config_version=self._cfg.feature_config_version,
            feature_schema_version=self._cfg.feature_schema_version,
            label_schema_version=self._cfg.label_schema_version,
            normalization_version=self._cfg.normalization_version,
            scaling_version=self._cfg.scaling_version,
            configuration_version=self._cfg.configuration_version,
            dataset_version=self._cfg.dataset_version,
        )
