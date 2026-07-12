"""Unit tests for Phase 14.7 prediction traceability.

Tests verify:
  - Prediction IDs are unique (UUID v4)
  - Metadata object is complete (all required fields populated)
  - Version information is correct
  - Inference timing is recorded and positive
  - TracedPrediction carries original predictions unmodified
  - Exported manifest metadata matches the loaded model
  - Config and schema hashing is stable and deterministic
  - InferenceContext captures environment correctly
  - TracedInferenceAdapter is a pure decorator — no prediction mutation
"""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.requires_torch
import torch

from trainer.exporters.metadata_manifest import MetadataManifestBuilder
from trainer.framework.traced_inference import TracedInferenceAdapter
from trainer.traceability import (
    InferenceContext,
    PredictionMetadata,
    hash_config,
    hash_schema,
    hash_string,
)
from trainer.traceability.config import TraceabilityConfig
from trainer.traceability.metadata import (
    InferenceInfo,
    ModelInfo,
    ReproducibilityInfo,
    SchemaVersions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """Minimal adapter stub returning fixed predictions."""

    def __init__(self, predictions: dict[str, Any]) -> None:
        self._fixed = predictions

    def predict(self, x: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        return dict(self._fixed)


def _make_adapter(
    predictions: dict[str, Any] | None = None,
    trace_config: TraceabilityConfig | None = None,
) -> TracedInferenceAdapter:
    preds = predictions or {
        "buy_prob": torch.tensor([0.7]),
        "sell_prob": torch.tensor([0.2]),
        "no_trade_prob": torch.tensor([0.1]),
        "confidence": torch.tensor([0.85]),
    }
    inner = _FakeAdapter(preds)
    cfg = trace_config or TraceabilityConfig(
        model_name="test_model",
        model_version="2.0.0",
        experiment_id="exp-unit-test",
        dataset_version="test-ds-v1",
        feature_schema_version="1.0.0",
        label_schema_version="1.0.0",
    )
    return TracedInferenceAdapter(inner, trace_config=cfg)


BATCH = 4
SEQ = 8
DIM = 16


@pytest.fixture
def x_batch() -> torch.Tensor:
    return torch.randn(BATCH, SEQ, DIM)


@pytest.fixture
def traced_adapter() -> TracedInferenceAdapter:
    return _make_adapter()


# ---------------------------------------------------------------------------
# PredictionMetadata — completeness and structure
# ---------------------------------------------------------------------------


class TestPredictionMetadata:
    def test_default_has_prediction_id(self) -> None:
        meta = PredictionMetadata()
        assert meta.prediction_id, "prediction_id must not be empty"
        # Validate UUID v4 format
        parsed = uuid.UUID(meta.prediction_id, version=4)
        assert str(parsed) == meta.prediction_id

    def test_default_has_prediction_timestamp(self) -> None:
        meta = PredictionMetadata()
        assert meta.prediction_timestamp
        # Must be parseable as ISO-8601
        from datetime import datetime

        dt = datetime.fromisoformat(meta.prediction_timestamp)
        assert dt is not None

    def test_all_sub_containers_present(self) -> None:
        meta = PredictionMetadata()
        assert isinstance(meta.model, ModelInfo)
        assert isinstance(meta.inference, InferenceInfo)
        assert isinstance(meta.reproducibility, ReproducibilityInfo)
        assert isinstance(meta.schemas, SchemaVersions)

    def test_to_dict_is_json_serialisable(self) -> None:
        meta = PredictionMetadata()
        d = meta.to_dict()
        json_str = json.dumps(d, default=str)
        assert isinstance(json_str, str)

    def test_to_flat_dict_has_prefixed_keys(self) -> None:
        meta = PredictionMetadata()
        flat = meta.to_flat_dict()
        assert "prediction_id" in flat
        assert "model.model_name" in flat
        assert "inference.device" in flat
        assert "reproducibility.git_commit_hash" in flat
        assert "schemas.output_schema_version" in flat

    def test_model_version_property(self) -> None:
        meta = PredictionMetadata()
        meta.model.model_version = "9.9.9"
        assert meta.model_version == "9.9.9"

    def test_inference_duration_property(self) -> None:
        meta = PredictionMetadata()
        meta.inference.inference_duration_ms = 42.0
        assert meta.inference_duration_ms == 42.0


# ---------------------------------------------------------------------------
# Prediction ID uniqueness
# ---------------------------------------------------------------------------


class TestPredictionIdUniqueness:
    def test_ids_are_unique_across_calls(self, traced_adapter, x_batch) -> None:
        results = [traced_adapter.predict(x_batch) for _ in range(100)]
        ids = [r.metadata.prediction_id for r in results]
        assert len(set(ids)) == 100, "All 100 prediction IDs must be distinct"

    def test_ids_are_uuid4(self, traced_adapter, x_batch) -> None:
        result = traced_adapter.predict(x_batch)
        pid = result.metadata.prediction_id
        parsed = uuid.UUID(pid, version=4)
        assert str(parsed) == pid

    def test_separate_adapters_produce_different_ids(self, x_batch) -> None:
        a1 = _make_adapter()
        a2 = _make_adapter()
        r1 = a1.predict(x_batch)
        r2 = a2.predict(x_batch)
        assert r1.metadata.prediction_id != r2.metadata.prediction_id


# ---------------------------------------------------------------------------
# Metadata completeness
# ---------------------------------------------------------------------------


class TestMetadataCompleteness:
    REQUIRED_MODEL_FIELDS = [
        "model_name",
        "model_version",
        "architecture",
        "framework",
        "precision",
    ]
    REQUIRED_INFERENCE_FIELDS = [
        "inference_timestamp",
        "inference_duration_ms",
        "batch_size",
        "device",
        "pytorch_version",
    ]
    REQUIRED_REPRO_FIELDS = [
        "git_commit_hash",
        "trainer_version",
        "framework_version",
    ]
    REQUIRED_SCHEMA_FIELDS = [
        "output_schema_version",
        "prediction_schema_version",
        "feature_schema_version",
        "label_schema_version",
    ]

    def test_model_fields_present(self, traced_adapter, x_batch) -> None:
        meta = traced_adapter.predict(x_batch).metadata
        for f in self.REQUIRED_MODEL_FIELDS:
            val = getattr(meta.model, f, None)
            assert val is not None, f"model.{f} must be present"

    def test_inference_fields_present(self, traced_adapter, x_batch) -> None:
        meta = traced_adapter.predict(x_batch).metadata
        for f in self.REQUIRED_INFERENCE_FIELDS:
            val = getattr(meta.inference, f, None)
            assert val is not None, f"inference.{f} must be present"

    def test_reproducibility_fields_present(self, traced_adapter, x_batch) -> None:
        meta = traced_adapter.predict(x_batch).metadata
        for f in self.REQUIRED_REPRO_FIELDS:
            val = getattr(meta.reproducibility, f, None)
            assert val is not None, f"reproducibility.{f} must be present"

    def test_schema_version_fields_present(self, traced_adapter, x_batch) -> None:
        meta = traced_adapter.predict(x_batch).metadata
        for f in self.REQUIRED_SCHEMA_FIELDS:
            val = getattr(meta.schemas, f, None)
            assert val is not None, f"schemas.{f} must be present"

    def test_model_name_from_config(self, x_batch) -> None:
        adapter = _make_adapter(
            trace_config=TraceabilityConfig(model_name="quant_pro", model_version="3.0.0")
        )
        meta = adapter.predict(x_batch).metadata
        assert meta.model.model_name == "quant_pro"
        assert meta.model.model_version == "3.0.0"

    def test_experiment_id_propagated(self, x_batch) -> None:
        cfg = TraceabilityConfig(experiment_id="exp-999")
        adapter = _make_adapter(trace_config=cfg)
        meta = adapter.predict(x_batch).metadata
        assert meta.model.experiment_id == "exp-999"

    def test_dataset_version_propagated(self, x_batch) -> None:
        cfg = TraceabilityConfig(dataset_version="dataset-v42")
        adapter = _make_adapter(trace_config=cfg)
        meta = adapter.predict(x_batch).metadata
        assert meta.reproducibility.dataset_version == "dataset-v42"

    def test_schema_versions_propagated(self, x_batch) -> None:
        cfg = TraceabilityConfig(
            feature_schema_version="2.1.0",
            label_schema_version="1.5.0",
            output_schema_version="4.0.0",
        )
        adapter = _make_adapter(trace_config=cfg)
        meta = adapter.predict(x_batch).metadata
        assert meta.schemas.feature_schema_version == "2.1.0"
        assert meta.schemas.label_schema_version == "1.5.0"
        assert meta.schemas.output_schema_version == "4.0.0"

    def test_random_seed_propagated(self, x_batch) -> None:
        cfg = TraceabilityConfig(random_seed=42)
        adapter = _make_adapter(trace_config=cfg)
        meta = adapter.predict(x_batch).metadata
        assert meta.reproducibility.random_seed == 42

    def test_mlflow_run_id_propagated(self, x_batch) -> None:
        cfg = TraceabilityConfig(mlflow_run_id="abc123run")
        adapter = _make_adapter(trace_config=cfg)
        meta = adapter.predict(x_batch).metadata
        assert meta.model.mlflow_run_id == "abc123run"


# ---------------------------------------------------------------------------
# Version information correctness
# ---------------------------------------------------------------------------


class TestVersionInformation:
    def test_pytorch_version_is_string(self, traced_adapter, x_batch) -> None:
        meta = traced_adapter.predict(x_batch).metadata
        assert isinstance(meta.inference.pytorch_version, str)
        assert len(meta.inference.pytorch_version) > 0

    def test_python_version_is_string(self, traced_adapter, x_batch) -> None:
        meta = traced_adapter.predict(x_batch).metadata
        assert isinstance(meta.inference.python_version, str)
        assert "." in meta.inference.python_version

    def test_trainer_version_matches(self, traced_adapter, x_batch) -> None:
        import trainer

        meta = traced_adapter.predict(x_batch).metadata
        assert meta.reproducibility.trainer_version == trainer.__version__

    def test_inference_engine_version_from_config(self, x_batch) -> None:
        cfg = TraceabilityConfig(inference_engine_version="2.5.0")
        adapter = _make_adapter(trace_config=cfg)
        meta = adapter.predict(x_batch).metadata
        assert meta.inference.inference_engine_version == "2.5.0"

    def test_git_commit_hash_is_string(self, traced_adapter, x_batch) -> None:
        meta = traced_adapter.predict(x_batch).metadata
        ghash = meta.reproducibility.git_commit_hash
        assert isinstance(ghash, str)
        assert len(ghash) > 0

    def test_framework_version_matches_pytorch(self, traced_adapter, x_batch) -> None:
        import torch

        meta = traced_adapter.predict(x_batch).metadata
        assert meta.reproducibility.framework_version == torch.__version__


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------


class TestInferenceTiming:
    def test_duration_is_positive(self, traced_adapter, x_batch) -> None:
        result = traced_adapter.predict(x_batch)
        assert result.metadata.inference.inference_duration_ms > 0

    def test_duration_is_in_milliseconds(self, traced_adapter, x_batch) -> None:
        result = traced_adapter.predict(x_batch)
        ms = result.metadata.inference.inference_duration_ms
        # Sanity: should be < 60_000 ms (60 seconds) for a test forward pass
        assert ms < 60_000, f"Duration {ms}ms seems unreasonably large"

    def test_batch_size_recorded(self, traced_adapter, x_batch) -> None:
        result = traced_adapter.predict(x_batch)
        assert result.metadata.inference.batch_size == BATCH

    def test_slow_adapter_has_larger_duration(self) -> None:
        class SlowAdapter:
            def predict(self, x, **kw):
                time.sleep(0.05)  # 50 ms
                return {"val": torch.tensor([1.0])}

        cfg = TraceabilityConfig()
        adapter = TracedInferenceAdapter(SlowAdapter(), trace_config=cfg)
        x = torch.zeros(1, 1, 1)
        result = adapter.predict(x)
        assert result.metadata.inference.inference_duration_ms >= 40.0  # allow some slack

    def test_inference_timestamp_is_recent(self, traced_adapter, x_batch) -> None:
        from datetime import datetime

        result = traced_adapter.predict(x_batch)
        ts_str = result.metadata.inference.inference_timestamp
        ts = datetime.fromisoformat(ts_str)
        now = datetime.now(tz=UTC)
        assert abs((now - ts).total_seconds()) < 5, "Timestamp should be within 5 seconds of now"


# ---------------------------------------------------------------------------
# TracedPrediction — predictions unmodified
# ---------------------------------------------------------------------------


class TestPredictionsUnmodified:
    def test_all_prediction_keys_preserved(self, x_batch) -> None:
        original_preds = {
            "buy_prob": torch.tensor([0.7, 0.6, 0.5, 0.4]),
            "sell_prob": torch.tensor([0.2, 0.3, 0.3, 0.4]),
            "no_trade_prob": torch.tensor([0.1, 0.1, 0.2, 0.2]),
            "confidence": torch.tensor([0.85, 0.75, 0.65, 0.55]),
            "expected_move": torch.tensor([12.0, 8.0, 5.0, 3.0]),
        }
        adapter = _make_adapter(predictions=original_preds)
        result = adapter.predict(x_batch)
        for key in original_preds:
            assert key in result.predictions, f"Key '{key}' missing from predictions"

    def test_prediction_values_unchanged(self, x_batch) -> None:
        fixed_val = torch.tensor([0.12345, 0.98765, 0.55555, 0.11111])
        preds = {"buy_prob": fixed_val}
        adapter = _make_adapter(predictions=preds)
        result = adapter.predict(x_batch)
        assert torch.allclose(result.predictions["buy_prob"], fixed_val)

    def test_metadata_key_not_in_predictions(self, traced_adapter, x_batch) -> None:
        result = traced_adapter.predict(x_batch)
        assert "prediction_id" not in result.predictions
        assert "metadata" not in result.predictions
        assert "model_version" not in result.predictions

    def test_to_dict_separates_namespaces(self, traced_adapter, x_batch) -> None:
        result = traced_adapter.predict(x_batch)
        d = result.to_dict()
        assert "predictions" in d
        assert "metadata" in d
        # prediction keys must not appear at the top level
        for key in result.predictions:
            assert key not in d or key in ("predictions", "metadata")

    def test_to_flat_dict_prefixes_predictions(self, traced_adapter, x_batch) -> None:
        result = traced_adapter.predict(x_batch)
        flat = result.to_flat_dict()
        for key in result.predictions:
            assert f"pred.{key}" in flat

    def test_prediction_id_property(self, traced_adapter, x_batch) -> None:
        result = traced_adapter.predict(x_batch)
        assert result.prediction_id == result.metadata.prediction_id

    def test_model_version_property(self, traced_adapter, x_batch) -> None:
        result = traced_adapter.predict(x_batch)
        assert result.model_version == "2.0.0"


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------


class TestHashingUtilities:
    def test_hash_config_is_deterministic(self) -> None:
        cfg = {"lr": 0.001, "batch_size": 32, "epochs": 10}
        h1 = hash_config(cfg)
        h2 = hash_config(cfg)
        assert h1 == h2

    def test_hash_config_field_order_independent(self) -> None:
        cfg_a = {"lr": 0.001, "batch_size": 32}
        cfg_b = {"batch_size": 32, "lr": 0.001}
        assert hash_config(cfg_a) == hash_config(cfg_b)

    def test_hash_config_different_values_differ(self) -> None:
        cfg_a = {"lr": 0.001}
        cfg_b = {"lr": 0.002}
        assert hash_config(cfg_a) != hash_config(cfg_b)

    def test_hash_schema_is_deterministic(self) -> None:
        schema = {"features": ["open", "high", "low", "close"], "version": "1.0.0"}
        assert hash_schema(schema) == hash_schema(schema)

    def test_hash_string_stable(self) -> None:
        h = hash_string("signalium_quant_v1")
        assert hash_string("signalium_quant_v1") == h

    def test_hash_length_default(self) -> None:
        h = hash_config({"x": 1})
        assert len(h) == 12

    def test_hash_length_custom(self) -> None:
        h = hash_config({"x": 1}, length=8)
        assert len(h) == 8

    def test_hash_dataclass(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class Cfg:
            lr: float = 0.001
            batch: int = 32

        h1 = hash_config(Cfg())
        h2 = hash_config(Cfg())
        assert h1 == h2


# ---------------------------------------------------------------------------
# InferenceContext
# ---------------------------------------------------------------------------


class TestInferenceContext:
    def test_device_cpu(self) -> None:
        ctx = InferenceContext(device=torch.device("cpu"))
        assert ctx.device == torch.device("cpu")

    def test_device_name_cpu(self) -> None:
        ctx = InferenceContext(device=torch.device("cpu"))
        assert ctx.device_name == "cpu"

    def test_pytorch_version_nonempty(self) -> None:
        ctx = InferenceContext(device=torch.device("cpu"))
        assert len(ctx.pytorch_version) > 0

    def test_python_version_nonempty(self) -> None:
        ctx = InferenceContext(device=torch.device("cpu"))
        assert "." in ctx.python_version

    def test_git_commit_nonempty(self) -> None:
        ctx = InferenceContext(device=torch.device("cpu"))
        assert len(ctx.git_commit_hash) > 0

    def test_inference_engine_version_propagated(self) -> None:
        ctx = InferenceContext(device=torch.device("cpu"), inference_engine_version="7.7.7")
        assert ctx.inference_engine_version == "7.7.7"

    def test_to_dict_has_required_keys(self) -> None:
        ctx = InferenceContext(device=torch.device("cpu"))
        d = ctx.to_dict()
        required = ["device", "device_name", "pytorch_version", "python_version", "git_commit_hash"]
        for key in required:
            assert key in d

    def test_memory_used_mb_nonnegative(self) -> None:
        ctx = InferenceContext(device=torch.device("cpu"))
        assert ctx.memory_used_mb() >= 0.0


# ---------------------------------------------------------------------------
# Metadata manifest builder
# ---------------------------------------------------------------------------


class TestMetadataManifestBuilder:
    def test_build_contains_traceability(self) -> None:
        builder = MetadataManifestBuilder(
            model_id="test_model",
            version="2.0.0",
            framework="pytorch",
            architecture="QuantModelV1",
        )
        manifest = builder.build()
        assert "traceability" in manifest
        t = manifest["traceability"]
        assert "model" in t
        assert "reproducibility" in t
        assert "schemas" in t

    def test_build_is_json_serialisable(self) -> None:
        builder = MetadataManifestBuilder(
            model_id="test_model",
            version="2.0.0",
            framework="torchscript",
            architecture="QuantModelV1",
        )
        json_str = json.dumps(builder.build(), default=str)
        assert isinstance(json_str, str)

    def test_from_trace_config(self) -> None:
        cfg = TraceabilityConfig(
            model_name="quant_pro",
            model_version="3.0.0",
            experiment_id="exp-777",
        )
        builder = MetadataManifestBuilder.from_trace_config(
            trace_config=cfg,
            framework="onnx",
            architecture="QuantModelV1",
        )
        manifest = builder.build()
        assert manifest["model_id"] == "quant_pro"
        assert manifest["version"] == "3.0.0"
        assert manifest["traceability"]["model"]["experiment_id"] == "exp-777"

    def test_enabled_heads_in_manifest(self) -> None:
        builder = MetadataManifestBuilder(
            model_id="m", version="1.0", framework="pytorch", architecture="Q"
        )
        heads = ["direction", "confidence", "mfe", "uncertainty"]
        builder.set_enabled_heads(heads)
        manifest = builder.build()
        assert manifest["traceability"]["model"]["enabled_heads"] == heads

    def test_num_parameters_in_manifest(self) -> None:
        builder = MetadataManifestBuilder(
            model_id="m", version="1.0", framework="pytorch", architecture="Q"
        )
        builder.set_num_parameters(1_030_000)
        manifest = builder.build()
        assert manifest["traceability"]["model"]["num_parameters"] == 1_030_000

    def test_write_manifest_creates_file(self) -> None:
        builder = MetadataManifestBuilder(
            model_id="m", version="1.0", framework="pytorch", architecture="Q"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = builder.write(Path(tmpdir))
            assert path.exists()
            content = json.loads(path.read_text())
            assert "traceability" in content

    def test_write_metadata_sidecar(self) -> None:
        builder = MetadataManifestBuilder(
            model_id="m", version="1.0", framework="pytorch", architecture="Q"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = builder.write_metadata_sidecar(Path(tmpdir))
            assert sidecar.exists()
            content = json.loads(sidecar.read_text())
            assert content["document_type"] == "prediction_metadata_sidecar"
            assert "model" in content
            assert "reproducibility" in content
            assert "schemas" in content

    def test_manifest_model_version_matches_config(self) -> None:
        cfg = TraceabilityConfig(model_version="5.1.3")
        builder = MetadataManifestBuilder.from_trace_config(
            trace_config=cfg,
            framework="pytorch",
            architecture="Q",
        )
        manifest = builder.build()
        assert manifest["version"] == "5.1.3"
        assert manifest["traceability"]["model"]["model_version"] == "5.1.3"

    def test_schemas_match_config(self) -> None:
        cfg = TraceabilityConfig(
            output_schema_version="3.0.0",
            prediction_schema_version="1.0.0",
            feature_schema_version="2.0.0",
            label_schema_version="1.5.0",
        )
        builder = MetadataManifestBuilder.from_trace_config(
            trace_config=cfg, framework="pytorch", architecture="Q"
        )
        schemas = builder.build()["traceability"]["schemas"]
        assert schemas["output_schema_version"] == "3.0.0"
        assert schemas["prediction_schema_version"] == "1.0.0"
        assert schemas["feature_schema_version"] == "2.0.0"
        assert schemas["label_schema_version"] == "1.5.0"


# ---------------------------------------------------------------------------
# TracedInferenceAdapter — pure decorator semantics
# ---------------------------------------------------------------------------


class TestTracedAdapterDecoratorSemantics:
    def test_adapter_property_returns_inner(self, x_batch) -> None:
        inner = _FakeAdapter({"x": torch.ones(1)})
        traced = TracedInferenceAdapter(inner, trace_config=TraceabilityConfig())
        assert traced.adapter is inner

    def test_context_property_returns_context(self, traced_adapter) -> None:
        ctx = traced_adapter.context
        assert isinstance(ctx, InferenceContext)

    def test_device_inferred_from_adapter_attribute(self, x_batch) -> None:
        inner = _FakeAdapter({"x": torch.ones(1)})
        inner.device = torch.device("cpu")
        traced = TracedInferenceAdapter(inner, trace_config=TraceabilityConfig())
        assert traced.context.device == torch.device("cpu")

    def test_default_device_is_cpu_when_not_available(self, x_batch) -> None:
        inner = _FakeAdapter({"x": torch.ones(1)})
        traced = TracedInferenceAdapter(inner, trace_config=TraceabilityConfig())
        assert traced.context.device.type == "cpu"

    def test_no_exception_on_minimal_config(self, x_batch) -> None:
        inner = _FakeAdapter({"val": torch.tensor([1.0])})
        traced = TracedInferenceAdapter(inner)
        result = traced.predict(x_batch)
        assert "val" in result.predictions
