# Prediction Traceability Guide
## Phase 14.7 — Signalium Quant

---

## Overview

Every inference produced by the Signalium Quant model is wrapped in a
`TracedPrediction` object that carries a `PredictionMetadata` record alongside
the original prediction dict.

**Prediction values are never modified.** Metadata travels in a dedicated
container that does not pollute the prediction namespace.

---

## Quick Start

```python
from trainer.traceability.config import TraceabilityConfig
from trainer.framework.traced_inference import TracedInferenceAdapter
from trainer.framework.inference import PyTorchInferenceAdapter

# 1. Build the existing adapter (no changes required)
inner = PyTorchInferenceAdapter(model=model, device=device)

# 2. Define traceability configuration
config = TraceabilityConfig(
    model_name="signalium_quant_v1",
    model_version="2.0.0",
    architecture="QuantModelV1",
    experiment_id="exp-2026-Q1",
    dataset_version="oanda-eurusd-v3",
    feature_schema_version="2.0.0",
    label_schema_version="1.0.0",
    random_seed=42,
)

# 3. Wrap with traceability
adapter = TracedInferenceAdapter(inner, trace_config=config)

# 4. Predict — returns TracedPrediction
result = adapter.predict(x)

# 5. Access predictions (unchanged)
buy_prob   = result.predictions["buy_prob"]
confidence = result.predictions["confidence"]

# 6. Access metadata
prediction_id  = result.metadata.prediction_id        # UUID v4
duration_ms    = result.metadata.inference_duration_ms
model_version  = result.metadata.model.model_version
git_hash       = result.metadata.reproducibility.git_commit_hash
```

---

## Prediction Metadata Specification

### Root Object — `PredictionMetadata`

| Field | Type | Description |
|-------|------|-------------|
| `prediction_id` | `str` (UUID v4) | Globally unique per inference call |
| `prediction_timestamp` | `str` (ISO-8601 UTC) | Timestamp of the inference call |
| `model` | `ModelInfo` | Static model information |
| `inference` | `InferenceInfo` | Runtime inference information |
| `reproducibility` | `ReproducibilityInfo` | Reproducibility anchors |
| `schemas` | `SchemaVersions` | All schema version identifiers |

---

### `ModelInfo`

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `str` | Human-readable model name |
| `model_version` | `str` | Semantic version (e.g. "2.0.0") |
| `architecture` | `str` | Architecture class name |
| `num_parameters` | `int` | Trainable parameter count |
| `input_shape` | `str` | Input shape description |
| `output_shape` | `str` | Output shape description |
| `enabled_heads` | `list[str]` | Active prediction head names |
| `framework` | `str` | "pytorch" / "torchscript" / "onnx" |
| `precision` | `str` | "fp32" / "fp16" / "int8" |
| `model_registry_id` | `str?` | Model registry identifier |
| `experiment_id` | `str?` | Experiment / study identifier |
| `mlflow_run_id` | `str?` | MLflow run ID |
| `training_timestamp` | `str?` | UTC ISO-8601 training completion time |
| `model_export_timestamp` | `str?` | UTC ISO-8601 export time |

---

### `InferenceInfo`

| Field | Type | Description |
|-------|------|-------------|
| `inference_timestamp` | `str` (ISO-8601 UTC) | Moment inference was called |
| `inference_duration_ms` | `float` | Wall-clock forward-pass time (ms) |
| `batch_size` | `int` | Number of samples in the batch |
| `device` | `str` | "cpu" / "cuda:0" / "mps" |
| `device_name` | `str` | Full device name (e.g. "NVIDIA A100 80GB") |
| `cuda_version` | `str?` | CUDA version if GPU, else `null` |
| `pytorch_version` | `str` | Installed PyTorch version |
| `python_version` | `str` | Python runtime version |
| `memory_used_mb` | `float` | GPU / RAM memory consumed (MB) |
| `inference_engine_version` | `str` | Inference adapter version |

---

### `ReproducibilityInfo`

| Field | Type | Description |
|-------|------|-------------|
| `git_commit_hash` | `str` | 12-char short SHA of model-producing commit |
| `trainer_version` | `str` | Signalium Trainer library version |
| `framework_version` | `str` | PyTorch version string |
| `random_seed` | `int?` | Training random seed |
| `training_config_hash` | `str?` | SHA-256 prefix of training config |
| `inference_config_hash` | `str?` | SHA-256 prefix of inference config |
| `feature_config_version` | `str?` | Feature config document version |
| `feature_schema_version` | `str?` | Feature engineering schema version |
| `label_schema_version` | `str?` | Label generation schema version |
| `normalization_version` | `str?` | Data normalisation pipeline version |
| `scaling_version` | `str?` | Feature scaling pipeline version |
| `configuration_version` | `str?` | Overall config file version tag |
| `dataset_version` | `str?` | Training dataset version |

---

### `SchemaVersions`

| Field | Type | Description |
|-------|------|-------------|
| `output_schema_version` | `str` | Model output contract version |
| `prediction_schema_version` | `str` | Inference response schema version |
| `feature_schema_version` | `str` | Feature engineering schema version |
| `label_schema_version` | `str` | Label generation schema version |
| `configuration_version` | `str` | Overall configuration version |

---

## Traceability Guide

### How to trace a prediction back to its origin

Every `TracedPrediction` carries enough information to fully reconstruct the
provenance chain:

```
prediction_id → inference timestamp → model_version → model_registry_id
                                                    → experiment_id
                                                    → mlflow_run_id
                                                    → git_commit_hash
                                                    → training_config_hash
                                                    → dataset_version
                                                    → feature_schema_version
                                                    → label_schema_version
```

**Step 1 — Log every prediction**

```python
import json

result = adapter.predict(x)

# Write to your audit log (replace with your structured logger)
log_entry = result.to_dict()
audit_logger.info(json.dumps(log_entry, default=str))
```

**Step 2 — Query by prediction ID**

```python
# Retrieve from your log store
record = audit_log.get(prediction_id="550e8400-e29b-41d4-a716-446655440000")
model_ver = record["metadata"]["model"]["model_version"]
experiment = record["metadata"]["model"]["experiment_id"]
```

**Step 3 — Reproduce the prediction**

```python
# Every field needed to reproduce is in ReproducibilityInfo
repro = record["metadata"]["reproducibility"]
seed  = repro["random_seed"]
cfg_h = repro["training_config_hash"]
ds_v  = repro["dataset_version"]

# Reload the exact model version from your registry
model = registry.load(model_id=record["metadata"]["model"]["model_registry_id"])
```

---

## Reproducibility Guide

### Config hashing

Use `hash_config()` to generate stable, order-independent hashes of any
training or inference configuration:

```python
from trainer.traceability import hash_config

# Training config dataclass — order-independent SHA-256 prefix
cfg_hash = hash_config(training_config)          # e.g. "a3f8b2c91d7e"

# Inference config dict
inf_hash = hash_config({"device": "cuda", "batch_size": 128})
```

Embed these hashes in `TraceabilityConfig` before wrapping the adapter:

```python
config = TraceabilityConfig(
    training_config_hash=hash_config(train_cfg),
    inference_config_hash=hash_config(inference_cfg),
    random_seed=42,
)
```

### Schema hashing

```python
from trainer.traceability import hash_schema

feature_hash = hash_schema(feature_schema_dict)
label_hash   = hash_schema(label_schema_dict)
```

### Reproducibility checklist

| Item | How to capture |
|------|---------------|
| Random seed | `TraceabilityConfig(random_seed=42)` |
| Training config | `hash_config(train_cfg)` → `training_config_hash` |
| Model weights | SHA-256 in export manifest |
| Git commit | Captured automatically by `InferenceContext` |
| Dataset version | `TraceabilityConfig(dataset_version="ds-v3")` |
| Feature schema | `TraceabilityConfig(feature_schema_version="2.0.0")` |
| Label schema | `TraceabilityConfig(label_schema_version="1.0.0")` |
| PyTorch version | Captured automatically |
| Python version | Captured automatically |

---

## Inference Metadata API

### `TracedInferenceAdapter`

```python
TracedInferenceAdapter(
    adapter,                          # Any object with predict(x) → dict
    trace_config: TraceabilityConfig, # Optional, defaults to TraceabilityConfig()
    device: torch.device,             # Optional, inferred from adapter if not given
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(x, **kwargs)` | `TracedPrediction` | Run inference with full metadata |
| `adapter` | inner adapter | Access the wrapped adapter |
| `context` | `InferenceContext` | Access the captured environment |

---

### `TracedPrediction`

| Attribute / Method | Type | Description |
|--------------------|------|-------------|
| `predictions` | `dict[str, Any]` | Original prediction dict (unmodified) |
| `metadata` | `PredictionMetadata` | Full traceability metadata |
| `prediction_id` | `str` | UUID v4 shortcut |
| `inference_duration_ms` | `float` | Duration shortcut |
| `model_version` | `str` | Version shortcut |
| `to_dict()` | `dict` | Nested dict with "predictions" and "metadata" |
| `to_flat_dict()` | `dict` | Flat dict for DataFrame / CSV rows |

---

### `PredictionMetadata`

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | JSON-serialisable nested dict |
| `to_flat_dict()` | `dict` | Flat `section.field` dict for logging |

---

## Export Manifest with Traceability

Use `MetadataManifestBuilder` (a superset of `ManifestBuilder`) to embed
traceability into model export manifests:

```python
from trainer.exporters.metadata_manifest import MetadataManifestBuilder
from trainer.traceability.config import TraceabilityConfig
from pathlib import Path

cfg = TraceabilityConfig(
    model_name="signalium_quant_v1",
    model_version="2.0.0",
    experiment_id="exp-2026-Q1",
    dataset_version="oanda-eurusd-v3",
    training_config_hash=hash_config(train_cfg),
    random_seed=42,
)

builder = MetadataManifestBuilder.from_trace_config(
    trace_config=cfg,
    framework="torchscript",
    architecture="QuantModelV1",
)
builder.set_enabled_heads(model.enabled_heads)
builder.set_num_parameters(sum(p.numel() for p in model.parameters()))
builder.add_file(Path("exports/model.pt"))

# Full manifest (files + checksums + traceability)
builder.write(output_dir=Path("exports/"))

# Lightweight sidecar for tools that don't parse the full manifest
builder.write_metadata_sidecar(output_dir=Path("exports/"))
```

The resulting files:

```
exports/
├── model.pt                    ← TorchScript artifact
├── manifest.json               ← Full manifest with traceability section
└── prediction_metadata.json    ← Standalone traceability sidecar
```

### Manifest structure

```json
{
  "model_id": "signalium_quant_v1",
  "version": "2.0.0",
  "framework": "torchscript",
  "architecture": "QuantModelV1",
  "created_at": "2026-07-05T12:00:00Z",
  "files": [...],
  "traceability": {
    "model": {
      "model_name": "signalium_quant_v1",
      "model_version": "2.0.0",
      "enabled_heads": ["direction", "confidence", "mfe", "..."],
      "num_parameters": 1030000,
      "experiment_id": "exp-2026-Q1",
      ...
    },
    "reproducibility": {
      "git_commit_hash": "a3f8b2c91d7e",
      "trainer_version": "0.1.0",
      "random_seed": 42,
      "training_config_hash": "9f7e2a3b1c8d",
      "dataset_version": "oanda-eurusd-v3",
      ...
    },
    "schemas": {
      "output_schema_version": "3.0.0",
      "prediction_schema_version": "1.0.0",
      "feature_schema_version": "2.0.0",
      "label_schema_version": "1.0.0"
    },
    "manifest_generated_at": "2026-07-05T12:00:00Z"
  }
}
```

---

## Audit Trail Design

### Recommended log schema

```python
{
    # Partition / index keys
    "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
    "prediction_timestamp": "2026-07-05T12:00:00.000000+00:00",

    # Traceability lookups
    "model.model_name": "signalium_quant_v1",
    "model.model_version": "2.0.0",
    "model.experiment_id": "exp-2026-Q1",
    "model.mlflow_run_id": "abc123",

    # Reproducibility
    "reproducibility.git_commit_hash": "a3f8b2c91d7e",
    "reproducibility.dataset_version": "oanda-eurusd-v3",
    "reproducibility.random_seed": 42,
    "reproducibility.training_config_hash": "9f7e2a3b1c8d",

    # Performance
    "inference.inference_duration_ms": 3.2,
    "inference.device": "cuda:0",
    "inference.batch_size": 128,

    # Prediction outcomes (for monitoring drift and calibration)
    "pred.buy_prob": [...],
    "pred.confidence": [...],
}
```

### Querying audit records

```sql
-- All predictions from a specific model version
SELECT * FROM predictions
WHERE "model.model_version" = '2.0.0'
  AND "prediction_timestamp" > '2026-07-01'
ORDER BY "prediction_timestamp";

-- Predictions by experiment
SELECT * FROM predictions
WHERE "model.experiment_id" = 'exp-2026-Q1';

-- Slow inference calls (>10ms)
SELECT prediction_id, "inference.inference_duration_ms"
FROM predictions
WHERE "inference.inference_duration_ms" > 10;
```

---

## Operational Readiness

| Capability | Status | Implementation |
|-----------|--------|---------------|
| Unique prediction IDs | ✅ | UUID v4 per call |
| Full provenance chain | ✅ | All 23 metadata fields |
| Inference timing | ✅ | `time.perf_counter()` — nanosecond precision |
| Device tracking | ✅ | `InferenceContext` |
| Git commit pinning | ✅ | `subprocess git rev-parse HEAD` |
| Config hashing | ✅ | SHA-256, order-independent |
| Schema versioning | ✅ | 5 versioned schema identifiers |
| Export manifest | ✅ | `MetadataManifestBuilder` |
| Standalone sidecar | ✅ | `write_metadata_sidecar()` |
| JSON serialisation | ✅ | `to_dict()` / `to_flat_dict()` |
| Zero prediction mutation | ✅ | Pure decorator pattern |
| Backward compatibility | ✅ | Inner adapter API unchanged |
| Reproducibility anchors | ✅ | Seed + config hashes + schema versions |
| Audit log compatibility | ✅ | `to_flat_dict()` → DataFrame / OLAP |
