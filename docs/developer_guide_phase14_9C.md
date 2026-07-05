# Signalium Trainer — Phase 14.9C Developer Guide

**Explainability, Plugin Architecture, and Future Interfaces**

---

## New Packages in Phase 14.9C

```
trainer/
├── explainability/           ← NEW: complete explainability layer
│   ├── __init__.py           ← public API
│   ├── attribution.py        ← IntegratedGradients, GradientAttribution, SmoothGrad
│   ├── attention.py          ← AttentionExtractor, AttentionMap
│   ├── permutation.py        ← PermutationImportance, FeatureImportanceResult
│   ├── stream_analysis.py    ← StreamContributionAnalyzer, StreamContribution
│   ├── explanation.py        ← ExplainabilityEngine, PredictionExplanation
│   └── visualization.py      ← 4 chart data classes + AttentionMapVisualizer
│
├── interfaces/               ← NEW: future encoder interface contracts
│   ├── __init__.py
│   ├── base.py               ← BaseExternalEncoder (abstract)
│   └── future_encoders.py    ← 9 future encoder stubs (interfaces only)
│
└── plugins/                  ← NEW: plugin registration system
    ├── __init__.py
    └── registry.py           ← PluginRegistry, EncoderPlugin
```

---

## Explainability — Developer Patterns

### Pattern 1: Explain a single prediction

```python
from trainer.explainability import ExplainabilityEngine

engine = ExplainabilityEngine(
    attribution_method="integrated_gradients",
    ig_steps=50,
    top_k_features=10,
)

with torch.no_grad():
    pass  # attribution handles grad mode internally

explanation = engine.explain(
    model=model,
    input_tensor=x,               # (1, seq_len, n_features) — single sample
    feature_names=feature_names,
    target_class=0,               # 0=BUY, 1=SELL, 2=NO_TRADE
)
print(explanation.summary())
```

### Pattern 2: Batch attribution for a dataset

```python
from trainer.explainability import IntegratedGradients, AttributionResult

ig = IntegratedGradients()
all_scores = []

for batch_x, batch_y in dataloader:
    result = ig.attribute(model, batch_x, target_class=0, steps=20)
    result.aggregate(reduction="abs_mean")
    all_scores.append(result.aggregated)  # (batch, n_features)

# Mean importance across the dataset
dataset_importance = torch.cat(all_scores, dim=0).mean(dim=0)
```

### Pattern 3: Save explanations as JSON

```python
import json
from pathlib import Path

explanation = engine.explain(model, x, feature_names=names)
path = Path("experiments/explanations") / f"{prediction_id}.json"
path.write_text(json.dumps(explanation.to_dict(), indent=2))
```

### Pattern 4: Log to MLflow

```python
import mlflow

d = explanation.to_dict()
mlflow.log_dict(d, "prediction_explanation.json")
mlflow.log_metric("top_feature_score", d["top_features"][0]["score"])
```

---

## Attribution — Implementation Notes

**Integrated Gradients computational cost:**
- Time: O(steps × batch × forward pass)
- Memory: same as a single forward pass (gradients freed each step)
- Recommendation: `steps=50` for production; `steps=10` for CI/testing

**SmoothGrad noise standard deviation:**
- `noise_std` is a fraction of the input range (not absolute)
- Default `0.15` = 15% of `max(x) - min(x)`
- Higher noise → smoother but less precise; lower → noisier but more faithful

**Permutation Importance computational cost:**
- Time: O(n_features × n_repeats × forward pass)
- For 100 features × 5 repeats = 500 forward passes
- Use smaller `n_repeats=2` during development

---

## Attention Extraction — Implementation Notes

`AttentionExtractor` uses `module.register_forward_hook()` on every `nn.MultiheadAttention` sub-module.

**Important:** PyTorch's `TransformerEncoderLayer` uses `nn.MultiheadAttention` internally, but attention weights are only available when `need_weights=True` (the default). If you switch to custom attention with `need_weights=False`, the extractor returns an empty map.

**LSTM/GRU models:** Return an empty `AttentionMap` with `is_empty=True`. Never raises an error — callers must check `is_empty`.

---

## Plugin System — Developer Notes

### Registry is a class-level singleton

The `PluginRegistry._plugins` dict is class-level — shared across all imports. In tests, always use `PluginRegistry.clear()` in an autouse fixture to prevent test pollution.

### Decorator executes at import time

```python
# This registers the encoder immediately when the module is imported:
@PluginRegistry.register("my_enc")
class MyEncoder(BaseExternalEncoder):
    ...
```

This means plugins are registered by importing the module that contains them. Add plugin module imports to a `plugins/` directory `__init__.py` so they're picked up at startup.

### Version policy

- Use semantic versioning: `"1.0.0"` for production, `"0.x.y"` for experimental
- Bump the minor version when the output schema changes
- Bump the major version when the input API changes (breaking)

---

## Metadata — What Changed in 14.9C

### ModelInfo new fields

```python
from trainer.traceability.metadata import ModelInfo

info = ModelInfo(
    phase="14.9C",
    backbone_type="transformer",
    active_streams=["raw_market", "engineered_features"],
    fusion_strategy="attention",
    encoder_versions={"backbone": "14.9B", "hybrid_encoder": "14.9A"},
)
```

All 5 new fields have safe defaults so existing code that constructs `ModelInfo()` is unchanged.

### get_export_metadata() additions

```python
meta = model.get_export_metadata()
# New keys guaranteed present:
meta["backbone_type"]     # e.g. "lstm"
meta["active_streams"]    # e.g. ["raw_market"] or []
meta["fusion_strategy"]   # e.g. "attention" or None
meta["encoder_versions"]  # e.g. {"backbone": "14.9B", ...}
# Phase is now "14.9C" when hybrid encoder is active
meta["phase"]             # "14.9C" | "14.6" | "14.5"
```

---

## Testing Phase 14.9C Code

```bash
# Run all explainability tests
pytest tests/unit/test_explainability.py -v

# Run just attribution tests
pytest tests/unit/test_explainability.py -k "Attribution" -v

# Run plugin registry tests
pytest tests/unit/test_explainability.py -k "PluginRegistry" -v

# Run metadata extension tests
pytest tests/unit/test_explainability.py -k "ModelInfo" -v
```

---

## Common Mistakes

### 1. Calling attribution inside `torch.no_grad()`

Attribution methods internally re-enable gradients via `torch.enable_grad()`. But if you've disabled grad globally with `torch.no_grad()` as a context, the inner `enable_grad()` will work correctly in Python — this is by design in PyTorch.

However, it's cleaner to call attribution outside of `no_grad` blocks:

```python
# Good
explanation = engine.explain(model, x)

# Also fine (enable_grad overrides no_grad inside attribution)
with torch.no_grad():
    explanation = engine.explain(model, x)  # internally re-enables grad
```

### 2. Forgetting to call `aggregate()` before `top_features()`

```python
result = ig.attribute(model, x, steps=50)
# result.aggregate() is called automatically by top_features()
# But for explicitness, call it manually:
result.aggregate(reduction="abs_mean")
top = result.top_features(k=10)
```

### 3. Plugin output_dim mismatch

The plugin's `output_dim` must match the fusion layer's `latent_dim`. Always call `validate_output()` before passing to the fusion layer.

### 4. Not clearing the registry in tests

Each test that registers a plugin must clean up:

```python
@pytest.fixture(autouse=True)
def clean_registry():
    PluginRegistry.clear()
    yield
    PluginRegistry.clear()
```
