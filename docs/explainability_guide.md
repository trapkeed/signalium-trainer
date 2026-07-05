# Signalium Quant — Explainability Guide

**Phase 14.9C** — Explainability & Interpretability

---

## Overview

The explainability layer answers three questions about every model prediction:

1. **Feature attribution** — which input features drove this prediction, and in which direction?
2. **Temporal attribution** — which timestep in the input sequence was most influential?
3. **Stream attribution** — for hybrid-encoder models, which processing stream contributed most to the latent representation?

All methods are **non-destructive** — they never modify model weights, never change prediction outputs, and always restore the model to its original training mode after analysis.

---

## Quick Start

```python
from trainer.explainability import ExplainabilityEngine

engine = ExplainabilityEngine(
    attribution_method="integrated_gradients",  # or "gradient"
    ig_steps=50,                                # integration steps
    top_k_features=10,                          # features in report
    extract_attention=True,                     # Transformer attention maps
)

explanation = engine.explain(
    model=model,
    input_tensor=x,                             # (batch, seq_len, n_features)
    feature_names=["open", "high", "low", "close", "volume", "rsi", ...],
    target_class=0,                             # 0=BUY, 1=SELL, 2=NO_TRADE
)

print(explanation.summary())
```

---

## Attribution Methods

### Integrated Gradients (recommended)

Computes the path integral of gradients from a baseline to the actual input. Satisfies the *Completeness* axiom:

```
sum(attribution) ≈ model(input)[class] - model(baseline)[class]
```

```python
from trainer.explainability import IntegratedGradients

ig = IntegratedGradients()
result = ig.attribute(
    model=model,
    input_tensor=x,         # (batch, seq_len, features)
    target_class=0,
    steps=50,               # higher = more accurate
    baseline=None,          # defaults to zeros
    feature_names=names,
)

# Aggregate over the sequence dimension
result.aggregate(reduction="abs_mean")

# Ranked list of (feature_name, score)
for name, score in result.top_features(k=10):
    print(f"{name:30s}  {score:+.4f}")
```

**When to use:** When you need accurate explanations that satisfy formal axioms. Use 50+ steps for production; 10–20 for fast development.

### Gradient × Input

Fast but approximate. Computes element-wise `gradient × input`:

```python
from trainer.explainability import GradientAttribution

result = GradientAttribution().attribute(model, x, target_class=0)
```

**When to use:** Fast iteration, debugging, or when IG is too slow.

### SmoothGrad

Averages gradients over `n_samples` noisy inputs to reduce noise:

```python
from trainer.explainability import SmoothGrad

result = SmoothGrad(n_samples=30, noise_std=0.10).attribute(
    model, x, target_class=0
)
```

**When to use:** When gradient maps are too noisy. Good for visualisation.

---

## Attention Visualization (Transformer Backbones)

Automatically extracts multi-head attention weights from `TransformerBackbone` and `HybridLSTMTransformerBackbone`:

```python
from trainer.explainability import AttentionExtractor, AttentionMapVisualizer

# Capture attention weights during a forward pass
attn_map = AttentionExtractor.extract(model, x)

# Which timestep was most attended?
ts = attn_map.most_attended_timestep(sample_idx=0)
print(f"Most attended timestep: t-{x.shape[1]-1-ts}")

# Per-layer focus (lower entropy = more focused)
for i, e in enumerate(attn_map.layer_entropy()):
    print(f"Layer {i}: entropy={e:.3f}")

# ASCII heatmap
viz = AttentionMapVisualizer(width=40, height=10)
print(viz.to_ascii(attn_map))
```

For LSTM/GRU/TCN backbones, `is_empty=True` is returned — no error is raised.

---

## Stream Contribution Analysis (Hybrid Encoder)

Measures how much each of the five hybrid encoder streams contributed to the final latent representation:

```python
from trainer.explainability import StreamContributionAnalyzer

# Option A: one-shot
contribution = StreamContributionAnalyzer.analyze(model, hybrid_input)

# Option B: after your own forward pass
with torch.no_grad():
    _ = model(hybrid_input)
contribution = StreamContributionAnalyzer.from_model(model)

# Ranked streams
for stream, weight in contribution.ranked_streams:
    print(f"{stream:<30s} {weight:.3f}")
```

**Streams tracked:**
| Stream | Description |
|---|---|
| `raw_market` | OHLCV sequence via LSTM/GRU |
| `multi_timeframe` | 7-scale timeframe encoder |
| `engineered_features` | Hand-crafted feature MLP |
| `context_embeddings` | Instrument / session / regime embeddings |
| `market_memory` | Market Memory Engine state |

---

## Permutation Feature Importance

Black-box importance ranking that does not require gradients — works with any model:

```python
from trainer.explainability import PermutationImportance

pi = PermutationImportance(
    metric="accuracy",   # or "loss"
    n_repeats=5,
    seed=42,
)
result = pi.compute(
    model=model,
    inputs=x,
    labels=y,
    feature_names=names,
)

# Top 10 most important features
for name, importance in result.top_k(10):
    print(f"{name:<30s} {importance:+.4f}")
```

---

## Prediction Explanation — Complete Output

The `PredictionExplanation` container combines all methods:

```python
explanation = engine.explain(model, x, feature_names=names)

# Human-readable summary
print(explanation.summary())

# JSON-safe dict (for logging, APIs, MLflow)
import json
data = explanation.to_dict()
json.dumps(data)  # safe — no tensors

# Individual fields
explanation.top_features             # [(name, score), ...]
explanation.most_influential_timeframe  # "t-3 (step 7/9)" or None
explanation.most_influential_memory     # "market_memory (weight=0.42)" or None
explanation.confidence_explanation   # "High confidence in BUY (65.3%)"
explanation.uncertainty_explanation  # populated when UncertaintyHead is enabled
explanation.stream_contributions     # StreamContribution object
explanation.attention_map            # AttentionMap object (may be empty)
```

---

## Visualization Chart Data

All chart data classes produce JSON-safe dicts — no plotting library required:

```python
from trainer.explainability.visualization import (
    FeatureImportanceChartData,
    FusionContributionChartData,
    AttentionMapChartData,
    EncoderContributionChartData,
)

# Feature bar chart data
chart = FeatureImportanceChartData.from_result(attribution_result, top_k=20)
chart.to_dict()  # {"feature_names": [...], "scores": [...], "colors": [...]}

# Hybrid stream pie chart
pie = FusionContributionChartData.from_contribution(stream_contrib)
pie.to_dict()    # {"stream_names": [...], "weights": [...]}

# Attention heatmap matrix
heatmap = AttentionMapChartData.from_attention_map(attn_map)
heatmap.to_dict()  # {"matrix": [[...], ...], "x_labels": [...], ...}
```

---

## API Reference

| Class | Method | Description |
|---|---|---|
| `ExplainabilityEngine` | `.explain(model, x, ...)` | Full orchestrated explanation |
| `ExplainabilityEngine` | `.explain_stream_contributions(...)` | Stream-only analysis |
| `IntegratedGradients` | `.attribute(model, x, ...)` | IG attribution |
| `GradientAttribution` | `.attribute(model, x, ...)` | Gradient × input |
| `SmoothGrad` | `.attribute(model, x, ...)` | Noise-averaged gradients |
| `AttributionResult` | `.aggregate(reduction)` | Reduce over seq dim |
| `AttributionResult` | `.top_features(k)` | Ranked feature list |
| `AttentionExtractor` | `.extract(model, x)` | One-shot extraction |
| `AttentionMap` | `.most_attended_timestep()` | Peak attention position |
| `AttentionMapVisualizer` | `.to_ascii(attn_map)` | ASCII heatmap |
| `PermutationImportance` | `.compute(model, x, y, ...)` | Black-box feature ranking |
| `StreamContributionAnalyzer` | `.analyze(model, x)` | Stream contribution |
| `StreamContributionAnalyzer` | `.from_model(model)` | Post-forward extraction |
