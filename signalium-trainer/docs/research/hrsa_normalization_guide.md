# HRSA Normalisation Guide

## Purpose

Level 1 raw metrics arrive in different units and scales.
Normalisation converts every raw value into a [0, 1] score before
category aggregation. This makes all metrics comparable.

## Direction

Every metric has a direction:

| Direction | Meaning | Normalised score |
|-----------|---------|-----------------|
| `higher_is_better` | Higher raw value = better | increases with value |
| `lower_is_better` | Lower raw value = better | `1 − normalised` |
| `neutral` | No direction | use as-is |

## Methods

### 1. Min-Max (default)

Linear mapping from expected range `[lo, hi]` → `[0, 1]`:

```
normalised = (value − lo) / (hi − lo)
```

Clipped to `[0, 1]`. Values outside range are capped.

**Best for**: metrics with well-known bounds (e.g. accuracy 0.50–0.75,
Sharpe 0.5–2.5).

```python
n.normalise(1.82, lo=0.5, hi=2.5, method="min_max", direction="higher_is_better")
# → 0.66
```

### 2. Z-Score

Standardises relative to the distribution of values:

```
z = (value − μ) / σ
normalised = (z + 3) / 6    # maps [−3, 3] → [0, 1]
```

**Best for**: metrics without known absolute bounds; relative ranking
matters more than absolute value.

```python
n.normalise(val, method="z_score", direction="higher_is_better",
            distribution=[1.2, 1.5, 1.8, 2.0, 1.6])
```

### 3. Percentile

Fraction of reference values ≤ the target value:

```
normalised = rank(value in distribution) / n
```

**Best for**: ordinal comparisons; robust to outliers.

### 4. Rank Scaling

Smooth rank interpolation over the unique sorted distribution values.
Similar to percentile but interpolates between unique values.

### 5. Robust Scaling

```
normalised = (value − median) / IQR
# mapped from [−2, 2] → [0, 1]
```

**Best for**: distributions with outliers that would distort Z-score.

### 6. Custom

Pass any callable `fn: float → float` that returns a value in `[0, 1]`:

```python
n.normalise(val, method="custom", direction="higher_is_better",
            custom_fn=lambda x: x / 100.0)
```

## Category Normaliser

`CategoryNormaliser` normalises all metrics in a category at once,
applies per-metric weights, and returns a weighted mean score in [0, 100]:

```python
from trainer.research import CategoryNormaliser, NormMethod

cn = CategoryNormaliser()
score_100, metric_results = cn.normalise_category(
    metric_values={"sharpe_ratio": 1.82, "max_drawdown": 0.067},
    metric_definitions=reg.for_category("trading_performance"),
    method=NormMethod.MIN_MAX,
    metric_weights={"sharpe_ratio": 3.0, "max_drawdown": 3.0},
)
```

## Per-Category Normalisation Configuration

Categories may use different normalisation methods via `conf/research/hrsa.yaml`:

```yaml
normalisation:
  global_default: min_max
  per_category:
    validation_quality: min_max
    trading_performance: min_max
    # ... override per category as needed
```

## Adding Custom Metric Bounds

To register a metric with custom normalisation bounds:

```python
from trainer.research import RawMetricsRegistry, MetricDefinition, MetricDirection

reg = RawMetricsRegistry.default()
reg.register(MetricDefinition(
    name="my_custom_alpha",
    direction=MetricDirection.HIGHER,
    category="trading_performance",
    unit="alpha",
    normalisation_lo=0.0,
    normalisation_hi=0.20,
))
```

## Confidence Impact of Normalisation

After normalisation, `CategoryNormaliser` computes confidence for each
category score based on:

1. **Coverage** (40%): fraction of expected metrics that have values
2. **Count** (35%): 10+ metrics → max confidence
3. **Score variance** (25%): low inter-metric variance → high confidence

A category with only 2/10 metrics provided will have low confidence
and therefore lower effective weight in the Level 3 composite.
