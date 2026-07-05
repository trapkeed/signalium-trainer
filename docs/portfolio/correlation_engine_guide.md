# Correlation Engine Guide — Phase 20

## Overview

The `CorrelationEngine` computes dynamic, multi-method correlations between
instrument return series. It maintains a rolling history of observations and
supports six correlation methods.

---

## Correlation Methods

| Method | Class Enum | Description |
|--------|-----------|-------------|
| Rolling (Pearson) | `ROLLING` | Standard Pearson over a sliding window |
| Rank (Spearman) | `RANK` | Rank-based — robust to outliers |
| Tail (lower) | `TAIL_LOWER` | Correlation in lower tail of distribution |
| Tail (upper) | `TAIL_UPPER` | Correlation in upper tail |
| Exponential | `EXPONENTIAL` | EWMA — recent data weighted more |
| Conditional | `CONDITIONAL` | Conditioned on a third series |

---

## Usage

```python
from trainer.portfolio import CorrelationEngine, CorrelationMethod

eng = CorrelationEngine(window=60, decay=0.94)

# Feed return observations
for i in range(100):
    eng.add_observations({
        "EURUSD": returns_eurusd[i],
        "GBPUSD": returns_gbpusd[i],
        "XAUUSD": returns_xauusd[i],
    })

# Compute full matrix
matrix = eng.compute_matrix(method=CorrelationMethod.ROLLING)
print(matrix.get("EURUSD", "GBPUSD"))  # 0.82

# High correlation pairs
pairs = matrix.high_correlation_pairs(threshold=0.70)
for a, b, corr in pairs:
    print(f"{a}-{b}: {corr:.2f}")

# Regime-aware matrix
matrix_risk_off = eng.regime_aware_matrix(regime="risk_off")
```

---

## Correlation Confidence

Confidence represents how reliable the correlation estimate is, based on
the number of observations relative to the configured window.

```python
conf = eng.correlation_confidence("EURUSD", "GBPUSD")
# 0.0 (no data) → 1.0 (full window)
```

---

## CorrelationMatrix

| Property | Description |
|----------|-------------|
| `instruments` | List of instrument symbols |
| `matrix` | Dict[(a, b)] → float |
| `method` | Method used |
| `get(a, b)` | Returns correlation (0.0 if unknown) |
| `high_correlation_pairs(threshold)` | Pairs above threshold |
| `to_heatmap_data()` | N×N list of lists |
| `to_dict()` | Serializable representation |

---

## Regime-Aware Behaviour

| Regime | Recommended Method |
|--------|------------------|
| `risk_off`, `crash`, `high_volatility` | TAIL_LOWER (tail correlations spike) |
| `risk_on`, `trending` | ROLLING |
| All others | EXPONENTIAL |
