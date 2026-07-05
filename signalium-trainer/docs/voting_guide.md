# Ensemble Voting Guide

## Overview

The AEIP supports 8 ensemble combination strategies, all configurable via YAML. Each strategy implements a different approach to combining expert predictions.

---

## Strategy Reference

### 1. Simple Averaging

```python
from trainer.ensemble import build_strategy
strategy = build_strategy("simple_averaging")
result = strategy.combine(predictions)
```

**Formula:** `P(class) = mean(P_expert(class))` across all experts.
**Best for:** Baseline, equal-confidence experts, debugging.

---

### 2. Weighted Averaging

```python
strategy = build_strategy("weighted_averaging")
result = strategy.combine(predictions, weights={"e1": 0.7, "e2": 0.3})
```

**Formula:** `P(class) = sum(w_i * P_i(class))` normalized by sum(w_i).
**Best for:** General use when expert reliabilities differ.

---

### 3. Performance-Weighted Averaging

```python
strategy = build_strategy("performance_weighted")
# Weights computed from recent accuracy/Sharpe by DynamicWeightEngine
```

**Best for:** Automatic adaptation to expert performance.

---

### 4. Hard Voting

```python
strategy = build_strategy("hard_voting")
result = strategy.combine(predictions)
# result.buy_prob is 1.0 if buy wins majority, 0.0 otherwise
```

**Formula:** Majority class wins. Weighted version accounts for expert weights.
**Best for:** High-conviction signals, clear consensus required.
**Output:** Discrete class (probability is 1.0 for winner, 0.0 for others).

---

### 5. Soft Voting (Recommended)

```python
strategy = build_strategy("soft_voting")
result = strategy.combine(predictions, weights)
```

**Formula:** `P(class) = weighted sum of expert class probabilities`.
Confidence is scaled by inter-expert agreement.
**Best for:** Default production ensemble — smooth, calibrated output.

---

### 6. Mixture of Experts

```python
strategy = build_strategy("mixture_of_experts", temperature=0.5)
result = strategy.combine(predictions, weights)
```

**Formula:** Softmax-normalized gating on raw weights.
**Temperature:** 
  - `temperature=0.1` → winner-take-all (highest-weight expert dominates)
  - `temperature=1.0` → proportional allocation (default)
  - `temperature=5.0` → near-uniform distribution

**Best for:** Regime-conditioned routing where one expert should dominate.

```python
# Top expert in the result metadata:
print(result.metadata["top_expert"])
print(result.metadata["gate_weights"])
```

---

### 7. Stacking

```python
strategy = build_strategy("stacking", learning_rate=0.01)
# Online update after observing ground truth:
strategy.update(predictions, true_class="buy")
result = strategy.combine(predictions, weights)
```

**Formula:** Per-expert, per-class learned weights updated via online SGD.
**Best for:** When expert signals correlate with specific classes.

---

### 8. Blending

```python
strategy = build_strategy("blending")
result = strategy.combine(predictions, holdout_weights)
```

**Formula:** Identical to weighted averaging but weights are expected to be
calibrated from a holdout validation set.
**Best for:** When you have labeled holdout data to calibrate blending weights.

---

## Voting History Tracking

Every prediction records votes per expert:

```python
# Get votes for a specific prediction
votes = engine.db.get_voting_history(bundle.bundle_id)
for vote in votes:
    print(f"  {vote['expert_id']}: {vote['voted_class']} "
          f"(conf={vote['confidence']:.2f}, weight={vote['weight']:.3f})")
```

---

## Voting Distribution Chart

```python
# Get voting distribution visualization
vote_records = engine.db.get_voting_history(result_id="...")
chart = engine.viz.voting_distribution(vote_records)
# Returns: bar_stacked chart with buy/sell/no_trade counts per expert
```

---

## Agreement Score

All strategies compute `agreement_score ∈ [0, 1]`:
- `1.0` = all experts agree on the same class (unanimous)
- `0.33` = random agreement (experts split 3 ways equally)

```python
result = strategy.combine(predictions, weights)
print(f"Expert agreement: {result.agreement_score:.2%}")
```

High agreement (≥ 0.75) combined with high confidence (≥ 0.80) → Grade A institutional forecast.

---

## EnsembleResult Fields

| Field                  | Type              | Description                              |
|------------------------|-------------------|------------------------------------------|
| `buy_prob`             | float             | BUY probability [0, 1]                   |
| `sell_prob`            | float             | SELL probability [0, 1]                  |
| `no_trade_prob`        | float             | NO TRADE probability [0, 1]              |
| `confidence`           | float             | Ensemble-level confidence [0, 1]         |
| `uncertainty`          | float             | Ensemble-level uncertainty [0, 1]        |
| `strategy`             | str               | Strategy name                            |
| `n_experts`            | int               | Number of experts in this prediction     |
| `contributing_experts` | List[str]        | Expert IDs that contributed              |
| `weight_map`           | Dict[str, float]  | expert_id → normalized weight            |
| `agreement_score`      | float             | Inter-expert agreement [0, 1]            |
| `institutional_forecast` | str             | Human-readable forecast string           |
