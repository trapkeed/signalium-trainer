# Prediction Head Guide

**Phase**: 14.5 | **Model**: QuantModelV1

---

## Overview

All prediction heads receive the **same context vector** from the shared backbone.
Each head is a small, independent feed-forward network that produces a single
targeted output.

---

## 1. Direction Head (ClassificationHead)

**Status**: Always on  
**Output**: Logits → Softmax probabilities over {BUY, SELL, NO_TRADE}  
**Shape**: `(batch, 3)`

```
Linear(context_dim → 128) → LayerNorm → GELU → Dropout(0.3) → Linear(128 → 3)
```

The softmax is applied in the OutputLayer (not in the head itself) so that raw
logits are always available for CrossEntropyLoss.

**Class indices**: 0=BUY, 1=SELL, 2=NO_TRADE

---

## 2. Confidence Head (ConfidenceHead)

**Status**: Always on  
**Output**: Scalar confidence ∈ [0, 1]  
**Shape**: `(batch, 1)`

```
Linear(context_dim → 64) → GELU → Dropout(0.2) → Linear(64 → 1) → Sigmoid
```

Represents the model's internal certainty about its own direction prediction.
Trained with MSE against a calibrated confidence target derived from historical
per-bin accuracy. Use `gate_by_confidence(threshold)` on the ModelOutput to
suppress low-confidence signals.

---

## 3. Expected Move Head (ExpectedMoveHead)

**Status**: Optional (disabled by default)  
**Output**: Signed scalar move size  
**Shape**: `(batch, 1)`  
**Units**: pips | points | atr (configurable via `expected_move_head.unit`)

```
Linear(context_dim → 64) → GELU → Dropout(0.2) → Linear(64 → 1)
```

No output activation — the output is unrestricted. Positive = expected upward
move, negative = expected downward move.

**Label generation**: `y_move = entry_price - exit_price` (normalised to the
configured unit).

**Recommended loss**: Huber (robust to outlier moves).

---

## 4. Stop Loss Head (StopLossHead)

**Status**: Optional (disabled by default)  
**Output**: Non-negative SL distance from entry  
**Shape**: `(batch, 1)`  
**Units**: pips | atr | points (configurable via `stop_loss_head.unit`)

```
Linear(context_dim → 64) → GELU → Dropout(0.2) → Linear(64 → 1) → Softplus
```

Softplus guarantees output > 0 for all inputs, with smooth gradients everywhere
(unlike ReLU). This is critical because a zero or negative stop loss has no
physical meaning.

**Label generation**: `y_sl = abs(entry_price - optimal_sl)` in configured units.

**Recommended loss**: Huber.

---

## 5. Take Profit Head (TakeProfitHead)

**Status**: Optional (disabled by default)  
**Output**: Non-negative TP distance from entry  
**Shape**: `(batch, 1)`  
**Units**: pips | atr | points (configurable via `take_profit_head.unit`)

```
Linear(context_dim → 64) → GELU → Dropout(0.2) → Linear(64 → 1) → Softplus
```

Same architecture as StopLossHead with the same non-negativity guarantee.

**Label generation**: `y_tp = abs(entry_price - optimal_tp)` in configured units.

**Recommended loss**: Huber.

---

## 6. Holding Time Head (HoldingTimeHead)

**Status**: Optional (disabled by default)  
**Output**: Non-negative expected duration  
**Shape**: `(batch, 1)`  
**Units**: minutes | bars | candles (configurable via `holding_time_head.unit`)

```
Linear(context_dim → 64) → GELU → Dropout(0.2) → Linear(64 → 1) → Softplus
```

Softplus guarantees non-negative duration. Consider log-normalising the holding
time labels before training if the distribution is heavy-tailed.

**Label generation**: `y_holding_time = trade_exit_time - trade_entry_time` in
configured units.

**Recommended loss**: Huber (log-scale labels help with long-tail distributions).

---

## 7. TP Probability Head (TPProbabilityHead)

**Status**: Optional (disabled by default)  
**Output**: P(Take Profit reached before Stop Loss) ∈ [0, 1]  
**Shape**: `(batch, 1)`

```
Linear(context_dim → 64) → GELU → Dropout(0.2) → Linear(64 → 1) → Sigmoid
```

Binary classification output. A value near 1.0 indicates the model predicts
that the trade will reach TP before SL with high probability.

**Label generation**: `y_tp_prob = 1` if exit was at TP, `0` if exit was at SL.
Trades with ambiguous exits (time-based, manual) should be excluded or labelled
with an interpolated probability.

**Recommended loss**: BinaryCrossEntropy  
**Evaluation metric**: ROC-AUC (since the class balance may be skewed)

---

## Enabling / Disabling Heads

In your YAML config:

```yaml
stop_loss_head:
  enabled: true    # ← enables the head and its gradient computation
  hidden_dim: 64
  dropout: 0.2
  unit: pips
  loss_type: huber
  huber_delta: 1.0

expected_move_head:
  enabled: false   # ← head is not instantiated; no parameters, no compute
```

When a head is disabled:
- No parameters are added to the model
- No compute runs in `forward_multi_task()`
- The key is absent from the prediction dict
- The `MultiTaskLoss` silently skips that head

---

## Label Schema (MultiTaskParquetDataset)

| Parquet Column   | Labels Dict Key  | Type  | Description                     |
|------------------|------------------|-------|---------------------------------|
| `label`          | `y_direction`    | int   | 0=BUY, 1=SELL, 2=NO_TRADE       |
| `expected_move`  | `y_move`         | float | signed move in configured units |
| `stop_loss`      | `y_sl`           | float | SL distance ≥ 0                 |
| `take_profit`    | `y_tp`           | float | TP distance ≥ 0                 |
| `holding_time`   | `y_holding_time` | float | duration ≥ 0 in config units    |
| `tp_probability` | `y_tp_prob`      | float | 0 or 1 (binary)                 |
| `confidence`     | `y_confidence`   | float | target confidence in [0, 1]     |

All optional columns. Missing columns produce no key in the labels dict; the
`MultiTaskLoss` skips the head gracefully.
