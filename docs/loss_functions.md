# Loss Function Guide

**Phase**: 14.5 | **Model**: QuantModelV1

---

## Overview

Phase 14.5 introduces `MultiTaskLoss` — a single composite loss module that
computes a **weighted sum** of all enabled head losses in one pass.

```
L_total = w_dir   × L_direction
        + w_move  × L_expected_move    (if head enabled + labels present)
        + w_sl    × L_stop_loss        (if head enabled + labels present)
        + w_tp    × L_take_profit      (if head enabled + labels present)
        + w_time  × L_holding_time     (if head enabled + labels present)
        + w_conf  × L_confidence       (if labels present)
        + w_tpprob× L_tp_probability   (if head enabled + labels present)
```

---

## Loss Functions by Head

### Direction Loss
**Type**: `CrossEntropyLoss`  
**Target**: Integer class label in {0, 1, 2}

Standard multi-class cross-entropy.  Supports:
- Class weights (`direction_class_weights` tensor)
- Label smoothing (`label_smoothing` float)

Optionally replace with Focal Loss for severe class imbalance:

```python
from trainer.losses.focal import FocalLoss
direction_loss = FocalLoss(gamma=2.0)
```

---

### Confidence Loss
**Type**: `MSELoss` (Mean Squared Error)  
**Target**: Float in [0, 1] representing calibrated target confidence

The confidence head is trained to predict the fraction of similar predictions
that are correct in the training set (per-bin calibration).

---

### Expected Move Loss
**Type**: Configurable — Huber (default), MSE, or MAE  
**Target**: Signed float (move size in configured units)

Huber loss is recommended because move size targets may contain outliers:

```
L_Huber(ŷ, y) = 0.5 × (ŷ - y)²          if |ŷ - y| ≤ δ
                δ × (|ŷ - y| - 0.5 × δ)  otherwise
```

Set `huber_delta` to approximately 1 ATR for good robustness.

---

### Stop Loss Distance Loss
**Type**: Configurable — Huber (default), MSE, or MAE  
**Target**: Non-negative float (SL distance in configured units)

Note: The StopLossHead outputs via Softplus, so the model can never predict a
negative SL. However, the loss function still operates on the raw Softplus
output — no additional clamping needed.

---

### Take Profit Distance Loss
**Type**: Configurable — Huber (default), MSE, or MAE  
**Target**: Non-negative float (TP distance in configured units)

Same architecture and recommendations as the Stop Loss head.

---

### Holding Time Loss
**Type**: Configurable — Huber (default), MSE, or MAE  
**Target**: Non-negative float (duration in configured units)

For heavy-tailed holding time distributions (common in trend-following),
consider log-normalising the targets before training:

```python
y_log_holding_time = torch.log1p(y_holding_time)
```

---

### TP Probability Loss
**Type**: `BinaryCrossEntropyLoss`  
**Target**: Binary float (0.0 or 1.0)

Standard binary cross-entropy between the Sigmoid output and the binary label.

```
L_BCE(p, y) = -[y × log(p) + (1-y) × log(1-p)]
```

---

## Configuration

### YAML

```yaml
multi_task_loss:
  direction_weight:     1.0
  move_weight:          0.5
  sl_weight:            0.5
  tp_weight:            0.5
  holding_time_weight:  0.3
  confidence_weight:    0.2
  tp_probability_weight: 0.5
```

### Python

```python
from trainer.losses.multi_task import MultiTaskLoss
from trainer.models.quant_v1.config import MultiTaskLossConfig, RegressionLossType

criterion = MultiTaskLoss(
    config=MultiTaskLossConfig(
        direction_weight=1.0,
        move_weight=0.5,
        sl_weight=0.5,
        tp_weight=0.5,
    ),
    expected_move_loss_type=RegressionLossType.HUBER,
    huber_delta=1.5,
    label_smoothing=0.1,
)
```

---

## Loss Weighting Guidelines

| Head             | Suggested Weight | Rationale                                              |
|------------------|-----------------|--------------------------------------------------------|
| direction        | 1.0             | Primary task — highest weight                          |
| tp_probability   | 0.5             | High-value secondary signal                            |
| stop_loss        | 0.5             | Risk management critical                               |
| take_profit      | 0.5             | Risk/reward critical                                   |
| expected_move    | 0.3–0.5         | Useful but noisier than SL/TP                          |
| holding_time     | 0.2–0.3         | Lower priority; high label noise in live markets       |
| confidence       | 0.1–0.2         | Calibration auxiliary; don't let it dominate           |

Start with the default weights and adjust based on per-head validation metrics.
If a regression head shows high MAE, increase its weight. If direction accuracy
drops, reduce the competing head weights.

---

## Per-Head Loss Tracking

`MultiTaskLossOutput.to_dict()` returns all per-head losses as scalar floats:

```python
loss_out = criterion(predictions, labels)
metrics = loss_out.to_dict()
# {
#   "loss_total": 1.23,
#   "loss_direction": 0.85,
#   "loss_expected_move": 0.41,
#   "loss_stop_loss": 0.32,
#   ...
# }
```

All keys are prefixed with `loss_` and are compatible with MLflow / W&B logging.

---

## Adding a New Loss

1. Add a new `RegressionLossType` variant to `config.py`.
2. Add a branch to `_regression_loss()` in `losses/multi_task.py`.
3. No changes needed to the model or training engine.
