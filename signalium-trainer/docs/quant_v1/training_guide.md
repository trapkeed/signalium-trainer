# QuantModelV1 — Training Guide

## Quick Start

```bash
# 1. Download and validate the dataset
trainer download <dataset_id> --output-dir data/datasets
trainer validate data/datasets

# 2. Train QuantModelV1
trainer train quant_v1 \
  --dataset-dir data/datasets \
  --epochs 100 \
  --run-name "quant_v1_baseline"

# 3. Evaluate
trainer evaluate checkpoints/best.pt \
  --dataset-dir data/datasets \
  --output-dir reports/quant_v1

# 4. Export
trainer export checkpoints/best.pt \
  --output-dir exports/quant_v1

# 5. Upload
trainer upload exports/quant_v1 \
  --model-id quant_v1 \
  --version 1.0.0
```

---

## Key Configuration Decisions

### Class Imbalance

Trading datasets are almost always imbalanced. NO TRADE is typically the dominant class.
Recommended strategies (in order of preference):

1. **Class weights** (default) — fast, no data duplication:
   ```yaml
   balancing:
     strategy: class_weights
   ```

2. **Weighted Sampler** — equal class frequency per batch:
   ```yaml
   balancing:
     strategy: weighted_sampler
   ```

3. **Focal Loss** — dynamically down-weights easy examples:
   ```yaml
   loss:
     type: focal
     focal_gamma: 2.0
   ```

### Loss Function

| Loss | When to use |
|---|---|
| `cross_entropy` | Balanced datasets, baseline |
| `weighted_cross_entropy` | Imbalanced, want explicit weight control |
| `focal` | Strongly imbalanced, hard examples |

### Backbone Selection

| Backbone | Config | Characteristics |
|---|---|---|
| LSTM | `type: lstm` | Best default; captures long-range dependencies |
| GRU | `type: gru` | Faster; fewer parameters; often similar quality |

### Window Length

`backbone.window_length` controls how many historical bars the model sees.
Longer windows capture more context but require more memory and compute.

| Timeframe | Recommended window |
|---|---|
| 1-minute bars | 60–120 |
| 5-minute bars | 48–96 |
| 1-hour bars | 24–72 |
| Daily bars | 20–60 |

---

## Hyper-parameter Recommendations

### Optimizer
```yaml
optimizer:
  type: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
```

### Scheduler
```yaml
scheduler:
  type: cosine
  T_max: 100     # Set to epochs
  eta_min: 1.0e-6
```

### Regularization
```yaml
backbone:
  dropout: 0.3
classification_head:
  dropout: 0.3
training:
  gradient_clip_val: 1.0
  early_stopping_patience: 10
```

---

## Monitoring Training

MLflow tracks every run automatically. Key metrics to watch:

| Metric | Good trend | Warning |
|---|---|---|
| `train_loss` | Decreasing | Plateaus early → LR too low |
| `val_loss` | Decreasing | Diverges from train → overfitting |
| `val_f1_macro` | Increasing | Stuck at 0.33 → class collapse |
| `val_roc_auc` | > 0.7 | < 0.55 → random performance |

---

## Preventing Common Failures

1. **Class collapse** — model predicts NO TRADE for everything.
   Fix: Check class weights, verify dataset balance, try focal loss.

2. **Gradient explosion** — loss becomes NaN.
   Fix: Lower learning rate, enable gradient clipping (`gradient_clip_val: 1.0`).

3. **Overfitting** — val_loss increases while train_loss decreases.
   Fix: Increase dropout, enable early stopping, reduce model capacity.

4. **Never validate** — model does not improve past random.
   Fix: Check feature normalisation, verify label quality, inspect confusion matrix.
