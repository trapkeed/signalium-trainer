# Signalium Trainer — Developer Guide

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.x
- See `requirements.txt` for full dependency list

---

## Phase 14.5 — Multi-Task Quick Start

### 1. Enable Multi-Task Heads

Copy `conf/model/quant_v1_multitask.yaml` and set `enabled: true` on each head:

```yaml
stop_loss_head:
  enabled: true
  unit: pips
  loss_type: huber

take_profit_head:
  enabled: true
  unit: pips

tp_probability_head:
  enabled: true
```

### 2. Prepare Multi-Task Labels

Your Parquet files must include the optional label columns:

| Column           | Type  | Description                              |
|------------------|-------|------------------------------------------|
| `label`          | int   | 0=BUY, 1=SELL, 2=NO_TRADE (required)     |
| `expected_move`  | float | signed move in configured units          |
| `stop_loss`      | float | SL distance ≥ 0                          |
| `take_profit`    | float | TP distance ≥ 0                          |
| `holding_time`   | float | duration in configured units ≥ 0         |
| `tp_probability` | float | 0 or 1 (reached TP before SL)            |
| `confidence`     | float | target confidence in [0, 1]              |

### 3. Use MultiTaskParquetDataset

```python
from trainer.datasets.multi_task import MultiTaskParquetDataset, collate_multi_task
from torch.utils.data import DataLoader

dataset = MultiTaskParquetDataset(root=Path("data/"), split="train")
loader = DataLoader(
    dataset,
    batch_size=64,
    collate_fn=collate_multi_task,
    num_workers=4,
    pin_memory=True,
)
# Each batch: (x, labels_dict)
# x: (batch, seq_len, feature_dim)
# labels_dict: {"y_direction": ..., "y_sl": ..., "y_tp": ..., ...}
```

### 4. Build the Model

```python
from trainer.models.quant_v1.model import QuantModelV1
from trainer.models.quant_v1.config import (
    QuantV1Config, RegressionHeadConfig, TPProbabilityHeadConfig, RegressionUnit,
)

cfg = QuantV1Config()
cfg.stop_loss_head = RegressionHeadConfig(enabled=True, unit=RegressionUnit.PIPS)
cfg.take_profit_head = RegressionHeadConfig(enabled=True, unit=RegressionUnit.PIPS)
cfg.tp_probability_head = TPProbabilityHeadConfig(enabled=True)

model = QuantModelV1(trainer_config, model_config=cfg)
print(model.enabled_heads)
# ['direction', 'confidence', 'stop_loss', 'take_profit', 'tp_probability']
```

### 5. Build the Loss

```python
from trainer.losses.multi_task import MultiTaskLoss
from trainer.models.quant_v1.config import MultiTaskLossConfig

criterion = MultiTaskLoss(
    config=MultiTaskLossConfig(
        direction_weight=1.0,
        sl_weight=0.5,
        tp_weight=0.5,
        tp_probability_weight=0.5,
    )
)
```

### 6. Train with MultiTaskTrainingEngine

```python
from trainer.training.multi_task_engine import MultiTaskTrainingEngine

engine = MultiTaskTrainingEngine(model, trainer_config, criterion)
results = engine.fit(train_loader, val_loader)
```

### 7. Inference

```python
from trainer.framework.inference import InferenceAdapterFactory

adapter = InferenceAdapterFactory.from_model(model)
result = adapter.predict(x)

print(result["buy_prob"])           # P(BUY) per sample
print(result["stop_loss"])          # SL distance per sample
print(result["tp_probability"])     # P(TP before SL)
print(result["prediction_timestamp"])
```

---

## Backward Compatibility

Existing code using `QuantModelV1` requires **no changes**:

```python
# Original code — still works identically
model = QuantModelV1(trainer_config)     # all new heads disabled by default
logits = model(x)                        # returns (batch, 3) direction logits
loss = nn.CrossEntropyLoss()(logits, y)  # standard CE loss works as before
```

---

## Running the Tests

```bash
# All unit tests
pytest tests/unit/ -v

# Multi-task specific
pytest tests/unit/test_multi_task_model.py -v
pytest tests/unit/test_multi_task_loss.py -v
pytest tests/unit/test_multi_task_inference.py -v
```

---

## Adding a New Prediction Head

1. Add config dataclass to `trainer/models/quant_v1/config.py` and add field to `QuantV1Config`.
2. Add head class to `trainer/models/quant_v1/head.py`.
3. Wire head into model: `__init__`, `forward_multi_task()`, `_forward_full()`, `enabled_heads`.
4. Add loss weight to `MultiTaskLossConfig` and loss computation in `MultiTaskLoss.forward()`.
5. Add label column to `_OPTIONAL_LABEL_COLUMNS` in `trainer/datasets/multi_task.py`.
6. Add metric to `trainer/metrics/multi_task.py`.
7. Add shape and gradient tests.

---

## Troubleshooting

| Symptom                                  | Fix                                                          |
|------------------------------------------|--------------------------------------------------------------|
| Loss NaN / Inf                           | Check label ranges; ensure y_sl, y_tp ≥ 0; enable grad clip |
| Direction accuracy drops after new heads | Increase `direction_weight` to 1.5–2.0                      |
| Regression head predicts constant value  | Increase head weight; verify label variance                  |
| `KeyError` for head output in result     | Check head is enabled in config                              |
