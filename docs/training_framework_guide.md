# Signalium Trainer — Training Framework Guide

## Overview

The training framework provides a complete, generic training loop that works with any
model inheriting from `BaseQuantModel`. Future models plug in without modifying the engine.

## Core Components

### `TrainingEngine`

The central orchestrator. Manages:
- Device placement and AMP scaler
- Optimizer and scheduler construction
- Training and validation loops
- Gradient clipping
- Checkpoint saving and resuming
- Early stopping

```python
from trainer.training.engine import TrainingEngine
from trainer.config.schema import TrainerConfig
import torch.nn as nn

engine = TrainingEngine(
    model=my_model,
    config=TrainerConfig(),
    criterion=nn.CrossEntropyLoss(),
)

result = engine.fit(train_loader, val_loader)
print(result["best_metrics"])
```

### `EarlyStopping`

Monitors a validation metric and signals when to stop:

```python
from trainer.training.early_stopping import EarlyStopping

es = EarlyStopping(patience=10, mode="min", min_delta=1e-4)
for epoch in range(100):
    val_loss = run_validation()
    if es.step(val_loss):
        break
```

### `CheckpointManager`

Saves `latest.pt`, `best.pt`, and per-epoch snapshots with top-K rotation:

```python
from trainer.checkpoints.manager import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir=Path("checkpoints"),
    keep_top_k=3,
    monitor_metric="val_loss",
    monitor_mode="min",
)
manager.save(epoch=5, model=model, optimizer=optimizer,
             scheduler=scheduler, metrics=metrics, config=config)
```

### `MetricsTracker`

Records per-epoch metrics and provides aggregation:

```python
from trainer.metrics.tracker import MetricsTracker

tracker = MetricsTracker()
tracker.update({"train_loss": 0.5, "val_loss": 0.4})
best_val = tracker.best("val_loss", mode="min")
history = tracker.history()
```

### `MLflowTracker`

Wraps MLflow for experiment tracking:

```python
from trainer.training.mlflow_tracker import MLflowTracker
from trainer.config.schema import MLflowConfig

tracker = MLflowTracker(config=MLflowConfig())
run_id = tracker.start_run(trainer_config)
tracker.log_metrics({"val_loss": 0.3, "accuracy": 0.95}, step=10)
tracker.end_run()
```

## Resuming Training

Pass the checkpoint path to `engine.fit()`:

```bash
trainer resume checkpoints/latest.pt --model my_model --dataset-dir data/datasets
```

## Mixed Precision

Enabled by default when CUDA is available. Controlled by `training.mixed_precision: true`.
Uses `torch.cuda.amp.GradScaler` and `torch.autocast`.

## Gradient Clipping

Controlled by `training.gradient_clip_val`. Set to `0` to disable.
Applied after unscaling the AMP scaler, before `optimizer.step()`.

## Seed Reproducibility

`set_seed(seed, deterministic=False)` seeds Python, NumPy, PyTorch, and CUDA RNGs.
Set `deterministic=True` for fully reproducible results (at the cost of speed).
