# Experiment Guide

## Purpose

`ExperimentRecord` and `ExperimentTracker` give every training run a
persistent, searchable identity — including the git hash, hardware snapshot,
dataset version, and full metric history.

## ExperimentRecord

```python
from trainer.research import ExperimentRecord

record = ExperimentRecord(
    name="lstm-baseline-v1",
    model_name="quant_v1",
    model_version="1.0.0",
    dataset_version="2.1.0",
    feature_version="1.0.0",
    label_version="1.0.0",
    random_seed=42,
    configuration={"backbone": "lstm", "hidden_size": 256},
    hyperparameters={"lr": 1e-3, "dropout": 0.2},
)

# ... run training ...

record.complete(metrics={
    "accuracy": 0.742,
    "f1_macro": 0.718,
    "val_loss": 0.831,
})

# On failure:
# record.fail("GPU OOM at epoch 47")
```

All fields are automatically populated where possible:
- `experiment_id` — UUID4
- `git_commit_hash` — captured via `git rev-parse --short HEAD`
- `hardware_info` — CPU, GPU, RAM snapshot
- `start_time` / `end_time` — UTC datetimes
- `training_duration_seconds` — computed from timestamps

## ExperimentTracker

```python
from trainer.research import ExperimentTracker
from pathlib import Path

tracker = ExperimentTracker(store_path=Path("experiments/records.json"))

# Save on start
tracker.save(record)

# Update after completion
record.complete({"accuracy": 0.74})
tracker.save(record)

# Query
completed = tracker.filter(status="completed", model_name="quant_v1")
best = tracker.best_by("f1_macro", model_name="quant_v1")
all_records = tracker.load_all()
```

## Filtering and Ranking

```python
# Filter by multiple criteria
records = tracker.filter(
    model_name="quant_v1",
    dataset_version="2.1.0",
    status="completed",
)

# Custom predicate
high_f1 = tracker.filter(predicate=lambda r: r.metrics.get("f1_macro", 0) > 0.75)

# Best record by any metric
best_acc = tracker.best_by("accuracy")
best_loss = tracker.best_by("val_loss", higher_is_better=False)
```

## HyperparameterRecord

```python
from trainer.research import HyperparameterRecord

hp = HyperparameterRecord(
    learning_rate=5e-4,
    hidden_size=512,
    num_layers=3,
    dropout=0.15,
    optimizer="adamw",
    scheduler="cosine_annealing",
    epochs=150,
    batch_size=512,
    loss_type="focal",
    focal_gamma=2.0,
)

# Attach to experiment
record.hyperparameters = hp.to_dict()

# Compare two runs
diff = hp.diff(other_hp)
# → {"learning_rate": (5e-4, 1e-3), "hidden_size": (512, 256)}
```
