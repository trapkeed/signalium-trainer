# Model Registry Guide — Phase 18

## Overview

The Model Registry is the single source of truth for every model in Signalium's production stack. No model may enter production without a complete registry record.

---

## Registry Fields

Every `ModelRecord` stores:

| Field                        | Type              | Description                                              |
|------------------------------|-------------------|----------------------------------------------------------|
| `model_id`                   | UUID str          | Unique identifier                                        |
| `name`                       | str               | Human-readable model name                               |
| `version`                    | str               | Semantic version (e.g. "1.0.0")                         |
| `training_date`              | ISO-8601 str      | When training completed                                  |
| `architecture`               | str               | Neural architecture (lstm, transformer, etc.)           |
| `backbone`                   | str               | Specific backbone variant                               |
| `parameter_count`            | int               | Total trainable parameters                              |
| `dataset_version`            | str               | Training dataset version tag                            |
| `feature_version`            | str               | Feature engineering pipeline version                   |
| `label_version`              | str               | Label generation pipeline version                       |
| `git_commit`                 | str               | Training codebase commit hash                           |
| `training_config`            | dict              | Full training hyperparameter configuration              |
| `validation_report`          | dict              | Phase 15 validation results                             |
| `promotion_status`           | str               | pending / approved / rejected                           |
| `deployment_status`          | str               | not_deployed / deployed / rolled_back                   |
| `retirement_status`          | str               | active / retired                                        |
| `approval_history`           | List[ApprovalRecord] | Full approval/rejection history with actor+reason    |
| `hardware_used`              | str               | Training hardware description                           |
| `training_duration_seconds`  | float             | Wall-clock training time                               |
| `parent_model_id`            | str?              | If fine-tuned, the parent model's ID                   |
| `child_model_ids`            | List[str]         | IDs of models derived from this one                    |
| `lifecycle_state`            | str               | Current LifecycleState value                            |
| `notes`                      | str               | Free-form operational notes                             |
| `metadata`                   | dict              | Arbitrary additional metadata                           |

---

## Registering a Model

```python
from trainer.mlops import MLOpsEngine, ModelRecord

engine = MLOpsEngine.from_config({...})

model_id = engine.register_model({
    "name": "signalium_quant",
    "version": "2.1.0",
    "architecture": "transformer_encoder",
    "backbone": "transformer_4h_256d_4l",
    "parameter_count": 8_241_920,
    "dataset_version": "v4.0",
    "feature_version": "v3.0",
    "label_version": "v2.0",
    "git_commit": "f8a3b2c1d0e9f7a6",
    "hardware_used": "4× NVIDIA A100 40GB",
    "training_duration_seconds": 14400.0,
    "training_config": {
        "optimizer": "AdamW",
        "lr": 1e-4,
        "batch_size": 512,
        "epochs": 100,
        "early_stopping": True,
    },
    "validation_report": {
        "accuracy": 0.72,
        "sharpe_ratio": 1.85,
        "calibration_error": 0.032,
        "max_drawdown": -0.08,
    },
    "notes": "Successor to LSTM v1. Transformer backbone with 4 attention heads.",
    "parent_model_id": "model_01_legacy_id",
})
```

---

## Querying the Registry

```python
# Get a specific model
model = engine.registry.get(model_id)
print(model.full_name)   # "signalium_quant@2.1.0"
print(model.is_champion)
print(model.approval_history)

# List all models in a state
candidates = engine.registry.by_state("candidate")
active_models = engine.registry.active()
champion = engine.registry.champion()

# Latest version of a model by name
latest = engine.registry.latest_version("signalium_quant")

# Registry summary
summary = engine.registry.summary()
print(summary["total_models"])
print(summary["by_state"])
```

---

## API Registry Endpoints

```python
# Register
resp = engine.api.register_model(model_dict)
model_id = resp["data"]["model_id"]

# Get
resp = engine.api.get_model(model_id)
print(resp["data"]["lifecycle_state"])

# List (with optional state filter)
resp = engine.api.list_models(lifecycle_state="candidate")
print(resp["data"]["count"])

# Update notes
resp = engine.api.update_model_notes(model_id,
    notes="Production performance exceeds baseline by 8%.",
    actor="head_quant")
```

---

## Database Persistence

All registry records are persisted to SQLite:

```python
# Direct DB access (advanced)
rows = engine.db.list_models(lifecycle_state="production")
row = engine.db.get_model(model_id)
engine.db.update_model_state(model_id, "retired")
```

## Export Registry

```python
report = engine.export_registry(
    path="exports/registry_2026-07-05.json"
)
```

The export includes full model records, approval history, and registry summary.

---

## Champion / Challenger Registry

See the [Governance Guide](governance_guide.md) and [Deployment Guide](deployment_guide.md) for champion promotion and challenger comparison workflows.
