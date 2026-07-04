# Champion / Challenger Guide

## Purpose

The Champion/Challenger framework tracks the lifecycle of every model version
from initial development through to production deployment.

**No automatic promotion occurs.** All transitions are manual and auditable.

## Model Lifecycle States

```
EXPERIMENTAL → CANDIDATE → CHALLENGER → CHAMPION
                                           │
                                        RETIRED
```

| Status | Description |
|---|---|
| `EXPERIMENTAL` | New model under active development |
| `CANDIDATE` | Passed initial quality gates; eligible for benchmarking |
| `CHALLENGER` | Actively competing with the champion; A/B testing eligible |
| `CHAMPION` | Current production model |
| `RETIRED` | Superseded or deprecated — terminal state |

## Basic Usage

```python
from trainer.research import ChampionRegistry, ModelStatus
from pathlib import Path

registry = ChampionRegistry(store_path=Path("experiments/champions.json"))

# 1. Register a new model after training
registry.register(
    model_name="quant_v1",
    model_version="2.0.0",
    experiment_id="abc123",
    metrics={"f1_macro": 0.734, "accuracy": 0.758},
    notes="Double hidden size, focal loss.",
)

# 2. Promote after passing quality gates
registry.promote("quant_v1", "2.0.0", ModelStatus.CANDIDATE)
registry.promote("quant_v1", "2.0.0", ModelStatus.CHALLENGER,
                 notes="A/B test started 2024-07-01.")

# 3. Crown the champion (auto-retires the previous champion)
registry.promote("quant_v1", "2.0.0", ModelStatus.CHAMPION,
                 notes="Beat v1.0.0 by +4pp F1 on Q2 live data.")

# 4. Inspect
champion = registry.get_champion("quant_v1")
challengers = registry.get_challengers("quant_v1")
all_entries = registry.list_all("quant_v1")
summary = registry.summary()
```

## Rules

1. **Only one CHAMPION per model name** — promoting a new champion automatically
   retires the old one.
2. **RETIRED is terminal** — call `retire()` explicitly; it cannot be reversed
   through `promote()`.
3. **EXPERIMENTAL models are excluded from the leaderboard** by default.
4. **State is persisted** — `ChampionRegistry` writes to JSON on every change.

## Retiring a Model

```python
registry.retire(
    "quant_v1", "1.0.0",
    notes="Superseded by v2.0.0 with superior calibration."
)
```

## Integration with the Benchmark Pipeline

```python
runner = BenchmarkRunner(conditions=conditions, experiment_tracker=tracker)
result = runner.run(train_data, val_data)

# After a successful benchmark, register and promote
best_run = result.completed_runs()[0]
registry.register(
    model_name=best_run.model_name,
    model_version="2.1.0",
    experiment_id=best_run.experiment_id,
    metrics=best_run.metrics,
)
registry.promote(best_run.model_name, "2.1.0", ModelStatus.CANDIDATE,
                 notes="Benchmark passed quality gate.")
```
