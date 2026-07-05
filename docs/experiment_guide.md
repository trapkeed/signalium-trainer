# Experiment Guide

## Overview

Every hyperparameter evaluation in the AutoML platform is tracked as a **Trial** within a **SearchJob**. All data is persisted to SQLite and can be queried via the AutoML API.

---

## Experiment Lifecycle

```
SearchJob (created)
  └── Trial (pending)
       ├── Trial.start()          → status: running
       ├── [training + evaluation]
       ├── Trial.add_intermediate(step, score)  → pruning check
       ├── Prune?
       │    ├── YES → Trial.prune(step, reason)  → status: pruned
       │    └── NO  → Trial.complete(score, metrics) → status: completed
       └── ExperimentRecord saved to AutoMLDatabase
```

---

## Creating and Running a Search Job

```python
from trainer.automl import (
    AutoMLEngine, AutoMLConfig, SearchJob, Trial, TrialStatus
)

# Define job
config = AutoMLConfig(
    name="sharpe_sweep",
    strategy="bayesian",
    n_trials=30,
    objective="composite_research_score",
    dataset_version="v4.2",
    feature_version="v3.1",
    label_version="v2.0",
    git_commit="abc1234",
    hardware="A100-40GB",
)

engine = AutoMLEngine(config=config)
result = engine.run(train_fn=my_train_fn, model_id="quant_v1")
```

---

## Trial Data Model

Every Trial records:

| Field                    | Description                                     |
|--------------------------|-------------------------------------------------|
| `trial_id`               | Unique UUID                                    |
| `job_id`                 | Parent search job UUID                          |
| `experiment_id`          | Optional external experiment tracker ID         |
| `parent_trial_id`        | For population-based training lineage          |
| `config`                 | Hyperparameter configuration dict               |
| `backbone`               | Backbone name used                              |
| `random_seed`            | Seed for reproducibility                        |
| `dataset_version`        | Dataset tag                                     |
| `feature_version`        | Feature engineering tag                         |
| `label_version`          | Label version tag                               |
| `git_commit`             | Git commit hash at run time                     |
| `model_version`          | Generated model version string                  |
| `hardware`               | Hardware descriptor                             |
| `status`                 | pending → running → completed/pruned/failed     |
| `score`                  | Primary objective score                         |
| `metrics`                | Full metrics dict                               |
| `intermediate_scores`    | [(step, score), …] for pruning                 |
| `training_duration_seconds` | Elapsed training time                        |
| `validation_duration_seconds` | Elapsed validation time                    |
| `resource_usage`         | CPU/GPU/RAM snapshot                           |
| `promotion_eligible`     | Whether trial passed promotion criteria         |

---

## Querying Experiments

```python
from trainer.automl import AutoMLDatabase

db = AutoMLDatabase("experiments/automl/automl.db")

# All jobs
jobs = db.list_jobs()

# Completed trials for a job
records = db.get_records_for_job(job_id, status="completed")

# Best overall
best = db.best_record()
print(f"Best: {best['trial_id']} score={best['primary_score']:.4f}")

# Per-backbone performance
summary = db.backbone_summary()
for row in summary:
    print(f"{row['backbone']}: avg={row['avg_score']:.3f} max={row['max_score']:.3f}")

# Score history (convergence)
history = db.score_history(job_id)

# Statistics
stats = db.stats()
print(stats)

db.close()
```

---

## Experiment Recovery (Resume after Interruption)

The engine checkpoints the job state every `checkpoint_frequency` trials. If an experiment is interrupted:

1. The SQLite database retains all completed trial records.
2. Resume by re-running the engine with the same `db_path`.
3. Already-completed trials are visible to the strategy and pruner.
4. Job status is updated from `"running"` to `"completed"` on clean exit.

---

## Multi-Objective Tracking

Every trial records scores for all configured objectives:

```python
from trainer.automl import ExperimentRecord, ExperimentRanker

ranker = ExperimentRanker()
records = db.list_records(status="completed", limit=100)
ranking = ranker.full_ranking(records)

# Overall champion
print(ranking.champion.trial_id)

# Per-instrument champions
eurusd_rank1 = ranking.by_instrument["EURUSD"][0]

# Per-timeframe champions
h1_rank1 = ranking.by_timeframe["H1"][0]

# Per-regime champions
trending_rank1 = ranking.by_regime["trending_up"][0]
```

---

## Retry Policy

Failed trials are automatically retried up to `max_retries` times:

```yaml
automl:
  max_retries: 2
```

A trial in `RETRYING` status will be re-queued on the next engine iteration.

---

## Intermediate Score Reporting

For pruning to work, trials must report intermediate scores:

```python
# Inside train_fn — called by AutoMLEngine automatically
trial.add_intermediate(step=epoch, score=val_score)
```

The engine calls `pruner.should_prune()` after each intermediate score. If the pruner fires, the trial is stopped and marked `PRUNED`.
