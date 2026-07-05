# Institutional AutoML Platform Guide

## Overview

The Institutional AutoML and Research Optimization Platform (Phase 16) automatically discovers better-performing model configurations by integrating tightly with the existing trainer, model registry, and the Phase 15 Institutional Research Validation Platform.

**Key principles:**
- Do NOT modify production models directly
- All candidates are validated before promotion
- Full backward compatibility with all prior phases
- Complete configurability via YAML — no code changes required

---

## Architecture

```
AutoMLEngine
├── AutoMLConfig (YAML-driven)
├── SearchSpace (hyperparameter definitions)
├── BaseStrategy (grid | random | bayesian | manual)
├── BasePruner (none | median | successive_halving)
├── ResourceManager (CPU/GPU/RAM slot management)
├── BackboneCatalog (lstm | gru | transformer | cnn | hybrids)
├── ObjectiveScorer (composite + per-objective)
├── ExperimentRanker (overall + per-instrument/timeframe/regime)
├── AutoMLDatabase (SQLite — jobs, records, pruning, promotion)
├── AutoMLVisualizationBuilder (JSON chart data)
└── AutoMLAPI (structured endpoint layer)
```

---

## Quick Start

```python
from trainer.automl import AutoMLEngine, AutoMLConfig

# Option 1: programmatic config
config = AutoMLConfig(
    name="sharpe_sweep",
    strategy="bayesian",
    n_trials=50,
    objective="composite_research_score",
    seed=42,
)
engine = AutoMLEngine(config=config)

# Option 2: YAML config
engine = AutoMLEngine.from_yaml("conf/automl/automl.yaml")

# Run the search (mock evaluator by default)
result = engine.run(model_id="quant_v1")
print(result.summary())
print(f"Best trial: {result.champion_trial.trial_id}")
print(f"Best score: {result.job.best_score:.4f}")
```

---

## Running with a Custom Trainer

```python
def my_train_fn(config: dict, backbone_spec) -> dict:
    """
    Train a model with the given hyperparameter config.
    Returns a metrics dict compatible with Phase 15.
    """
    model = build_model(backbone_spec, config)
    metrics = train_and_evaluate(model, config)
    return metrics

result = engine.run(
    train_fn=my_train_fn,
    model_id="quant_v1",
)
```

---

## Search Strategies

| Strategy  | Use Case                                    |
|-----------|---------------------------------------------|
| `manual`  | Evaluate a fixed list of configurations     |
| `grid`    | Exhaustive grid search (small spaces)       |
| `random`  | Fast broad exploration                      |
| `bayesian`| Efficient optimization (recommended)        |

---

## Configuration Reference

All settings live in `conf/automl/automl.yaml`:

```yaml
automl:
  name: my_search
  strategy: bayesian
  n_trials: 50
  objective: composite_research_score
  seed: 42
  backbones: [lstm, gru, transformer_encoder]
  pruner: median
  pruner_config:
    n_startup_trials: 5
    percentile: 50.0
  promotion_thresholds:
    min_validation_score: 0.55
    min_sharpe_ratio: 0.5
    max_max_drawdown: 0.35
  auto_promote: false
```

---

## Results and Reports

```python
result = engine.run()

# Summary string
print(result.summary())

# Champion trial
champion = result.champion_trial
print(f"Backbone: {champion.backbone}")
print(f"Config: {champion.config}")
print(f"Metrics: {champion.metrics}")

# Ranking
for entry in result.ranking.overall[:5]:
    print(f"Rank {entry.rank}: {entry.trial_id[:8]} score={entry.score:.4f}")

# Dashboard charts (JSON-serializable)
import json
print(json.dumps(result.dashboard["search_convergence"], indent=2))

# Full report
print(json.dumps(result.report, indent=2))
```

---

## Integration with Phase 15

The AutoML engine can optionally pass every trial's output through the Phase 15 Institutional Validation Platform before scoring:

```python
from trainer.validation import InstitutionalValidationEngine

validation_engine = InstitutionalValidationEngine(
    db_path="experiments/validation.db",
    leaderboard_path="experiments/leaderboard.json",
)

result = engine.run(
    train_fn=my_train_fn,
    validation_engine=validation_engine,
    model_id="quant_v1",
)
```

---

## Database Access

```python
from trainer.automl import AutoMLDatabase

db = AutoMLDatabase("experiments/automl/automl.db")
jobs = db.list_jobs()
records = db.list_records(status="completed", limit=50)
best = db.best_record()
stats = db.stats()
db.close()
```

---

## API Layer

```python
from trainer.automl import AutoMLAPI, AutoMLDatabase

db = AutoMLDatabase("experiments/automl/automl.db")
api = AutoMLAPI(db)

# List all jobs
resp = api.list_jobs()

# Get leaderboard
resp = api.get_leaderboard(limit=20)

# Champion/challenger comparison
resp = api.champion_challenger_compare(champion_id, challenger_id)

# Approve promotion
resp = api.approve_promotion(trial_id, job_id, model_version="3.0.0")
```
