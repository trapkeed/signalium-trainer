# Research Lab — Experiment Flow

## Overview

An experiment is the fundamental unit of Research Lab work. Every training run, every new QDE version candidate, and every hyperparameter configuration is tracked as an experiment with full metadata, timing, metrics, and deployment status.

---

## Tracked Fields

| Field | Description |
|---|---|
| `experiment_id` | Unique identifier (`exp-{uuid}`) |
| `name` | Human-readable experiment name |
| `qde_version` | QDE version label (e.g. `QDE v13`) |
| `training_dataset` | Dataset ID used for training |
| `status` | `PENDING \| RUNNING \| COMPLETED \| FAILED \| ARCHIVED` |
| `training_started_at` | UTC timestamp when training began |
| `training_completed_at` | UTC timestamp when training finished |
| `training_duration_s` | Wall-clock training duration in seconds |
| `sharpe_ratio` | Risk-adjusted return (Sharpe) |
| `sortino_ratio` | Risk-adjusted return (Sortino) |
| `max_drawdown` | Maximum drawdown recorded |
| `avg_drawdown` | Average drawdown across periods |
| `recovery_factor` | Net profit / max drawdown |
| `expected_value` | Expected value per decision |
| `expected_value_per_trade` | EV per individual trade |
| `profit_factor` | Gross profit / gross loss |
| `win_rate` | Fraction of winning trades |
| `decision_quality` | Entry/exit prediction precision |
| `capital_efficiency` | Return per unit of margin deployed |
| `deployment_status` | `NOT_DEPLOYED \| SHADOW \| CANARY \| ACTIVE \| RETIRED \| ROLLED_BACK` |
| `researcher` | Who ran this experiment |
| `hyperparameters` | Full hyperparameter dict |
| `tags` | Free-form classification tags |
| `notes` | Researcher notes |

---

## Experiment Lifecycle

```
PENDING
   │
   │  POST /admin/research/experiments/:id/start
   ▼
RUNNING
   │
   ├── [success] POST /admin/research/experiments/:id/complete
   │               { "metrics": { ... } }
   │               ▼
   │            COMPLETED
   │
   └── [failure] POST /admin/research/experiments/:id/fail
                  { "error": "..." }
                  ▼
               FAILED
```

---

## Creating an Experiment

```
POST /api/admin/research/experiments
Authorization: Bearer <token>

{
  "name": "QDE v13 — enhanced regime detection",
  "qde_version": "QDE v13",
  "training_dataset": "ds-a4f2c1d0",
  "researcher": "dr-chen",
  "tags": ["regime", "transformer", "2026"],
  "hyperparameters": {
    "learning_rate": 0.0003,
    "dropout": 0.15,
    "sequence_length": 60,
    "n_heads": 8
  },
  "notes": "Testing multi-head attention on regime features"
}
```

---

## Completing with Metrics

```
POST /api/admin/research/experiments/:experimentId/complete
Authorization: Bearer <token>

{
  "metrics": {
    "sharpe_ratio": 1.74,
    "sortino_ratio": 2.10,
    "max_drawdown": 0.087,
    "profit_factor": 2.1,
    "win_rate": 0.593,
    "decision_quality": 0.681,
    "capital_efficiency": 0.214,
    "expected_value": 0.0031,
    "expected_value_per_trade": 0.0024,
    "recovery_factor": 8.2
  }
}
```

---

## Model Comparison

Compare up to N QDE versions on the 8 standard metrics:

| Metric | Comparison Direction |
|---|---|
| Sharpe Ratio | Higher is better |
| Profit Factor | Higher is better |
| Max Drawdown | Lower is better |
| Recovery Factor | Higher is better |
| Win Rate | Higher is better |
| Decision Quality | Higher is better |
| Expected Value | Higher is better |
| Capital Efficiency | Higher is better |

```
POST /api/admin/research/comparisons
Authorization: Bearer <token>

{
  "version_ids": ["QDE v11", "QDE v12", "QDE v13"],
  "metrics_per_version": {
    "QDE v11": { "sharpe_ratio": 1.2, "max_drawdown": 0.14, ... },
    "QDE v12": { "sharpe_ratio": 1.5, "max_drawdown": 0.11, ... },
    "QDE v13": { "sharpe_ratio": 1.74, "max_drawdown": 0.087, ... }
  },
  "notes": "Q3 2026 candidate comparison"
}
```

The response includes per-metric rankings and an `overall_winner` determined by most first-place finishes.

---

## Parameter Search

```
POST /api/admin/research/searches
Authorization: Bearer <token>

{
  "name": "learning-rate sweep v13",
  "strategy": "BAYESIAN",
  "search_space": {
    "learning_rate": [0.0001, 0.001],
    "dropout": [0.1, 0.3]
  },
  "objective": "sharpe_ratio",
  "maximize": true,
  "n_trials": 50,
  "qde_version": "QDE v13"
}
```

Record each trial result as training completes:
```
POST /api/admin/research/searches/:searchId/trials
{ "params": { "learning_rate": 0.0003, "dropout": 0.15 }, "score": 1.74 }
```

---

## Listing and Filtering

```
GET /api/admin/research/experiments?qde_version=QDE+v13&status=COMPLETED
GET /api/admin/research/experiments/summary
GET /api/admin/research/experiments/:experimentId
```
