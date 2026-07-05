# Institutional Model Leaderboard Guide

## Phase 15 — Signalium Quant

---

## Overview

The Institutional Leaderboard is a permanent, sortable, filterable record of every model that has been evaluated through the validation platform. It serves as the single source of truth for model performance history at Signalium.

---

## Leaderboard Fields

| Field | Type | Description |
|-------|------|-------------|
| model_id | str | Unique model identifier |
| model_version | str | Semantic version (e.g. "2.0.0") |
| model_name | str | Human-readable name |
| training_date | str | ISO-8601 training date |
| architecture | str | Model architecture label |
| backbone | str | Backbone identifier |
| phase | str | Signalium development phase |
| n_parameters | int | Total parameter count |
| training_time_seconds | float | Wall-clock training duration |
| dataset_version | str | Dataset version used |
| feature_version | str | Feature set version |
| label_version | str | Label set version |
| validation_accuracy | float | Walk-forward mean accuracy |
| validation_f1 | float | Walk-forward mean F1 |
| ece | float | Expected Calibration Error |
| brier_score | float | Multi-class Brier score |
| roc_auc | float | ROC AUC |
| profit_factor | float | Gross profit / gross loss |
| sharpe_ratio | float | Annualised Sharpe |
| sortino_ratio | float | Annualised Sortino |
| calmar_ratio | float | CAGR / max drawdown |
| max_drawdown | float | Maximum drawdown fraction |
| win_rate | float | Trade win rate |
| expectancy | float | Expected P&L per trade |
| inference_latency_ms | float | Median inference latency |
| model_size_mb | float | Parameter size in MB |
| promotion_status | str | experimental/candidate/challenger/champion/retired |
| is_champion | bool | Current champion flag |
| deployment_history | list | Deployment records |
| is_retired | bool | Retirement flag |
| overall_score | float | Weighted composite score [0, 1] |
| experiment_id | str | Linked experiment UUID |
| git_commit | str | Git commit hash |
| n_validation_folds | int | Number of walk-forward folds |
| notes | str | Free-text notes |
| added_at | str | ISO-8601 timestamp when added |

---

## Usage

### Creating a Leaderboard

```python
from trainer.validation import InstitutionalLeaderboard, LeaderboardEntry

lb = InstitutionalLeaderboard(
    store_path=Path("experiments/leaderboard.json"),
    score_weights={
        "validation_accuracy": 0.15,
        "validation_f1": 0.15,
        "sharpe_ratio": 0.25,
        "profit_factor": 0.15,
        "max_drawdown": -0.10,   # negative = lower is better
        "ece": -0.10,
        "win_rate": 0.10,
    },
)
```

### Adding an Entry

Entries are created automatically by the `InstitutionalValidationEngine`. To add manually:

```python
entry = LeaderboardEntry(
    model_id="quant_v1",
    model_version="2.0.0",
    model_name="Quant V1 Hybrid",
    architecture="hybrid_transformer_lstm",
    backbone="HybridEncoder",
    validation_accuracy=0.612,
    sharpe_ratio=1.45,
    profit_factor=1.72,
    max_drawdown=0.12,
    win_rate=0.58,
    ece=0.04,
    promotion_status="candidate",
)
lb.add(entry)
```

### Querying the Leaderboard

```python
# Top 10 by overall score
top10 = lb.top(10)

# Top 5 by Sharpe ratio
best_sharpe = lb.top(5, sort_by="sharpe_ratio")

# Top by accuracy (ascending for min drawdown)
safest = lb.top(10, sort_by="max_drawdown", ascending=True)

# Filter by status
champions = lb.filter(promotion_status="champion")
candidates = lb.filter(promotion_status="candidate")
active = lb.filter(is_retired=False)

# Filter by performance thresholds
strong = lb.filter(min_accuracy=0.60, max_drawdown_limit=0.15)

# Filter by architecture
transformers = lb.filter(architecture="hybrid_transformer_lstm")
```

### Searching

```python
# Full-text search across name, notes, architecture
results = lb.search("bitcoin")
results = lb.search("phase 14")
results = lb.search("transformer")
```

### Comparing Entries

```python
comparison = lb.compare("quant_v1_champion", "quant_v1_challenger")
# Returns side-by-side dict with all key metrics
```

### Printing the Leaderboard

```python
print(lb.format_table(k=20, sort_by="overall_score"))
```

Output:
```
---------------------------------------------------------------------------
#   Model                          Version   Acc      F1    Sharpe     PF     DD   Score   Status
---------------------------------------------------------------------------
*1  quant_v1_2_0                   2.0.0    0.6120  0.5980   1.450  1.720  0.120  0.7423  champion
 2  quant_v1_1_5                   1.5.0    0.5980  0.5810   1.210  1.540  0.150  0.6882  retired
 3  quant_v1_candidate             2.1.0    0.6250  0.6120   1.520  1.810  0.110  0.7890  candidate
---------------------------------------------------------------------------
(* = Champion)  3 total entries  sorted by overall_score
```

---

## Scoring

The overall score is a weighted composite:

```
score = Σ(weight_i × metric_i)
```

Default weights:

| Metric | Weight | Direction |
|--------|--------|-----------|
| validation_accuracy | 0.15 | Higher is better |
| validation_f1 | 0.15 | Higher is better |
| sharpe_ratio | 0.20 | Higher is better |
| profit_factor | 0.15 | Higher is better |
| max_drawdown | -0.10 | Lower is better |
| ece | -0.10 | Lower is better |
| win_rate | 0.15 | Higher is better |

Score is clamped to [0, 1].

---

## Lifecycle States

```
EXPERIMENTAL → CANDIDATE → CHALLENGER → CHAMPION → RETIRED
```

| Status | Meaning |
|--------|---------|
| experimental | Just registered, not yet evaluated |
| candidate | Passed initial evaluation thresholds |
| challenger | Competing against the current champion |
| champion | Current production model |
| retired | No longer active |

---

## Persistence

The leaderboard auto-saves to JSON after every modification:

```python
# Manual save
lb.save(Path("leaderboard_backup.json"))

# Load from file
lb2 = InstitutionalLeaderboard.load(Path("leaderboard.json"))
```

JSON format:
```json
{
  "entries": [
    {
      "model_id": "quant_v1",
      "sharpe_ratio": 1.45,
      "promotion_status": "champion",
      "is_champion": true,
      ...
    }
  ]
}
```
