# Champion/Challenger Guide

## Overview

The Champion/Challenger workflow allows new candidate models to be compared against the current production champion before promotion. Integration is provided both at the AutoML engine level and through the Phase 15 Institutional Research Validation Platform.

---

## Concepts

| Term           | Definition                                                           |
|----------------|----------------------------------------------------------------------|
| **Champion**   | The current best model in production (highest validation score)      |
| **Challenger** | A new candidate model being evaluated against the champion           |
| **Promotion**  | The act of replacing the champion with a better challenger           |
| **Rollback**   | Reverting to the previous champion if the new one underperforms      |

---

## Promotion Workflow

```
AutoML Search
    ↓
[Best Trial Selected as Challenger]
    ↓
Phase 15 Validation (if require_validation: true)
    ↓
Promotion Eligibility Check
    ├── Pass all thresholds?
    │    ├── NO  → Not eligible, stays as challenger
    │    └── YES → Promotion eligible
    │         ├── auto_promote: false → Awaits manual approval
    │         └── auto_promote: true  → Auto-promoted to champion
```

---

## Configuring Promotion Thresholds

```yaml
automl:
  promotion_thresholds:
    min_validation_score: 0.55   # accuracy/val score threshold
    min_sharpe_ratio: 0.5        # minimum acceptable Sharpe
    max_max_drawdown: 0.35       # maximum acceptable drawdown
    min_profit_factor: 1.0       # minimum profit factor
    min_win_rate: 0.45           # minimum win rate
  auto_promote: false            # require manual sign-off
  require_validation: true       # enforce Phase 15 validation
```

Threshold keys follow the pattern:
- `min_<metric>` — metric must be ≥ threshold (maximize metrics)
- `max_<metric>` — metric must be ≤ threshold (minimize metrics)

---

## Running a Champion/Challenger Comparison

### Via the API

```python
from trainer.automl import AutoMLAPI, AutoMLDatabase

db = AutoMLDatabase("experiments/automl/automl.db")
api = AutoMLAPI(db)

resp = api.champion_challenger_compare(
    champion_trial_id="trial-uuid-champion",
    challenger_trial_id="trial-uuid-challenger",
)

data = resp["data"]
print(f"Winner: {data['winner']}")
print(f"Score delta: {data['score_delta']:+.4f}")
print(f"Challenger improves: {data['challenger_improves']}")
```

### Via the Engine

```python
result = engine.run(train_fn=my_train_fn)

# Compare champion (best trial) vs any specific trial
resp = engine.api.champion_challenger_compare(
    champion_trial_id=result.job.champion_trial_id,
    challenger_trial_id="some-other-trial-id",
)
```

---

## Manual Promotion

When `auto_promote: false`, the promotion workflow requires human approval:

```python
# 1. Get promotion candidates
resp = api.get_promotion_candidates(job_id=job_id)
candidates = resp["data"]["candidates"]

# 2. Review the best candidate
best = candidates[0]
print(f"Trial: {best['trial_id']}")
print(f"Score: {best['primary_score']:.4f}")
print(f"Backbone: {best['backbone']}")

# 3. Approve
resp = api.approve_promotion(
    trial_id=best["trial_id"],
    job_id=best["job_id"],
    model_version="3.0.0",
    notes="Approved after manual review on 2026-07-05",
)
```

---

## Promotion History

```python
history = db.get_promotion_history(limit=20)
for record in history:
    print(f"{record['decided_at']} | "
          f"trial={record['trial_id'][:8]} | "
          f"eligible={record['is_eligible']} | "
          f"champion={record['champion_status']} | "
          f"auto={record['auto_approved']}")
```

---

## Rollback

Rollback eligibility is recorded in the promotion decision. To execute a rollback:

1. Identify the previous champion from `get_promotion_history()`.
2. Set the previous trial as the new champion via `approve_promotion()`.
3. Record the rollback reason in `notes`.

```python
api.approve_promotion(
    trial_id=previous_champion_id,
    job_id=job_id,
    model_version="2.1.0",
    notes="Rollback: v3.0.0 underperformed in live trading. Reverting to v2.1.0.",
)
```

---

## Integration with Phase 15 Leaderboard

When the AutoML engine runs with `require_validation: true`, each promoted trial is automatically submitted to the Phase 15 Institutional Model Leaderboard:

```python
from trainer.validation import InstitutionalValidationEngine, PromotionPolicy

phase15 = InstitutionalValidationEngine(
    db_path="experiments/validation.db",
    leaderboard_path="experiments/leaderboard.json",
    policy=PromotionPolicy(
        min_accuracy=0.55,
        min_sharpe=0.5,
        max_drawdown=0.35,
    ),
)

result = engine.run(
    train_fn=my_train_fn,
    validation_engine=phase15,
)
```

The Phase 15 platform runs the full 17-step validation pipeline for each promoted candidate, including walk-forward analysis, Monte Carlo simulation, and failure analysis.
