# AutoML Platform API Reference

## Overview

The `AutoMLAPI` class provides a structured endpoint layer for the AutoML platform. All methods return a standardized response dict:

```json
{
  "status": "ok" | "error",
  "message": "...",
  "data": { ... },
  "timestamp": "2026-07-05T12:00:00Z"
}
```

---

## Initialization

```python
from trainer.automl import AutoMLAPI, AutoMLDatabase, ExperimentRanker

db = AutoMLDatabase("experiments/automl/automl.db")
api = AutoMLAPI(db)

# With custom scorer and ranker
from trainer.automl import build_research_scorer
api = AutoMLAPI(
    database=db,
    ranker=ExperimentRanker(),
    scorer=build_research_scorer(),
)
```

---

## Search Job Endpoints

### `create_job(job_config)`

Create and persist a new search job.

**Parameters:**
```python
job_config = {
    "name": "sharpe_sweep",
    "strategy": "bayesian",       # random | grid | bayesian | manual
    "n_trials": 30,
    "objective": "composite_research_score",
    "objectives": ["sharpe_ratio", "max_drawdown"],
    "search_space_name": "signalium_standard",
    "budget_type": "trials",      # trials | time | metric
    "budget_value": 30,
    "pruner_type": "median",
    "parallel_workers": 1,
    "max_retries": 2,
    "metadata": {},
}
resp = api.create_job(job_config)
job_id = resp["data"]["job_id"]
```

---

### `get_job(job_id)`

Get full job status and metadata.

```python
resp = api.get_job("job-uuid")
print(resp["data"]["status"])        # running | completed | failed
print(resp["data"]["n_completed"])   # completed trial count
print(resp["data"]["best_score"])    # best primary score so far
```

---

### `list_jobs(status=None, limit=50)`

List search jobs with optional status filter.

```python
resp = api.list_jobs(status="completed", limit=10)
jobs = resp["data"]["jobs"]
```

---

### `pause_job(job_id)` / `resume_job(job_id)`

Pause or resume a running search job.

```python
api.pause_job(job_id)
api.resume_job(job_id)
```

---

## Trial Endpoints

### `get_trial(record_id)`

Retrieve a single experiment record.

```python
resp = api.get_trial("record-uuid")
record = resp["data"]
print(record["backbone"], record["primary_score"])
```

---

### `list_trials(job_id=None, status=None, backbone=None, limit=100, offset=0)`

List experiment records with optional filters.

```python
resp = api.list_trials(
    job_id="job-uuid",
    status="completed",
    backbone="lstm",
    limit=20,
)
trials = resp["data"]["trials"]
```

---

## Optimization History

### `get_optimization_history(job_id)`

Return score history and convergence chart for a job.

```python
resp = api.get_optimization_history(job_id)
history = resp["data"]["history"]          # list of {created_at, score}
chart = resp["data"]["chart"]              # line chart data dict
```

---

## Leaderboard

### `get_leaderboard(job_id=None, limit=20, sort_by="primary_score")`

Return ranked experiment leaderboard.

```python
resp = api.get_leaderboard(job_id=job_id, limit=10)
lb = resp["data"]["leaderboard"]
for entry in lb:
    print(f"Rank {entry['rank']}: {entry['trial_id'][:8]} "
          f"score={entry['score']:.4f} backbone={entry['backbone']}")
```

---

## Pareto Front

### `get_pareto_front(job_id=None)`

Return Pareto-optimal trials across all configured objectives.

```python
resp = api.get_pareto_front(job_id=job_id)
pareto = resp["data"]["pareto"]
n_pareto = resp["data"]["n_pareto"]
```

---

## Promotion Endpoints

### `get_promotion_candidates(job_id=None, limit=20)`

Return trials that passed all promotion eligibility criteria.

```python
resp = api.get_promotion_candidates(job_id=job_id)
candidates = resp["data"]["candidates"]
history = resp["data"]["promotion_history"]
```

---

### `approve_promotion(trial_id, job_id, model_version="", notes="")`

Manually approve a trial for champion promotion.

```python
resp = api.approve_promotion(
    trial_id="trial-uuid",
    job_id="job-uuid",
    model_version="3.0.0",
    notes="Approved after review on 2026-07-05",
)
assert resp["data"]["champion_status"] is True
```

---

## Dashboard

### `get_dashboard(job_id=None)`

Return all visualization charts in one call.

```python
resp = api.get_dashboard(job_id=job_id)
charts = resp["data"]["charts"]
backbone_summary = resp["data"]["backbone_summary"]
stats = resp["data"]["stats"]

# Access individual charts
opt_hist = charts["optimization_history"]
convergence = charts["search_convergence"]
param_importance = charts["parameter_importance"]
timeline = charts["experiment_timeline"]
pareto_chart = charts["objective_tradeoff"]
distribution = charts["score_distribution"]
```

---

## Champion/Challenger

### `champion_challenger_compare(champion_trial_id, challenger_trial_id)`

Side-by-side comparison of two trials.

```python
resp = api.champion_challenger_compare(
    champion_trial_id="champ-uuid",
    challenger_trial_id="challenger-uuid",
)
data = resp["data"]
print(f"Winner: {data['winner']}")           # "champion" | "challenger"
print(f"Delta: {data['score_delta']:+.4f}")  # challenger score - champion score
print(f"Improves: {data['challenger_improves']}")
```

---

## Statistics

### `get_stats()`

Return overall platform statistics.

```python
resp = api.get_stats()
stats = resp["data"]
print(f"Total jobs: {stats['total_jobs']}")
print(f"Completed records: {stats['completed_records']}")
print(f"Pruned records: {stats['pruned_records']}")
print(f"Eligible promotions: {stats['eligible_promotions']}")
```

---

## Error Responses

All errors return `status: "error"` with a `code` and `message`:

```json
{
  "status": "error",
  "code": "NOT_FOUND",
  "message": "Job 'nonexistent-id' not found",
  "data": null,
  "timestamp": "2026-07-05T12:00:00Z"
}
```

Common error codes:
- `NOT_FOUND` — resource does not exist
- `CREATE_JOB_ERROR` — job creation failed
- `GET_JOB_ERROR` — job retrieval failed
- `LIST_TRIALS_ERROR` — trial listing failed
- `LEADERBOARD_ERROR` — leaderboard computation failed
- `PARETO_ERROR` — Pareto front computation failed
- `DASHBOARD_ERROR` — dashboard generation failed
- `CHAMPION_CHALLENGER_ERROR` — comparison failed
- `APPROVE_PROMOTION_ERROR` — promotion approval failed
