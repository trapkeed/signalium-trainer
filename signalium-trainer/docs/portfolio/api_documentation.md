# API Documentation — Phase 20 Portfolio Intelligence

## PortfolioApi

The unified entry point for the entire Phase 20 engine.

```python
from trainer.portfolio import PortfolioApi

api = PortfolioApi(
    db_path="data/portfolio.db",   # SQLite path (":memory:" for in-process)
    risk_free_rate=0.045,
    correlation_window=60,
)
```

---

## Endpoints

All endpoints return `{"status": "ok", "data": ...}` on success,
or `{"status": "error", "message": "..."}` on failure.

---

### `recommend(forecasts, method, run_conflict_check, run_budget_check)`

Full portfolio recommendation pipeline.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `forecasts` | `List[ModelForecast]` | required | Model outputs |
| `method` | `str` | `"hybrid"` | Allocation method |
| `run_conflict_check` | `bool` | `True` | Whether to resolve conflicts first |
| `run_budget_check` | `bool` | `True` | Whether to enforce budget |

**Response data keys:** `rec_id`, `regime`, `allocation`, `portfolio_metrics`, `exposure`, `conflict_report`, `dependency_graph`, `weights`

---

### `exposure(forecasts, weights)`

Calculates and audits portfolio exposure.

**Response:** `{exp_id, exposure}` where `exposure` is `ExposureReport.to_dict()`.

---

### `correlation(observations, method, instruments)`

Adds return observations and computes the correlation matrix.

**Parameters:**

| Name | Description |
|------|-------------|
| `observations` | `Dict[str, float]` — instrument → return |
| `method` | `"rolling"` \| `"rank"` \| `"exponential"` \| `"tail_lower"` \| `"tail_upper"` |

---

### `optimize(forecasts, objective, compare_all)`

Portfolio weight optimization.

**Parameters:**

| Name | Values |
|------|--------|
| `objective` | `"max_sharpe"` \| `"min_volatility"` \| `"max_diversification"` \| `"max_risk_adjusted_return"` \| `"balanced"` |
| `compare_all` | Run all objectives and return comparison |

**Response:** `{opt_id, best, all_objectives, efficient_frontier}`

---

### `scenario(forecasts, weights, scenarios)`

Macro scenario stress testing.

**Parameters:**

| Name | Description |
|------|-------------|
| `scenarios` | List of scenario names (omit for all defaults) |

**Response:** `ScenarioReport.to_dict()` — includes `worst_scenario`, `best_scenario`, `average_resilience`.

---

### `conflict_check(forecasts)`

Detect conflicting signals without full recommendation.

**Response:** `ConflictReport.to_dict()`

---

### `regime(observations)`

Detect current market regime.

**Response:** `RegimeAssessment.to_dict()`

---

### `risk_budget_check(forecasts, weights)`

Check current weights against risk budgets.

**Response:** `BudgetCheckResult.to_dict()`

---

### `allocation_history(limit=20)`

Recent allocation decisions from the database.

---

### `recommendations_history(limit=20)`

Recent full portfolio recommendations from the database.

---

### `scenario_history(scenario_type, limit=20)`

Recent scenario results, optionally filtered by scenario type.

---

### `dashboard_data(forecasts, weights, correlation_matrix)`

All 8 visualisation data structures in one call.

**Response keys:** `correlation_heatmap`, `exposure_chart`, `risk_allocation`,
`dependency_graph`, `portfolio_timeline`, `optimization_dashboard`,
`scenario_dashboard`, `portfolio_scorecard`

---

### `stats()`

System statistics: database record counts, registry info, correlation engine state.

---

### `registry_info()`

Full instrument registry — all 16 default instruments and their metadata.

---

## Response Codes

| `status` | Meaning |
|----------|---------|
| `"ok"` | Success — `data` contains the result |
| `"error"` | Failure — `message` contains the reason |
