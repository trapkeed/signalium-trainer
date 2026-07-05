# HRSA What-If Simulation Guide

## Overview

The `HRSAWhatIfSimulator` answers counterfactual questions about score
impact without requiring a full retraining cycle.

## Core Questions

- What if calibration improved from 62 to 85?
- What if drawdown reduced from 0.12 to 0.05?
- What if inference latency doubled?
- Which 3 improvements would have the greatest composite impact?

## Category-Level Simulation

The fastest simulation path: change one or more **Level 2 category scores**
and estimate the resulting **Level 3 composite change**.

Uses linear approximation:
```
Δ composite ≈ Σ (Δ category_score_i × effective_weight_i) / total_weight
```

```python
from trainer.research import HRSAWhatIfSimulator, HierarchicalResearchScorer
from trainer.research import RawMetricsBundle

scorer = HierarchicalResearchScorer()
bundle = RawMetricsBundle("my_model")
# ... set metrics ...
hs = scorer.score(bundle)

sim = HRSAWhatIfSimulator()

# Single category
result = sim.simulate_category(hs, "calibration", new_score=85.0)
print(f"Composite would increase by: {result.delta:+.2f} pts")
print(f"New composite: {result.simulated_composite:.1f}")
print(f"Grade change: {result.grade_change}")

# Multiple categories simultaneously
result = sim.simulate_categories(hs, {
    "calibration":   85.0,
    "robustness":    82.0,
    "risk_management": 80.0,
})
```

## Metric-Level Simulation (exact)

For exact results at the raw metric level, pass the full scorer:

```python
sim = HRSAWhatIfSimulator(scorer=scorer)

# What if max_drawdown dropped to 0.04?
result = sim.simulate_metric(hs, bundle, "max_drawdown", new_value=0.04)
print(f"Delta: {result.delta:+.2f}")

# Multiple metrics
result = sim.simulate_multiple_metric_changes(hs, bundle, {
    "max_drawdown": 0.04,
    "expected_calibration_error": 0.015,
    "monte_carlo_stability": 0.92,
})
```

## Optimal Improvement Budget

Find the highest-impact category improvements within a budget:

```python
results = sim.optimal_improvements(hs, budget_categories=3)
for r in results:
    print(f"  {r.scenario_name}: {r.delta:+.2f} pts → {r.simulated_composite:.1f}")
```

The combined scenario is automatically appended as the last result.

## Batch Simulation

Run multiple named scenarios at once:

```python
scenarios = [
    {"name": "Conservative improvement", "changes": {"calibration": 75.0}},
    {"name": "Aggressive improvement",   "changes": {
        "calibration": 90.0, "robustness": 88.0, "risk_management": 85.0
    }},
    {"name": "Latency crisis (negative)", "changes": {"efficiency": 20.0}},
]
results = sim.run_batch(hs, scenarios)
for r in results:
    sign = "+" if r.improved else ""
    print(f"  {r.scenario_name}: {sign}{r.delta:.2f} pts")
```

## SimulationResult

Every simulation returns a `SimulationResult`:

```python
result.simulation_id          # UUID
result.scenario_name          # human label
result.original_composite     # before
result.simulated_composite    # after
result.delta                  # simulated − original
result.improved               # delta > 0
result.category_impacts       # {category: contribution to delta}
result.changes_applied        # list of changes
result.grade_change           # "Candidate → Production Ready" or None
```

## API Integration

```python
api = HRSAApi()
b = RawMetricsBundle("my_exp")
# ... set metrics, then score once to cache bundle
api.score(b)

# Category what-if via API
resp = api.what_if("my_exp", [
    {"name": "Improve calibration", "changes": {"calibration": 85.0}}
])

# Optimal budget
resp = api.what_if_optimal("my_exp", budget=3)
improvements = resp["data"]["optimal_improvements"]
```

## Simulation Persistence

All simulations are saved to `hrsa_whatif` database table:

```python
db.whatif_history("my_exp")
# [{"scenario_name": ..., "delta": ..., "simulated_at": ...}, ...]
```
