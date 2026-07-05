# IRSE Benchmark Guide

## Overview

The IRSE Benchmarker compares any model's Research Score against:

- The current **champion** (model in production)
- **Historical champions** (prior promotions)
- **Baseline** models (fixed reference points)
- **Experimental** models (under active development)
- **Candidate** models (promotion-ready)

---

## Model Roles

```python
from trainer.research.benchmarking import ModelRole

ModelRole.CHAMPION     # current production model
ModelRole.BASELINE     # fixed reference (never changes)
ModelRole.CANDIDATE    # awaiting promotion review
ModelRole.EXPERIMENTAL # active research
ModelRole.HISTORICAL   # former champion — archived
```

---

## Basic Usage

```python
from trainer.research.benchmarking import IRSEBenchmarker, ModelRole

bm = IRSEBenchmarker()

# Register scored experiments
bm.register("exp_baseline", baseline_rs, ModelRole.BASELINE)
bm.register("exp_champion", champion_rs, ModelRole.CHAMPION)
bm.register("exp_candidate", candidate_rs, ModelRole.CANDIDATE)
bm.register("exp_experimental_v2", experimental_rs, ModelRole.EXPERIMENTAL)

# Set the champion
bm.set_champion("exp_champion")
```

---

## Pairwise Comparison

```python
# Compare candidate vs champion (default: vs champion)
comp = bm.compare("exp_candidate")

print(comp.summary())
# Benchmark: candidate_v3 vs champion_v2 [champion]
#   Composite delta: +4.12
#   Stronger categories: validation_quality, trading_performance
#   Weaker categories: efficiency

# Access details
print(f"Delta: {comp.composite_delta:+.2f}")
print(f"Improved: {comp.composite_improved}")
print(f"Category deltas: {comp.category_deltas}")

# Compare against specific model
comp2 = bm.compare("exp_experimental_v2", against="exp_baseline")
```

---

## Benchmark Reports

```python
# Full report for one experiment
report = bm.benchmark_report("exp_candidate")
print(report["champion_comparison"])
print(report["all_comparisons"])

# Summary across all registered models
full = bm.full_benchmark_report()
print(f"Best model: {full['ranking'][0]['name']} "
      f"({full['ranking'][0]['composite']:.1f})")

# Compare all models against baseline
comparisons = bm.compare_all_against("exp_baseline")
```

---

## API-Level Benchmarking

```python
# Register via API (also persists to DB)
api.register_benchmark("exp_champion", champion_rs, role="champion")
api.register_benchmark("exp_candidate", candidate_rs, role="candidate")

# Compare
resp = api.compare_experiments("exp_candidate", "exp_champion")
delta = resp["data"]["composite_delta"]

# Full report
resp = api.benchmark_report("exp_candidate")
resp = api.full_benchmark()
```

---

## Dashboard — Benchmark Comparison

```python
dash = IRSEScoreDashboard(lb, trends, bm, db)

# Side-by-side category scores for all benchmark roles
bm_dash = dash.benchmark_comparison(subject_id="exp_candidate")
# Returns: { type, models: [{name, role, composite, category_scores}], ... }
```

---

## Promotion Workflow

```python
# 1. Score the candidate
resp = api.score_experiment(candidate_inputs)

# 2. Register for benchmarking
api.register_benchmark(candidate_id, candidate_rs, role="candidate")

# 3. Compare against champion
comp = api.compare_experiments(candidate_id, champion_id)

# 4. Check improvement
if comp["data"]["composite_improved"]:
    delta = comp["data"]["composite_delta"]
    stronger = comp["data"]["stronger_categories"]
    print(f"Candidate beats champion by {delta:+.2f} pts")
    print(f"Stronger in: {', '.join(stronger)}")

    # 5. Get promotion eligibility
    candidates = api.get_promotion_candidates()
    eligible = any(c["experiment_id"] == candidate_id
                   for c in candidates["data"]["candidates"])
    if eligible:
        print("Ready for promotion review")
```

---

## Interpreting Comparisons

| Delta      | Interpretation                                 |
|------------|------------------------------------------------|
| > +5.0     | Clear improvement — strong promotion case      |
| +2.0 to +5 | Moderate improvement — review category details |
| -2.0 to +2 | No significant difference                      |
| < -2.0     | Regression — investigate weaker categories     |

Category delta ≥ +2.0 → stronger.  Category delta ≤ -2.0 → weaker.
