# HRSA Benchmark Guide

## Overview

The `HRSABenchmarkEngine` compares a model's HierarchicalScore against
multiple reference points to provide context for the score.

## Benchmark Roles

| Role | Description |
|------|-------------|
| `champion` | Current production champion |
| `previous_champion` | Last displaced champion |
| `baseline` | Fixed institutional reference |
| `current_generation` | Models in current training run |
| `previous_generation` | Models from prior training run |
| `public_benchmark` | Externally published benchmark |
| `internal_benchmark` | Firm-specific historical target |
| `candidate` | Model under evaluation |
| `experimental` | Experimental / research model |

## Usage

```python
from trainer.research import HRSABenchmarkEngine, BenchmarkRole, HRSAApi

api = HRSAApi(db_path="data/hrsa.db")

# Register the current champion
api.register_benchmark("champion_v5", champion_bundle,
                        role="champion", name="Champion v5")

# Register a baseline
api.register_benchmark("baseline_2025", baseline_bundle,
                        role="baseline", name="2025 Baseline")

# Compare a candidate
resp = api.compare_to_benchmarks("candidate_v6", candidate_bundle)
suite = resp["data"]
print(f"Beats champion: {suite['beats_champion']}")
for comparison in suite["comparisons"]:
    delta = comparison["composite_delta"]
    print(f"  vs {comparison['reference']['name']}: {delta:+.2f} pts")
```

## Pairwise Comparison

Each `BenchmarkDelta` includes:

```python
delta.composite_delta       # subject − reference (pts)
delta.improved              # True if composite_delta > 0
delta.stronger              # categories where subject > ref + 2 pts
delta.weaker                # categories where subject < ref − 2 pts
delta.category_deltas       # {category: delta} for all categories
```

## Champion Lifecycle

When a new model is registered as `champion`:
1. The previous champion is automatically demoted to `previous_champion`.
2. Its composite is added to the historical average.
3. `engine.champion_record()` returns the current champion.

```python
engine = HRSABenchmarkEngine()
engine.register("v1", hs_v1, role="champion")  # v1 = champion
engine.register("v2", hs_v2, role="champion")  # v2 = champion, v1 = previous_champion
```

## Full Report

```python
report = api.benchmark_report()["data"]
# {
#   "total_registered": 5,
#   "champion_id": "v2",
#   "historical_average": 82.3,
#   "role_distribution": {"champion": 1, "baseline": 2, ...},
#   "ranking": [...],
#   "score_range": {"min": 74.1, "max": 86.2, "mean": 80.5}
# }
```

## Historical Average

When champions are registered over time, the engine tracks their
composites and computes a virtual `historical_average` reference
included in every benchmark comparison.

## Category-Level Benchmarking

Every BenchmarkDelta includes `category_deltas` so you can see exactly
which dimensions the subject improved or regressed on:

```python
delta = suite.vs_champion()
best_cats = sorted(delta.category_deltas.items(), key=lambda x: x[1])
print("Best improvements:")
for cat, d in best_cats[-3:]:
    print(f"  {cat}: {d:+.1f} pts")
```

## Persisting Benchmarks

```python
# Save benchmark record to DB
api.register_benchmark("champion_v5", bundle, role="champion",
                        generation="gen_2026_q1")
```

The HRSA database table `hrsa_benchmarks` stores all registered benchmark
records with their category scores for future retrieval.
