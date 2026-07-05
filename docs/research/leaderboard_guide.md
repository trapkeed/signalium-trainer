# Leaderboard Guide

## Purpose

The Leaderboard ranks models from a benchmark run using a configurable
weighted composite score. It produces ranked tables, summary dicts,
and ASCII console output.

## Building the Leaderboard

```python
from trainer.research import Leaderboard, ScoringWeights

lb = Leaderboard()  # Default weights
lb.ingest(benchmark_result)
```

## Scoring Weights

The `overall_score` is a weighted combination of normalised metrics.
You can tune the weights to match your deployment priorities:

```python
# Prioritise predictive quality (default balance)
weights = ScoringWeights(
    accuracy=0.25,
    f1_macro=0.30,
    f1_weighted=0.15,
    calibration_error=0.15,
    inference_speed=0.15,
)

# Prioritise low latency (e.g. high-frequency context)
weights = ScoringWeights(
    accuracy=0.10,
    f1_macro=0.15,
    f1_weighted=0.10,
    calibration_error=0.05,
    inference_speed=0.60,
)

lb = Leaderboard(weights=weights)
lb.ingest(benchmark_result)
```

## Reading Results

```python
# All entries sorted by rank
entries = lb.ranked_entries()

# Top 5
top5 = lb.top_n(5)

# Current champion (rank 1)
champ = lb.champion()
print(f"Champion: {champ.model_name}, Score: {champ.overall_score:.4f}")

# ASCII table
print(lb.format_table())

# Serialise
lb_dicts = lb.to_dict()  # List of dicts, JSON-serialisable
```

## LeaderboardEntry Fields

| Field | Type | Description |
|---|---|---|
| `rank` | int | 1 = best |
| `model_name` | str | Model name from benchmark |
| `architecture` | str | From ModelSpec (if registered) |
| `accuracy` | float | Validation accuracy |
| `f1_macro` | float | Macro-averaged F1 |
| `f1_weighted` | float | Weighted F1 |
| `calibration_error` | float | ECE (lower = better) |
| `inference_latency_ms` | float | Median per-sample latency |
| `training_time_seconds` | float | Total training wall-clock |
| `model_size_mb` | float | Parameter size in MB |
| `overall_score` | float | Weighted composite [0, 1] |
| `status` | str | `"completed"` or `"failed"` |

## Visualising the Leaderboard

```python
from trainer.research.visualization import plot_leaderboard
from pathlib import Path

plot_leaderboard(
    lb.to_dict(),
    output_path=Path("reports/leaderboard.png"),
    title="Model Benchmark — Q3 2024",
    metric="f1_macro",
)
```
