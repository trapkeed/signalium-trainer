# Signalium Research Guide

## Overview

The Signalium Research & Benchmarking Framework is a complete environment for
objective, repeatable comparison of quantitative trading models.

It provides:
- **Experiment Tracking** — persistent records of every training run
- **Benchmark Runner** — identical-conditions multi-model evaluation
- **Leaderboard** — automatic configurable-weight ranking
- **Cross-Validation** — time-series aware fold generation
- **Champion/Challenger** — model lifecycle management
- **Report Generation** — HTML, Markdown, JSON, CSV export
- **Visualization** — training curves, ROC, confusion matrix, calibration

## Quick Start

```python
from trainer.research import (
    BenchmarkConditions, BenchmarkRunner, Leaderboard,
    ReportConfig, ReportGenerator,
)
from trainer.framework import ModelFactory
from trainer.config.schema import TrainerConfig
from pathlib import Path

# 1. Define identical conditions for all models
conditions = BenchmarkConditions(
    random_seed=42,
    max_epochs=100,
    batch_size=256,
    train_ratio=0.70,
    val_ratio=0.15,
    device="cuda",
)

# 2. Register models
runner = BenchmarkRunner(conditions=conditions)
runner.register_model(
    "quant_v1_lstm",
    model_factory=lambda: ModelFactory.create("quant_v1", TrainerConfig()),
)

# 3. Run benchmark (pass your train/val tensors)
result = runner.run(train_data=(x_train, y_train), val_data=(x_val, y_val))

# 4. Build leaderboard
lb = Leaderboard()
lb.ingest(result)
print(lb.format_table())

# 5. Generate reports
gen = ReportGenerator(ReportConfig(
    formats=["html", "markdown", "json", "csv"],
    output_dir=Path("reports"),
))
gen.generate_benchmark_report(result, lb)
```

## Package Structure

```
trainer/research/
├── experiment/       ExperimentRecord, ExperimentTracker
├── benchmark/        BenchmarkConditions, BenchmarkRunner, BenchmarkResult
├── leaderboard/      Leaderboard, ScoringWeights, LeaderboardEntry
├── validation/       WalkForwardValidator, RollingWindowValidator
├── hyperparameter/   HyperparameterRecord
├── report/           ReportGenerator, ReportConfig, export_*
├── visualization/    plot_training_curves, plot_roc_curves, ...
└── champion/         ModelStatus, ChampionRegistry, ModelEntry
```

## Key Principles

1. **Reproducibility** — every benchmark run applies the same global seed before each model.
2. **Fair Comparison** — all models use the same dataset, split, and evaluation logic.
3. **Persistence** — ExperimentTracker and ChampionRegistry write JSON to disk automatically.
4. **No Training Engine Changes** — the research framework sits on top of the existing engine.
5. **Optional Dependencies** — visualization requires `matplotlib`; all other features are dependency-light.
