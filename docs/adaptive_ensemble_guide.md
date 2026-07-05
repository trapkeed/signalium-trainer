# Adaptive Ensemble Intelligence Platform Guide

## Overview

Phase 17 builds the **Adaptive Ensemble Intelligence Platform (AEIP)** — transforming Signalium from a single-model forecasting system into an adaptive ensemble of specialized quantitative models.

**Key principles:**
- The existing Signalium Quant model is Expert 1. It is NEVER removed.
- Every new model becomes an additional expert.
- Experts train independently with their own weights, checkpoints, validation, metrics, version, and metadata.
- The ensemble intelligently combines expert outputs via configurable strategies.
- Model weights adapt automatically based on performance, regime, instrument, and timeframe.
- Full backward compatibility — zero modifications to Phases 1–16.

---

## Architecture

```
EnsembleEngine
├── ExpertRegistry
│    ├── Expert 1: Signalium Quant v1 (lstm, champion)
│    ├── Expert 2: Transformer Expert (transformer_encoder, trending)
│    └── Expert N: Future experts...
│
├── ExpertRouter
│    ├── DynamicWeightEngine
│    │    └── WeightFactorConfig (7 factors)
│    └── RouterConfig (filter by instrument/timeframe/regime)
│
├── BaseEnsembleStrategy (8 strategies)
│    ├── SimpleAveraging | WeightedAveraging | PerformanceWeighted
│    ├── HardVoting | SoftVoting
│    ├── MixtureOfExperts | Stacking | Blending
│
├── EnsembleMetaModel (linear meta-learner with online SGD)
│
├── ExpertCalibrator (per-expert + ensemble ECE, temperature scaling)
├── ExpertFailureAnalyzer (regime/agreement/uncertainty tracking)
├── OnlineScorer (rolling accuracy, Sharpe, win rate per window)
│
├── EnsembleDatabase (SQLite — 7 tables)
├── EnsembleLeaderboard (experts + ensemble + meta model)
├── EnsembleVisualizationBuilder (8 JSON chart types)
├── EnsembleAPI (15 structured endpoints)
└── EnsembleExporter (bundle, inference contract)
```

---

## Quick Start

```python
from trainer.ensemble import EnsembleEngine, EnsembleConfig, ExpertModel

# Create engine — Expert 1 (Signalium Quant v1) is registered automatically
engine = EnsembleEngine.from_config({
    "db_path": "experiments/ensemble/ensemble.db",
    "strategy": "soft_voting",      # any of 8 strategies
    "rolling_window": 200,
})

# Register a new expert (Expert 2)
engine.register_expert(ExpertModel(
    expert_id="expert_02",
    name="Transformer Expert",
    model_family="transformer_encoder",
    specializations=["trending"],
    version="1.0.0",
))

# Run a prediction
bundle = engine.predict(
    features={"close": [1.12, 1.13, 1.14]},
    context={
        "regime": "trending_up",
        "instrument": "EURUSD",
        "timeframe": "H1",
    },
)

print(bundle.to_dict()["institutional_forecast"])
# "[B] LONG | trending_up | conf=0.72 | agree=0.75 | n_experts=2"

# Record outcome (ground truth)
engine.record_outcome(bundle.bundle_id, true_class="buy", profit_pnl=85.0)

# Dashboard
resp = engine.get_dashboard()
charts = resp["data"]["charts"]
```

---

## Expert Model Registration

Every expert has:

| Field              | Description                                                 |
|--------------------|-------------------------------------------------------------|
| `expert_id`        | Unique UUID                                                 |
| `name`             | Human-readable name                                         |
| `model_family`     | Architecture: lstm, gru, transformer_encoder, temporal_cnn, hybrid_* |
| `specializations`  | Market conditions: trending, ranging, high_volatility, ...  |
| `instruments`      | Instrument filter (empty = all instruments)                 |
| `timeframes`       | Timeframe filter (empty = all timeframes)                   |
| `version`          | Semantic version                                            |
| `checkpoint_path`  | Path to model weights                                       |
| `status`           | active | champion | retired | experimental                |
| `weight`           | Current ensemble weight (auto-updated)                      |

```python
from trainer.ensemble import ExpertModel, ModelFamily, SpecializationType

expert = ExpertModel(
    expert_id="expert_03",
    name="High-Vol TCN",
    model_family=ModelFamily.TEMPORAL_CNN,
    specializations=[SpecializationType.HIGH_VOLATILITY],
    instruments=["GBPUSD", "USDJPY"],
    timeframes=["M5", "M15"],
    version="1.0.0",
    checkpoint_path="models/tcn_high_vol_v1.pt",
)
engine.register_expert(expert)
```

---

## Ensemble Strategies

| Strategy              | Description                                           |
|-----------------------|-------------------------------------------------------|
| `simple_averaging`    | Unweighted mean of all expert probabilities           |
| `weighted_averaging`  | Weight-adjusted mean                                  |
| `performance_weighted`| Weights from recent expert performance                |
| `hard_voting`         | Majority class gets probability 1.0                  |
| `soft_voting`         | Weighted sum of class probabilities (recommended)    |
| `mixture_of_experts`  | Softmax-normalized gating with temperature control   |
| `stacking`            | Learned linear combiner (online SGD)                 |
| `blending`            | Holdout-calibrated blending                           |

Change strategy without retraining:

```yaml
# conf/ensemble/ensemble.yaml
ensemble:
  strategy: mixture_of_experts
```

---

## Prediction Output

Every `engine.predict()` call returns a `PredictionBundle`:

```python
bundle = engine.predict(features, context)
d = bundle.to_dict()

print(d["final"]["buy_prob"])             # float [0, 1]
print(d["final"]["sell_prob"])            # float [0, 1]
print(d["final"]["no_trade_prob"])        # float [0, 1]
print(d["final"]["confidence"])           # float [0, 1]
print(d["institutional_forecast"])        # "[A] LONG | trending_up | conf=0.85 | agree=0.90"
print(d["n_experts"])                     # int
print(d["routing_decision"]["regime"])    # str
```

---

## YAML Configuration

```yaml
# conf/ensemble/ensemble.yaml
ensemble:
  db_path: experiments/ensemble/ensemble.db
  strategy: soft_voting
  rolling_window: 200
  weight_adjust_interval: 50
  meta_min_samples: 50
  auto_retire_threshold: null   # e.g. 0.35 to auto-retire underperformers
```
