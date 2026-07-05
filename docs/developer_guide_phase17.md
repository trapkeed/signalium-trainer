# Developer Guide — Phase 17: Adaptive Ensemble Intelligence Platform

## Package Structure

```
trainer/ensemble/               # 15 modules
├── __init__.py                 # Public API exports (60+ symbols)
├── expert.py                   # SpecializationType, ModelFamily, ExpertPrediction,
│                               #   ExpertModel, ExpertRegistry, build_default_registry
├── strategies.py               # EnsembleResult, 8 combination strategies, build_strategy
├── weighting.py                # WeightFactorConfig, DynamicWeightEngine, REGIME_FAMILY_AFFINITY
├── routing.py                  # RoutingDecision, RouterConfig, ExpertRouter
├── meta_model.py               # MetaPrediction, MetaFeatureBuilder, EnsembleMetaModel
├── calibration.py              # CalibrationReport, ExpertCalibrator
├── failure_analysis.py         # FailureEvent, ExpertFailureReport, ExpertFailureAnalyzer
├── online_scoring.py           # PredictionRecord, RollingMetrics, OnlineScorer
├── database.py                 # EnsembleDatabase (SQLite, 7 tables)
├── leaderboard.py              # LeaderboardEntry, EnsembleLeaderboard
├── visualization.py            # EnsembleVisualizationBuilder (8 chart types)
├── api.py                      # EnsembleAPI (15 endpoints)
├── export.py                   # EnsembleExporter (bundle, inference contract)
└── engine.py                   # EnsembleConfig, PredictionBundle, EnsembleEngine
```

Configuration: `conf/ensemble/ensemble.yaml`

Tests: `tests/unit/test_ensemble_platform.py` (160+ tests)

Docs:
- `docs/adaptive_ensemble_guide.md`
- `docs/meta_learning_guide.md`
- `docs/expert_routing_guide.md`
- `docs/voting_guide.md`
- `docs/developer_guide_phase17.md` (this file)
- `docs/ensemble_architecture_report.md`

---

## Design Decisions

### Expert 1 is never removed

`build_default_registry()` always registers Signalium Quant v1 as `expert_01` with champion status. This is called automatically by `EnsembleEngine.__init__()`.

### Evaluation-only inference contract

The `ExpertModel.predict_fn` is an injected callable — the engine never imports PyTorch directly. This keeps the ensemble layer dependency-free.

### Pure Python meta model

The `LinearMetaLayer` uses hand-rolled matrix ops + softmax. No PyTorch/sklearn in the meta layer. Scales to ~10 experts efficiently.

### Zero new runtime dependencies

All functionality uses Python standard library + existing `loguru` + existing `PyYAML`.

---

## Adding a New Expert

```python
from trainer.ensemble import ExpertModel, ModelFamily, SpecializationType

engine.register_expert(ExpertModel(
    expert_id="expert_03",
    name="GRU Ranging Specialist",
    model_family=ModelFamily.GRU,
    specializations=[SpecializationType.RANGING],
    instruments=["EURUSD", "GBPUSD"],
    timeframes=["H1", "H4"],
))
```

Or via the API:

```python
resp = engine.api.register_expert({
    "name": "GRU Ranging Specialist",
    "model_family": "gru",
    "specializations": ["ranging"],
    "instruments": ["EURUSD"],
    "version": "1.0.0",
})
```

---

## Adding a New Ensemble Strategy

1. Subclass `BaseEnsembleStrategy` in `trainer/ensemble/strategies.py`.
2. Set `name` class attribute.
3. Implement `combine(predictions, weights) → EnsembleResult`.
4. Add to `_STRATEGY_MAP`.
5. Reference by name in YAML: `strategy: my_strategy`.

```python
class MyStrategy(BaseEnsembleStrategy):
    name = "my_strategy"

    def combine(self, predictions, weights=None):
        w = self._normalize_weights(predictions, weights)
        # ... your combination logic ...
        return EnsembleResult(
            buy_prob=..., sell_prob=..., no_trade_prob=...,
            strategy=self.name, n_experts=len(predictions),
            ...
        )

_STRATEGY_MAP["my_strategy"] = MyStrategy
```

---

## Database Schema

| Table                | Rows Stored                         |
|----------------------|-------------------------------------|
| `expert_models`      | One per registered expert           |
| `ensemble_results`   | One per prediction cycle            |
| `meta_model_state`   | Checkpoints of meta model           |
| `weight_history`     | One per weight computation event    |
| `routing_decisions`  | One per route() call                |
| `performance_history`| One per expert prediction outcome   |
| `voting_history`     | One per expert vote per prediction  |

---

## Backward Compatibility

Phase 17:
- Does NOT modify any Phase 1–16 modules
- Does NOT modify `trainer/validation/` (Phase 15)
- Does NOT modify `trainer/automl/` (Phase 16)
- Does NOT modify any existing model architectures
- Adds only: `trainer/ensemble/` + `conf/ensemble/` + docs + tests

---

## Testing

```bash
# Run all Phase 17 tests
pytest tests/unit/test_ensemble_platform.py -v

# Run specific group
pytest tests/unit/test_ensemble_platform.py::TestEnsembleEngine -v

# With coverage
pytest tests/unit/test_ensemble_platform.py --cov=trainer.ensemble
```

Tests cover 160+ cases across all 15 modules.

---

## Full Ensemble Run Example

```python
from trainer.ensemble import EnsembleEngine, EnsembleConfig, ExpertModel, ModelFamily

# Initialize
cfg = EnsembleConfig(
    db_path="experiments/ensemble/ensemble.db",
    strategy="soft_voting",
    rolling_window=200,
    meta_min_samples=50,
    auto_retire_threshold=0.35,
)
engine = EnsembleEngine(config=cfg)

# Register experts
engine.register_expert(ExpertModel(
    expert_id="expert_02",
    name="Transformer Trending Expert",
    model_family=ModelFamily.TRANSFORMER_ENCODER,
    specializations=["trending"],
    predict_fn=my_transformer_predict,
))

# Prediction loop
for features, context, true_class, pnl in live_data:
    bundle = engine.predict(features, context)
    print(bundle.to_dict()["institutional_forecast"])
    engine.record_outcome(bundle.bundle_id, true_class, pnl)

# Summary
print(engine.summary())

# Export for deployment
bundle = engine.export_bundle("experiments/ensemble/exports/v1.json")
contract = engine.export_inference_contract()

# Close
engine.close()
```
