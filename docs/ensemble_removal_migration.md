# Ensemble Removal Migration Report

**Repository:** signalium-trainer  
**Date:** 2026-07-08  
**Author:** Architecture Migration  
**Standard:** One Model. One Brain. One Trader. One Decision Maker.

---

## Objective

Remove every system that allows multiple models to make or influence production trading decisions.

The following systems violate the Unified QDE philosophy and are deleted in this migration:

- Expert Router
- Expert Selection / Expert Registry
- Model Ensemble
- Meta Models
- Voting Systems (8 combination strategies)
- Consensus / Arbitration Systems
- Canary deployment mode (routes live traffic to a challenger model)

---

## What Is Being Removed

### trainer/ensemble/ — 15 files, 4,800+ lines

| File | Class | Violation |
|---|---|---|
| `engine.py` | `EnsembleEngine` | Central orchestrator of N competing models. Embodies the entire multi-brain pattern. |
| `expert.py` | `ExpertModel`, `ExpertRegistry` | Defines "experts" as independent decision makers. Registry explicitly states "Expert 1 is never removed." |
| `routing.py` | `ExpertRouter` | Routes market data to different experts based on regime classification. |
| `strategies.py` | `SimpleAveragingStrategy`, `WeightedAveragingStrategy`, `PerformanceWeightedStrategy`, `HardVotingStrategy`, `SoftVotingStrategy`, `MixtureOfExpertsStrategy`, `StackingStrategy`, `BlendingStrategy` | 8 voting/consensus algorithms. All assume N competing outputs to combine. |
| `meta_model.py` | `EnsembleMetaModel`, `MetaFeatureBuilder` | A model that overrides all other models. Second decision maker above the ensemble. |
| `weighting.py` | `DynamicWeightEngine` | Computes per-expert weights from performance and routing signals. |
| `calibration.py` | `ExpertCalibrator`, `CalibrationReport` | Per-expert output calibration. Superseded by `trainer/calibration/`. |
| `failure_analysis.py` | `ExpertFailureAnalyzer`, `FailureEvent` | Tracks which expert failed on which regime. No experts means no role. |
| `online_scoring.py` | `OnlineScorer`, `PredictionRecord` | Online per-expert scoring. Superseded by `trainer/mlops/monitoring.py`. |
| `leaderboard.py` | `EnsembleLeaderboard` | Expert ranking leaderboard. Incompatible with One Model. |
| `database.py` | `EnsembleDatabase` | SQLite persistence for multi-expert voting history and expert records. |
| `api.py` | `EnsembleAPI` | REST API for multi-expert dashboards and operations. |
| `export.py` | `EnsembleExporter` | Exports an ensemble of N models. Superseded by `trainer/exporters/`. |
| `visualization.py` | `EnsembleVisualizationBuilder` | Expert comparison and voting dashboards. |
| `__init__.py` | — | Module public exports. |

### tests/unit/test_ensemble_platform.py — 1,580 lines

The complete test suite for the ensemble module. Deleted alongside the module.

### CANARY mode in trainer/qde/ — 4 files

| File | Change |
|---|---|
| `qde_models.py` | Remove `QDEABTestType.CANARY` enum value. Remove `canary_traffic_fraction` field from `QDEABTest`. |
| `qde_ab_testing.py` | Remove `create_canary()` method. |
| `qde_api.py` | Remove `create_canary_test()` method. Renumber API endpoint list. |
| `qde_server.py` | Remove `CanaryTestRequest` model. Remove `POST /qde/ab/canary` endpoint. |

**Why CANARY is removed:** CANARY routes a configurable fraction of live traffic to a challenger model. This means two models simultaneously make live trading decisions — violating One Decision Maker. SHADOW and SIDE_BY_SIDE are retained because the challenger observes or compares only; it never executes.

### Ensemble-era field renames — 2 model files

| File | Old field | New field | Reason |
|---|---|---|---|
| `trainer/icari/icari_models.py` | `ensemble_confidence` | removed | `model_confidence` already exists on the same model; `ensemble_confidence` was a second, redundant slot for ensemble-era voting confidence |
| `trainer/memory/decision_models.py` | `ensemble_output` | `model_output` | The single model produces one output; there is no ensemble output |
| `trainer/memory/decision_models.py` | `expert_weights` | removed | Ensemble-era field storing per-expert vote weights; no longer meaningful |

---

## What Is NOT Removed

The following remain unchanged because they serve the one model without replacing it:

| Retained | Reason |
|---|---|
| `qde/qde_ab_testing.py` SHADOW mode | Challenger observes live data but its decisions are not executed. Safe for Research Lab use. |
| `qde/qde_ab_testing.py` SIDE_BY_SIDE mode | Both versions receive identical inputs; neither affects execution. Purely analytical. |
| `trainer/mlops/monitoring.py` | Single-model production monitoring. |
| `trainer/mlops/champion.py` | Tracks the one champion model. |
| `trainer/calibration/` | Output calibration for one model's predictions. |
| `trainer/registry/` | Model registry (one active version at a time). |
| `trainer/research_lab/` | Admin: experiments, backtesting, rollback. Multiple models may exist here for research purposes only. |

---

## Dependency Analysis

**External consumers of `trainer.ensemble`:** None found in `apps/`, `trainer/` (outside `ensemble/`), or any other package.

**The ensemble module was self-contained.** No KEEP module imports from it. The QDE pipeline, MLOps, and Research Lab have no coupling to the ensemble.

**Test file impact:** `tests/unit/test_ensemble_platform.py` is the only test file with ensemble imports. Two other test files reference ensemble-era field names (`ensemble_confidence`, `ensemble_output`) — these are updated to match renamed fields.

---

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Breaking import errors in other modules | **None** — ensemble is isolated | Verified by grep: zero imports from `trainer.ensemble` outside the module itself |
| Loss of expert calibration capability | **Low** | `trainer/calibration/` (Platt, isotonic, temperature) provides full calibration for one model |
| Loss of production monitoring | **None** | `trainer/mlops/monitoring.py` is unaffected |
| Loss of model comparison capability | **None** | SHADOW and SIDE_BY_SIDE test modes remain; Research Lab backtesting remains |
| Loss of institutional knowledge | **Low** | All heuristics documented in `docs/qde_architectural_violations.md` and `docs/architecture_audit_v2.md` before deletion |

---

## Files Changed

### Deleted (17 files)

```
trainer/ensemble/__init__.py
trainer/ensemble/api.py
trainer/ensemble/calibration.py
trainer/ensemble/database.py
trainer/ensemble/engine.py
trainer/ensemble/expert.py
trainer/ensemble/export.py
trainer/ensemble/failure_analysis.py
trainer/ensemble/leaderboard.py
trainer/ensemble/meta_model.py
trainer/ensemble/online_scoring.py
trainer/ensemble/routing.py
trainer/ensemble/strategies.py
trainer/ensemble/visualization.py
trainer/ensemble/weighting.py
trainer/ensemble/  (directory)
tests/unit/test_ensemble_platform.py
```

### Modified (6 files)

```
trainer/qde/qde_models.py       — remove CANARY enum, canary_traffic_fraction field
trainer/qde/qde_ab_testing.py   — remove create_canary() method
trainer/qde/qde_api.py          — remove create_canary_test(), renumber endpoint list
trainer/qde/qde_server.py       — remove CanaryTestRequest, remove /qde/ab/canary
trainer/icari/icari_models.py   — remove ensemble_confidence field
trainer/memory/decision_models.py — rename ensemble_output→model_output, remove expert_weights
tests/unit/test_icari.py        — remove ensemble_confidence= kwargs
tests/unit/test_memory.py       — update ensemble_output→model_output
```

---

## Post-Migration State

After this migration:

- **One model** is registered as ACTIVE in the QDE registry at any time.
- **No voting, no routing, no meta-model** exists in any code path.
- **Research Lab** is the only place where multiple model versions exist simultaneously, and they are never co-executing in production.
- **SHADOW testing** allows a new version to observe live data without making decisions.
- **SIDE_BY_SIDE testing** allows analytical comparison without affecting execution.
- **Promotion** requires explicit admin approval and advances exactly one version to ACTIVE.
- **Rollback** reverts exactly one version to ACTIVE.
