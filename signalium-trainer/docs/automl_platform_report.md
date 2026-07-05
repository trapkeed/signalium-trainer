# Institutional AutoML Platform Report — Phase 16

**Date:** 2026-07-05
**Phase:** 16 — Institutional AutoML and Research Optimization Platform
**Repository:** signalium-trainer
**Status:** Complete

---

## 1. Architecture

### 1.1 Package Overview

```
trainer/automl/              # 13 modules
├── search_space.py          # Parameter space: Float, Int, Categorical, Conditional
├── objectives.py            # Objective functions and composite scorer
├── pruning.py               # MedianPruner, SuccessiveHalvingPruner, NoPruner
├── strategies.py            # Grid, Random, Bayesian, Manual strategies
├── experiment.py            # Trial, SearchJob, ExperimentRecord data models
├── resource_manager.py      # CPU/GPU/RAM slot management
├── backbone_search.py       # BackboneCatalog, variant generation
├── ranking.py               # Multi-dimensional experiment ranking
├── database.py              # SQLite persistence (4 tables)
├── visualization.py         # 8 chart types (JSON-safe)
├── api.py                   # 15 structured endpoints
├── engine.py                # Orchestration engine + config
└── __init__.py              # 52 public exports
```

### 1.2 System Diagram

```
YAML Config
    ↓
AutoMLConfig
    ↓
AutoMLEngine
    ├── SearchSpace (17+ hyperparameters)
    ├── BaseStrategy (grid | random | bayesian | manual)
    ├── BasePruner (none | median | successive_halving)
    ├── ResourceManager (slots, snapshots)
    ├── BackboneCatalog (6 backbones)
    │
    ├── Trial Loop
    │    ├── Suggest config
    │    ├── Evaluate (train_fn OR mock)
    │    ├── Optional: Phase 15 validation
    │    ├── Score (composite_research_score)
    │    ├── Prune check
    │    └── Persist ExperimentRecord → SQLite
    │
    ├── ExperimentRanker (overall + 14 instruments + 8 TFs + 5 regimes)
    ├── Promotion eligibility check
    ├── AutoMLVisualizationBuilder (8 charts)
    └── AutoMLAPI (15 endpoints)
```

---

## 2. Optimization Workflow

### 2.1 Search Pipeline (per trial)

1. `strategy.suggest(trial_id)` → `Suggestion(config, backbone)`
2. `train_fn(config, backbone_spec)` → `metrics_dict`
3. `validation_engine.evaluate_metrics(metrics)` (optional Phase 15)
4. `compute_composite_research_score(metrics)`
5. `trial.add_intermediate(step, score)`
6. `pruner.should_prune(trial_id, step, score, history)`
   - YES → `trial.prune()`, persist pruning decision
   - NO  → `trial.complete()`, `strategy.report(score)`
7. `_check_promotion(metrics, thresholds)` → `promotion_eligible`
8. `db.save_record(ExperimentRecord)`

### 2.2 Bayesian Optimization (pure Python GP)

The Bayesian strategy uses a Gaussian Process surrogate with:
- **Kernel:** Radial Basis Function (RBF)
- **Acquisition:** Expected Improvement (EI)
- **Solver:** Pure-Python Cholesky decomposition
- **Candidates:** `n_candidates` random samples evaluated per step
- **Warm-up:** `n_initial_random` random trials before GP activates

**Complexity:** O(n³) per suggestion (Cholesky), scales to ~200 trials before becoming slow. For larger budgets, use `strategy: random`.

### 2.3 Pruning

**MedianPruner:** Prunes if current score < Pₙ-th percentile of all other trials at the same step. Safe defaults (5 startup trials, 0 warmup steps).

**SuccessiveHalvingPruner:** Implements ASHA. At each rung level (min_resource × eta^k), keeps the top 1/eta fraction. Configurable reduction factor.

---

## 3. Experiment Lifecycle

```
SearchJob created → started → [n_trials × Trial] → completed
                                    ↓
                            Trial: pending → running → completed
                                                     → pruned
                                                     → failed → retrying
```

All state transitions are persisted to SQLite in real time, enabling recovery after interruption.

---

## 4. Database Schema

### Tables

| Table                  | Rows Stored                              |
|------------------------|------------------------------------------|
| `search_jobs`          | One per AutoML search run                |
| `experiment_records`   | One per trial                            |
| `pruning_decisions`    | One per pruning evaluation               |
| `promotion_decisions`  | One per promotion eligibility evaluation |

### Indexes

```sql
CREATE INDEX idx_records_job   ON experiment_records(job_id);
CREATE INDEX idx_records_status ON experiment_records(status);
CREATE INDEX idx_records_score ON experiment_records(primary_score DESC);
CREATE INDEX idx_pruning_trial ON pruning_decisions(trial_id);
CREATE INDEX idx_promotion_trial ON promotion_decisions(trial_id);
```

---

## 5. API Design

15 structured endpoints organized into 6 groups:

| Group             | Endpoints                                               |
|-------------------|---------------------------------------------------------|
| Search Jobs       | create_job, get_job, list_jobs, pause_job, resume_job  |
| Trials            | get_trial, list_trials                                  |
| Optimization      | get_optimization_history                                |
| Leaderboard       | get_leaderboard, get_pareto_front                       |
| Promotion         | get_promotion_candidates, approve_promotion             |
| Analysis          | get_dashboard, get_stats, champion_challenger_compare   |

All endpoints return `{"status": "ok"|"error", "data": ..., "timestamp": ...}`.

---

## 6. Resource Management

### Worker Slots

- `ResourceManager` maintains a fixed pool of `max_workers` slots.
- Each trial acquires a slot before execution and releases it on completion.
- Slot overflow returns `None` — caller must queue or wait.

### Throttling

Automatic throttling when:
- RAM > 90% of `max_ram_gb`
- VRAM > 95% of `gpu_memory_gb`

### Checkpointing

Job state is checkpointed every `checkpoint_frequency` trials. On restart, all completed records are available in the database for strategy warm-up.

---

## 7. Scalability

### Current Limitations

| Dimension         | Current Scale       | Notes                               |
|-------------------|---------------------|-------------------------------------|
| Trials            | ~1,000 per job      | SQLite scales fine                  |
| Parallel workers  | 1–16               | Thread-safe, no distributed compute |
| Bayesian GP       | ~200 trials         | O(n³) Cholesky; switch to random above this |
| Database size     | ~100MB per 10K trials | WAL mode, one file                |

### Future Extensibility

- **Distributed workers:** Replace `ResourceManager` with a task queue (Celery, Ray).
- **Hyperband:** Add to `BasePruner` — `SuccessiveHalvingPruner` is Hyperband-ready.
- **Population-based training:** `Trial.parent_trial_id` supports lineage tracking.
- **External surrogate:** Replace `BayesianStrategy._gp_predict` with `scikit-learn` or `GPyTorch`.

---

## 8. Visualization

8 chart types, all JSON-safe (no matplotlib/plotly):

| Chart                  | Type       | Description                                      |
|------------------------|------------|--------------------------------------------------|
| `optimization_history` | line       | Score per trial index                            |
| `search_convergence`   | line       | Running best score (convergence curve)           |
| `parameter_importance` | bar        | Absolute correlation of each param with score   |
| `leaderboard_progression` | bar     | Top-N trials ranked by score                    |
| `experiment_timeline`  | gantt      | Trial start time + duration                      |
| `resource_utilization` | gauge      | CPU/GPU/RAM peaks                                |
| `objective_tradeoff`   | scatter    | 2D Pareto scatter for any two objectives         |
| `score_distribution`   | histogram  | Distribution of trial scores                     |

---

## 9. Operational Readiness

### Production Checklist

| Item                          | Status   |
|-------------------------------|----------|
| Zero new runtime dependencies | ✅ Done  |
| YAML-configurable (all params)| ✅ Done  |
| SQLite persistence            | ✅ Done  |
| Thread-safe database          | ✅ Done  |
| Retry on failure              | ✅ Done  |
| Checkpoint/resume             | ✅ Done  |
| Promotion eligibility checks  | ✅ Done  |
| Manual approval workflow      | ✅ Done  |
| Phase 15 integration          | ✅ Done  |
| 130+ unit tests               | ✅ Done  |
| Full documentation (7 docs)   | ✅ Done  |
| Backward compatible           | ✅ Done  |

### Running Tests

```bash
pytest tests/unit/test_automl_platform.py -v
```

### Starting a Search

```bash
python -c "
from trainer.automl import AutoMLEngine
engine = AutoMLEngine.from_yaml('conf/automl/automl.yaml')
result = engine.run(model_id='quant_v1')
print(result.summary())
"
```

---

## 10. Deliverables

| Deliverable                                  | Location                                          |
|----------------------------------------------|---------------------------------------------------|
| AutoML package (13 modules)                  | `trainer/automl/`                                 |
| YAML configuration                           | `conf/automl/automl.yaml`                         |
| Unit tests (130+ cases)                      | `tests/unit/test_automl_platform.py`              |
| AutoML Guide                                 | `docs/automl_guide.md`                            |
| Hyperparameter Optimization Guide            | `docs/hyperparameter_optimization_guide.md`       |
| Experiment Guide                             | `docs/experiment_guide.md`                        |
| Champion/Challenger Guide                    | `docs/champion_challenger_guide.md`               |
| Developer Guide                              | `docs/developer_guide_phase16.md`                 |
| API Reference                                | `docs/automl_api_reference.md`                    |
| Platform Report                              | `docs/automl_platform_report.md`                  |
