# Developer Guide — Phase 16: Institutional AutoML Platform

## Package Structure

```
trainer/automl/
├── __init__.py          # Public API exports (50+ symbols)
├── search_space.py      # FloatParameter, IntParameter, CategoricalParameter,
│                        #   ConditionalParameter, SearchSpace, build_standard_space
├── objectives.py        # Objective, ObjectiveScorer, STANDARD_OBJECTIVES,
│                        #   compute_composite_research_score, build_research_scorer
├── pruning.py           # NoPruner, MedianPruner, SuccessiveHalvingPruner,
│                        #   PruningDecision, build_pruner
├── strategies.py        # ManualStrategy, GridStrategy, RandomStrategy,
│                        #   BayesianStrategy, Suggestion, build_strategy
├── experiment.py        # Trial, TrialStatus, SearchJob, ExperimentRecord
├── resource_manager.py  # ResourceManager, ResourceSnapshot, WorkerSlot
├── backbone_search.py   # BackboneSpec, BackboneCatalog, BackboneSearcher,
│                        #   BackboneEvalResult
├── ranking.py           # ExperimentRanker, RankingEntry, RankingResult
├── database.py          # AutoMLDatabase (SQLite, thread-safe)
├── visualization.py     # AutoMLVisualizationBuilder (JSON chart data)
├── api.py               # AutoMLAPI (structured endpoint layer)
└── engine.py            # AutoMLEngine, AutoMLConfig, AutoMLResult
```

Configuration: `conf/automl/automl.yaml`

Tests: `tests/unit/test_automl_platform.py` (130+ tests)

Docs:
- `docs/automl_guide.md`
- `docs/hyperparameter_optimization_guide.md`
- `docs/experiment_guide.md`
- `docs/champion_challenger_guide.md`
- `docs/developer_guide_phase16.md` (this file)
- `docs/automl_api_reference.md`
- `docs/automl_platform_report.md`

---

## Design Decisions

### No External AutoML Library

Phase 16 uses pure Python for all optimization logic (Bayesian GP, Cholesky decomposition, EI acquisition). This avoids dependencies on `optuna`, `hyperopt`, or `ray[tune]`.

**Trade-off:** Less efficient than production AutoML libraries, but zero dependency risk and full auditability.

### SQLite Backend

Same approach as Phase 15 — `sqlite3` from the standard library. Thread-safe via `threading.local` (connection-per-thread).

### JSON-Only Visualization

All chart data is returned as Python dicts (JSON-serializable). No `matplotlib` or `plotly` dependency. Frontend can render with any library.

### Evaluation-Only Engine

`AutoMLEngine.run()` never trains models directly. It calls the caller-supplied `train_fn(config, backbone_spec) → metrics_dict`. If `train_fn` is None, the mock evaluator is used.

---

## Adding a New Parameter

1. Open `trainer/automl/search_space.py`.
2. Add your parameter to `build_standard_space()`:
   ```python
   space.add(IntParameter("my_new_param", low=1, high=10))
   ```
3. Or add it to `conf/automl/automl.yaml` via `custom_search_space`.
4. No other code changes required.

---

## Adding a New Backbone

1. Open `trainer/automl/backbone_search.py`.
2. Add to `_DEFAULT_BACKBONES`:
   ```python
   "my_backbone": BackboneSpec(
       name="my_backbone",
       hidden_size=256,
       num_layers=4,
   ),
   ```
3. Or register at runtime:
   ```python
   catalog = BackboneCatalog()
   catalog.register(BackboneSpec(name="my_backbone", hidden_size=256))
   ```

---

## Adding a New Objective

1. Add to `STANDARD_OBJECTIVES` in `trainer/automl/objectives.py`:
   ```python
   "my_metric": Objective("my_metric", "maximize", clip_min=0.0, clip_max=1.0),
   ```
2. Reference in YAML:
   ```yaml
   automl:
     objectives: [composite_research_score, my_metric]
   ```

---

## Adding a New Search Strategy

1. Subclass `BaseStrategy` in `trainer/automl/strategies.py`.
2. Implement `suggest(trial_id, completed) → Suggestion`.
3. Add to `build_strategy()` registry.
4. Reference by name in YAML: `strategy: my_strategy`.

---

## Adding a New Pruner

1. Subclass `BasePruner` in `trainer/automl/pruning.py`.
2. Implement `should_prune(...) → PruningDecision`.
3. Add to `build_pruner()` registry.
4. Reference by name in YAML: `pruner: my_pruner`.

---

## Database Schema

### `search_jobs`
| Column                | Type    | Description                          |
|-----------------------|---------|--------------------------------------|
| `job_id`              | TEXT PK | UUID                                 |
| `name`                | TEXT    | Human name                           |
| `strategy`            | TEXT    | Search strategy                      |
| `n_trials`            | INTEGER | Target trials                        |
| `objective`           | TEXT    | Primary objective name               |
| `objectives_json`     | TEXT    | JSON list of all objectives          |
| `status`              | TEXT    | pending/running/completed/failed     |
| `budget_type`         | TEXT    | trials/time/metric                   |
| `budget_value`        | REAL    | Budget limit                         |
| `pruner_type`         | TEXT    | Pruner name                          |
| `best_score`          | REAL    | Best score so far                    |
| `champion_trial_id`   | TEXT    | Best trial UUID                      |
| `created_at`          | TEXT    | UTC ISO-8601                         |

### `experiment_records`
| Column                | Type    | Description                          |
|-----------------------|---------|--------------------------------------|
| `record_id`           | TEXT PK | UUID                                 |
| `trial_id`            | TEXT    | Trial UUID                           |
| `job_id`              | TEXT FK | Parent job UUID                      |
| `backbone`            | TEXT    | Backbone name                        |
| `status`              | TEXT    | Trial status                         |
| `primary_score`       | REAL    | Primary objective score (indexed)    |
| `composite_research_score` | REAL | Combined research score          |
| `promotion_eligible`  | INTEGER | 0/1                                  |
| `config_json`         | TEXT    | Hyperparameter config                |
| `validation_results_json` | TEXT | Phase 15 metrics                  |
| `pruned`              | INTEGER | 0/1                                  |

### `pruning_decisions`
Stores every pruning evaluation result for audit and analysis.

### `promotion_decisions`
Stores every promotion evaluation result including manual approvals.

---

## Testing

```bash
# Run all Phase 16 tests
pytest tests/unit/test_automl_platform.py -v

# Run a specific class
pytest tests/unit/test_automl_platform.py::TestBayesianStrategy -v

# Run with coverage
pytest tests/unit/test_automl_platform.py --cov=trainer.automl
```

Tests cover: 130+ test cases across all 13 modules. Includes full engine runs with mock evaluator, edge cases, error handling, and database round-trips.

---

## Dependencies

Phase 16 adds **zero new Python dependencies**. All functionality uses:
- Python standard library: `sqlite3`, `threading`, `math`, `statistics`, `uuid`, `json`, `hashlib`
- Existing project dependencies: `loguru`

Optional (for YAML config loading): `PyYAML` (already in project).

---

## Backward Compatibility

Phase 16:
- Does NOT modify any Phase 1–15 modules
- Does NOT modify `trainer/validation/` (Phase 15)
- Does NOT modify `trainer/research/` (prior phases)
- Adds only: `trainer/automl/` package + `conf/automl/` + docs + tests
- All Phase 15 public APIs remain unchanged
