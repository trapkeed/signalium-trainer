# Developer Guide — Phase 21.6 Decision Memory

## Architecture

```
trainer/memory/
├── decision_types.py        Enums: DecisionType, OutcomeType, QualityFactor, MarketRegime
├── decision_models.py       Dataclasses: DecisionRecord, Snapshot, Reasoning, Outcome, Score
├── decision_memory.py       In-process registry — stores/filters/retrieves records
├── outcome_tracker.py       Aggregate outcome analytics across decision sets
├── quality_engine.py        10-factor quality scoring
├── counterfactual_engine.py 6 counterfactual scenarios per decision
├── pattern_discovery.py     Auto-discovers patterns (regime/session/instrument/timing)
├── self_evaluation.py       Periodic AI self-assessments
├── knowledge_base.py        Searchable institutional knowledge base
├── decision_lineage.py      Decision → Model → Dataset → Git lineage graph
├── decision_benchmarking.py Benchmark vs historical champions and baselines
├── memory_search.py         Structured institutional search
├── memory_database.py       SQLite persistence (9 tables)
├── memory_dashboard.py      8 dashboard views
├── memory_api.py            16-endpoint unified API
└── __init__.py              Public exports
```

## Module Load Order (for spec_from_file_location tests)

```
decision_types → decision_models → decision_memory → outcome_tracker →
quality_engine → counterfactual_engine → pattern_discovery → self_evaluation →
knowledge_base → decision_lineage → decision_benchmarking → memory_search →
memory_database → memory_dashboard → memory_api
```

## Key Design Decisions

### 1. In-process + Persistent

`DecisionMemory` holds all records in a Python dict (fast access). `MemoryDatabase` provides SQLite persistence. They are kept separate — the API layer syncs them via `persist=True` flag.

### 2. No torch, no numpy, no pandas

Every module uses Python stdlib + loguru only. All numerical computations are pure Python floats.

### 3. SQLite :memory: pattern

For in-process SQLite (used in tests), a persistent connection is stored at init:

```python
self._persistent_conn: Optional[sqlite3.Connection] = None
```

Never call `sqlite3.connect(":memory:")` inside a context manager that discards the connection — every new `:memory:` connection creates an empty database.

### 4. Structured reasoning only

`DecisionReasoning` enforces structured lists. No string blob storage. No chain-of-thought.

### 5. Outcome-before-scoring

`QualityEngine` can score without an outcome (using forecast probabilities) but produces best results when `record.outcome` is set. `CounterfactualEngine` requires an outcome — returns `[]` otherwise.

## Testing

```bash
cd signalium-trainer
python3 -m unittest tests/unit/test_memory.py
```

Run all phases together:

```bash
python3 -m unittest tests/unit/test_portfolio.py tests/unit/test_execution.py tests/unit/test_lifecycle.py tests/unit/test_memory.py
```

Tests use `spec_from_file_location` — never `import trainer.memory.*` directly (would pull in torch via other modules).

## API Quick Reference

| Endpoint | Description |
|---|---|
| `record_decision()` | Create and store a new decision record |
| `get_decision(id)` | Retrieve a decision by ID |
| `record_outcome()` | Attach outcome to a decision |
| `score_decision(id)` | Score with 10-factor quality engine |
| `score_all()` | Batch score all unscored decisions |
| `analyse_counterfactuals(id)` | Generate counterfactual scenarios |
| `summarise_outcomes()` | Aggregate outcome analytics |
| `discover_patterns()` | Auto-discover performance patterns |
| `self_evaluate()` | Generate self-assessment |
| `search_decisions(query)` | Structured search over memory |
| `knowledge_search()` | Search knowledge base |
| `index_decision_in_knowledge_base(id)` | Auto-index decision |
| `build_lineage(id, ...)` | Create decision lineage |
| `benchmark()` | Benchmark vs baselines |
| `get_dashboard(name)` | Get a dashboard view |
| `system_stats()` | Aggregate system statistics |

## Extending the Quality Engine

Add a new quality factor:

1. Add to `QualityFactor` enum in `decision_types.py`
2. Add weight to `QUALITY_WEIGHTS` (ensure sum = 1.0)
3. Add scoring logic in `quality_engine.py` `score()` method
4. Add sub_score computation with `_clamp()` bounds check
5. Update `conf/memory/memory.yaml` factors section

## Adding a New Dashboard View

1. Add method to `MemoryDashboard` in `memory_dashboard.py`
2. Register in `all_dashboards()` dict
3. Register in `MemoryAPI.get_dashboard()` fns dict
4. Add to `dashboards.views` list in `memory.yaml`

## Decision Lineage

A complete lineage traces 8 levels:

```
Decision → Forecast → Model → Dataset → FeatureSet → TrainingRun → Experiment → GitCommit
```

```python
lin = api.build_lineage(
    decision_id   = record.decision_id,
    model_version = "v3.2",
    dataset_name  = "EURUSD_2Y_2025",
    feature_set   = "momentum_trend_v4",
    training_run  = "run_2025_03_01",
    experiment    = "exp_attention_heads",
    git_commit    = "a1b2c3d4",
)
print(f"Lineage depth: {lin['depth']}")  # 8
```

## Common Patterns

### Score all decisions after batch outcome recording

```python
for decision_id in decision_ids:
    api.record_outcome(decision_id, OutcomeType.HIT_TP, ...)

result = api.score_all()
print(f"Scored {result['scored']} decisions, avg quality: {result['avg_quality']:.2f}")
```

### Monthly quality report

```python
assessment = api.self_evaluate(period="MONTHLY")
print(f"Grade: {assessment['grade']}")
for opp in assessment["opportunities"]:
    print(f"  Opportunity: {opp}")
```

### Benchmark against previous quarter

```python
# Create Q1 baseline from records
q1_records = api.memory.filter(since=q1_start_ts)
api.benchmarking.create_baseline_from_records("Q1-2025", q1_records, "Q1-2025")

# Benchmark Q2 vs Q1
report = api.benchmark(baselines=["Q1-2025"])
print(f"Q2 vs Q1: {report['overall_verdict']}")
```
