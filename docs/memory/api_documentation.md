# API Documentation — Phase 21.6 Decision Memory

## MemoryAPI — Unified Interface

```python
from trainer.memory import MemoryAPI
api = MemoryAPI(db_path=":memory:")   # use file path in production
```

---

## Endpoint 1 — record_decision

**Records a new Signalium decision.**

```python
record = api.record_decision(
    decision_type  : DecisionType,
    instrument     : str,
    snapshot       : DecisionSnapshot,
    reasoning      : DecisionReasoning,
    tags           : Optional[List[str]] = None,
    trade_id       : str = "",
    lineage_id     : str = "",
    persist        : bool = True,
) -> DecisionRecord
```

Returns the full `DecisionRecord` with a UUID `decision_id`.

---

## Endpoint 2 — get_decision

**Retrieve a decision by ID.**

```python
d = api.get_decision(decision_id: str) -> Optional[Dict]
```

Returns `None` if not found. Returns `record.to_dict()` if found.

---

## Endpoint 3 — record_outcome

**Attach an outcome to a completed decision.**

```python
ok = api.record_outcome(
    decision_id        : str,
    outcome_type       : OutcomeType,
    actual_pnl         : float = 0.0,
    actual_pnl_pct     : float = 0.0,
    hit_tp             : bool = False,
    hit_sl             : bool = False,
    holding_time_h     : float = 0.0,
    direction_correct  : Optional[bool] = None,
    market_event       : str = "",
    notes              : str = "",
    persist            : bool = True,
) -> bool
```

Returns `True` if the decision was found and outcome attached.

---

## Endpoint 4 — score_decision

**Score a decision on 10 quality factors.**

```python
score = api.score_decision(decision_id: str) -> Optional[Dict]
# Returns: {"overall": 0.72, "grade": "GOOD", "sub_scores": {...}, "notes": [...]}
```

---

## Endpoint 5 — score_all

**Batch score all decisions.**

```python
result = api.score_all() -> Dict
# Returns: {"scored": 150, "avg_quality": 0.70}
```

---

## Endpoint 6 — analyse_counterfactuals

**Generate counterfactual scenarios for a completed decision.**

```python
scenarios = api.analyse_counterfactuals(decision_id: str) -> List[Dict]
# Requires outcome to be recorded first.
# Returns list of CounterfactualResult dicts.
```

---

## Endpoint 7 — summarise_outcomes

**Aggregate outcome statistics.**

```python
summary = api.summarise_outcomes(
    instrument     : Optional[str] = None,
    decision_type  : Optional[DecisionType] = None,
) -> Dict
# Returns OutcomeSummary.to_dict() with win_rate, avg_pnl, by_session, by_regime, etc.
```

---

## Endpoint 8 — discover_patterns

**Automatically discover performance patterns.**

```python
report = api.discover_patterns() -> Dict
# Returns PatternReport.to_dict() with patterns list, top_positive, top_negative, summary.
```

---

## Endpoint 9 — self_evaluate

**Generate a periodic self-assessment.**

```python
assessment = api.self_evaluate(period: str = "CURRENT") -> Dict
# Returns SelfAssessment.to_dict() with grade, strengths, weaknesses, etc.
```

---

## Endpoint 10 — search_decisions

**Structured institutional search.**

```python
result = api.search_decisions(query: SearchQuery) -> Dict
```

**SearchQuery fields:**

| Field | Type | Description |
|---|---|---|
| text | str | Free-text search across instrument, tags, reasoning |
| instrument | Optional[str] | Filter by instrument |
| decision_type | Optional[str] | Filter by DecisionType value |
| outcome_type | Optional[str] | Filter by OutcomeType value |
| regime | Optional[str] | Filter by market regime |
| session | Optional[str] | Filter by session |
| grade | Optional[str] | Filter by quality grade |
| min_confidence | float | Minimum forecast confidence (0–1) |
| max_confidence | float | Maximum forecast confidence (0–1) |
| min_pnl | Optional[float] | Minimum actual PnL |
| max_pnl | Optional[float] | Maximum actual PnL |
| direction_correct | Optional[bool] | Filter by direction correctness |
| hit_tp | Optional[bool] | Filter by TP hit |
| hit_sl | Optional[bool] | Filter by SL hit |
| profitable | Optional[bool] | Filter by PnL > 0 |
| limit | int | Maximum results (default 100) |
| sort_by | str | timestamp \| confidence \| pnl \| quality |

**Example queries:**

```python
from trainer.memory import SearchQuery

# Incorrect BUY decisions on EURUSD
result = api.search_decisions(SearchQuery(
    instrument        = "EURUSD",
    decision_type     = "BUY",
    direction_correct = False,
))

# High-confidence decisions (≥95%)
result = api.search_decisions(SearchQuery(
    min_confidence = 0.95,
))

# Profitable decisions sorted by PnL
result = api.search_decisions(SearchQuery(
    profitable = True,
    sort_by    = "pnl",
    limit      = 50,
))

# Failed breakout trades
result = api.search_decisions(SearchQuery(
    regime  = "BREAKOUT",
    hit_sl  = True,
))

# Successful pullback entries
result = api.search_decisions(SearchQuery(
    text     = "pullback",
    hit_tp   = True,
    profitable = True,
))
```

---

## Endpoint 11 — knowledge_search

**Search the institutional knowledge base.**

```python
results = api.knowledge_search(
    query       : str = "",
    entry_type  : Optional[str] = None,
    tags        : Optional[List[str]] = None,
    limit       : int = 50,
) -> List[Dict]
```

---

## Endpoint 12 — index_decision_in_knowledge_base

**Auto-index a decision into the knowledge base.**

```python
entry = api.index_decision_in_knowledge_base(decision_id: str) -> Optional[Dict]
```

---

## Endpoint 13 — build_lineage

**Build a decision lineage graph.**

```python
lin = api.build_lineage(
    decision_id    : str,
    model_version  : str = "",
    dataset_name   : str = "",
    feature_set    : str = "",
    training_run   : str = "",
    experiment     : str = "",
    git_commit     : str = "",
) -> Optional[Dict]
```

Returns 8-level lineage: Decision → Forecast → Model → Dataset → FeatureSet → TrainingRun → Experiment → GitCommit.

---

## Endpoint 14 — benchmark

**Benchmark current decisions against baselines.**

```python
report = api.benchmark(
    baselines : Optional[List[str]] = None,
) -> Dict
# Returns BenchmarkReport with verdict: BETTER | WORSE | SIMILAR
```

Default baselines: RANDOM, GOOD_TRADER, EXCELLENT_TRADER.

---

## Endpoint 15 — get_dashboard

**Get a dashboard view.**

```python
data = api.get_dashboard(name: str = "all") -> Dict
```

Available views:

| Name | Description |
|---|---|
| `all` | All 8 dashboards combined |
| `decision_timeline` | Decisions bucketed by day/hour/month |
| `decision_tree` | Instrument × decision type × outcome tree |
| `quality_distribution` | Grade distribution and score histogram |
| `failure_analysis` | Failure breakdown by regime/session/type |
| `counterfactual_explorer` | Alternative scenario summary |
| `knowledge_graph` | Decision ↔ instrument ↔ regime node/edge graph |
| `learning_timeline` | Quality and win rate over time |
| `decision_heatmaps` | Win rate heatmap: session × regime |

---

## Endpoint 16 — system_stats

**System-wide statistics.**

```python
stats = api.system_stats() -> Dict
# {
#   "memory":   {"total_decisions": 150, "with_outcome": 120, ...},
#   "database": {"decisions": 150, "outcomes": 120, "scores": 90, "counterfactuals": 450},
#   "knowledge": {"total_entries": 180, "by_type": {...}},
#   "lineage":  {"total_lineages": 150, "avg_depth": 7.8},
#   "baselines": 3
# }
```
