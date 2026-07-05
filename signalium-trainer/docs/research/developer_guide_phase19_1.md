# Developer Guide — Phase 19.1

## Overview

Phase 19.1 adds **Experiment Lineage and Reproducibility** to the Signalium Research Intelligence Platform. This guide covers the architecture, module map, integration patterns, and extension points.

---

## Module Map

```
trainer/research/
├── reproducibility.py      ← EnvironmentSnapshot — captures git, python, hardware
├── snapshots.py            ← DatasetSnapshot, FeatureSnapshot, ModelSnapshot,
│                              ValidationSnapshot
├── lineage.py              ← LineageNode, LineageTree, ExperimentStatus
├── differencing.py         ← DatasetDiff, FeatureDiff, ModelDiff,
│                              ValidationDiff, ExperimentDiffer
├── knowledge_graph.py      ← KnowledgeGraph, GraphNode, GraphEdge,
│                              NodeType, EdgeType
├── timeline.py             ← ResearchTimeline, TimelineEvent, TimelineEventType
├── cost_tracker.py         ← CostTracker, ExperimentCost
├── dashboard.py            ← DashboardBuilder (7 dashboard types)
├── export.py               ← ResearchExporter (5 export types)
├── lineage_database.py     ← LineageDatabase (SQLite, 10 tables)
└── lineage_api.py          ← LineageAPI (structured endpoint layer)
```

### Pre-existing Phase 19 modules (unchanged)

```
observatory.py, market_evolution.py, feature_lifecycle.py,
label_quality.py, dataset_evolution.py, model_health_evolution.py,
retraining_advisor.py, experiment_engine.py, knowledge_base.py,
research_assistant.py, database.py, api.py, visualization.py, engine.py
```

---

## Architecture

```
LineageAPI
    ├── LineageTree           (in-memory lineage graph)
    │     └── LineageNode[]
    │           ├── EnvironmentSnapshot
    │           ├── DatasetSnapshot
    │           ├── FeatureSnapshot
    │           ├── ModelSnapshot
    │           └── ValidationSnapshot
    ├── ResearchTimeline      (chronological event log)
    ├── CostTracker           (per-experiment cost accumulator)
    ├── KnowledgeGraph        (entity relationship graph)
    ├── ExperimentDiffer      (experiment comparison)
    ├── DashboardBuilder      (7 dashboard payloads)
    ├── ResearchExporter      (5 export types)
    └── LineageDatabase       (SQLite persistence)
```

---

## Data Flow

### Experiment Registration

```
register_experiment()
  ├── EnvironmentSnapshot.capture()  → auto-captures git/python/hardware
  ├── LineageTree.register()          → creates LineageNode
  ├── ResearchTimeline.record()       → logs experiment_started event
  └── KnowledgeGraph.add_node()       → creates experiment node
        └── if parent_id: add PARENT_OF edge
```

### Snapshot Attachment

```
attach_dataset_snapshot()
  ├── DatasetSnapshot.from_dict()
  ├── LineageTree.attach_dataset_snapshot()
  └── KnowledgeGraph.register_dataset_node()
        └── add USES_DATASET edge from experiment node
```

### Cost Recording

```
record_cost()
  ├── CostTracker.record()            → computes USD from rates
  ├── LineageTree.record_cost()       → stores on node
  └── ResearchTimeline.record_cost()  → logs cost_recorded event
```

---

## Public API Reference

### `LineageAPI` — primary interface

All methods return the standard envelope:
```python
{
    "status": "ok" | "error",
    "code": "lineage.xxx",
    "message": "ok",
    "data": { ... },
    "timestamp": "2026-07-05T12:00:00+00:00"
}
```

| Method | Description |
|--------|-------------|
| `register_experiment(name, ...)` | Register new experiment, capture environment |
| `start_experiment(node_id)` | Mark as running |
| `complete_experiment(node_id, status)` | Mark as succeeded/failed/abandoned |
| `get_experiment(node_id)` | Get full node data |
| `get_lineage_report(node_id)` | Full lineage report |
| `get_tree_summary()` | Platform-wide tree summary |
| `get_ancestors(node_id)` | All ancestor nodes |
| `get_descendants(node_id)` | All descendant nodes |
| `get_by_branch(branch)` | Filter by branch |
| `get_by_project(project)` | Filter by project |
| `get_by_status(status)` | Filter by status |
| `attach_environment(node_id, env_dict)` | Attach environment snapshot |
| `attach_dataset_snapshot(node_id, ds)` | Attach dataset snapshot |
| `attach_feature_snapshot(node_id, fs)` | Attach feature snapshot |
| `attach_model_snapshot(node_id, ms)` | Attach model snapshot |
| `attach_validation_snapshot(node_id, vs)` | Attach validation snapshot |
| `record_cost(entity_id, ...)` | Record cost breakdown |
| `record_cost_estimate(entity_id, seconds)` | Estimate cost from duration |
| `compare_experiments(a, b)` | Full experiment diff |
| `compare_models(a, b)` | Model-only diff |
| `compare_datasets(a, b)` | Dataset-only diff |
| `compare_features(a, b)` | Feature-only diff |
| `get_dashboard(type)` | Build dashboard payload |
| `get_all_dashboards()` | Build all 7 dashboards |
| `export_experiment_card(node_id)` | One-page experiment card |
| `export_lineage_report(node_id)` | Full lineage report |
| `export_research_report()` | Platform-wide report |
| `export_experiment_tree()` | Hierarchical tree export |
| `export_knowledge_graph()` | Full graph export |
| `get_cost_report()` | Platform cost summary |
| `get_timeline_report()` | Timeline summary |
| `get_stats()` | Platform-wide stats |

---

## Key Design Decisions

### No automatic actions

- `RetrainingAdvisor(auto_retrain=True)` raises `ValueError`
- `promotion_decision` in `ValidationSnapshot` is informational only
- No experiment is automatically promoted or deployed

### No new runtime dependencies

All Phase 19.1 modules use only Python stdlib:
- `sqlite3`, `threading`, `uuid`, `json`, `pathlib`
- `subprocess`, `platform`, `sys`, `os`
- `dataclasses`, `collections`, `typing`
- `loguru` (already a project dependency)

### Backward compatibility

Phase 19.1 adds **only new files**. No Phase 1–18 module is modified.

### Thread safety

`LineageDatabase` uses `threading.local()` for per-thread connections and SQLite WAL journal mode.

---

## Extension Points

### Custom node types

```python
class NodeType:
    MY_ENTITY = "my_entity"

nid = kg.add_node(NodeType.MY_ENTITY, "my_label", {"key": "val"})
```

### Custom edge types

```python
class EdgeType:
    MY_RELATION = "MY_RELATION"

kg.add_edge(source_id, target_id, EdgeType.MY_RELATION)
```

### Custom timeline events

```python
from trainer.research.timeline import TimelineEventType

api.record_timeline_event(
    "custom_event_type",
    entity_id="exp_001",
    description="Custom event recorded",
    attributes={"custom_key": "custom_value"},
)
```

### Custom dashboard

Extend `DashboardBuilder` and add to `DASHBOARD_TYPES`:

```python
class MyDashboardBuilder(DashboardBuilder):
    DASHBOARD_TYPES = DashboardBuilder.DASHBOARD_TYPES + ["my_dashboard"]

    def _build_my_dashboard(self) -> Dict[str, Any]:
        return {"my_data": "..."}
```

---

## Testing

All Phase 19.1 classes are covered in:

```
tests/unit/test_research_lineage.py
```

Run with:

```bash
pytest tests/unit/test_research_lineage.py -v
```

The test file covers 60+ test cases across all 11 new modules.
