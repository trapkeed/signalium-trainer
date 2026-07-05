# Knowledge Graph Guide

## Overview

The Signalium Research Knowledge Graph links every research entity into a queryable directed graph. It provides full lineage navigation, "what used X" queries, and impact analysis for any entity.

---

## Graph Structure

### Node Types

| Type | Description |
|------|-------------|
| `experiment` | A single training run / lineage node |
| `dataset` | A dataset version |
| `model` | A trained model artifact |
| `feature` | An individual feature |
| `validation_run` | A validation execution |
| `deployment` | A production deployment |
| `champion_model` | Current or former production champion |
| `retired_model` | A previously deployed model |

### Edge Types

| Type | Description |
|------|-------------|
| `USES_DATASET` | Experiment → Dataset |
| `HAS_MODEL` | Experiment → Model |
| `USES_FEATURE` | Experiment → Feature |
| `VALIDATED_BY` | Model → ValidationRun |
| `DEPLOYED_AS` | Model → Deployment |
| `PROMOTED_TO` | Model → ChampionModel |
| `RETIRED` | ChampionModel → RetiredModel |
| `PARENT_OF` | Experiment → Child Experiment |
| `DERIVED_FROM` | Model → Parent Model |
| `COMPARED_WITH` | Experiment → Experiment |

---

## Usage

### Direct API

```python
from trainer.research.knowledge_graph import KnowledgeGraph, NodeType, EdgeType

kg = KnowledgeGraph()

# Add nodes
exp_id = kg.add_node(NodeType.EXPERIMENT, "exp_001", {"researcher": "alice"})
ds_id  = kg.add_node(NodeType.DATASET, "ds_v2.3.1", {"sample_count": 1_250_000})
mdl_id = kg.add_node(NodeType.MODEL, "model_001", {"architecture": "transformer"})

# Add edges
kg.add_edge(exp_id, ds_id,  EdgeType.USES_DATASET)
kg.add_edge(exp_id, mdl_id, EdgeType.HAS_MODEL)
```

### Via LineageAPI (auto-registered)

When you use the `LineageAPI`, nodes and edges are automatically registered:

- `register_experiment()` → creates `experiment` node
- `attach_dataset_snapshot()` → creates `dataset` node + `USES_DATASET` edge
- `attach_feature_snapshot()` → creates `feature` nodes + `USES_FEATURE` edges (up to 50)
- `attach_model_snapshot()` → creates `model` node + `HAS_MODEL` edge
- Parent–child experiment links → `PARENT_OF` edges

---

## Graph Queries

### Neighbours

```python
# Outgoing neighbours
datasets_used = kg.neighbours(exp_id, direction="out", edge_type=EdgeType.USES_DATASET)

# Incoming neighbours  
experiments_using_ds = kg.neighbours(ds_id, direction="in", edge_type=EdgeType.USES_DATASET)

# Both directions
all_linked = kg.neighbours(exp_id, direction="both")
```

### Full Reachability

```python
# All experiments reachable from this one (descendants)
all_descendants = kg.reachable(exp_id, direction="out")
```

### Shortest Path

```python
# Find how two nodes are connected
path = kg.shortest_path(source_id, target_id)
# Returns list of node_ids in order
```

### Typed Queries

```python
# All experiments
experiments = kg.nodes_by_type(NodeType.EXPERIMENT)

# All experiments that used a specific dataset
users = kg.experiments_using_dataset(dataset_node_id)

# All experiments that used a specific feature
users = kg.experiments_using_feature(feature_node_id)

# Full model lineage
lineage = kg.model_lineage(model_node_id)
# Returns: parent_models, experiments, validation_runs, deployments
```

---

## Via LineageAPI

```python
# Graph statistics
resp = api.get_graph_report()

# Neighbours
resp = api.get_node_neighbours(node_id, direction="out", edge_type="USES_DATASET")

# Model lineage
resp = api.get_model_lineage(model_node_id)

# Shortest path
resp = api.get_shortest_path(source_id, target_id)

# Add custom nodes/edges
resp = api.add_graph_node("deployment", "prod_deployment_001")
resp = api.add_graph_edge(model_node_id, deployment_node_id, "DEPLOYED_AS")

# Full export
resp = api.export_knowledge_graph()
```

---

## Knowledge Graph Dashboard

The `knowledge_graph` dashboard provides the full graph data formatted for visualisation:

```python
resp = api.get_dashboard("knowledge_graph")
data = resp["data"]["data"]

# data contains:
# - nodes: list of all nodes with type, label, attributes
# - edges: list of all edges with source, target, type
# - adjacency: simplified source/target/type list
# - node_type_distribution: count per type
# - edge_type_distribution: count per type
# - champion_models, retired_models, deployments: counts
```

---

## Serialisation and Persistence

```python
# Save to dict (JSON-serialisable)
d = kg.to_dict()

# Restore from dict
kg2 = KnowledgeGraph.from_dict(d)

# Persist via LineageDatabase
from trainer.research.lineage_database import LineageDatabase
db = LineageDatabase("experiments/research/lineage.db")
db.save_knowledge_graph(kg.to_dict())

# Reload
blob = db.load_knowledge_graph()
kg3 = KnowledgeGraph.from_dict(blob)
```
