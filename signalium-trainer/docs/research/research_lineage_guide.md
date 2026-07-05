# Research Lineage Guide

## Overview

Phase 19.1 introduces **Experiment Lineage and Reproducibility** — the permanent research memory of Signalium Quant. Every experiment, dataset, model, feature set, and validation run is tracked with full lineage, enabling any historical experiment to be reproduced with identical inputs and identical configuration.

---

## Core Concepts

### Experiment Lineage Tree

Every experiment is a node in a directed acyclic tree. Each node records:

| Field | Description |
|-------|-------------|
| `node_id` | Unique UUID |
| `name` | Human-readable experiment name |
| `researcher` | Who ran the experiment |
| `purpose` | Why the experiment was run |
| `parent_id` | Parent experiment this was derived from |
| `child_ids` | Experiments derived from this one |
| `branch` | Git-style research branch |
| `project` | Research project grouping |
| `status` | `planned`, `running`, `succeeded`, `failed`, `abandoned`, `archived` |

### Reproducibility Snapshot

Each node captures the complete execution environment:

```
Git Commit Hash    ← exact code version
Git Branch         ← research branch
Python Version     ← runtime version
PyTorch Version    ← framework version
CUDA Version       ← GPU driver version
Random Seed        ← deterministic training
OS / CPU / GPU     ← hardware identity
RAM                ← memory capacity
Docker Image       ← container image (if used)
Env Vars           ← non-sensitive environment variables
```

**Reproducibility Score:** 0–100% based on how much environment information is captured. Grade A (≥90%) means the experiment is fully reproducible.

### Four Snapshot Types

Each experiment node can carry four immutable point-in-time snapshots:

1. **DatasetSnapshot** — version, hash, instruments, timeframes, coverage, sample count, quality score
2. **FeatureSnapshot** — feature names, parameters, importance, statistics, drift status, lifecycle status
3. **ModelSnapshot** — architecture, backbone, optimizer, learning rate, loss functions, checkpoint hash
4. **ValidationSnapshot** — walk-forward results, calibration, trading metrics, stress test results, promotion decision

---

## Usage

### Registering an Experiment

```python
from trainer.research.lineage_api import LineageAPI
from trainer.research.lineage import LineageTree
from trainer.research.timeline import ResearchTimeline
from trainer.research.cost_tracker import CostTracker
from trainer.research.knowledge_graph import KnowledgeGraph

api = LineageAPI(
    tree=LineageTree(),
    timeline=ResearchTimeline(),
    cost_tracker=CostTracker(),
    knowledge_graph=KnowledgeGraph(),
)

resp = api.register_experiment(
    "transformer_4h_v2",
    researcher="alice",
    purpose="Test 4-head attention sweep on hourly data",
    branch="feature/attention-sweep",
    project="q4-2026",
    parent_id=parent_node_id,   # optional
    random_seed=42,
    capture_environment=True,   # captures git, python, pytorch, cuda, hardware
)
node_id = resp["data"]["node_id"]
```

### Attaching Snapshots

```python
from trainer.research.snapshots import DatasetSnapshot, ModelSnapshot

# Dataset snapshot
api.attach_dataset_snapshot(node_id, {
    "dataset_version": "v2.3.1",
    "dataset_hash": "sha256:abc123",
    "instruments": ["ES", "NQ", "CL"],
    "sample_count": 1_250_000,
    "data_quality_score": 0.97,
})

# Model snapshot
api.attach_model_snapshot(node_id, {
    "architecture": "transformer",
    "backbone": "transformer_4h",
    "parameter_count": 2_500_000,
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "checkpoint_hash": "sha256:ckpt_001",
})
```

### Completing and Recording Cost

```python
api.complete_experiment(node_id, status="succeeded", accuracy=0.692)

api.record_cost(node_id,
    gpu_hours=4.5,
    cpu_hours=2.0,
    storage_gb_months=5.0,
    project="q4-2026",
    researcher="alice",
)
```

---

## Lineage Navigation

```python
# Full lineage report for one experiment
resp = api.get_lineage_report(node_id)

# All ancestors up to root
resp = api.get_ancestors(node_id)

# All descendants
resp = api.get_descendants(node_id)

# Filter by branch / project / researcher / status
resp = api.get_by_branch("feature/attention-sweep")
resp = api.get_by_project("q4-2026")
resp = api.get_by_researcher("alice")
resp = api.get_by_status("succeeded")
```

---

## Experiment Differencing

Compare any two experiments to automatically identify what changed:

```python
# Full experiment diff
resp = api.compare_experiments(node_id_a, node_id_b)

# Model-only diff
resp = api.compare_models(node_id_a, node_id_b)

# Dataset-only diff
resp = api.compare_datasets(node_id_a, node_id_b)

# Feature set-only diff
resp = api.compare_features(node_id_a, node_id_b)
```

Diff output includes:
- Changed parameters (before → after, with delta)
- Performance differences
- Feature additions/removals
- Validation metric changes
- Whether performance improved

---

## Reproducibility Score

| Grade | Score | Description |
|-------|-------|-------------|
| A | ≥ 90% | Fully reproducible. Git hash + seed + all snapshots |
| B | 75–89% | Very reproducible. Minor gaps |
| C | 55–74% | Partially reproducible |
| D | 35–54% | Significant gaps |
| F | < 35% | Not reproducible |

To achieve Grade A:
- Always call `capture_environment=True` (default)
- Always set `random_seed`
- Attach all four snapshot types
- Ensure git repo is clean before training

---

## Persistence

All lineage data is persisted via `LineageDatabase`:

```python
from trainer.research.lineage_database import LineageDatabase

db = LineageDatabase("experiments/research/lineage.db")
db.save_node(node.to_dict())
db.save_environment(node_id, env.to_dict())
db.save_cost(cost.to_dict())
db.save_timeline_event(event.to_dict())
db.close()
```

The database uses SQLite with WAL journal mode for thread safety. No external dependencies required.
