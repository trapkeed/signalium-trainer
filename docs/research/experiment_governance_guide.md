# Experiment Governance Guide

## Overview

The Signalium Quant Experiment Governance Framework ensures that every research experiment is conducted with full institutional accountability, documented purpose, traceable lineage, and reproducible results.

Governance is non-blocking: all recommendations are advisory. No experiment is automatically rejected, and no automatic retraining or deployment occurs.

---

## Governance Principles

1. **Every experiment has a purpose.** Purpose must be stated at registration time.
2. **Every experiment has an owner.** Researcher identity is required.
3. **Every experiment is traceable.** Parent lineage is captured at registration.
4. **Every experiment is reproducible.** Environment is captured automatically.
5. **Every experiment is costed.** GPU/CPU/storage estimates are recorded.
6. **No automatic actions.** All recommendations require human review.

---

## Experiment Lifecycle

```
PLANNED ──start()──▶ RUNNING ──complete()──▶ SUCCEEDED
                              └──complete()──▶ FAILED
                              └──complete()──▶ ABANDONED
SUCCEEDED / FAILED ──archive()──▶ ARCHIVED
```

### State Transitions

| Action | API Call | Description |
|--------|----------|-------------|
| Register | `api.register_experiment(name, ...)` | Creates node in PLANNED state |
| Start | `api.start_experiment(node_id)` | Marks as RUNNING |
| Complete | `api.complete_experiment(node_id, status)` | Marks as SUCCEEDED/FAILED/ABANDONED |
| Archive | Set status to `archived` | Long-term storage |

---

## Research Branches and Projects

Experiments are organised into **branches** (like git branches) and **projects**:

```python
# Architecture sweep on feature/attention-sweep branch
api.register_experiment("transformer_8h",
    branch="feature/attention-sweep",
    project="q4-2026-architecture",
)

# Filter by context
experiments = api.get_by_branch("feature/attention-sweep")
experiments = api.get_by_project("q4-2026-architecture")
```

### Recommended Branch Naming

| Pattern | Purpose |
|---------|---------|
| `main` | Stable, champion-grade experiments |
| `feature/<name>` | Architecture or feature engineering sweeps |
| `experiment/<name>` | Exploratory investigations |
| `hotfix/<name>` | Urgent production fixes |
| `ablation/<name>` | Ablation studies |

---

## Research Cost Governance

Every experiment should have a cost estimate recorded:

```python
api.record_cost(node_id,
    gpu_hours=4.5,
    cpu_hours=2.0,
    storage_gb_months=5.0,
    validation_gpu_hours=0.5,
    project="q4-2026",
    researcher="alice",
)
```

Or estimate from training duration:

```python
api.record_cost_estimate(node_id,
    training_seconds=14400.0,  # 4 hours
    gpu_count=2,
    cpu_cores=16,
    storage_gb=10.0,
)
```

Cost reports are available per experiment, per project, per researcher, and platform-wide.

---

## Promotion Governance

Promotion decisions come from the ValidationSnapshot attached to each experiment:

```python
api.attach_validation_snapshot(node_id, {
    "walk_forward_accuracy": 0.692,
    "sharpe_ratio": 1.82,
    "promotion_decision": "promote",
    "promotion_rationale": "Exceeds all Tier 1 thresholds.",
})
```

**No automatic promotion occurs.** The `promotion_decision` field is informational. A human reviewer must approve any champion change.

---

## Experiment Differencing for Peer Review

Before promoting an experiment, compare it to the current champion:

```python
resp = api.compare_experiments(candidate_node_id, champion_node_id)
diff = resp["data"]

print(f"Total changes: {diff['total_changes']}")
print(f"Performance improved: {diff['performance_improved']}")
print(f"Architecture changed: {diff['model_diff']['architecture_changed']}")
print(f"Cost delta: ${diff['cost_diff_usd']:+.2f}")
```

This automatically identifies:
- Changed hyperparameters
- Architecture differences
- Feature set changes
- Validation metric deltas

---

## Export and Reporting

### Experiment Card (per-experiment summary)

```python
resp = api.export_experiment_card(node_id)
md_resp = api.export_experiment_card(node_id, as_markdown=True)
```

### Research Report (platform-wide)

```python
resp = api.export_research_report()
md_resp = api.export_research_report(as_markdown=True)
```

### Full Experiment Tree

```python
resp = api.export_experiment_tree()
```

---

## Governance Checklist

Before closing any experiment:

- [ ] `purpose` documented at registration
- [ ] `researcher` identified
- [ ] `branch` and `project` set
- [ ] `EnvironmentSnapshot` captured (via `capture_environment=True`)
- [ ] `random_seed` set
- [ ] `DatasetSnapshot` attached
- [ ] `FeatureSnapshot` attached
- [ ] `ModelSnapshot` attached with checkpoint hash
- [ ] `ValidationSnapshot` attached
- [ ] Cost recorded
- [ ] Status set to `succeeded` or `failed`
- [ ] Experiment card exported and archived

A complete record yields a Reproducibility Grade of A (≥90%).
