# Deployment Guide — Phase 18

## Overview

Deployment governance in Signalium ensures that every production deployment is controlled, traceable, and reversible. All deployments are recorded and included in the audit trail.

---

## Deployment Targets

| Target       | Description                                              |
|--------------|----------------------------------------------------------|
| `champion`   | Primary production model — serves all live trades        |
| `production` | Production model (non-champion, canary, or AB)           |
| `shadow`     | Shadow mode — receives predictions but takes no action   |

---

## Deployment Types

| Type               | Description                                              |
|--------------------|----------------------------------------------------------|
| `manual`           | Explicit human-initiated deployment                      |
| `automatic`        | Auto-promoted when thresholds are met (if enabled)       |
| `scheduled`        | Time-triggered deployment                                |
| `rollback`         | Planned rollback                                         |
| `emergency_rollback` | Immediate rollback, skips additional checks            |

---

## Manual Deployment Workflow

```python
from trainer.mlops import MLOpsEngine

engine = MLOpsEngine.from_config({...})

# Register and advance through lifecycle
model_id = engine.register_model({...})
engine.transition(model_id, "training", actor="trainer_service")
engine.transition(model_id, "validating", actor="validator_service")
engine.transition(model_id, "candidate", actor="validator_service")
engine.approve(model_id, approver="head_quant",
               reason="Validated against institutional criteria")

# Deploy as champion
resp = engine.set_champion(model_id, approver="head_quant",
                            reason="Production-ready v2.1.0")
print(resp["status"])  # "ok"

# Or deploy as production (non-champion)
resp = engine.deploy(model_id, target="production",
                      approver="ops_lead",
                      reason="Canary deployment — 10% traffic")
```

---

## Champion/Challenger Workflow

```python
# Current champion is model_01

# Add model_02 as challenger
engine.add_challenger(model_id_02, reason="New transformer backbone")

# Compare metrics
comparison = engine.compare_models(model_id_01, model_id_02,
    metrics_a={"accuracy": 0.68, "calibration_error": 0.04},
    metrics_b={"accuracy": 0.72, "calibration_error": 0.03},
)
print(comparison["data"]["overall_winner_name"])

# Promote challenger to champion
engine.api.promote_challenger(model_id_02, approver="head_quant",
                               reason="Outperforms champion on all key metrics")
```

---

## Rollback Procedures

### Standard Rollback

```python
engine.rollback(model_id, actor="ops_lead",
                reason="Scheduled rollback", emergency=False)
```

### Emergency Rollback

```python
# Rollback a deployed model
engine.rollback(model_id, actor="system",
                reason="Confidence drift 0.18 > threshold 0.10",
                emergency=True)

# Rollback the champion
engine.api.rollback_champion(actor="ops_lead",
                              reason="Latency degradation detected")
```

---

## Automatic Promotion (Optional)

When `governance_auto_promote=True`:

```python
from trainer.mlops.governance import GovernanceConfig

config = GovernanceConfig(
    auto_promote_enabled=True,
    auto_promote_min_validation_score=0.60,    # accuracy threshold
    auto_promote_max_calibration_error=0.10,   # ECE threshold
)
# Auto-promotion fires when an APPROVED model meets both thresholds
result = engine.governance.auto_promote_if_eligible(model_id)
if result:
    print(f"Auto-promoted: {result.deployment_id}")
```

Automatic promotion is **disabled by default**.

---

## Deployment History and Audit

```python
# All deployments
resp = engine.api.list_deployments()
print(resp["data"]["deployments"])

# Specific model
resp = engine.api.list_deployments(model_id=model_id)

# Active deployments
resp = engine.api.active_deployments()

# Full audit trail
report = engine.export_deployment_report(
    path="exports/deployment_report.json"
)
```

---

## Database Schema — Deployments Table

| Column           | Description                         |
|------------------|-------------------------------------|
| `deployment_id`  | UUID                                |
| `model_id`       | Deployed model                      |
| `target`         | champion / production / shadow      |
| `deploy_type`    | manual / automatic / rollback       |
| `approver`       | Named approver                      |
| `actor`          | Who executed                        |
| `reason`         | Deployment rationale                |
| `status`         | active / rolled_back / superseded   |
| `from_state`     | Lifecycle state before deployment   |
| `to_state`       | Lifecycle state after deployment    |
| `deployed_at`    | UTC timestamp                       |
| `rolled_back_at` | Rollback timestamp (if any)         |
| `rollback_reason`| Rollback reason (if any)            |

---

## Governance Configuration Reference

```python
from trainer.mlops.governance import GovernanceConfig

GovernanceConfig(
    auto_promote_enabled=False,                  # default: manual only
    auto_promote_min_validation_score=0.60,
    auto_promote_max_calibration_error=0.10,
    require_approver=True,                       # named approver required
    allow_emergency_rollback=True,
    max_candidates=5,
    rollback_on_alert=False,                     # auto rollback on CRITICAL
)
```
