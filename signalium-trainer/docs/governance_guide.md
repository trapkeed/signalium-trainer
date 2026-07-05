# Model Governance Guide — Phase 18

## Overview

The governance layer ensures that every model deployment is controlled, auditable, and reversible. No model may enter production without completing the full lifecycle and receiving explicit approval.

---

## Lifecycle States

```
Draft → Training → Validating → Candidate → Approved → Champion → Production
                                                    ↓
                                                  Shadow
                                                    ↓
                                                 Archived → Retired
                                                    ↑
                                               Rolled Back
```

| State          | Description                                              |
|----------------|----------------------------------------------------------|
| `draft`        | Model registered, not yet training                       |
| `training`     | Model actively training                                  |
| `validating`   | Model under validation (Phase 15 validation suite)       |
| `candidate`    | Passed validation, eligible for promotion                |
| `approved`     | Approved by designated approver                          |
| `champion`     | Current champion model                                   |
| `production`   | Running in production (non-champion)                     |
| `shadow`       | Running in shadow mode (inference only, no action taken) |
| `archived`     | No longer active, preserved for audit                    |
| `retired`      | Permanently decommissioned                               |
| `rolled_back`  | Emergency or planned rollback                            |

### Terminal states

`archived`, `retired`, and `rolled_back` → `archived` are terminal or near-terminal. Models in these states cannot re-enter production without explicit manual intervention.

---

## Transition Graph

| From           | Valid Next States                                        |
|----------------|----------------------------------------------------------|
| `draft`        | training, archived                                       |
| `training`     | validating, draft, archived                              |
| `validating`   | candidate, draft, archived                               |
| `candidate`    | approved, draft, archived                                |
| `approved`     | champion, production, shadow, candidate, archived        |
| `champion`     | production, shadow, rolled_back, archived                |
| `production`   | champion, shadow, rolled_back, archived                  |
| `shadow`       | production, champion, candidate, archived                |
| `archived`     | retired                                                  |
| `retired`      | *(none — terminal)*                                      |
| `rolled_back`  | archived, candidate                                      |

---

## Approval Workflow

```python
from trainer.mlops import MLOpsEngine

engine = MLOpsEngine.from_config({...})

# Step 1: Train (external process, register when done)
model_id = engine.register_model({
    "name": "signalium_quant",
    "version": "2.0.0",
    ...
})

# Step 2: Begin governance lifecycle
engine.transition(model_id, "training", actor="trainer_service")
engine.transition(model_id, "validating", actor="validator_service")
engine.transition(model_id, "candidate", actor="validator_service")

# Step 3: Approval gate (requires named approver)
engine.approve(model_id, approver="head_quant",
               reason="Accuracy 0.70, Sharpe 1.8, ECE 0.03 — all thresholds met")

# Step 4: Deploy to champion
engine.set_champion(model_id, approver="head_quant")
```

Every step is recorded in the audit trail and lifecycle history.

---

## Rollback

### Planned Rollback

```python
# Planned rollback (not emergency)
engine.rollback(model_id, actor="ops_lead",
                reason="Scheduled rollback for maintenance",
                emergency=False)
```

### Emergency Rollback

```python
# Emergency rollback (skips additional checks)
engine.rollback(model_id, actor="system",
                reason="Confidence drift 0.18 > threshold 0.10",
                emergency=True)
```

### Champion Rollback

```python
# Roll back the current champion
resp = engine.api.rollback_champion(actor="ops_lead",
                                     reason="Drift detected in prod")
```

---

## Governance Configuration

```python
from trainer.mlops.governance import GovernanceConfig

config = GovernanceConfig(
    auto_promote_enabled=False,                  # manual only
    auto_promote_min_validation_score=0.60,      # min accuracy for auto-promote
    auto_promote_max_calibration_error=0.10,     # max ECE for auto-promote
    require_approver=True,                       # named approver required
    allow_emergency_rollback=True,
    max_candidates=5,
    rollback_on_alert=False,                     # auto rollback on critical alert
)
```

---

## Audit Trail

Every governance event is recorded:

```python
# Query all approvals for a model
events = engine.api.get_audit_events(
    event_type="manual_approval",
    model_id=model_id,
)["data"]["events"]

# Full model history
history = engine.api.get_lifecycle_history(model_id)["data"]["history"]

# Full audit report
report = engine.api.get_audit_report()["data"]
```

See [Operational Runbook](operational_runbook.md) for operational procedures.
