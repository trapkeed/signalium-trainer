# Developer Guide — Phase 18: Institutional MLOps Platform

## Package Structure

```
trainer/mlops/                      # 14 modules
├── __init__.py                     # Public API (50+ symbols)
├── registry.py                     # LifecycleState, ApprovalRecord,
│                                   #   ModelRecord (20 fields), ModelRegistry
├── lifecycle.py                    # LifecycleTransition, LifecycleError,
│                                   #   LifecycleManager (11 states + graph)
├── governance.py                   # DeploymentRecord, GovernanceConfig,
│                                   #   DeploymentGovernance
├── monitoring.py                   # InferenceRecord, MonitoringSnapshot,
│                                   #   ProductionMonitor
├── health.py                       # HealthRecord, HealthReport,
│                                   #   DriftThresholds, ModelHealthMonitor
├── alerting.py                     # AlertLevel, AlertCategory, Alert,
│                                   #   AlertingSystem (16 categories)
├── champion.py                     # ChampionEvent, ChampionChallengerManager
├── audit.py                        # AuditEventType, AuditEvent,
│                                   #   AuditSystem (20 event types)
├── retraining.py                   # RetrainingTrigger, RetrainingPolicy,
│                                   #   RetrainingRecommendation,
│                                   #   RetrainingPolicyManager,
│                                   #   build_default_policies()
├── database.py                     # MLOpsDatabase (SQLite, 8 tables)
├── api.py                          # MLOpsAPI (40+ endpoints)
├── visualization.py                # MLOpsVisualizationBuilder (7 charts)
├── export.py                       # MLOpsExporter (6 export types)
└── engine.py                       # MLOpsConfig, MLOpsEngine
```

Configuration: `conf/mlops/mlops.yaml`

Tests: `tests/unit/test_mlops_platform.py` (160+ tests)

Docs (7):
- `docs/mlops_guide.md`
- `docs/governance_guide.md`
- `docs/registry_guide.md`
- `docs/deployment_guide.md`
- `docs/operational_runbook.md`
- `docs/developer_guide_phase18.md` (this file)
- `docs/mlops_readiness_report.md`

---

## Design Decisions

### No automatic retraining — ever

`RetrainingPolicyManager(auto_retrain=True)` raises `ValueError`.
The `auto_retrain` flag is always `False`. Advisory recommendations only.

### Strict lifecycle enforcement

`LifecycleManager(strict=True)` — any invalid transition raises `LifecycleError`.
The full valid transition graph is in `lifecycle.py::_VALID_TRANSITIONS`.

### Zero new dependencies

Pure stdlib: `sqlite3`, `threading`, `json`, `uuid`, `math`, `collections`, `pathlib`.
Plus existing `loguru`.

### Immutable audit trail

Audit events are append-only. No deletion or modification is supported.
The `AuditSystem` enforces this contractually.

### Thread-safe database

`MLOpsDatabase` uses `threading.local()` for connection-per-thread + WAL journal mode.

---

## Adding a New Lifecycle State

1. Add the state constant to `LifecycleState` in `registry.py`.
2. Add transitions to/from the new state in `_VALID_TRANSITIONS` in `lifecycle.py`.
3. Update `LifecycleState.ALL`, `ACTIVE`, and `TERMINAL` lists as appropriate.
4. Add tests for the new transitions in `test_mlops_platform.py`.

---

## Adding a New Alert Category

1. Add the category constant to `AlertCategory.ALL` in `alerting.py`.
2. Add an emit helper if needed (like `emit_from_health_report`).
3. Use via: `engine.emit_alert(AlertLevel.WARNING, model_id, AlertCategory.MY_NEW_CAT, "msg")`.

---

## Adding a New Retraining Trigger

1. Add the trigger constant to `RetrainingTrigger.ALL` in `retraining.py`.
2. Add the metric key mapping in `RetrainingPolicyManager.evaluate()`.
3. Add a default policy in `build_default_policies()` if needed.

---

## Database Schema Overview

| Table                        | Rows                                        |
|------------------------------|---------------------------------------------|
| `model_registry`             | One per registered model                    |
| `lifecycle_transitions`      | One per lifecycle state change              |
| `deployments`                | One per deployment event                    |
| `health_metrics`             | Rolling health/drift snapshots              |
| `alerts`                     | All generated alerts                        |
| `audit_events`               | Full immutable audit trail                  |
| `retraining_recommendations` | Advisory recommendations only               |

---

## Testing

```bash
# Run all Phase 18 tests
pytest tests/unit/test_mlops_platform.py -v

# Run specific group
pytest tests/unit/test_mlops_platform.py::TestMLOpsEngine -v

# With coverage
pytest tests/unit/test_mlops_platform.py --cov=trainer.mlops
```

---

## Complete Engine Usage

```python
from trainer.mlops import MLOpsEngine, MLOpsConfig, AlertLevel, AlertCategory

# Initialize
engine = MLOpsEngine(MLOpsConfig(
    db_path="experiments/mlops/mlops.db",
    monitoring_window=500,
    governance_require_approver=True,
    enable_default_retraining_policies=True,
))

# Register + lifecycle
model_id = engine.register_model({
    "name": "signalium_quant", "version": "2.0.0",
    "architecture": "transformer_encoder",
    "parameter_count": 8_241_920, ...
})
engine.transition(model_id, "training", actor="trainer_service")
engine.transition(model_id, "validating", actor="validator_service")
engine.transition(model_id, "candidate", actor="validator_service")
engine.approve(model_id, approver="head_quant")
engine.set_champion(model_id, approver="head_quant")

# Monitor production
for prediction in live_predictions:
    engine.record_inference(
        model_id,
        latency_ms=prediction.latency,
        confidence=prediction.confidence,
        predicted_class=prediction.cls,
        cpu_pct=42.0,
        memory_mb=2100.0,
    )

# Check health (every N predictions)
resp = engine.check_health(model_id)
if not resp["data"]["is_healthy"]:
    print(resp["data"]["drift_alerts"])

# Retraining evaluation (periodic, advisory only)
resp = engine.evaluate_retraining(model_id, {
    "performance_degradation": health_report["performance_degradation"],
    "confidence_degradation": abs(
        health_report["recent_avg_confidence"] -
        health_report["baseline_avg_confidence"]
    ),
})

# Compliance export
engine.export_registry("compliance/registry.json")
engine.export_audit_report(path="compliance/audit.json")

# Shutdown
engine.close()
```
