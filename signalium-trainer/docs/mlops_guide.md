# Institutional MLOps Guide — Phase 18

## Overview

Phase 18 builds the **Institutional MLOps, Model Governance, and Continuous Monitoring Platform** — the governance backbone for every model in Signalium's production stack.

**Core principles:**
- No production model may exist outside this governance framework.
- No neural network is modified by this platform.
- No model is automatically retrained — all retraining is advisory.
- Full backward compatibility with Phases 1–17.

---

## Quick Start

```python
from trainer.mlops import MLOpsEngine, MLOpsConfig, ModelRecord

# Initialize the platform
engine = MLOpsEngine.from_config({
    "db_path": "experiments/mlops/mlops.db",
})

# Register a model
model_id = engine.register_model({
    "name": "signalium_quant",
    "version": "1.0.0",
    "architecture": "lstm",
    "backbone": "lstm_256_2l",
    "parameter_count": 2_458_624,
    "dataset_version": "v3.2",
    "feature_version": "v2.1",
    "label_version": "v1.0",
    "git_commit": "abc1234def5678",
    "hardware_used": "NVIDIA A100 40GB",
    "training_duration_seconds": 3600.0,
    "validation_report": {
        "accuracy": 0.68,
        "sharpe_ratio": 1.5,
        "calibration_error": 0.04,
    },
})

# Govern the lifecycle
engine.transition(model_id, "training", actor="trainer_service")
engine.transition(model_id, "validating", actor="validator_service")
engine.transition(model_id, "candidate", actor="validator_service")
engine.approve(model_id, approver="head_quant", reason="Meets institutional criteria")
engine.set_champion(model_id, approver="head_quant")

# Monitor production inferences
engine.record_inference(model_id,
    latency_ms=11.8, confidence=0.78,
    predicted_class="buy", buy_prob=0.78, sell_prob=0.12, no_trade_prob=0.10,
    cpu_pct=42.0, memory_mb=2100.0, regime="trending_up")

# Check health
resp = engine.check_health(model_id)
print(resp["data"]["is_healthy"])

# Full readiness report
report = engine.get_readiness_report()
print(report["institutional_readiness"])

engine.close()
```

---

## Platform Components

| Component                  | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `ModelRegistry`            | Central registry — all 20 model metadata fields             |
| `LifecycleManager`         | 11 lifecycle states + transition graph + enforcement        |
| `DeploymentGovernance`     | Approval workflows, rollback, deployment history            |
| `ProductionMonitor`        | Latency/throughput/CPU/GPU/memory/confidence monitoring     |
| `ModelHealthMonitor`       | Drift detection across 7 dimensions                         |
| `AlertingSystem`           | 16 alert categories, 3 severity levels, dedup, ack          |
| `ChampionChallengerManager`| Champion designation, challenger queue, comparison          |
| `AuditSystem`              | Immutable event trail (20 event types)                      |
| `RetrainingPolicyManager`  | Advisory-only retraining recommendations                    |
| `MLOpsDatabase`            | SQLite persistence (8 tables, 10 indexes)                   |
| `MLOpsAPI`                 | Structured endpoints (40+ endpoints, standard envelope)     |
| `MLOpsVisualizationBuilder`| 7 chart types + full dashboard                              |
| `MLOpsExporter`            | 6 export types + readiness report                           |
| `MLOpsEngine`              | Central orchestrator                                        |

---

## Configuration

```python
from trainer.mlops import MLOpsEngine, MLOpsConfig

cfg = MLOpsConfig(
    db_path="experiments/mlops/mlops.db",
    monitoring_window=500,
    health_baseline_window=200,
    health_recent_window=50,
    latency_alert_ms=100.0,
    memory_alert_mb=8192.0,
    confidence_alert_threshold=0.40,
    governance_require_approver=True,
    governance_auto_promote=False,   # manual promotion only by default
    export_dir="experiments/mlops/exports",
    enable_default_retraining_policies=True,
)
engine = MLOpsEngine(cfg)
```

Or from YAML:

```python
engine = MLOpsEngine.from_yaml("conf/mlops/mlops.yaml")
```

---

## Backward Compatibility

Phase 18 only adds `trainer/mlops/` — zero modifications to:
- Phases 1–16: All training, validation, and AutoML modules
- Phase 17 AEIP ensemble modules
- Any neural network architecture
- Any prediction head
