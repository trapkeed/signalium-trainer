# Operational Runbook — Phase 18 MLOps Platform

## Purpose

This runbook documents standard operating procedures for the Signalium Institutional MLOps Platform.

---

## 1. Daily Health Checks

### 1.1 Check Active Alerts

```python
from trainer.mlops import MLOpsEngine

engine = MLOpsEngine.from_config({
    "db_path": "experiments/mlops/mlops.db"
})

# Check active critical alerts
resp = engine.api.get_alerts(level="CRITICAL", active_only=True)
print(f"Critical alerts: {resp['data']['count']}")
for a in resp["data"]["alerts"]:
    print(f"  [{a['category']}] {a['message']} — {a['created_at']}")
```

### 1.2 Check Champion Health

```python
champion = engine.champion_manager.current_champion
if champion:
    resp = engine.check_health(champion["model_id"])
    report = resp["data"]
    print(f"Champion health: {'OK' if report['is_healthy'] else 'DEGRADED'}")
    if report["drift_alerts"]:
        print(f"  Drift alerts: {report['drift_alerts']}")
```

### 1.3 Check Monitoring Snapshot

```python
if champion:
    snap = engine.monitor.get_snapshot(champion["model_id"])
    print(f"Avg latency: {snap.avg_latency_ms:.1f}ms")
    print(f"Avg confidence: {snap.avg_confidence:.3f}")
    print(f"Confidence trend: {snap.confidence_trend}")
    print(f"Signal rate: {snap.signal_rate:.1%}")
```

### 1.4 Check Pending Retraining Recommendations

```python
resp = engine.api.list_retraining_recommendations()
pending = resp["data"]["pending"]
if pending:
    print(f"Pending retraining recommendations: {len(pending)}")
    for r in pending:
        print(f"  Priority {r['priority']}: {r['reason']}")
```

---

## 2. Deploying a New Champion

### Prerequisites

1. Model trained and registered with full metadata
2. Phase 15 validation completed (`validation_report` populated)
3. Architecture and backbone documented
4. Approval from Head Quant or designated approver

### Procedure

```bash
# Step 1: Advance through lifecycle
python -c "
from trainer.mlops import MLOpsEngine
engine = MLOpsEngine.from_config({'db_path': 'experiments/mlops/mlops.db'})
engine.transition('MODEL_ID', 'training', actor='trainer_service')
engine.transition('MODEL_ID', 'validating', actor='validator_service')
engine.transition('MODEL_ID', 'candidate', actor='validator_service')
"

# Step 2: Review validation report
# (view in dashboard or query API)

# Step 3: Approve
python -c "
from trainer.mlops import MLOpsEngine
engine = MLOpsEngine.from_config({'db_path': 'experiments/mlops/mlops.db'})
engine.approve('MODEL_ID', approver='head_quant',
               reason='Accuracy 0.72, Sharpe 1.85, ECE 0.032 — all thresholds met')
"

# Step 4: Set as champion
python -c "
from trainer.mlops import MLOpsEngine
engine = MLOpsEngine.from_config({'db_path': 'experiments/mlops/mlops.db'})
engine.set_champion('MODEL_ID', approver='head_quant')
"
```

---

## 3. Emergency Rollback

**Trigger this if:**
- Confidence drift > 0.15 (system-generated CRITICAL alert)
- Model returns unexpected classes at > 20% rate
- Latency p99 > 250ms sustained for > 5 minutes
- Memory exhaustion alert

```bash
python -c "
from trainer.mlops import MLOpsEngine
engine = MLOpsEngine.from_config({'db_path': 'experiments/mlops/mlops.db'})
champion = engine.champion_manager.current_champion
if champion:
    engine.rollback(champion['model_id'],
                    actor='ops_lead',
                    reason='Emergency: confidence_drift=0.18',
                    emergency=True)
    print('Champion rolled back. Designate new champion.')
"
```

After rollback:
1. Check `historical_champions` for the previous champion
2. If previous champion is eligible, promote it: `engine.set_champion(prev_id)`
3. Investigate the rolled-back model's health report
4. Open a retraining recommendation if drift persists

---

## 4. Acknowledging Alerts

```python
# Acknowledge a specific alert
resp = engine.api.acknowledge_alert(alert_id, ack_by="ops_lead")

# Acknowledge all active critical alerts for a model
resp = engine.api.get_alerts(model_id=model_id, level="CRITICAL")
for alert in resp["data"]["alerts"]:
    engine.api.acknowledge_alert(alert["alert_id"], ack_by="ops_lead")
```

---

## 5. Retiring a Model

```python
# Retire a model that is no longer used
engine.retire_model(model_id, actor="head_quant",
                     reason="Superseded by v3.0.0. Archived for compliance.")
```

---

## 6. Exporting Compliance Reports

```python
# Monthly compliance export
engine.export_registry(path="compliance/registry_2026-07.json")
engine.export_audit_report(path="compliance/audit_2026-07.json")
engine.export_deployment_report(path="compliance/deployments_2026-07.json")
engine.export_health_report(path="compliance/health_2026-07.json")

# Full institutional readiness report
report = engine.get_readiness_report()
engine.exporter.save(report, "compliance/readiness_2026-07.json")
```

---

## 7. Retraining Recommendation Handling

**IMPORTANT:** The platform never automatically retrains any model.
All recommendations are advisory.

```python
# Review pending recommendations
resp = engine.api.list_retraining_recommendations()
for rec in resp["data"]["pending"]:
    print(f"Priority {rec['priority']}: {rec['trigger']} — {rec['reason']}")

# Act on a recommendation (means: schedule retraining manually)
engine.retraining.acknowledge(recommendation_id, acted_by="head_quant")

# Dismiss a recommendation
engine.retraining.dismiss(recommendation_id)
```

---

## 8. Drift Alert Thresholds

| Drift Type              | Default Threshold | Alert Level |
|-------------------------|------------------|-------------|
| Confidence drift        | > 0.10           | WARNING     |
| Prediction drift        | > 0.15           | WARNING     |
| Calibration drift       | > 0.05           | WARNING     |
| Performance degradation | > 0.05           | WARNING     |
| Latency degradation     | > 20ms           | WARNING     |
| Distribution drift      | > 0.15           | WARNING     |

To modify thresholds:

```python
from trainer.mlops.health import DriftThresholds, ModelHealthMonitor

thresholds = DriftThresholds(
    confidence_drift=0.08,
    prediction_drift=0.12,
    calibration_drift=0.03,
    performance_degradation=0.04,
    latency_degradation_ms=15.0,
    distribution_drift=0.12,
)
engine.health = ModelHealthMonitor(thresholds=thresholds)
```

---

## 9. Dashboard

```python
# Full dashboard
resp = engine.get_dashboard()
dashboard_data = resp["data"]

# Specific charts
charts = engine.viz.full_dashboard(
    models=[m.to_dict() for m in engine.registry.all_models()],
    health_reports={mid: r.to_dict() for mid, r in engine.health.all_health_reports().items()},
    deployments=engine.db.list_deployments(),
    comparison=None,
    alerts=engine.alerting.all_alerts(),
    monitoring_snapshots={mid: s.to_dict() for mid, s in engine.monitor.all_snapshots().items()},
)
```

---

## 10. Incident Response Flow

```
1. CRITICAL alert received
   └→ Acknowledge alert (ack_by=ops_lead)
   └→ Check champion health report
   └→ Check monitoring snapshot (latency, confidence, distribution)

2. If drift confirmed:
   └→ Evaluate retraining recommendation
   └→ Add previous champion as challenger
   └→ Compare champion vs challenger
   └→ If challenger wins: promote_challenger()
   └→ If no alternative: emergency rollback

3. Root cause analysis:
   └→ Export audit report for model
   └→ Export health report
   └→ Review lifecycle history
   └→ File retraining recommendation with head_quant

4. Post-incident:
   └→ Export full deployment report
   └→ Update model notes with incident summary
   └→ Archive incident audit events
```
