"""Phase 18 — Institutional MLOps Platform Unit Tests.

Covers:
  - ModelRecord, ModelRegistry, LifecycleState
  - LifecycleManager: all transitions, rollback, approve, reject, retire
  - DeploymentGovernance: deploy, rollback, auto-promote, history
  - ProductionMonitor: record_inference, snapshots, latency, confidence
  - ModelHealthMonitor: drift detection (confidence, prediction, calibration,
                         performance, latency, distribution)
  - AlertingSystem: emit, acknowledge, resolve, dedup, health/monitoring alerts
  - ChampionChallengerManager: set_champion, rollback, add_challenger,
                                promote_challenger, compare, retire
  - AuditSystem: log, query, model_history, get_report, export
  - RetrainingPolicyManager: add_policy, evaluate, acknowledge, dismiss,
                               no auto-retrain, build_default_policies
  - MLOpsDatabase: save/get model, transitions, deployments, health, alerts,
                    audit, recommendations
  - MLOpsAPI: all endpoint groups
  - MLOpsVisualizationBuilder: all 7 chart types + full_dashboard
  - MLOpsExporter: all 5 export types + readiness report + save/load
  - MLOpsEngine: register, lifecycle, deploy, rollback, champion, monitoring,
                  health, alerting, retraining, dashboard, summary, export
  - PublicAPI: __all__ completeness
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_dict(**overrides) -> dict[str, Any]:
    return {
        "model_id": str(uuid.uuid4()),
        "name": "signalium_quant",
        "version": "1.0.0",
        "architecture": "lstm",
        "backbone": "lstm_256_2l",
        "parameter_count": 2_458_624,
        "dataset_version": "v3.2",
        "feature_version": "v2.1",
        "label_version": "v1.0",
        "git_commit": "abc1234def567",
        "hardware_used": "NVIDIA A100",
        "training_duration_seconds": 3600.0,
        "validation_report": {
            "accuracy": 0.68,
            "sharpe_ratio": 1.5,
            "calibration_error": 0.04,
        },
        **overrides,
    }


def _make_registry(n: int = 2) -> ModelRegistry:
    from trainer.mlops.registry import ModelRecord, ModelRegistry

    r = ModelRegistry()
    for i in range(n):
        r.register(
            ModelRecord.from_dict(_model_dict(model_id=f"model_{i:02d}", version=f"{i + 1}.0.0"))
        )
    return r


def _make_engine(tmpdir: str) -> MLOpsEngine:
    from trainer.mlops.engine import MLOpsConfig, MLOpsEngine

    return MLOpsEngine(
        MLOpsConfig(
            db_path=os.path.join(tmpdir, "test_mlops.db"),
            governance_require_approver=False,
            enable_default_retraining_policies=True,
        )
    )


# ===========================================================================
# Registry Tests
# ===========================================================================


class TestModelRecord:
    def test_default_state_is_draft(self):
        from trainer.mlops.registry import LifecycleState, ModelRecord

        r = ModelRecord(name="m1")
        assert r.lifecycle_state == LifecycleState.DRAFT

    def test_full_name(self):
        from trainer.mlops.registry import ModelRecord

        r = ModelRecord(name="quant", version="2.1.0")
        assert r.full_name == "quant@2.1.0"

    def test_is_active_false_on_draft(self):
        from trainer.mlops.registry import ModelRecord

        r = ModelRecord()
        assert not r.is_active

    def test_is_champion_false_initially(self):
        from trainer.mlops.registry import ModelRecord

        r = ModelRecord()
        assert not r.is_champion

    def test_add_approval(self):
        from trainer.mlops.registry import ModelRecord

        r = ModelRecord(model_id="m1")
        approval = r.add_approval("ops", "approved", "looks good", "candidate", "approved")
        assert approval.approver == "ops"
        assert len(r.approval_history) == 1

    def test_add_child(self):
        from trainer.mlops.registry import ModelRecord

        r = ModelRecord(model_id="parent")
        r.add_child("child_01")
        r.add_child("child_02")
        r.add_child("child_01")  # duplicate — not added
        assert r.child_model_ids == ["child_01", "child_02"]

    def test_to_from_dict(self):
        from trainer.mlops.registry import ModelRecord

        r = ModelRecord.from_dict(_model_dict())
        d = r.to_dict()
        r2 = ModelRecord.from_dict(d)
        assert r2.model_id == r.model_id
        assert r2.version == r.version

    def test_is_terminal_false_by_default(self):
        from trainer.mlops.registry import ModelRecord

        r = ModelRecord()
        assert not r.is_terminal


class TestModelRegistry:
    def test_register_and_get(self):
        from trainer.mlops.registry import ModelRecord, ModelRegistry

        r = ModelRegistry()
        rec = ModelRecord(model_id="m1", name="quant")
        r.register(rec)
        assert r.get("m1") is rec

    def test_get_or_raise_missing(self):
        from trainer.mlops.registry import ModelRegistry

        r = ModelRegistry()
        with pytest.raises(KeyError):
            r.get_or_raise("nonexistent")

    def test_all_models(self):
        r = _make_registry(3)
        assert r.n_models == 3

    def test_active_returns_only_active(self):

        r = _make_registry(3)
        # Set one to champion (active)
        r.set_champion("model_00")
        active = r.active()
        assert any(m.is_champion for m in active)

    def test_by_state_draft(self):
        from trainer.mlops.registry import LifecycleState

        r = _make_registry(3)
        drafts = r.by_state(LifecycleState.DRAFT)
        assert len(drafts) == 3

    def test_set_champion_exclusive(self):
        r = _make_registry(3)
        r.set_champion("model_00")
        r.set_champion("model_01")
        champions = [m for m in r.all_models() if m.is_champion]
        assert len(champions) == 1
        assert champions[0].model_id == "model_01"

    def test_summary(self):
        r = _make_registry(2)
        s = r.summary()
        assert s["total_models"] == 2

    def test_latest_version(self):
        from trainer.mlops.registry import ModelRecord, ModelRegistry

        r = ModelRegistry()
        r.register(ModelRecord(model_id="a1", name="quant", version="1.0.0"))
        r.register(ModelRecord(model_id="a2", name="quant", version="2.0.0"))
        r.register(ModelRecord(model_id="a3", name="quant", version="1.5.0"))
        latest = r.latest_version("quant")
        assert latest is not None
        assert latest.version == "2.0.0"

    def test_to_from_dict(self):
        from trainer.mlops.registry import ModelRegistry

        r = _make_registry(2)
        d = r.to_dict()
        r2 = ModelRegistry.from_dict(d)
        assert r2.n_models == 2


# ===========================================================================
# Lifecycle Tests
# ===========================================================================


class TestLifecycleManager:
    def setup_method(self):
        from trainer.mlops.lifecycle import LifecycleManager
        from trainer.mlops.registry import ModelRecord

        self.lm = LifecycleManager(strict=True)
        self.record = ModelRecord.from_dict(_model_dict())

    def test_transition_draft_to_training(self):
        from trainer.mlops.registry import LifecycleState

        t = self.lm.transition(self.record, LifecycleState.TRAINING, actor="trainer")
        assert self.record.lifecycle_state == LifecycleState.TRAINING
        assert t.to_state == LifecycleState.TRAINING

    def test_full_happy_path(self):
        from trainer.mlops.registry import LifecycleState

        for state in [LifecycleState.TRAINING, LifecycleState.VALIDATING, LifecycleState.CANDIDATE]:
            self.lm.transition(self.record, state)
        self.lm.approve(self.record, approver="ops")
        assert self.record.lifecycle_state == LifecycleState.APPROVED
        assert self.record.promotion_status == "approved"

    def test_invalid_transition_raises(self):
        from trainer.mlops.lifecycle import LifecycleError
        from trainer.mlops.registry import LifecycleState

        with pytest.raises(LifecycleError):
            self.lm.transition(self.record, LifecycleState.RETIRED)

    def test_invalid_strict_false_no_raise(self):
        from trainer.mlops.lifecycle import LifecycleManager
        from trainer.mlops.registry import LifecycleState, ModelRecord

        lm = LifecycleManager(strict=False)
        rec = ModelRecord.from_dict(_model_dict())
        lm.transition(rec, LifecycleState.RETIRED)  # should not raise

    def test_rollback(self):
        from trainer.mlops.registry import LifecycleState

        for state in [
            LifecycleState.TRAINING,
            LifecycleState.VALIDATING,
            LifecycleState.CANDIDATE,
            LifecycleState.APPROVED,
            LifecycleState.CHAMPION,
        ]:
            self.lm.transition(self.record, state)
        t = self.lm.rollback(self.record, actor="ops", reason="emergency")
        assert self.record.lifecycle_state == LifecycleState.ROLLED_BACK
        assert t.metadata.get("rollback") is True

    def test_history_recorded(self):
        from trainer.mlops.registry import LifecycleState

        self.lm.transition(self.record, LifecycleState.TRAINING)
        self.lm.transition(self.record, LifecycleState.VALIDATING)
        history = self.lm.history(self.record.model_id)
        assert len(history) == 2

    def test_promote_to_candidate(self):
        from trainer.mlops.registry import LifecycleState

        self.lm.transition(self.record, LifecycleState.TRAINING)
        self.lm.transition(self.record, LifecycleState.VALIDATING)
        t = self.lm.promote_to_candidate(self.record, validation_score=0.72)
        assert self.record.lifecycle_state == LifecycleState.CANDIDATE
        assert t.metadata["validation_score"] == pytest.approx(0.72)

    def test_reject(self):
        from trainer.mlops.registry import LifecycleState

        self.lm.transition(self.record, LifecycleState.TRAINING)
        self.lm.transition(self.record, LifecycleState.VALIDATING)
        self.lm.transition(self.record, LifecycleState.CANDIDATE)
        self.lm.reject(self.record, approver="ops", reason="poor calibration")
        assert self.record.lifecycle_state == LifecycleState.DRAFT
        assert self.record.promotion_status == "rejected"

    def test_retire(self):
        from trainer.mlops.registry import LifecycleState

        transitions = self.lm.retire(self.record, reason="end of life")
        assert self.record.lifecycle_state == LifecycleState.RETIRED
        assert self.record.retirement_status == "retired"
        assert len(transitions) == 2

    def test_same_state_is_noop(self):
        from trainer.mlops.registry import LifecycleState

        t = self.lm.transition(self.record, LifecycleState.DRAFT)
        assert "no-op" in t.reason

    def test_state_summary(self):
        from trainer.mlops.registry import LifecycleState

        self.lm.transition(self.record, LifecycleState.TRAINING)
        summary = self.lm.state_summary()
        assert summary["models_tracked"] >= 1

    def test_can_transition_true(self):
        from trainer.mlops.lifecycle import LifecycleManager
        from trainer.mlops.registry import LifecycleState

        lm = LifecycleManager()
        assert lm.can_transition(LifecycleState.DRAFT, LifecycleState.TRAINING)

    def test_can_transition_false(self):
        from trainer.mlops.lifecycle import LifecycleManager
        from trainer.mlops.registry import LifecycleState

        lm = LifecycleManager()
        assert not lm.can_transition(LifecycleState.DRAFT, LifecycleState.RETIRED)


# ===========================================================================
# Governance Tests
# ===========================================================================


class TestDeploymentGovernance:
    def setup_method(self):
        from trainer.mlops.governance import DeploymentGovernance, GovernanceConfig
        from trainer.mlops.lifecycle import LifecycleManager

        self.registry = _make_registry(2)
        self.lm = LifecycleManager(strict=False)
        self.gov = DeploymentGovernance(
            registry=self.registry,
            lifecycle=self.lm,
            config=GovernanceConfig(require_approver=False),
        )

    def _approve_model(self, model_id: str) -> None:
        from trainer.mlops.registry import LifecycleState

        model = self.registry.get(model_id)
        for state in [
            LifecycleState.TRAINING,
            LifecycleState.VALIDATING,
            LifecycleState.CANDIDATE,
            LifecycleState.APPROVED,
        ]:
            self.lm.transition(model, state)

    def test_deploy_production(self):

        self._approve_model("model_00")
        rec = self.gov.deploy("model_00", target="production", approver="ops", reason="release v1")
        assert rec.status == "active"
        assert rec.target == "production"

    def test_deploy_champion(self):

        self._approve_model("model_00")
        self.gov.deploy("model_00", target="champion", approver="head_quant")
        model = self.registry.get("model_00")
        assert model.is_champion

    def test_rollback(self):
        from trainer.mlops.registry import LifecycleState

        self._approve_model("model_00")
        self.gov.deploy("model_00", target="production", approver="ops")
        rb_rec = self.gov.rollback("model_00", actor="ops", reason="high drift", emergency=True)
        assert rb_rec.deploy_type == "emergency_rollback"
        model = self.registry.get("model_00")
        assert model.lifecycle_state == LifecycleState.ROLLED_BACK

    def test_active_deployments(self):
        self._approve_model("model_00")
        self.gov.deploy("model_00", target="production", approver="ops")
        active = self.gov.active_deployments()
        assert len(active) >= 1

    def test_superseded_on_new_deploy(self):
        self._approve_model("model_00")
        self._approve_model("model_01")
        self.gov.deploy("model_00", target="production", approver="ops")
        self.gov.deploy("model_01", target="production", approver="ops")
        active = self.gov.active_deployments()
        active_ids = [d["model_id"] for d in active]
        assert "model_01" in active_ids

    def test_history(self):
        self._approve_model("model_00")
        self.gov.deploy("model_00", target="production", approver="ops")
        history = self.gov.history("model_00")
        assert len(history) >= 1

    def test_stats(self):
        stats = self.gov.stats()
        assert "total_deployments" in stats

    def test_auto_promote_disabled_by_default(self):
        from trainer.mlops.governance import DeploymentGovernance, GovernanceConfig

        gov = DeploymentGovernance(
            registry=self.registry,
            lifecycle=self.lm,
            config=GovernanceConfig(auto_promote_enabled=False),
        )
        result = gov.auto_promote_if_eligible("model_00")
        assert result is None

    def test_auto_promote_with_eligible_model(self):
        from trainer.mlops.governance import DeploymentGovernance, GovernanceConfig

        gov = DeploymentGovernance(
            registry=self.registry,
            lifecycle=self.lm,
            config=GovernanceConfig(
                auto_promote_enabled=True,
                auto_promote_min_validation_score=0.5,
                auto_promote_max_calibration_error=0.15,
            ),
        )
        self._approve_model("model_00")
        result = gov.auto_promote_if_eligible("model_00")
        assert result is not None
        assert result.deploy_type == "automatic"


# ===========================================================================
# Monitoring Tests
# ===========================================================================


class TestProductionMonitor:
    def setup_method(self):
        from trainer.mlops.monitoring import ProductionMonitor

        self.monitor = ProductionMonitor(window=100, latency_alert_ms=50.0)

    def test_record_inference(self):
        rec = self.monitor.record_inference(
            "m1", latency_ms=12.5, confidence=0.78, predicted_class="buy"
        )
        assert rec.model_id == "m1"
        assert rec.latency_ms == 12.5

    def test_snapshot_empty(self):
        from trainer.mlops.monitoring import MonitoringSnapshot

        snap = self.monitor.get_snapshot("nonexistent")
        assert isinstance(snap, MonitoringSnapshot)
        assert snap.n_records == 0

    def test_avg_latency(self):
        for i in range(10):
            self.monitor.record_inference("m1", latency_ms=float(i * 10))
        snap = self.monitor.get_snapshot("m1")
        assert snap.avg_latency_ms == pytest.approx(45.0)

    def test_latency_alert(self):
        for _ in range(10):
            self.monitor.record_inference("m1", latency_ms=100.0)
        snap = self.monitor.get_snapshot("m1")
        assert snap.latency_alert is True

    def test_no_latency_alert(self):
        for _ in range(5):
            self.monitor.record_inference("m1", latency_ms=10.0)
        snap = self.monitor.get_snapshot("m1")
        assert snap.latency_alert is False

    def test_confidence_trend_insufficient(self):
        self.monitor.record_inference("m1", confidence=0.7)
        snap = self.monitor.get_snapshot("m1")
        assert snap.confidence_trend == "insufficient"

    def test_confidence_trend_declining(self):
        for _ in range(50):
            self.monitor.record_inference("m1", confidence=0.9)
        for _ in range(50):
            self.monitor.record_inference("m1", confidence=0.5)
        snap = self.monitor.get_snapshot("m1")
        assert snap.confidence_trend == "declining"

    def test_class_distribution(self):
        for _ in range(6):
            self.monitor.record_inference("m1", predicted_class="buy")
        for _ in range(4):
            self.monitor.record_inference("m1", predicted_class="sell")
        snap = self.monitor.get_snapshot("m1")
        assert snap.buy_rate == pytest.approx(0.6)
        assert snap.sell_rate == pytest.approx(0.4)

    def test_signal_rate(self):
        for _ in range(7):
            self.monitor.record_inference("m1", predicted_class="buy")
        for _ in range(3):
            self.monitor.record_inference("m1", predicted_class="no_trade")
        snap = self.monitor.get_snapshot("m1")
        assert snap.signal_rate == pytest.approx(0.7)

    def test_all_snapshots(self):
        self.monitor.record_inference("m1", confidence=0.7)
        self.monitor.record_inference("m2", confidence=0.8)
        snaps = self.monitor.all_snapshots()
        assert "m1" in snaps and "m2" in snaps

    def test_to_dict(self):
        self.monitor.record_inference("m1", latency_ms=15.0, confidence=0.75)
        snap = self.monitor.get_snapshot("m1")
        d = snap.to_dict()
        assert "avg_latency_ms" in d
        assert "avg_confidence" in d

    def test_n_inferences(self):
        for _ in range(5):
            self.monitor.record_inference("m1")
        assert self.monitor.n_inferences("m1") == 5


# ===========================================================================
# Health Tests
# ===========================================================================


class TestModelHealthMonitor:
    def setup_method(self):
        from trainer.mlops.health import DriftThresholds, ModelHealthMonitor

        self.monitor = ModelHealthMonitor(
            baseline_window=20,
            recent_window=10,
            thresholds=DriftThresholds(confidence_drift=0.05),
        )

    def test_record_returns_health_record(self):
        from trainer.mlops.health import HealthRecord

        rec = self.monitor.record_prediction("m1", confidence=0.7, predicted_class="buy")
        assert isinstance(rec, HealthRecord)

    def test_insufficient_data_is_healthy(self):
        for _ in range(5):
            self.monitor.record_prediction("m1", confidence=0.7)
        report = self.monitor.get_health_report("m1")
        assert report.is_healthy  # not enough data to detect drift

    def test_confidence_drift_detected(self):
        from trainer.mlops.health import DriftThresholds, ModelHealthMonitor

        monitor = ModelHealthMonitor(
            baseline_window=20,
            recent_window=10,
            thresholds=DriftThresholds(confidence_drift=0.05),
        )
        for _ in range(20):
            monitor.record_prediction("m1", confidence=0.80)
        for _ in range(10):
            monitor.record_prediction("m1", confidence=0.50)
        report = monitor.get_health_report("m1")
        assert report.confidence_drift > 0.05

    def test_performance_degradation(self):
        from trainer.mlops.health import DriftThresholds, ModelHealthMonitor

        monitor = ModelHealthMonitor(
            baseline_window=20,
            recent_window=10,
            thresholds=DriftThresholds(performance_degradation=0.05),
        )
        for _ in range(20):
            monitor.record_prediction("m1", predicted_class="buy", true_class="buy")
        for _ in range(10):
            monitor.record_prediction("m1", predicted_class="buy", true_class="sell")
        report = monitor.get_health_report("m1")
        assert report.performance_degradation > 0.0

    def test_distribution_drift(self):
        from trainer.mlops.health import DriftThresholds, ModelHealthMonitor

        monitor = ModelHealthMonitor(
            baseline_window=20,
            recent_window=10,
            thresholds=DriftThresholds(distribution_drift=0.05),
        )
        for _ in range(20):
            monitor.record_prediction("m1", predicted_class="buy")
        for _ in range(10):
            monitor.record_prediction("m1", predicted_class="no_trade")
        report = monitor.get_health_report("m1")
        assert report.distribution_drift > 0.0

    def test_latency_degradation(self):
        from trainer.mlops.health import DriftThresholds, ModelHealthMonitor

        monitor = ModelHealthMonitor(
            baseline_window=20,
            recent_window=10,
            thresholds=DriftThresholds(latency_degradation_ms=5.0),
        )
        for _ in range(20):
            monitor.record_prediction("m1", latency_ms=10.0)
        for _ in range(10):
            monitor.record_prediction("m1", latency_ms=100.0)
        report = monitor.get_health_report("m1")
        assert report.latency_degradation > 5.0

    def test_unhealthy_models_list(self):
        from trainer.mlops.health import DriftThresholds, ModelHealthMonitor

        monitor = ModelHealthMonitor(
            baseline_window=20,
            recent_window=10,
            thresholds=DriftThresholds(confidence_drift=0.05),
        )
        for _ in range(20):
            monitor.record_prediction("m1", confidence=0.9)
        for _ in range(10):
            monitor.record_prediction("m1", confidence=0.4)
        unhealthy = monitor.unhealthy_models()
        assert "m1" in unhealthy

    def test_health_report_to_dict(self):
        self.monitor.record_prediction("m1", confidence=0.7)
        report = self.monitor.get_health_report("m1")
        d = report.to_dict()
        assert "confidence_drift" in d
        assert "is_healthy" in d
        assert "drift_alerts" in d


# ===========================================================================
# Alerting Tests
# ===========================================================================


class TestAlertingSystem:
    def setup_method(self):
        from trainer.mlops.alerting import AlertingSystem

        self.alerting = AlertingSystem()

    def test_emit_warning(self):
        from trainer.mlops.alerting import AlertLevel

        alert = self.alerting.emit(
            AlertLevel.WARNING, "m1", "confidence_drift", "Conf dropped 0.12"
        )
        assert alert.level == "WARNING"
        assert alert.is_active

    def test_emit_critical(self):
        from trainer.mlops.alerting import AlertCategory, AlertLevel

        alert = self.alerting.emit(
            AlertLevel.CRITICAL, "m1", AlertCategory.LATENCY_SPIKE, "p99 latency 250ms"
        )
        assert alert.is_critical

    def test_acknowledge(self):
        from trainer.mlops.alerting import AlertLevel

        alert = self.alerting.emit(AlertLevel.WARNING, "m1", "confidence_drift", "test")
        ok = self.alerting.acknowledge(alert.alert_id, ack_by="ops_lead")
        assert ok
        assert alert.status == "acknowledged"
        assert alert.acknowledged_by == "ops_lead"

    def test_resolve(self):
        from trainer.mlops.alerting import AlertLevel

        alert = self.alerting.emit(AlertLevel.INFO, "m1", "system", "info msg")
        ok = self.alerting.resolve(alert.alert_id)
        assert ok
        assert alert.status == "resolved"

    def test_suppress(self):
        from trainer.mlops.alerting import AlertLevel

        alert = self.alerting.emit(AlertLevel.WARNING, "m1", "system", "noise")
        self.alerting.suppress(alert.alert_id)
        assert alert.status == "suppressed"

    def test_dedup_returns_same_alert(self):
        from trainer.mlops.alerting import AlertLevel

        a1 = self.alerting.emit(AlertLevel.WARNING, "m1", "confidence_drift", "1st")
        a2 = self.alerting.emit(AlertLevel.WARNING, "m1", "confidence_drift", "2nd")
        assert a1.alert_id == a2.alert_id

    def test_dedup_cleared_after_acknowledge(self):
        from trainer.mlops.alerting import AlertLevel

        a1 = self.alerting.emit(AlertLevel.WARNING, "m1", "confidence_drift", "1st")
        self.alerting.acknowledge(a1.alert_id, "ops")
        a2 = self.alerting.emit(AlertLevel.WARNING, "m1", "confidence_drift", "2nd")
        assert a1.alert_id != a2.alert_id

    def test_active_alerts_filtered_by_model(self):
        from trainer.mlops.alerting import AlertLevel

        self.alerting.emit(AlertLevel.WARNING, "m1", "confidence_drift", "m1 alert")
        self.alerting.emit(AlertLevel.WARNING, "m2", "latency_spike", "m2 alert")
        active = self.alerting.active_alerts(model_id="m1")
        assert all(a["model_id"] == "m1" for a in active)

    def test_active_alerts_filtered_by_level(self):
        from trainer.mlops.alerting import AlertLevel

        self.alerting.emit(AlertLevel.CRITICAL, "m1", "latency_spike", "crit")
        self.alerting.emit(AlertLevel.WARNING, "m1", "confidence_drift", "warn")
        critical = self.alerting.active_alerts(level="CRITICAL")
        assert all(a["level"] == "CRITICAL" for a in critical)

    def test_critical_count(self):
        from trainer.mlops.alerting import AlertCategory, AlertLevel

        for _ in range(3):
            self.alerting.emit(
                AlertLevel.CRITICAL, f"m{_}", AlertCategory.MODEL_FAILURE, "failure", dedup=False
            )
        assert self.alerting.critical_count() == 3

    def test_emit_from_health_report(self):
        from unittest.mock import MagicMock

        report = MagicMock()
        report.model_id = "m1"
        report.drift_alerts = ["confidence_drift=0.15", "prediction_drift=0.20"]
        emitted = self.alerting.emit_from_health_report(report)
        assert len(emitted) == 2

    def test_emit_from_monitoring_snapshot_latency(self):
        from unittest.mock import MagicMock

        snap = MagicMock()
        snap.model_id = "m1"
        snap.latency_alert = True
        snap.memory_alert = False
        snap.avg_confidence = 0.7
        snap.avg_latency_ms = 150.0
        snap.n_records = 50
        emitted = self.alerting.emit_from_monitoring_snapshot(snap)
        assert len(emitted) >= 1
        assert any("latency" in str(e.category).lower() for e in emitted)

    def test_stats(self):
        stats = self.alerting.stats()
        assert "total_alerts" in stats
        assert "active" in stats


# ===========================================================================
# Champion Tests
# ===========================================================================


class TestChampionChallengerManager:
    def setup_method(self):
        from trainer.mlops.champion import ChampionChallengerManager
        from trainer.mlops.governance import DeploymentGovernance, GovernanceConfig
        from trainer.mlops.lifecycle import LifecycleManager

        self.registry = _make_registry(3)
        self.lm = LifecycleManager(strict=False)
        self.gov = DeploymentGovernance(
            registry=self.registry,
            lifecycle=self.lm,
            config=GovernanceConfig(require_approver=False),
        )
        self.cc = ChampionChallengerManager(
            registry=self.registry,
            lifecycle=self.lm,
            governance=self.gov,
        )

    def test_set_champion(self):
        event = self.cc.set_champion("model_00", approver="head_quant")
        assert event.event_type == "promotion"
        assert self.cc.current_champion is not None
        assert self.cc.current_champion["model_id"] == "model_00"

    def test_set_champion_exclusive(self):
        self.cc.set_champion("model_00")
        self.cc.set_champion("model_01")
        champ = self.cc.current_champion
        assert champ["model_id"] == "model_01"

    def test_rollback_champion(self):
        self.cc.set_champion("model_00")
        event = self.cc.rollback_champion(actor="ops", reason="drift")
        assert event.event_type == "rollback"
        assert self.cc.current_champion is None

    def test_rollback_no_champion(self):
        result = self.cc.rollback_champion()
        assert result is None

    def test_add_challenger(self):
        self.cc.add_challenger("model_01", reason="improved calibration")
        assert len(self.cc.challengers) == 1
        assert self.cc.challengers[0]["model_id"] == "model_01"

    def test_promote_challenger(self):
        self.cc.set_champion("model_00")
        self.cc.add_challenger("model_01")
        event = self.cc.promote_challenger("model_01", approver="head_quant")
        assert event.event_type == "promotion"
        assert self.cc.current_champion["model_id"] == "model_01"

    def test_promote_challenger_not_in_queue_raises(self):
        with pytest.raises(ValueError):
            self.cc.promote_challenger("model_99", approver="ops")

    def test_compare_returns_winner(self):
        comparison = self.cc.compare(
            "model_00",
            "model_01",
            metrics_a={"accuracy": 0.70, "sharpe_ratio": 1.5, "calibration_error": 0.04},
            metrics_b={"accuracy": 0.65, "sharpe_ratio": 1.2, "calibration_error": 0.06},
        )
        assert comparison["overall_winner"] == "model_00"

    def test_retire_champion(self):
        self.cc.set_champion("model_00")
        event = self.cc.retire_champion("model_00", reason="obsolete")
        assert event.event_type == "retirement"
        assert len(self.cc.retired_champions) == 1

    def test_event_history(self):
        self.cc.set_champion("model_00")
        self.cc.add_challenger("model_01")
        history = self.cc.event_history()
        assert len(history) >= 1

    def test_summary(self):
        summary = self.cc.summary()
        assert "current_champion" in summary
        assert "n_challengers" in summary


# ===========================================================================
# Audit Tests
# ===========================================================================


class TestAuditSystem:
    def setup_method(self):
        from trainer.mlops.audit import AuditSystem

        self.audit = AuditSystem()

    def test_log_records_event(self):
        from trainer.mlops.audit import AuditEventType

        event = self.audit.log(
            AuditEventType.DEPLOYMENT,
            actor="ops",
            model_id="m1",
            description="deployed to production",
        )
        assert self.audit.n_events == 1
        assert event.actor == "ops"

    def test_query_by_type(self):
        from trainer.mlops.audit import AuditEventType

        self.audit.log(AuditEventType.DEPLOYMENT, model_id="m1")
        self.audit.log(AuditEventType.PROMOTION, model_id="m2")
        events = self.audit.query(event_type=AuditEventType.DEPLOYMENT)
        assert all(e["event_type"] == AuditEventType.DEPLOYMENT for e in events)

    def test_query_by_model(self):
        from trainer.mlops.audit import AuditEventType

        self.audit.log(AuditEventType.DEPLOYMENT, model_id="m1")
        self.audit.log(AuditEventType.DEPLOYMENT, model_id="m2")
        events = self.audit.query(model_id="m1")
        assert all(e["model_id"] == "m1" for e in events)

    def test_model_history(self):
        from trainer.mlops.audit import AuditEventType

        for et in [AuditEventType.TRAINING, AuditEventType.VALIDATION, AuditEventType.PROMOTION]:
            self.audit.log(et, model_id="m1")
        self.audit.log(AuditEventType.DEPLOYMENT, model_id="m2")
        history = self.audit.model_history("m1")
        assert len(history) == 3

    def test_get_report(self):
        from trainer.mlops.audit import AuditEventType

        self.audit.log(AuditEventType.DEPLOYMENT, model_id="m1", actor="ops")
        report = self.audit.get_report()
        assert report["total_events"] == 1
        assert "by_type" in report

    def test_get_report_by_model(self):
        from trainer.mlops.audit import AuditEventType

        self.audit.log(AuditEventType.DEPLOYMENT, model_id="m1")
        self.audit.log(AuditEventType.DEPLOYMENT, model_id="m2")
        report = self.audit.get_report(model_id="m1")
        assert report["total_events"] == 1

    def test_export_all(self):
        from trainer.mlops.audit import AuditEventType

        for _ in range(5):
            self.audit.log(AuditEventType.SYSTEM)
        exported = self.audit.export_all()
        assert len(exported) == 5
        assert all("event_id" in e for e in exported)

    def test_session_tracking(self):
        from trainer.mlops.audit import AuditEventType

        session_id = self.audit.start_session()
        self.audit.log(AuditEventType.TRAINING, model_id="m1")
        self.audit.end_session()
        self.audit.log(AuditEventType.SYSTEM)
        events = self.audit.query()
        session_events = [e for e in events if e.get("session_id") == session_id]
        assert len(session_events) == 1

    def test_stats(self):
        from trainer.mlops.audit import AuditEventType

        self.audit.log(AuditEventType.DEPLOYMENT, model_id="m1")
        stats = self.audit.stats()
        assert stats["total_events"] == 1


# ===========================================================================
# Retraining Tests
# ===========================================================================


class TestRetrainingPolicyManager:
    def setup_method(self):
        from trainer.mlops.retraining import (
            RetrainingPolicy,
            RetrainingPolicyManager,
            RetrainingTrigger,
        )

        self.mgr = RetrainingPolicyManager(auto_retrain=False)
        self.mgr.add_policy(
            RetrainingPolicy(
                policy_id="perf_test",
                trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                threshold=0.05,
                cooldown_hours=0,
            )
        )

    def test_auto_retrain_forbidden(self):
        from trainer.mlops.retraining import RetrainingPolicyManager

        with pytest.raises(ValueError):
            RetrainingPolicyManager(auto_retrain=True)

    def test_no_recommendation_below_threshold(self):
        recs = self.mgr.evaluate("m1", {"performance_degradation": 0.02})
        assert len(recs) == 0

    def test_recommendation_above_threshold(self):
        recs = self.mgr.evaluate("m1", {"performance_degradation": 0.10})
        assert len(recs) == 1
        assert recs[0].trigger == "performance_degradation"

    def test_recommendation_is_advisory_only(self):
        recs = self.mgr.evaluate("m1", {"performance_degradation": 0.10})
        d = recs[0].to_dict()
        assert "RECOMMENDATION ONLY" in d["note"]

    def test_cooldown_prevents_duplicate(self):
        from trainer.mlops.retraining import (
            RetrainingPolicy,
            RetrainingPolicyManager,
            RetrainingTrigger,
        )

        mgr = RetrainingPolicyManager()
        mgr.add_policy(
            RetrainingPolicy(
                trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                threshold=0.01,
                cooldown_hours=1,
            )
        )
        mgr.evaluate("m1", {"performance_degradation": 0.10})
        recs2 = mgr.evaluate("m1", {"performance_degradation": 0.10})
        assert len(recs2) == 0

    def test_acknowledge_recommendation(self):
        recs = self.mgr.evaluate("m1", {"performance_degradation": 0.10})
        ok = self.mgr.acknowledge(recs[0].recommendation_id, acted_by="human")
        assert ok
        pending = self.mgr.pending_recommendations()
        assert len(pending) == 0

    def test_dismiss_recommendation(self):
        recs = self.mgr.evaluate("m1", {"performance_degradation": 0.10})
        ok = self.mgr.dismiss(recs[0].recommendation_id)
        assert ok

    def test_force_manual(self):
        recs = self.mgr.evaluate("m1", {}, force_manual=True)
        assert any(r.trigger == "manual" for r in recs)

    def test_build_default_policies(self):
        from trainer.mlops.retraining import build_default_policies

        mgr = build_default_policies()
        assert mgr.stats()["total_policies"] >= 4

    def test_stats(self):
        stats = self.mgr.stats()
        assert stats["auto_retrain"] is False
        assert "RECOMMENDATION ONLY" in stats["note"]

    def test_remove_policy(self):
        ok = self.mgr.remove_policy("perf_test")
        assert ok
        assert self.mgr.stats()["total_policies"] == 0


# ===========================================================================
# Database Tests
# ===========================================================================


class TestMLOpsDatabase:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _db(self):
        from trainer.mlops.database import MLOpsDatabase

        return MLOpsDatabase(os.path.join(self.tmpdir, "test_mlops.db"))

    def test_save_and_get_model(self):
        from trainer.mlops.registry import ModelRecord

        db = self._db()
        rec = ModelRecord.from_dict(_model_dict(model_id="m1"))
        db.save_model(rec)
        row = db.get_model("m1")
        assert row is not None
        assert row["name"] == "signalium_quant"
        db.close()

    def test_list_models_by_state(self):
        from trainer.mlops.registry import LifecycleState, ModelRecord

        db = self._db()
        r1 = ModelRecord.from_dict(_model_dict(model_id="m1"))
        r2 = ModelRecord.from_dict(_model_dict(model_id="m2"))
        r2.lifecycle_state = LifecycleState.TRAINING
        db.save_model(r1)
        db.save_model(r2)
        draft_models = db.list_models(lifecycle_state=LifecycleState.DRAFT)
        assert len(draft_models) == 1
        assert draft_models[0]["model_id"] == "m1"
        db.close()

    def test_update_model_state(self):
        from trainer.mlops.registry import ModelRecord

        db = self._db()
        rec = ModelRecord.from_dict(_model_dict(model_id="m1"))
        db.save_model(rec)
        db.update_model_state("m1", "training")
        row = db.get_model("m1")
        assert row["lifecycle_state"] == "training"
        db.close()

    def test_save_and_get_lifecycle_history(self):
        from trainer.mlops.registry import ModelRecord

        db = self._db()
        rec = ModelRecord.from_dict(_model_dict(model_id="m1"))
        db.save_model(rec)
        db.save_transition(
            {
                "model_id": "m1",
                "from_state": "draft",
                "to_state": "training",
                "actor": "trainer",
                "reason": "started",
            }
        )
        history = db.get_lifecycle_history("m1")
        assert len(history) == 1
        db.close()

    def test_save_and_list_deployments(self):
        db = self._db()
        db.save_deployment(
            {
                "model_id": "m1",
                "target": "production",
                "deploy_type": "manual",
                "approver": "ops",
                "status": "active",
            }
        )
        deps = db.list_deployments()
        assert len(deps) == 1
        assert deps[0]["target"] == "production"
        db.close()

    def test_save_and_get_health_report(self):
        db = self._db()
        db.save_health_report(
            {
                "model_id": "m1",
                "confidence_drift": 0.12,
                "is_healthy": False,
                "drift_alerts": ["confidence_drift=0.12"],
                "n_records": 100,
            }
        )
        latest = db.get_latest_health("m1")
        assert latest is not None
        assert latest["confidence_drift"] == pytest.approx(0.12)
        db.close()

    def test_save_and_list_alerts(self):
        db = self._db()
        db.save_alert(
            {
                "level": "CRITICAL",
                "model_id": "m1",
                "category": "latency_spike",
                "message": "p99=250ms",
                "status": "active",
            }
        )
        alerts = db.list_alerts(status="active")
        assert len(alerts) == 1
        assert alerts[0]["level"] == "CRITICAL"
        db.close()

    def test_save_and_list_audit_events(self):
        db = self._db()
        db.save_audit_event(
            {
                "event_type": "deployment",
                "actor": "ops",
                "model_id": "m1",
                "description": "deployed v1",
            }
        )
        events = db.list_audit_events(event_type="deployment")
        assert len(events) == 1
        db.close()

    def test_save_and_list_recommendations(self):
        db = self._db()
        db.save_recommendation(
            {
                "model_id": "m1",
                "trigger": "performance_degradation",
                "reason": "accuracy dropped 0.07",
                "priority": 1,
            }
        )
        recs = db.list_recommendations(model_id="m1", status="pending")
        assert len(recs) == 1
        db.close()

    def test_stats(self):
        db = self._db()
        stats = db.stats()
        assert "total_models" in stats
        assert "total_alerts" in stats
        assert "total_audit_events" in stats
        db.close()

    def test_get_nonexistent_model(self):
        db = self._db()
        assert db.get_model("nonexistent") is None
        db.close()


# ===========================================================================
# API Tests
# ===========================================================================


class TestMLOpsAPI:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _api(self):
        engine = _make_engine(self.tmpdir)
        return engine.api, engine

    def test_register_model(self):
        api, engine = self._api()
        resp = api.register_model(_model_dict())
        assert resp["status"] == "ok"
        assert "model_id" in resp["data"]
        engine.close()

    def test_get_model_not_found(self):
        api, engine = self._api()
        resp = api.get_model("nonexistent")
        assert resp["status"] == "error"
        assert resp["code"] == "NOT_FOUND"
        engine.close()

    def test_list_models_empty(self):
        api, engine = self._api()
        resp = api.list_models()
        assert resp["status"] == "ok"
        engine.close()

    def test_transition(self):
        api, engine = self._api()
        mid = api.register_model(_model_dict())["data"]["model_id"]
        resp = api.transition(mid, "training", actor="trainer")
        assert resp["status"] == "ok"
        engine.close()

    def test_approve_model(self):
        api, engine = self._api()
        mid = api.register_model(_model_dict())["data"]["model_id"]
        api.transition(mid, "training")
        api.transition(mid, "validating")
        api.transition(mid, "candidate")
        resp = api.approve_model(mid, approver="ops")
        assert resp["status"] == "ok"
        engine.close()

    def test_reject_model(self):
        api, engine = self._api()
        mid = api.register_model(_model_dict())["data"]["model_id"]
        api.transition(mid, "training")
        api.transition(mid, "validating")
        api.transition(mid, "candidate")
        resp = api.reject_model(mid, approver="ops", reason="poor metrics")
        assert resp["status"] == "ok"
        engine.close()

    def test_deploy(self):
        api, engine = self._api()
        mid = api.register_model(_model_dict())["data"]["model_id"]
        for state in ["training", "validating", "candidate", "approved"]:
            api.transition(mid, state)
        resp = api.deploy(mid, target="production", approver="ops")
        assert resp["status"] == "ok"
        engine.close()

    def test_rollback(self):
        api, engine = self._api()
        mid = api.register_model(_model_dict())["data"]["model_id"]
        for state in ["training", "validating", "candidate", "approved", "production"]:
            api.transition(mid, state)
        resp = api.rollback(mid, actor="ops", reason="drift", emergency=True)
        assert resp["status"] == "ok"
        engine.close()

    def test_get_health_report(self):
        api, engine = self._api()
        mid = api.register_model(_model_dict())["data"]["model_id"]
        resp = api.get_health_report(mid, save_to_db=False)
        assert resp["status"] == "ok"
        assert "confidence_drift" in resp["data"]
        engine.close()

    def test_get_alerts_empty(self):
        api, engine = self._api()
        resp = api.get_alerts()
        assert resp["status"] == "ok"
        engine.close()

    def test_emit_and_acknowledge_alert(self):
        from trainer.mlops.alerting import AlertCategory, AlertLevel

        api, engine = self._api()
        resp = api.emit_alert(
            AlertLevel.WARNING, "m1", AlertCategory.CONFIDENCE_DRIFT, "Confidence dropped"
        )
        alert_id = resp["data"]["alert_id"]
        ack_resp = api.acknowledge_alert(alert_id, ack_by="ops")
        assert ack_resp["status"] == "ok"
        engine.close()

    def test_get_champion_empty(self):
        api, engine = self._api()
        resp = api.get_champion()
        assert resp["status"] == "ok"
        assert resp["data"]["champion"] is None
        engine.close()

    def test_set_champion(self):
        api, engine = self._api()
        mid = api.register_model(_model_dict())["data"]["model_id"]
        for state in ["training", "validating", "candidate", "approved"]:
            api.transition(mid, state)
        resp = api.set_champion(mid, approver="head_quant")
        assert resp["status"] == "ok"
        champion_resp = api.get_champion()
        assert champion_resp["data"]["champion"]["model_id"] == mid
        engine.close()

    def test_evaluate_retraining_advisory_only(self):
        api, engine = self._api()
        resp = api.evaluate_retraining("m1", {"performance_degradation": 0.10})
        assert resp["status"] == "ok"
        assert "ADVISORY ONLY" in resp["data"]["note"]
        engine.close()

    def test_get_stats(self):
        api, engine = self._api()
        resp = api.get_stats()
        assert resp["status"] == "ok"
        assert "total_models" in resp["data"]
        engine.close()

    def test_get_dashboard(self):
        api, engine = self._api()
        resp = api.get_dashboard()
        assert resp["status"] == "ok"
        assert "registry_summary" in resp["data"]
        engine.close()


# ===========================================================================
# Visualization Tests
# ===========================================================================


class TestMLOpsVisualizationBuilder:
    def setup_method(self):
        from trainer.mlops.visualization import MLOpsVisualizationBuilder

        self.viz = MLOpsVisualizationBuilder()

    def test_registry_status_chart(self):
        models = [
            {"lifecycle_state": "draft"},
            {"lifecycle_state": "draft"},
            {"lifecycle_state": "production"},
        ]
        chart = self.viz.registry_status_chart(models)
        assert chart["chart_type"] == "bar"
        assert chart["total_models"] == 3

    def test_model_health_chart(self):
        health_reports = {
            "m1": {"confidence_drift": 0.12, "is_healthy": False},
            "m2": {"confidence_drift": 0.03, "is_healthy": True},
        }
        chart = self.viz.model_health_chart(health_reports)
        assert chart["chart_type"] == "bar_grouped"
        assert "m1" in chart["unhealthy"]

    def test_deployment_history_chart(self):
        deployments = [
            {
                "deploy_type": "manual",
                "target": "production",
                "status": "active",
                "deployed_at": "2026-07-01",
                "model_id": "m1",
            },
            {
                "deploy_type": "rollback",
                "target": "rollback",
                "status": "rolled_back",
                "deployed_at": "2026-07-02",
                "model_id": "m1",
            },
        ]
        chart = self.viz.deployment_history_chart(deployments)
        assert chart["chart_type"] == "timeline"
        assert chart["total_deployments"] == 2

    def test_champion_comparison_chart(self):
        comparison = {
            "model_a": {"model_id": "m1", "name": "Champion"},
            "model_b": {"model_id": "m2", "name": "Challenger"},
            "overall_winner_name": "Champion",
            "metrics": {
                "accuracy": {"model_a": 0.70, "model_b": 0.65, "winner": "m1"},
            },
        }
        chart = self.viz.champion_comparison_chart(comparison)
        assert chart["chart_type"] == "bar_grouped"
        assert chart["overall_winner"] == "Champion"

    def test_alert_distribution_chart(self):
        alerts = [
            {"category": "confidence_drift", "level": "WARNING"},
            {"category": "latency_spike", "level": "CRITICAL"},
            {"category": "confidence_drift", "level": "CRITICAL"},
        ]
        chart = self.viz.alert_distribution_chart(alerts)
        assert chart["total_alerts"] == 3
        assert chart["by_level"]["CRITICAL"] == 2

    def test_confidence_trend_chart(self):
        history = [0.8 - i * 0.01 for i in range(50)]
        chart = self.viz.confidence_trend_chart("m1", history, window=10)
        assert chart["chart_type"] == "line"
        assert chart["n_points"] == 50

    def test_drift_heatmap(self):
        health_reports = {
            "m1": {
                "confidence_drift": 0.12,
                "prediction_drift": 0.08,
                "calibration_drift": 0.03,
                "performance_degradation": 0.05,
                "latency_degradation": 15.0,
                "distribution_drift": 0.10,
            },
        }
        chart = self.viz.drift_heatmap(health_reports)
        assert chart["chart_type"] == "heatmap"
        assert "m1" in chart["matrix"]

    def test_latency_chart(self):
        snaps = {
            "m1": {
                "avg_latency_ms": 12.5,
                "p95_latency_ms": 30.0,
                "p99_latency_ms": 80.0,
                "latency_alert": False,
            },
            "m2": {
                "avg_latency_ms": 120.0,
                "p95_latency_ms": 200.0,
                "p99_latency_ms": 300.0,
                "latency_alert": True,
            },
        }
        chart = self.viz.latency_chart(snaps)
        assert chart["chart_type"] == "bar_grouped"
        assert "m2" in chart["alerts"]

    def test_full_dashboard(self):
        charts = self.viz.full_dashboard(
            models=[{"lifecycle_state": "production"}],
            health_reports={
                "m1": {
                    "confidence_drift": 0.0,
                    "is_healthy": True,
                    "prediction_drift": 0.0,
                    "calibration_drift": 0.0,
                    "performance_degradation": 0.0,
                    "latency_degradation": 0.0,
                    "distribution_drift": 0.0,
                }
            },
            deployments=[
                {
                    "deploy_type": "manual",
                    "target": "production",
                    "status": "active",
                    "deployed_at": "2026-07-01",
                    "model_id": "m1",
                }
            ],
            comparison=None,
            alerts=[{"category": "system", "level": "INFO"}],
            monitoring_snapshots={
                "m1": {
                    "avg_latency_ms": 12.0,
                    "p95_latency_ms": 25.0,
                    "p99_latency_ms": 40.0,
                    "latency_alert": False,
                }
            },
        )
        expected = [
            "registry_status",
            "model_health",
            "deployment_history",
            "alert_distribution",
            "drift_heatmap",
            "latency",
        ]
        for key in expected:
            assert key in charts


# ===========================================================================
# Export Tests
# ===========================================================================


class TestMLOpsExporter:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _engine(self):
        return _make_engine(self.tmpdir)

    def test_export_registry(self):
        engine = self._engine()
        engine.register_model(_model_dict())
        report = engine.exporter.export_registry()
        assert report["export_type"] == "model_registry"
        assert report["total_models"] == 1
        engine.close()

    def test_export_audit_report(self):
        engine = self._engine()
        engine.register_model(_model_dict())
        report = engine.exporter.export_audit_report()
        assert report["export_type"] == "audit_report"
        assert report["total_events"] >= 1
        engine.close()

    def test_export_deployment_report(self):
        engine = self._engine()
        report = engine.exporter.export_deployment_report()
        assert report["export_type"] == "deployment_report"
        assert "history" in report
        engine.close()

    def test_export_health_report(self):
        engine = self._engine()
        report = engine.exporter.export_health_report()
        assert report["export_type"] == "health_report"
        engine.close()

    def test_export_alert_history(self):
        engine = self._engine()
        report = engine.exporter.export_alert_history()
        assert report["export_type"] == "alert_history"
        engine.close()

    def test_export_readiness_report(self):
        engine = self._engine()
        engine.register_model(_model_dict())
        report = engine.exporter.export_mlops_readiness_report()
        assert report["report_type"] == "mlops_readiness_report"
        assert "registry" in report
        assert "governance" in report
        assert "monitoring" in report
        assert "health" in report
        assert "alerting" in report
        assert "audit" in report
        assert "retraining" in report
        assert "ADVISORY ONLY" in report["retraining"]["note"]
        assert report["backward_compatibility"]["neural_network_unchanged"]
        engine.close()

    def test_save_and_load(self):
        engine = self._engine()
        engine.register_model(_model_dict())
        report = engine.exporter.export_registry()
        path = os.path.join(self.tmpdir, "registry.json")
        saved = engine.exporter.save(report, path)
        assert os.path.exists(saved)
        loaded = engine.exporter.load(saved)
        assert loaded["export_type"] == "model_registry"
        engine.close()


# ===========================================================================
# Engine Tests
# ===========================================================================


class TestMLOpsEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _engine(self):
        return _make_engine(self.tmpdir)

    def test_register_model(self):
        engine = self._engine()
        mid = engine.register_model(_model_dict())
        assert engine.registry.get(mid) is not None
        engine.close()

    def test_full_lifecycle_happy_path(self):

        engine = self._engine()
        mid = engine.register_model(_model_dict())
        for state in ["training", "validating", "candidate"]:
            resp = engine.transition(mid, state)
            assert resp["status"] == "ok"
        resp = engine.approve(mid, approver="head_quant")
        assert resp["status"] == "ok"
        engine.close()

    def test_set_champion(self):
        engine = self._engine()
        mid = engine.register_model(_model_dict())
        for state in ["training", "validating", "candidate"]:
            engine.transition(mid, state)
        engine.approve(mid, approver="head_quant")
        resp = engine.set_champion(mid, approver="head_quant")
        assert resp["status"] == "ok"
        assert engine.champion_manager.current_champion["model_id"] == mid
        engine.close()

    def test_deploy(self):
        engine = self._engine()
        mid = engine.register_model(_model_dict())
        for state in ["training", "validating", "candidate", "approved"]:
            engine.transition(mid, state)
        resp = engine.deploy(mid, target="production", approver="ops")
        assert resp["status"] == "ok"
        engine.close()

    def test_rollback(self):
        engine = self._engine()
        mid = engine.register_model(_model_dict())
        for state in ["training", "validating", "candidate", "approved", "production"]:
            engine.transition(mid, state)
        resp = engine.rollback(mid, actor="ops", reason="drift")
        assert resp["status"] == "ok"
        engine.close()

    def test_record_inference_and_snapshot(self):
        engine = self._engine()
        mid = engine.register_model(_model_dict())
        for _ in range(10):
            engine.record_inference(mid, latency_ms=15.0, confidence=0.78, predicted_class="buy")
        snap = engine.monitor.get_snapshot(mid)
        assert snap.n_records == 10
        assert snap.avg_latency_ms == pytest.approx(15.0)
        engine.close()

    def test_check_health(self):
        engine = self._engine()
        mid = engine.register_model(_model_dict())
        resp = engine.check_health(mid)
        assert resp["status"] == "ok"
        engine.close()

    def test_emit_alert(self):
        from trainer.mlops.alerting import AlertCategory, AlertLevel

        engine = self._engine()
        resp = engine.emit_alert(
            AlertLevel.CRITICAL, "m1", AlertCategory.MODEL_FAILURE, "Model returned NaN"
        )
        assert resp["status"] == "ok"
        active = engine.active_alerts()
        assert len(active) >= 1
        engine.close()

    def test_evaluate_retraining_no_auto(self):
        engine = self._engine()
        resp = engine.evaluate_retraining("m1", {"performance_degradation": 0.10})
        assert resp["status"] == "ok"
        assert "ADVISORY ONLY" in resp["data"]["note"]
        assert engine.retraining.auto_retrain is False
        engine.close()

    def test_retire_model(self):
        from trainer.mlops.registry import LifecycleState

        engine = self._engine()
        mid = engine.register_model(_model_dict())
        engine.retire_model(mid, reason="end of life")
        model = engine.registry.get(mid)
        assert model.lifecycle_state == LifecycleState.RETIRED
        engine.close()

    def test_get_dashboard(self):
        engine = self._engine()
        resp = engine.get_dashboard()
        assert resp["status"] == "ok"
        engine.close()

    def test_get_readiness_report(self):
        engine = self._engine()
        engine.register_model(_model_dict())
        report = engine.get_readiness_report()
        assert report["report_type"] == "mlops_readiness_report"
        assert "readiness_checks" in report
        engine.close()

    def test_summary_string(self):
        engine = self._engine()
        s = engine.summary()
        assert "MLOps Platform" in s
        engine.close()

    def test_from_config(self):
        from trainer.mlops.engine import MLOpsEngine

        engine = MLOpsEngine.from_config(
            {
                "db_path": os.path.join(self.tmpdir, "from_config.db"),
            }
        )
        assert engine is not None
        engine.close()

    def test_add_challenger_and_compare(self):
        engine = self._engine()
        mid1 = engine.register_model(
            _model_dict(model_id="a1", version="1.0.0", validation_report={"accuracy": 0.70})
        )
        mid2 = engine.register_model(
            _model_dict(model_id="a2", version="2.0.0", validation_report={"accuracy": 0.65})
        )
        engine.set_champion(mid1)
        engine.add_challenger(mid2, reason="new version")
        resp = engine.compare_models(mid1, mid2)
        assert resp["status"] == "ok"
        assert resp["data"]["overall_winner"] == mid1
        engine.close()

    def test_export_registry(self):
        engine = self._engine()
        engine.register_model(_model_dict())
        report = engine.export_registry()
        assert "models" in report
        engine.close()


# ===========================================================================
# Public API (__all__ completeness)
# ===========================================================================


class TestPublicAPI:
    def test_all_exports_importable(self):
        pass

    def test_all_names_in_module(self):
        import trainer.mlops as m

        for name in m.__all__:
            assert hasattr(m, name), f"Missing from module: {name}"
