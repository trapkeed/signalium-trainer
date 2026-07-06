"""Phase 18 — MLOps Engine.

The MLOpsEngine is the central orchestrator for the Institutional MLOps,
Model Governance, and Continuous Monitoring Platform.

It coordinates:
  1. Model Registry (registration, versioning, metadata)
  2. Lifecycle Management (state transitions, approvals)
  3. Deployment Governance (workflows, rollback, audit)
  4. Production Monitoring (latency, throughput, resources)
  5. Model Health (drift detection across 7 dimensions)
  6. Alerting (drift, latency, confidence, API, model failures)
  7. Champion/Challenger Management
  8. Audit System (full immutable event trail)
  9. Retraining Policies (advisory recommendations only)
  10. Database Persistence (SQLite, 8 tables)
  11. Dashboard Generation
  12. Export and Reporting

No neural network is modified. No model is automatically retrained.
Full backward compatibility with Phases 1–17.

Usage::

    engine = MLOpsEngine.from_config({
        "db_path": "experiments/mlops/mlops.db",
    })
    model_id = engine.register_model({
        "name": "signalium_quant", "version": "1.0.0",
        "architecture": "lstm", ...
    })
    engine.set_champion(model_id)
    engine.record_inference(model_id, latency_ms=12.0, confidence=0.78)
    report = engine.get_readiness_report()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from trainer.mlops.alerting import AlertingSystem
from trainer.mlops.api import MLOpsAPI
from trainer.mlops.audit import AuditSystem, AuditEventType
from trainer.mlops.champion import ChampionChallengerManager
from trainer.mlops.database import MLOpsDatabase
from trainer.mlops.export import MLOpsExporter
from trainer.mlops.governance import DeploymentGovernance, GovernanceConfig
from trainer.mlops.health import DriftThresholds, ModelHealthMonitor
from trainer.mlops.lifecycle import LifecycleManager
from trainer.mlops.monitoring import ProductionMonitor
from trainer.mlops.registry import LifecycleState, ModelRecord, ModelRegistry
from trainer.mlops.retraining import (
    RetrainingPolicyManager,
    build_default_policies,
)
from trainer.mlops.visualization import MLOpsVisualizationBuilder

# ---------------------------------------------------------------------------
# Engine config
# ---------------------------------------------------------------------------


@dataclass
class MLOpsConfig:
    """Configuration for the MLOpsEngine.

    Attributes:
        db_path: SQLite database file path.
        monitoring_window: Rolling window for production monitoring.
        health_baseline_window: Baseline window for drift detection.
        health_recent_window: Recent window for drift detection.
        latency_alert_ms: Latency threshold for alerts.
        memory_alert_mb: Memory threshold for alerts.
        governance_require_approver: Whether deployments require a named approver.
        governance_auto_promote: Whether auto-promotion is enabled.
        export_dir: Directory for exports.
        enable_default_retraining_policies: Whether to load default policies.
    """

    db_path: str = "experiments/mlops/mlops.db"
    monitoring_window: int = 500
    health_baseline_window: int = 200
    health_recent_window: int = 50
    latency_alert_ms: float = 100.0
    memory_alert_mb: float = 8192.0
    confidence_alert_threshold: float = 0.40
    governance_require_approver: bool = True
    governance_auto_promote: bool = False
    export_dir: str = "experiments/mlops/exports"
    enable_default_retraining_policies: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MLOpsConfig":
        cfg = MLOpsConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# MLOps engine
# ---------------------------------------------------------------------------


class MLOpsEngine:
    """Central orchestrator for the Institutional MLOps Platform.

    No neural network changes. No automatic retraining.
    Full backward compatibility with Phases 1–17.

    Args:
        config: MLOpsConfig.
    """

    def __init__(self, config: Optional[MLOpsConfig] = None) -> None:
        self.config = config or MLOpsConfig()
        self.db = MLOpsDatabase(self.config.db_path)
        self.registry = ModelRegistry()
        self.lifecycle = LifecycleManager(strict=True)
        self.governance = DeploymentGovernance(
            registry=self.registry,
            lifecycle=self.lifecycle,
            config=GovernanceConfig(
                require_approver=self.config.governance_require_approver,
                auto_promote_enabled=self.config.governance_auto_promote,
            ),
        )
        self.monitor = ProductionMonitor(
            window=self.config.monitoring_window,
            latency_alert_ms=self.config.latency_alert_ms,
            memory_alert_mb=self.config.memory_alert_mb,
            confidence_alert_threshold=self.config.confidence_alert_threshold,
        )
        self.health = ModelHealthMonitor(
            baseline_window=self.config.health_baseline_window,
            recent_window=self.config.health_recent_window,
            thresholds=DriftThresholds(),
        )
        self.alerting = AlertingSystem()
        self.audit = AuditSystem()
        self.champion_manager = ChampionChallengerManager(
            registry=self.registry,
            lifecycle=self.lifecycle,
            governance=self.governance,
        )
        self.retraining = (
            build_default_policies()
            if self.config.enable_default_retraining_policies
            else RetrainingPolicyManager()
        )
        self.viz = MLOpsVisualizationBuilder()
        self.api = MLOpsAPI(
            db=self.db,
            registry=self.registry,
            lifecycle=self.lifecycle,
            governance=self.governance,
            health_monitor=self.health,
            monitor=self.monitor,
            alerting=self.alerting,
            audit=self.audit,
            champion_manager=self.champion_manager,
            retraining=self.retraining,
        )
        self.exporter = MLOpsExporter(
            db=self.db,
            registry=self.registry,
            audit=self.audit,
            alerting=self.alerting,
            governance=self.governance,
            health=self.health,
            monitor=self.monitor,
            retraining=self.retraining,
        )
        self.audit.log(
            AuditEventType.SYSTEM,
            actor="engine",
            description="MLOpsEngine initialized",
            metadata=self.config.to_dict(),
        )
        logger.info(f"MLOpsEngine initialized: db={self.config.db_path}")

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def register_model(self, model_dict: Dict[str, Any]) -> str:
        """Register a model in the registry and database. Returns model_id."""
        resp = self.api.register_model(model_dict)
        if resp["status"] == "error":
            raise ValueError(resp["message"])
        return resp["data"]["model_id"]

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        return self.registry.get(model_id)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def transition(
        self, model_id: str, to_state: str, actor: str = "system", reason: str = ""
    ) -> Dict[str, Any]:
        """Transition a model to a new lifecycle state."""
        return self.api.transition(model_id, to_state, actor=actor, reason=reason)

    def approve(self, model_id: str, approver: str, reason: str = "") -> Dict[str, Any]:
        """Approve a candidate model."""
        return self.api.approve_model(model_id, approver=approver, reason=reason)

    def reject(self, model_id: str, approver: str, reason: str = "") -> Dict[str, Any]:
        """Reject a candidate model."""
        return self.api.reject_model(model_id, approver=approver, reason=reason)

    def retire_model(self, model_id: str, actor: str = "system", reason: str = "") -> None:
        """Retire a model (ARCHIVED → RETIRED)."""
        model = self.registry.get_or_raise(model_id)
        self.lifecycle.retire(model, actor=actor, reason=reason)
        self.db.save_model(model)
        self.audit.log(
            AuditEventType.RETIREMENT,
            actor=actor,
            model_id=model_id,
            description=f"Model retired: {reason}",
        )

    # -----------------------------------------------------------------------
    # Deployment
    # -----------------------------------------------------------------------

    def deploy(
        self, model_id: str, target: str = "production", approver: str = "system", reason: str = ""
    ) -> Dict[str, Any]:
        """Deploy a model to production/champion/shadow."""
        return self.api.deploy(
            model_id, target=target, approver=approver, actor=approver, reason=reason
        )

    def rollback(
        self, model_id: str, actor: str = "ops", reason: str = "", emergency: bool = True
    ) -> Dict[str, Any]:
        """Roll back a deployed model."""
        return self.api.rollback(model_id, actor=actor, reason=reason, emergency=emergency)

    # -----------------------------------------------------------------------
    # Champion
    # -----------------------------------------------------------------------

    def set_champion(
        self, model_id: str, approver: str = "system", reason: str = ""
    ) -> Dict[str, Any]:
        """Set a model as champion."""
        return self.api.set_champion(model_id, approver=approver, reason=reason)

    def add_challenger(self, model_id: str, reason: str = "") -> Dict[str, Any]:
        """Add a model to the challenger queue."""
        return self.api.add_challenger(model_id, reason=reason)

    def compare_models(
        self,
        model_a_id: str,
        model_b_id: str,
        metrics_a: Optional[Dict[str, float]] = None,
        metrics_b: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compare two models."""
        return self.api.compare_models(
            model_a_id, model_b_id, metrics_a=metrics_a, metrics_b=metrics_b
        )

    # -----------------------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------------------

    def record_inference(
        self,
        model_id: str,
        latency_ms: float = 0.0,
        confidence: float = 0.5,
        predicted_class: str = "no_trade",
        buy_prob: float = 0.33,
        sell_prob: float = 0.33,
        no_trade_prob: float = 0.34,
        cpu_pct: float = 0.0,
        gpu_pct: float = 0.0,
        memory_mb: float = 0.0,
        true_class: Optional[str] = None,
        regime: str = "",
    ) -> None:
        """Record one inference event for monitoring and health tracking."""
        self.monitor.record_inference(
            model_id=model_id,
            latency_ms=latency_ms,
            confidence=confidence,
            predicted_class=predicted_class,
            buy_prob=buy_prob,
            sell_prob=sell_prob,
            no_trade_prob=no_trade_prob,
            cpu_pct=cpu_pct,
            gpu_pct=gpu_pct,
            memory_mb=memory_mb,
            regime=regime,
        )
        self.health.record_prediction(
            model_id=model_id,
            buy_prob=buy_prob,
            sell_prob=sell_prob,
            no_trade_prob=no_trade_prob,
            confidence=confidence,
            predicted_class=predicted_class,
            true_class=true_class,
            latency_ms=latency_ms,
            regime=regime,
        )

    def check_health(self, model_id: str) -> Dict[str, Any]:
        """Check model health and emit alerts if needed."""
        return self.api.get_health_report(model_id)

    def check_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health for all registered models."""
        return {
            m.model_id: self.api.get_health_report(m.model_id) for m in self.registry.all_models()
        }

    # -----------------------------------------------------------------------
    # Alerting
    # -----------------------------------------------------------------------

    def emit_alert(
        self,
        level: str,
        model_id: str,
        category: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Emit a manual alert."""
        return self.api.emit_alert(level, model_id, category, message, metadata=metadata)

    def active_alerts(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all active alerts."""
        return self.alerting.active_alerts(model_id=model_id)

    # -----------------------------------------------------------------------
    # Retraining
    # -----------------------------------------------------------------------

    def evaluate_retraining(self, model_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate retraining policies. Returns recommendations only."""
        return self.api.evaluate_retraining(model_id, metrics)

    # -----------------------------------------------------------------------
    # Exports and dashboard
    # -----------------------------------------------------------------------

    def get_dashboard(self) -> Dict[str, Any]:
        """Return the full MLOps dashboard."""
        return self.api.get_dashboard()

    def get_readiness_report(self) -> Dict[str, Any]:
        """Generate the full institutional MLOps readiness report."""
        return self.exporter.export_mlops_readiness_report()

    def export_registry(self, path: Optional[str] = None) -> Dict[str, Any]:
        data = self.exporter.export_registry()
        if path:
            self.exporter.save(data, path)
        return data

    def export_audit_report(
        self, model_id: Optional[str] = None, path: Optional[str] = None
    ) -> Dict[str, Any]:
        data = self.exporter.export_audit_report(model_id=model_id)
        if path:
            self.exporter.save(data, path)
        return data

    def export_deployment_report(self, path: Optional[str] = None) -> Dict[str, Any]:
        data = self.exporter.export_deployment_report()
        if path:
            self.exporter.save(data, path)
        return data

    def export_health_report(self, path: Optional[str] = None) -> Dict[str, Any]:
        data = self.exporter.export_health_report()
        if path:
            self.exporter.save(data, path)
        return data

    def summary(self) -> str:
        reg = self.registry.summary()
        champ = self.champion_manager.current_champion
        alerts = self.alerting.stats()
        return (
            f"=== Signalium MLOps Platform — Phase 18 ===\n"
            f"  DB: {self.config.db_path}\n"
            f"  Models: {reg['total_models']} total\n"
            f"  Champion: {champ['name'] if champ else 'None'}\n"
            f"  Active Alerts: {alerts.get('active', 0)} "
            f"(CRITICAL: {alerts.get('critical', 0)})\n"
            f"  Audit Events: {self.audit.n_events}\n"
            f"  Inferences Monitored: {self.monitor.total_inferences()}\n"
            f"  Retraining: advisory-only (no auto-retrain)\n"
        )

    def close(self) -> None:
        self.db.close()

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @staticmethod
    def from_config(config_dict: Dict[str, Any]) -> "MLOpsEngine":
        """Create MLOpsEngine from a config dict."""
        return MLOpsEngine(config=MLOpsConfig.from_dict(config_dict))

    @staticmethod
    def from_yaml(yaml_path: str) -> "MLOpsEngine":
        """Create MLOpsEngine from a YAML config file."""
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        cfg_data = data.get("mlops", data)
        return MLOpsEngine.from_config(cfg_data)
