"""Phase 18 — MLOps API.

Structured endpoint layer for the Institutional MLOps Platform.

All methods return::

    {
      "status": "ok" | "error",
      "code": "...",
      "message": "...",
      "data": { ... },
      "timestamp": "2026-07-05T12:00:00Z"
    }

Endpoint groups:
  Registry:    register_model, get_model, list_models, update_model_state
  Lifecycle:   transition, get_lifecycle_history
  Deployment:  deploy, rollback, list_deployments, active_deployments
  Health:      get_health_report, list_health_history, unhealthy_models
  Alerts:      get_alerts, acknowledge_alert, resolve_alert, emit_alert
  Audit:       get_audit_events, get_audit_report
  Champion:    get_champion, set_champion, add_challenger, promote_challenger,
               compare_models, rollback_champion, champion_history
  Retraining:  evaluate_policies, list_recommendations, dismiss_recommendation
  Stats:       get_stats, get_dashboard
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from trainer.mlops.alerting import AlertingSystem, AlertLevel, AlertCategory
from trainer.mlops.audit import AuditSystem, AuditEventType
from trainer.mlops.champion import ChampionChallengerManager
from trainer.mlops.database import MLOpsDatabase
from trainer.mlops.governance import DeploymentGovernance
from trainer.mlops.health import ModelHealthMonitor
from trainer.mlops.lifecycle import LifecycleManager
from trainer.mlops.monitoring import ProductionMonitor
from trainer.mlops.registry import ModelRecord, ModelRegistry
from trainer.mlops.retraining import RetrainingPolicyManager


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(data: Any, message: str = "ok") -> Dict[str, Any]:
    return {"status": "ok", "message": message, "data": data, "timestamp": _ts()}


def _err(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "code": code, "message": message, "data": None, "timestamp": _ts()}


class MLOpsAPI:
    """Institutional MLOps Platform API.

    All endpoints return a standard response envelope.
    """

    def __init__(
        self,
        db: MLOpsDatabase,
        registry: ModelRegistry,
        lifecycle: LifecycleManager,
        governance: DeploymentGovernance,
        health_monitor: ModelHealthMonitor,
        monitor: ProductionMonitor,
        alerting: AlertingSystem,
        audit: AuditSystem,
        champion_manager: ChampionChallengerManager,
        retraining: RetrainingPolicyManager,
    ) -> None:
        self.db = db
        self.registry = registry
        self.lifecycle = lifecycle
        self.governance = governance
        self.health = health_monitor
        self.monitor = monitor
        self.alerting = alerting
        self.audit = audit
        self.champion = champion_manager
        self.retraining = retraining

    # -----------------------------------------------------------------------
    # Registry endpoints
    # -----------------------------------------------------------------------

    def register_model(self, model_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            record = ModelRecord.from_dict(model_dict)
            self.registry.register(record)
            self.db.save_model(record)
            self.audit.log(
                AuditEventType.REGISTRY_REGISTER,
                actor=model_dict.get("registered_by", "system"),
                model_id=record.model_id,
                description=f"Model registered: {record.full_name}",
                metadata={"version": record.version, "architecture": record.architecture},
            )
            return _ok(record.to_dict(), "Model registered.")
        except Exception as e:
            logger.error(f"register_model: {e}")
            return _err("REGISTER_ERROR", str(e))

    def get_model(self, model_id: str) -> Dict[str, Any]:
        try:
            model = self.registry.get(model_id)
            if model is None:
                model_row = self.db.get_model(model_id)
                if model_row is None:
                    return _err("NOT_FOUND", f"Model '{model_id}' not found.")
                return _ok(model_row)
            return _ok(model.to_dict())
        except Exception as e:
            return _err("GET_MODEL_ERROR", str(e))

    def list_models(
        self, lifecycle_state: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        try:
            if lifecycle_state:
                models = [m.to_dict() for m in self.registry.by_state(lifecycle_state)]
            else:
                models = [m.to_dict() for m in self.registry.all_models()]
            return _ok({"models": models[:limit], "count": len(models[:limit])})
        except Exception as e:
            return _err("LIST_MODELS_ERROR", str(e))

    def update_model_notes(
        self, model_id: str, notes: str, actor: str = "system"
    ) -> Dict[str, Any]:
        try:
            model = self.registry.get_or_raise(model_id)
            model.notes = notes
            self.db.save_model(model)
            self.audit.log(
                AuditEventType.CONFIG_CHANGE,
                actor=actor,
                model_id=model_id,
                description="Notes updated",
            )
            return _ok({"model_id": model_id, "notes": notes})
        except Exception as e:
            return _err("UPDATE_NOTES_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Lifecycle endpoints
    # -----------------------------------------------------------------------

    def transition(
        self,
        model_id: str,
        to_state: str,
        actor: str = "system",
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            model = self.registry.get_or_raise(model_id)
            t = self.lifecycle.transition(
                model, to_state, actor=actor, reason=reason, metadata=metadata
            )
            self.db.save_model(model)
            self.db.save_transition(t.to_dict())
            self.audit.log(
                AuditEventType.LIFECYCLE_TRANSITION,
                actor=actor,
                model_id=model_id,
                description=f"State: {t.from_state} → {t.to_state}",
                metadata={"from": t.from_state, "to": t.to_state, "reason": reason},
            )
            return _ok(t.to_dict())
        except Exception as e:
            return _err("TRANSITION_ERROR", str(e))

    def approve_model(self, model_id: str, approver: str, reason: str = "") -> Dict[str, Any]:
        try:
            model = self.registry.get_or_raise(model_id)
            t = self.lifecycle.approve(model, approver=approver, reason=reason)
            self.db.save_model(model)
            self.db.save_transition(t.to_dict())
            self.audit.log(
                AuditEventType.MANUAL_APPROVAL,
                actor=approver,
                model_id=model_id,
                description=f"Model approved by {approver}: {reason}",
            )
            return _ok(t.to_dict())
        except Exception as e:
            return _err("APPROVE_ERROR", str(e))

    def reject_model(self, model_id: str, approver: str, reason: str = "") -> Dict[str, Any]:
        try:
            model = self.registry.get_or_raise(model_id)
            self.lifecycle.reject(model, approver=approver, reason=reason)
            self.db.save_model(model)
            self.audit.log(
                AuditEventType.MANUAL_REJECTION,
                actor=approver,
                model_id=model_id,
                description=f"Model rejected by {approver}: {reason}",
            )
            return _ok({"model_id": model_id, "status": "rejected"})
        except Exception as e:
            return _err("REJECT_ERROR", str(e))

    def get_lifecycle_history(self, model_id: str) -> Dict[str, Any]:
        try:
            history = self.db.get_lifecycle_history(model_id)
            return _ok({"model_id": model_id, "history": history, "count": len(history)})
        except Exception as e:
            return _err("LIFECYCLE_HISTORY_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Deployment endpoints
    # -----------------------------------------------------------------------

    def deploy(
        self,
        model_id: str,
        target: str = "production",
        approver: str = "system",
        actor: str = "system",
        reason: str = "",
        deploy_type: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            rec = self.governance.deploy(
                model_id,
                target=target,
                approver=approver,
                actor=actor,
                reason=reason,
                deploy_type=deploy_type,
                metadata=metadata,
            )
            self.db.save_deployment(rec.to_dict())
            model = self.registry.get(model_id)
            if model:
                self.db.save_model(model)
            self.audit.log(
                AuditEventType.DEPLOYMENT,
                actor=actor,
                model_id=model_id,
                description=f"Deployed to {target}",
                metadata={
                    "deployment_id": rec.deployment_id,
                    "target": target,
                    "type": deploy_type,
                },
            )
            return _ok(rec.to_dict(), f"Deployed to {target}.")
        except Exception as e:
            logger.error(f"deploy: {e}")
            return _err("DEPLOY_ERROR", str(e))

    def rollback(
        self, model_id: str, actor: str = "ops", reason: str = "", emergency: bool = False
    ) -> Dict[str, Any]:
        try:
            rec = self.governance.rollback(
                model_id, actor=actor, reason=reason, emergency=emergency
            )
            self.db.save_deployment(rec.to_dict())
            model = self.registry.get(model_id)
            if model:
                self.db.save_model(model)
            self.audit.log(
                AuditEventType.ROLLBACK,
                actor=actor,
                model_id=model_id,
                description=f"Rollback: {reason}",
                metadata={"emergency": emergency},
            )
            return _ok(rec.to_dict(), "Rollback complete.")
        except Exception as e:
            logger.error(f"rollback: {e}")
            return _err("ROLLBACK_ERROR", str(e))

    def list_deployments(self, model_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        try:
            deployments = self.db.list_deployments(model_id=model_id, limit=limit)
            return _ok({"deployments": deployments, "count": len(deployments)})
        except Exception as e:
            return _err("LIST_DEPLOYMENTS_ERROR", str(e))

    def active_deployments(self) -> Dict[str, Any]:
        try:
            active = self.governance.active_deployments()
            return _ok({"active": active, "count": len(active)})
        except Exception as e:
            return _err("ACTIVE_DEPLOYMENTS_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Health endpoints
    # -----------------------------------------------------------------------

    def get_health_report(self, model_id: str, save_to_db: bool = True) -> Dict[str, Any]:
        try:
            report = self.health.get_health_report(model_id)
            if save_to_db:
                self.db.save_health_report(report.to_dict())
            alerts = self.alerting.emit_from_health_report(report)
            for a in alerts:
                self.db.save_alert(a.to_dict())
            return _ok(report.to_dict())
        except Exception as e:
            return _err("HEALTH_REPORT_ERROR", str(e))

    def list_health_history(self, model_id: str, limit: int = 20) -> Dict[str, Any]:
        try:
            history = self.db.list_health_history(model_id, limit=limit)
            return _ok({"model_id": model_id, "history": history, "count": len(history)})
        except Exception as e:
            return _err("HEALTH_HISTORY_ERROR", str(e))

    def unhealthy_models(self) -> Dict[str, Any]:
        try:
            models = self.health.unhealthy_models()
            return _ok({"unhealthy": models, "count": len(models)})
        except Exception as e:
            return _err("UNHEALTHY_MODELS_ERROR", str(e))

    def get_monitoring_snapshot(self, model_id: str) -> Dict[str, Any]:
        try:
            snap = self.monitor.get_snapshot(model_id)
            alerts = self.alerting.emit_from_monitoring_snapshot(snap)
            for a in alerts:
                self.db.save_alert(a.to_dict())
            return _ok(snap.to_dict())
        except Exception as e:
            return _err("MONITORING_SNAPSHOT_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Alert endpoints
    # -----------------------------------------------------------------------

    def get_alerts(
        self,
        model_id: Optional[str] = None,
        level: Optional[str] = None,
        active_only: bool = True,
        limit: int = 50,
    ) -> Dict[str, Any]:
        try:
            if active_only:
                alerts = self.alerting.active_alerts(model_id=model_id, level=level)
            else:
                alerts = self.alerting.all_alerts(limit=limit, model_id=model_id)
            return _ok({"alerts": alerts, "count": len(alerts)})
        except Exception as e:
            return _err("GET_ALERTS_ERROR", str(e))

    def acknowledge_alert(self, alert_id: str, ack_by: str = "ops") -> Dict[str, Any]:
        try:
            ok = self.alerting.acknowledge(alert_id, ack_by=ack_by)
            if not ok:
                return _err("NOT_FOUND", f"Alert '{alert_id}' not found.")
            self.audit.log(
                AuditEventType.ALERT_ACKNOWLEDGED,
                actor=ack_by,
                description=f"Alert acknowledged: {alert_id}",
            )
            return _ok({"alert_id": alert_id, "status": "acknowledged"})
        except Exception as e:
            return _err("ACK_ALERT_ERROR", str(e))

    def resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        try:
            ok = self.alerting.resolve(alert_id)
            if not ok:
                return _err("NOT_FOUND", f"Alert '{alert_id}' not found.")
            return _ok({"alert_id": alert_id, "status": "resolved"})
        except Exception as e:
            return _err("RESOLVE_ALERT_ERROR", str(e))

    def emit_alert(
        self,
        level: str,
        model_id: str,
        category: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            alert = self.alerting.emit(level, model_id, category, message, metadata=metadata)
            self.db.save_alert(alert.to_dict())
            return _ok(alert.to_dict(), "Alert emitted.")
        except Exception as e:
            return _err("EMIT_ALERT_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Audit endpoints
    # -----------------------------------------------------------------------

    def get_audit_events(
        self, event_type: Optional[str] = None, model_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        try:
            events = self.db.list_audit_events(
                event_type=event_type, model_id=model_id, limit=limit
            )
            return _ok({"events": events, "count": len(events)})
        except Exception as e:
            return _err("AUDIT_EVENTS_ERROR", str(e))

    def get_audit_report(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            report = self.audit.get_report(model_id=model_id)
            return _ok(report)
        except Exception as e:
            return _err("AUDIT_REPORT_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Champion endpoints
    # -----------------------------------------------------------------------

    def get_champion(self) -> Dict[str, Any]:
        try:
            return _ok(
                {
                    "champion": self.champion.current_champion,
                    "challengers": self.champion.challengers,
                }
            )
        except Exception as e:
            return _err("GET_CHAMPION_ERROR", str(e))

    def set_champion(
        self, model_id: str, approver: str = "head_quant", reason: str = ""
    ) -> Dict[str, Any]:
        try:
            event = self.champion.set_champion(model_id, approver=approver, reason=reason)
            model = self.registry.get(model_id)
            if model:
                self.db.save_model(model)
            self.db.save_audit_event(
                self.audit.log(
                    AuditEventType.CHAMPION_CHANGE,
                    actor=approver,
                    model_id=model_id,
                    description=f"New champion: {model.full_name if model else model_id}",
                ).to_dict()
            )
            return _ok(event.to_dict(), "Champion set.")
        except Exception as e:
            logger.error(f"set_champion: {e}")
            return _err("SET_CHAMPION_ERROR", str(e))

    def add_challenger(self, model_id: str, reason: str = "") -> Dict[str, Any]:
        try:
            self.champion.add_challenger(model_id, reason=reason)
            model = self.registry.get(model_id)
            if model:
                self.db.save_model(model)
            return _ok({"model_id": model_id, "status": "challenger"})
        except Exception as e:
            return _err("ADD_CHALLENGER_ERROR", str(e))

    def promote_challenger(self, model_id: str, approver: str, reason: str = "") -> Dict[str, Any]:
        try:
            event = self.champion.promote_challenger(model_id, approver=approver, reason=reason)
            model = self.registry.get(model_id)
            if model:
                self.db.save_model(model)
            return _ok(event.to_dict(), "Challenger promoted to champion.")
        except Exception as e:
            return _err("PROMOTE_CHALLENGER_ERROR", str(e))

    def compare_models(
        self,
        model_a_id: str,
        model_b_id: str,
        metrics_a: Optional[Dict[str, float]] = None,
        metrics_b: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        try:
            comparison = self.champion.compare(
                model_a_id, model_b_id, metrics_a=metrics_a, metrics_b=metrics_b
            )
            return _ok(comparison)
        except Exception as e:
            return _err("COMPARE_MODELS_ERROR", str(e))

    def rollback_champion(self, actor: str = "ops", reason: str = "") -> Dict[str, Any]:
        try:
            event = self.champion.rollback_champion(actor=actor, reason=reason)
            if event is None:
                return _err("NO_CHAMPION", "No active champion to roll back.")
            return _ok(event.to_dict(), "Champion rolled back.")
        except Exception as e:
            return _err("ROLLBACK_CHAMPION_ERROR", str(e))

    def champion_history(self, limit: int = 20) -> Dict[str, Any]:
        try:
            return _ok(
                {
                    "events": self.champion.event_history(limit=limit),
                    "historical_champions": self.champion.historical_champions,
                    "retired_champions": self.champion.retired_champions,
                }
            )
        except Exception as e:
            return _err("CHAMPION_HISTORY_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Retraining endpoints
    # -----------------------------------------------------------------------

    def evaluate_retraining(
        self, model_id: str, metrics: Dict[str, float], force_manual: bool = False
    ) -> Dict[str, Any]:
        try:
            recs = self.retraining.evaluate(model_id, metrics, force_manual=force_manual)
            for r in recs:
                self.db.save_recommendation(r.to_dict())
                self.alerting.emit(
                    level=AlertLevel.WARNING,
                    model_id=model_id,
                    category=AlertCategory.RETRAINING_RECOMMENDED,
                    message=r.reason,
                    metadata=r.to_dict(),
                )
            return _ok(
                {
                    "recommendations": [r.to_dict() for r in recs],
                    "count": len(recs),
                    "note": "ADVISORY ONLY — no automatic retraining.",
                }
            )
        except Exception as e:
            return _err("EVALUATE_RETRAINING_ERROR", str(e))

    def list_retraining_recommendations(
        self, model_id: Optional[str] = None, limit: int = 50
    ) -> Dict[str, Any]:
        try:
            pending = self.retraining.pending_recommendations(model_id=model_id)
            return _ok({"pending": pending, "count": len(pending)})
        except Exception as e:
            return _err("LIST_RECOMMENDATIONS_ERROR", str(e))

    def dismiss_recommendation(self, recommendation_id: str) -> Dict[str, Any]:
        try:
            ok = self.retraining.dismiss(recommendation_id)
            if not ok:
                return _err("NOT_FOUND", f"Recommendation '{recommendation_id}' not found.")
            return _ok({"recommendation_id": recommendation_id, "status": "dismissed"})
        except Exception as e:
            return _err("DISMISS_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Stats and dashboard
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        try:
            return _ok(
                {
                    **self.db.stats(),
                    "registry": self.registry.summary(),
                    "champion": self.champion.summary(),
                    "alerting": self.alerting.stats(),
                    "audit": self.audit.stats(),
                    "retraining": self.retraining.stats(),
                }
            )
        except Exception as e:
            return _err("STATS_ERROR", str(e))

    def get_dashboard(self) -> Dict[str, Any]:
        try:
            models = self.registry.all_models()
            snapshots = {
                m.model_id: self.monitor.get_snapshot(m.model_id).to_dict() for m in models
            }
            health_reports = {
                m.model_id: self.health.get_health_report(m.model_id).to_dict() for m in models
            }
            return _ok(
                {
                    "registry_summary": self.registry.summary(),
                    "champion_summary": self.champion.summary(),
                    "active_alerts": self.alerting.active_alerts()[:20],
                    "alert_stats": self.alerting.stats(),
                    "monitoring_snapshots": snapshots,
                    "health_reports": health_reports,
                    "pending_recommendations": self.retraining.pending_recommendations(),
                    "governance_stats": self.governance.stats(),
                    "db_stats": self.db.stats(),
                }
            )
        except Exception as e:
            logger.error(f"get_dashboard: {traceback.format_exc()}")
            return _err("DASHBOARD_ERROR", str(e))
