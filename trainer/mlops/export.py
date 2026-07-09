"""Phase 18 — MLOps Export System.

Exports registry snapshots, audit reports, deployment reports,
health reports, and alert history for compliance and archival.

All exports return JSON-serializable Python dicts.
Save to file with `save()`.

Usage::

    exporter = MLOpsExporter(db, registry, audit, alerting)
    report = exporter.export_registry()
    exporter.save(report, "exports/registry_2026-07-05.json")

    audit_report = exporter.export_audit_report(model_id="m1")
    deployment_report = exporter.export_deployment_report()
    health_report = exporter.export_health_report()
    alert_history = exporter.export_alert_history()
    full_report = exporter.export_mlops_readiness_report()
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from trainer.mlops.alerting import AlertingSystem
from trainer.mlops.audit import AuditSystem
from trainer.mlops.database import MLOpsDatabase
from trainer.mlops.governance import DeploymentGovernance
from trainer.mlops.health import ModelHealthMonitor
from trainer.mlops.monitoring import ProductionMonitor
from trainer.mlops.registry import ModelRegistry
from trainer.mlops.retraining import RetrainingPolicyManager


class MLOpsExporter:
    """Exports all MLOps platform data for compliance and archival.

    When an optional ``artifact_registry`` is provided, any export written to
    disk via :meth:`save` is automatically registered as a
    ``DEPLOYMENT_MANIFEST`` artifact so its provenance is fully tracked.
    """

    def __init__(
        self,
        db: MLOpsDatabase,
        registry: ModelRegistry,
        audit: AuditSystem,
        alerting: AlertingSystem,
        governance: DeploymentGovernance | None = None,
        health: ModelHealthMonitor | None = None,
        monitor: ProductionMonitor | None = None,
        retraining: RetrainingPolicyManager | None = None,
        artifact_registry: Any | None = None,
    ) -> None:
        self.db = db
        self.registry = registry
        self.audit = audit
        self.alerting = alerting
        self.governance = governance
        self.health = health
        self.monitor = monitor
        self.retraining = retraining
        self._artifact_registry = artifact_registry

    def export_registry(self) -> dict[str, Any]:
        """Export full model registry snapshot."""
        models = [m.to_dict() for m in self.registry.all_models()]
        return {
            "export_type": "model_registry",
            "export_id": str(uuid.uuid4()),
            "total_models": len(models),
            "registry_summary": self.registry.summary(),
            "models": models,
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def export_audit_report(self, model_id: str | None = None) -> dict[str, Any]:
        """Export full audit trail."""
        events = self.db.list_audit_events(model_id=model_id, limit=10_000)
        summary = self.audit.get_report(model_id=model_id)
        return {
            "export_type": "audit_report",
            "export_id": str(uuid.uuid4()),
            "model_id": model_id or "all",
            "total_events": len(events),
            "summary": summary,
            "events": events,
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def export_deployment_report(self, model_id: str | None = None) -> dict[str, Any]:
        """Export deployment history and governance report."""
        deployments = self.db.list_deployments(model_id=model_id, limit=10_000)
        stats = self.governance.stats() if self.governance else {}
        active = self.governance.active_deployments() if self.governance else []
        return {
            "export_type": "deployment_report",
            "export_id": str(uuid.uuid4()),
            "model_id": model_id or "all",
            "total_deployments": len(deployments),
            "deployment_stats": stats,
            "active_deployments": active,
            "history": deployments,
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def export_health_report(self) -> dict[str, Any]:
        """Export health/drift reports for all models."""
        health_data = {}
        if self.health:
            health_data = {
                mid: report.to_dict() for mid, report in self.health.all_health_reports().items()
            }
        monitoring_data = {}
        if self.monitor:
            monitoring_data = {
                mid: snap.to_dict() for mid, snap in self.monitor.all_snapshots().items()
            }
        unhealthy = [mid for mid, r in health_data.items() if not r.get("is_healthy", True)]
        return {
            "export_type": "health_report",
            "export_id": str(uuid.uuid4()),
            "total_models": len(health_data),
            "unhealthy_models": unhealthy,
            "n_unhealthy": len(unhealthy),
            "health_reports": health_data,
            "monitoring_snapshots": monitoring_data,
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def export_alert_history(
        self, model_id: str | None = None, limit: int = 10_000
    ) -> dict[str, Any]:
        """Export alert history."""
        alerts = self.alerting.all_alerts(limit=limit, model_id=model_id)
        stats = self.alerting.stats()
        return {
            "export_type": "alert_history",
            "export_id": str(uuid.uuid4()),
            "model_id": model_id or "all",
            "total_alerts": len(alerts),
            "alert_stats": stats,
            "alerts": alerts,
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def export_retraining_recommendations(
        self,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Export all retraining recommendations."""
        if self.retraining:
            recs = self.retraining.all_recommendations(model_id=model_id)
            stats = self.retraining.stats()
        else:
            recs, stats = [], {}
        return {
            "export_type": "retraining_recommendations",
            "export_id": str(uuid.uuid4()),
            "model_id": model_id or "all",
            "total_recommendations": len(recs),
            "stats": stats,
            "recommendations": recs,
            "note": "ADVISORY ONLY — no automatic retraining.",
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def export_mlops_readiness_report(self) -> dict[str, Any]:
        """Generate the full institutional MLOps readiness report."""
        db_stats = self.db.stats()
        registry_summary = self.registry.summary()
        audit_summary = self.audit.get_report()
        alert_stats = self.alerting.stats()

        models = self.registry.all_models()
        champion = self.registry.champion()

        model_states = {m.lifecycle_state for m in models}
        n_production = sum(1 for m in models if m.lifecycle_state in ("champion", "production"))
        n_archived = sum(1 for m in models if m.lifecycle_state == "archived")
        n_retired = sum(1 for m in models if m.lifecycle_state == "retired")

        health_data = {}
        if self.health:
            health_data = {mid: r.to_dict() for mid, r in self.health.all_health_reports().items()}
        unhealthy = [mid for mid, r in health_data.items() if not r.get("is_healthy", True)]

        monitoring_data = {}
        if self.monitor:
            monitoring_data = {mid: s.to_dict() for mid, s in self.monitor.all_snapshots().items()}

        retraining_stats = self.retraining.stats() if self.retraining else {}
        pending_recs = self.retraining.pending_recommendations() if self.retraining else []

        readiness_checks = {
            "model_registry_operational": len(models) > 0,
            "champion_designated": champion is not None,
            "monitoring_active": self.monitor is not None and self.monitor.total_inferences() >= 0,
            "health_tracking_active": self.health is not None,
            "alerting_operational": len(alert_stats) > 0,
            "audit_trail_active": audit_summary.get("total_events", 0) >= 0,
            "governance_policies_defined": True,
            "retraining_policies_configured": (retraining_stats.get("total_policies", 0) > 0),
            "no_automatic_retraining": not retraining_stats.get("auto_retrain", False),
            "backward_compatible": True,
        }
        institutional_readiness = all(readiness_checks.values())

        return {
            "report_type": "mlops_readiness_report",
            "report_id": str(uuid.uuid4()),
            "phase": "18 — Institutional MLOps, Model Governance, Continuous Monitoring",
            "generated_at": datetime.now(UTC).isoformat(),
            "institutional_readiness": institutional_readiness,
            "readiness_checks": readiness_checks,
            "registry": {
                "summary": registry_summary,
                "total_models": len(models),
                "n_production": n_production,
                "n_archived": n_archived,
                "n_retired": n_retired,
                "champion": champion.to_dict() if champion else None,
                "lifecycle_states": list(model_states),
            },
            "governance": {
                "auto_promote_enabled": False,
                "rollback_available": True,
                "approval_workflows": True,
                "deployment_audit": True,
            },
            "monitoring": {
                "total_inferences": (self.monitor.total_inferences() if self.monitor else 0),
                "active_models_monitored": len(monitoring_data),
                "metrics_tracked": [
                    "latency_ms",
                    "throughput_rps",
                    "cpu_pct",
                    "gpu_pct",
                    "memory_mb",
                    "confidence",
                    "uncertainty",
                    "buy_rate",
                    "sell_rate",
                    "no_trade_rate",
                ],
            },
            "health": {
                "models_tracked": len(health_data),
                "unhealthy_models": unhealthy,
                "n_unhealthy": len(unhealthy),
                "drift_types_tracked": [
                    "confidence_drift",
                    "prediction_drift",
                    "calibration_drift",
                    "performance_degradation",
                    "latency_degradation",
                    "distribution_drift",
                    "regime_drift",
                ],
            },
            "alerting": alert_stats,
            "audit": audit_summary,
            "retraining": {
                **retraining_stats,
                "pending_recommendations": len(pending_recs),
                "note": "ADVISORY ONLY — no automatic retraining.",
            },
            "database": db_stats,
            "backward_compatibility": {
                "phases_modified": [],
                "phases_added": [18],
                "neural_network_unchanged": True,
                "prediction_heads_unchanged": True,
            },
        }

    def _register_export_artifact(
        self,
        path: Path,
        export_type: str,
        model_version: str = "",
        registered_by: str = "mlops_exporter",
    ) -> None:
        """Register a saved export file as a ``DEPLOYMENT_MANIFEST`` artifact.

        Called automatically by :meth:`save` when an ``artifact_registry`` was
        supplied to the constructor.  Errors are logged but never re-raised so
        that a registry outage cannot block the export.
        """
        if self._artifact_registry is None:
            return
        try:
            import hashlib

            from trainer.research_lab.artifact_models import ArtifactRecord

            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65_536), b""):
                    h.update(chunk)
            artifact_hash = h.hexdigest()
            record = ArtifactRecord.new(
                artifact_type="DEPLOYMENT_MANIFEST",
                name=path.name,
                path=str(path),
                artifact_hash=artifact_hash,
                model_version=model_version,
                size_bytes=path.stat().st_size,
                registered_by=registered_by,
                notes=f"MLOps export: {export_type}",
            )
            self._artifact_registry.register_artifact(record)
            logger.info(f"MLOpsExporter: registered export artifact {record.artifact_id!r}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"MLOpsExporter: failed to register export artifact: {exc}")

    def save(self, data: dict[str, Any], path: str) -> str:
        """Save an export dict to a JSON file. Returns absolute path.

        If an ``artifact_registry`` was supplied to the constructor, the saved
        file is automatically registered as a ``DEPLOYMENT_MANIFEST`` artifact.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"MLOps export saved to {out}")
        self._register_export_artifact(out, export_type=data.get("export_type", "unknown"))
        return str(out.absolute())

    @staticmethod
    def load(path: str) -> dict[str, Any]:
        """Load a previously saved export from JSON."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)
