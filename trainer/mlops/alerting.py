"""Phase 18 — MLOps Alerting System.

Generates and tracks alerts for:

  - High drift (confidence, prediction, distribution, calibration)
  - Low confidence
  - Latency spikes
  - Memory exhaustion
  - API failures
  - Model failures
  - Prediction anomalies
  - Unexpected behaviour

Alerts have severity levels: INFO, WARNING, CRITICAL.
All alerts are persisted and queryable.

Usage::

    alerting = AlertingSystem()
    alerting.emit(AlertLevel.CRITICAL, "model_01", "confidence_drift",
                  "Confidence dropped 0.12 below baseline")
    active = alerting.active_alerts()
    alerting.acknowledge("alert_id", ack_by="ops_lead")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Alert level
# ---------------------------------------------------------------------------


class AlertLevel:
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    ALL = [INFO, WARNING, CRITICAL]


# ---------------------------------------------------------------------------
# Alert category
# ---------------------------------------------------------------------------


class AlertCategory:
    CONFIDENCE_DRIFT = "confidence_drift"
    PREDICTION_DRIFT = "prediction_drift"
    DISTRIBUTION_DRIFT = "distribution_drift"
    CALIBRATION_DRIFT = "calibration_drift"
    LATENCY_SPIKE = "latency_spike"
    LOW_CONFIDENCE = "low_confidence"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    API_FAILURE = "api_failure"
    MODEL_FAILURE = "model_failure"
    PREDICTION_ANOMALY = "prediction_anomaly"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RETRAINING_RECOMMENDED = "retraining_recommended"
    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    LIFECYCLE = "lifecycle"
    SYSTEM = "system"

    ALL = [
        CONFIDENCE_DRIFT,
        PREDICTION_DRIFT,
        DISTRIBUTION_DRIFT,
        CALIBRATION_DRIFT,
        LATENCY_SPIKE,
        LOW_CONFIDENCE,
        MEMORY_EXHAUSTION,
        API_FAILURE,
        MODEL_FAILURE,
        PREDICTION_ANOMALY,
        PERFORMANCE_DEGRADATION,
        RETRAINING_RECOMMENDED,
        DEPLOYMENT,
        ROLLBACK,
        LIFECYCLE,
        SYSTEM,
    ]


# ---------------------------------------------------------------------------
# Alert record
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """One alert event.

    Attributes:
        alert_id: Unique UUID.
        level: AlertLevel value.
        model_id: Affected model (empty = system alert).
        category: AlertCategory value.
        message: Human-readable alert message.
        status: "active" | "acknowledged" | "resolved" | "suppressed".
        acknowledged_by: Who acknowledged the alert.
        acknowledged_at: When the alert was acknowledged.
        resolved_at: When the alert was resolved.
        metadata: Extra alert context (metric values, thresholds, etc.).
        created_at: UTC ISO-8601 alert creation timestamp.
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: str = AlertLevel.WARNING
    model_id: str = ""
    category: str = AlertCategory.SYSTEM
    message: str = ""
    status: str = "active"
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_critical(self) -> bool:
        return self.level == AlertLevel.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "level": self.level,
            "model_id": self.model_id,
            "category": self.category,
            "message": self.message,
            "status": self.status,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "is_critical": self.is_critical,
        }


# ---------------------------------------------------------------------------
# Alerting system
# ---------------------------------------------------------------------------


class AlertingSystem:
    """Generates, tracks, and manages all MLOps alerts.

    Args:
        max_alerts: Maximum alerts to keep in memory (rolling).
        dedup_window_seconds: Suppress duplicate alerts within this window.
    """

    def __init__(self, max_alerts: int = 10_000, dedup_window_seconds: int = 300) -> None:
        self.max_alerts = max_alerts
        self.dedup_window = dedup_window_seconds
        self._alerts: Dict[str, Alert] = {}
        self._order: List[str] = []
        self._dedup_keys: Dict[str, str] = {}  # dedup_key → alert_id

    def emit(
        self,
        level: str,
        model_id: str,
        category: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        dedup: bool = True,
    ) -> Alert:
        """Create and store a new alert.

        Args:
            level: AlertLevel.
            model_id: Affected model ID (empty for system alerts).
            category: AlertCategory.
            message: Human-readable description.
            metadata: Extra context dict.
            dedup: If True, suppress duplicate alerts within dedup_window.

        Returns:
            Alert object (may be an existing alert if deduped).
        """
        dedup_key = f"{model_id}:{category}"
        if dedup and dedup_key in self._dedup_keys:
            existing_id = self._dedup_keys[dedup_key]
            existing = self._alerts.get(existing_id)
            if existing and existing.is_active:
                existing.metadata["duplicate_count"] = (
                    existing.metadata.get("duplicate_count", 1) + 1
                )
                return existing

        alert = Alert(
            level=level,
            model_id=model_id,
            category=category,
            message=message,
            metadata=metadata or {},
        )
        self._alerts[alert.alert_id] = alert
        self._order.append(alert.alert_id)
        if dedup:
            self._dedup_keys[dedup_key] = alert.alert_id

        if level == AlertLevel.CRITICAL:
            logger.error(f"CRITICAL ALERT [{category}] {model_id}: {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"ALERT [{category}] {model_id}: {message}")
        else:
            logger.info(f"INFO ALERT [{category}] {model_id}: {message}")

        if len(self._order) > self.max_alerts:
            oldest_id = self._order.pop(0)
            self._alerts.pop(oldest_id, None)

        return alert

    def acknowledge(self, alert_id: str, ack_by: str = "ops") -> bool:
        """Acknowledge an alert."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            return False
        alert.status = "acknowledged"
        alert.acknowledged_by = ack_by
        alert.acknowledged_at = datetime.now(timezone.utc).isoformat()
        dedup_key = f"{alert.model_id}:{alert.category}"
        self._dedup_keys.pop(dedup_key, None)
        return True

    def resolve(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            return False
        alert.status = "resolved"
        alert.resolved_at = datetime.now(timezone.utc).isoformat()
        dedup_key = f"{alert.model_id}:{alert.category}"
        self._dedup_keys.pop(dedup_key, None)
        return True

    def suppress(self, alert_id: str) -> bool:
        """Suppress an alert (no action needed)."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            return False
        alert.status = "suppressed"
        return True

    def active_alerts(
        self, model_id: Optional[str] = None, level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all active alerts, optionally filtered."""
        alerts = [
            self._alerts[aid]
            for aid in self._order
            if aid in self._alerts and self._alerts[aid].is_active
        ]
        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        if level:
            alerts = [a for a in alerts if a.level == level]
        return [a.to_dict() for a in reversed(alerts)]

    def all_alerts(
        self, limit: int = 100, model_id: Optional[str] = None, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all alerts, optionally filtered, most recent first."""
        alerts = [self._alerts[aid] for aid in reversed(self._order) if aid in self._alerts]
        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        if category:
            alerts = [a for a in alerts if a.category == category]
        return [a.to_dict() for a in alerts[:limit]]

    def critical_count(self, model_id: Optional[str] = None) -> int:
        alerts = self.active_alerts(model_id=model_id, level=AlertLevel.CRITICAL)
        return len(alerts)

    def emit_from_health_report(self, health_report: Any) -> List[Alert]:
        """Auto-emit alerts from a HealthReport object."""
        emitted = []
        for drift_msg in health_report.drift_alerts:
            level = AlertLevel.CRITICAL if "drift" in drift_msg else AlertLevel.WARNING
            a = self.emit(
                level=level,
                model_id=health_report.model_id,
                category=AlertCategory.PREDICTION_DRIFT,
                message=f"Drift detected: {drift_msg}",
                metadata={"drift": drift_msg},
            )
            emitted.append(a)
        return emitted

    def emit_from_monitoring_snapshot(self, snapshot: Any) -> List[Alert]:
        """Auto-emit alerts from a MonitoringSnapshot object."""
        emitted = []
        if snapshot.latency_alert:
            a = self.emit(
                level=AlertLevel.CRITICAL,
                model_id=snapshot.model_id,
                category=AlertCategory.LATENCY_SPIKE,
                message=f"Latency spike: avg={snapshot.avg_latency_ms:.1f}ms",
                metadata={"avg_latency_ms": snapshot.avg_latency_ms},
            )
            emitted.append(a)
        if snapshot.memory_alert:
            a = self.emit(
                level=AlertLevel.CRITICAL,
                model_id=snapshot.model_id,
                category=AlertCategory.MEMORY_EXHAUSTION,
                message=f"Memory alert: peak={snapshot.peak_memory_mb:.0f}MB",
                metadata={"peak_memory_mb": snapshot.peak_memory_mb},
            )
            emitted.append(a)
        if snapshot.avg_confidence < 0.45 and snapshot.n_records >= 20:
            a = self.emit(
                level=AlertLevel.WARNING,
                model_id=snapshot.model_id,
                category=AlertCategory.LOW_CONFIDENCE,
                message=f"Low confidence: avg={snapshot.avg_confidence:.3f}",
                metadata={"avg_confidence": snapshot.avg_confidence},
            )
            emitted.append(a)
        return emitted

    def stats(self) -> Dict[str, Any]:
        all_a = list(self._alerts.values())
        return {
            "total_alerts": len(all_a),
            "active": sum(1 for a in all_a if a.is_active),
            "acknowledged": sum(1 for a in all_a if a.status == "acknowledged"),
            "resolved": sum(1 for a in all_a if a.status == "resolved"),
            "critical": sum(1 for a in all_a if a.level == AlertLevel.CRITICAL and a.is_active),
            "by_category": {
                cat: sum(1 for a in all_a if a.category == cat)
                for cat in AlertCategory.ALL
                if any(a.category == cat for a in all_a)
            },
        }
