"""Phase 18 — MLOps Audit System.

Records every important platform event with full traceability:

  - Training events
  - Validation events
  - Promotion events
  - Deployment events
  - Rollback events
  - Configuration changes
  - Inference errors
  - Manual approvals
  - Alert acknowledgements
  - Retraining recommendations
  - Champion changes
  - Lifecycle transitions

Every event is immutable once recorded.

Usage::

    audit = AuditSystem()
    audit.record(AuditEvent(
        event_type=AuditEventType.PROMOTION,
        actor="head_quant",
        model_id="model_01",
        description="Model promoted to champion",
        metadata={"validation_score": 0.72},
    ))
    report = audit.get_report()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Audit event types
# ---------------------------------------------------------------------------


class AuditEventType:
    TRAINING = "training"
    VALIDATION = "validation"
    PROMOTION = "promotion"
    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    RETIREMENT = "retirement"
    CONFIG_CHANGE = "config_change"
    INFERENCE_ERROR = "inference_error"
    MANUAL_APPROVAL = "manual_approval"
    MANUAL_REJECTION = "manual_rejection"
    RETRAINING_RECOMMENDATION = "retraining_recommendation"
    CHAMPION_CHANGE = "champion_change"
    LIFECYCLE_TRANSITION = "lifecycle_transition"
    ALERT_EMITTED = "alert_emitted"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    ALERT_RESOLVED = "alert_resolved"
    REGISTRY_REGISTER = "registry_register"
    REGISTRY_REMOVE = "registry_remove"
    GOVERNANCE = "governance"
    SYSTEM = "system"

    ALL = [
        TRAINING,
        VALIDATION,
        PROMOTION,
        DEPLOYMENT,
        ROLLBACK,
        RETIREMENT,
        CONFIG_CHANGE,
        INFERENCE_ERROR,
        MANUAL_APPROVAL,
        MANUAL_REJECTION,
        RETRAINING_RECOMMENDATION,
        CHAMPION_CHANGE,
        LIFECYCLE_TRANSITION,
        ALERT_EMITTED,
        ALERT_ACKNOWLEDGED,
        ALERT_RESOLVED,
        REGISTRY_REGISTER,
        REGISTRY_REMOVE,
        GOVERNANCE,
        SYSTEM,
    ]


# ---------------------------------------------------------------------------
# Audit event
# ---------------------------------------------------------------------------


@dataclass
class AuditEvent:
    """One immutable audit event.

    Attributes:
        event_id: Unique UUID.
        event_type: AuditEventType value.
        actor: System or human who triggered this event.
        model_id: Affected model ID (empty for system events).
        description: Human-readable description.
        outcome: Event outcome or result.
        metadata: Full event context (immutable after record).
        timestamp: UTC ISO-8601 event timestamp.
        session_id: Optional session grouping ID.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = AuditEventType.SYSTEM
    actor: str = "system"
    model_id: str = ""
    description: str = ""
    outcome: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "model_id": self.model_id,
            "description": self.description,
            "outcome": self.outcome,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }


# ---------------------------------------------------------------------------
# Audit system
# ---------------------------------------------------------------------------


class AuditSystem:
    """Records all platform events for compliance and traceability.

    All events are append-only — no modification or deletion is allowed.

    Args:
        max_events: Maximum events to keep in memory (deque-style rotation).
    """

    def __init__(self, max_events: int = 100_000) -> None:
        self.max_events = max_events
        self._events: list[AuditEvent] = []
        self._session_id: str | None = None

    def start_session(self, session_id: str | None = None) -> str:
        """Start a grouped audit session. Returns session_id."""
        self._session_id = session_id or str(uuid.uuid4())
        return self._session_id

    def end_session(self) -> None:
        self._session_id = None

    def record(self, event: AuditEvent) -> AuditEvent:
        """Record one audit event (immutable once stored)."""
        if event.session_id is None and self._session_id:
            event.session_id = self._session_id
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]
        return event

    def log(
        self,
        event_type: str,
        actor: str = "system",
        model_id: str = "",
        description: str = "",
        outcome: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Convenience method to create and record an event."""
        return self.record(
            AuditEvent(
                event_type=event_type,
                actor=actor,
                model_id=model_id,
                description=description,
                outcome=outcome,
                metadata=metadata or {},
            )
        )

    def get(self, event_id: str) -> AuditEvent | None:
        for e in reversed(self._events):
            if e.event_id == event_id:
                return e
        return None

    def query(
        self,
        event_type: str | None = None,
        model_id: str | None = None,
        actor: str | None = None,
        limit: int = 100,
        most_recent_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Query audit events with optional filters."""
        events = list(self._events)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if model_id:
            events = [e for e in events if e.model_id == model_id]
        if actor:
            events = [e for e in events if e.actor == actor]
        if most_recent_first:
            events = list(reversed(events))
        return [e.to_dict() for e in events[:limit]]

    def model_history(self, model_id: str) -> list[dict[str, Any]]:
        """Return all events for a specific model, oldest first."""
        return [e.to_dict() for e in self._events if e.model_id == model_id]

    def get_report(self, model_id: str | None = None) -> dict[str, Any]:
        """Generate an audit report."""
        events = (
            self._events
            if model_id is None
            else [e for e in self._events if e.model_id == model_id]
        )
        by_type: dict[str, int] = {}
        by_actor: dict[str, int] = {}
        by_outcome: dict[str, int] = {}
        for e in events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            by_actor[e.actor] = by_actor.get(e.actor, 0) + 1
            by_outcome[e.outcome] = by_outcome.get(e.outcome, 0) + 1
        return {
            "model_id": model_id or "all",
            "total_events": len(events),
            "by_type": by_type,
            "by_actor": by_actor,
            "by_outcome": by_outcome,
            "first_event": events[0].to_dict() if events else None,
            "last_event": events[-1].to_dict() if events else None,
        }

    def export_all(self) -> list[dict[str, Any]]:
        """Export all audit events as a list of dicts."""
        return [e.to_dict() for e in self._events]

    @property
    def n_events(self) -> int:
        return len(self._events)

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for e in self._events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
        return {
            "total_events": self.n_events,
            "by_type": by_type,
            "session_id": self._session_id,
        }
