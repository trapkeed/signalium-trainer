"""Phase 19.1 — Research Timeline.

Maintains a chronological log of every significant research event.

Event types:

  experiment_started    experiment_completed   experiment_failed
  champion_changed      feature_added          feature_deprecated
  dataset_updated       architecture_changed   deployment_created
  validation_completed  retraining_recommended cost_recorded

Usage::

    tl = ResearchTimeline()
    tl.record("experiment_started",
               entity_id="exp_001",
               description="Started transformer 4H sweep",
               attributes={"researcher": "alice"})
    events = tl.since("2026-01-01T00:00:00+00:00")
    report = tl.timeline_report()
    md = tl.to_markdown()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------


class TimelineEventType:
    EXPERIMENT_STARTED      = "experiment_started"
    EXPERIMENT_COMPLETED    = "experiment_completed"
    EXPERIMENT_FAILED       = "experiment_failed"
    CHAMPION_CHANGED        = "champion_changed"
    FEATURE_ADDED           = "feature_added"
    FEATURE_DEPRECATED      = "feature_deprecated"
    DATASET_UPDATED         = "dataset_updated"
    ARCHITECTURE_CHANGED    = "architecture_changed"
    DEPLOYMENT_CREATED      = "deployment_created"
    VALIDATION_COMPLETED    = "validation_completed"
    RETRAINING_RECOMMENDED  = "retraining_recommended"
    COST_RECORDED           = "cost_recorded"
    KNOWLEDGE_ENTRY_ADDED   = "knowledge_entry_added"
    REGIME_CHANGE_DETECTED  = "regime_change_detected"
    LINEAGE_NODE_REGISTERED = "lineage_node_registered"

    ALL = [
        EXPERIMENT_STARTED, EXPERIMENT_COMPLETED, EXPERIMENT_FAILED,
        CHAMPION_CHANGED, FEATURE_ADDED, FEATURE_DEPRECATED,
        DATASET_UPDATED, ARCHITECTURE_CHANGED, DEPLOYMENT_CREATED,
        VALIDATION_COMPLETED, RETRAINING_RECOMMENDED, COST_RECORDED,
        KNOWLEDGE_ENTRY_ADDED, REGIME_CHANGE_DETECTED,
        LINEAGE_NODE_REGISTERED,
    ]


# ---------------------------------------------------------------------------
# TimelineEvent
# ---------------------------------------------------------------------------


@dataclass
class TimelineEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    entity_id: str = ""
    entity_type: str = ""
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    researcher: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "description": self.description,
            "attributes": self.attributes,
            "recorded_at": self.recorded_at,
            "researcher": self.researcher,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TimelineEvent":
        return cls(
            event_id=d.get("event_id", str(uuid.uuid4())),
            event_type=d.get("event_type", ""),
            entity_id=d.get("entity_id", ""),
            entity_type=d.get("entity_type", ""),
            description=d.get("description", ""),
            attributes=d.get("attributes", {}),
            recorded_at=d.get("recorded_at", datetime.now(timezone.utc).isoformat()),
            researcher=d.get("researcher", "system"),
        )


# ---------------------------------------------------------------------------
# ResearchTimeline
# ---------------------------------------------------------------------------


class ResearchTimeline:
    """Chronological log of all research events."""

    def __init__(self) -> None:
        self._events: List[TimelineEvent] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        event_type: str,
        *,
        entity_id: str = "",
        entity_type: str = "",
        description: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        researcher: str = "system",
    ) -> str:
        """Record a timeline event and return its event_id."""
        event = TimelineEvent(
            event_type=event_type,
            entity_id=entity_id,
            entity_type=entity_type,
            description=description,
            attributes=attributes or {},
            researcher=researcher,
        )
        self._events.append(event)
        return event.event_id

    def record_experiment_started(
        self, node_id: str, name: str, researcher: str = "system"
    ) -> str:
        return self.record(
            TimelineEventType.EXPERIMENT_STARTED,
            entity_id=node_id,
            entity_type="experiment",
            description=f"Experiment started: {name}",
            attributes={"name": name},
            researcher=researcher,
        )

    def record_experiment_completed(
        self,
        node_id: str,
        name: str,
        status: str,
        accuracy: Optional[float] = None,
    ) -> str:
        attrs: Dict[str, Any] = {"name": name, "status": status}
        if accuracy is not None:
            attrs["accuracy"] = accuracy
        return self.record(
            TimelineEventType.EXPERIMENT_COMPLETED,
            entity_id=node_id,
            entity_type="experiment",
            description=f"Experiment {status}: {name}",
            attributes=attrs,
        )

    def record_champion_changed(
        self,
        old_champion: str,
        new_champion: str,
        reason: str = "",
    ) -> str:
        return self.record(
            TimelineEventType.CHAMPION_CHANGED,
            entity_id=new_champion,
            entity_type="model",
            description=f"Champion changed: {old_champion} → {new_champion}",
            attributes={"old_champion": old_champion, "reason": reason},
        )

    def record_dataset_updated(
        self, dataset_version: str, sample_count: int = 0
    ) -> str:
        return self.record(
            TimelineEventType.DATASET_UPDATED,
            entity_id=dataset_version,
            entity_type="dataset",
            description=f"Dataset updated to {dataset_version}",
            attributes={"sample_count": sample_count},
        )

    def record_architecture_changed(
        self, model_id: str, old_arch: str, new_arch: str
    ) -> str:
        return self.record(
            TimelineEventType.ARCHITECTURE_CHANGED,
            entity_id=model_id,
            entity_type="model",
            description=f"Architecture changed: {old_arch} → {new_arch}",
            attributes={"old_architecture": old_arch, "new_architecture": new_arch},
        )

    def record_deployment(self, model_id: str, environment: str = "production") -> str:
        return self.record(
            TimelineEventType.DEPLOYMENT_CREATED,
            entity_id=model_id,
            entity_type="deployment",
            description=f"Model {model_id} deployed to {environment}",
            attributes={"environment": environment},
        )

    def record_validation_completed(
        self,
        model_id: str,
        accuracy: float,
        sharpe: float,
        promotion_decision: str,
    ) -> str:
        return self.record(
            TimelineEventType.VALIDATION_COMPLETED,
            entity_id=model_id,
            entity_type="validation_run",
            description=(
                f"Validation completed: acc={accuracy:.3f}, "
                f"sharpe={sharpe:.2f}, decision={promotion_decision}"
            ),
            attributes={
                "accuracy": accuracy,
                "sharpe_ratio": sharpe,
                "promotion_decision": promotion_decision,
            },
        )

    def record_cost(
        self,
        entity_id: str,
        total_cost_usd: float,
        gpu_hours: float = 0.0,
    ) -> str:
        return self.record(
            TimelineEventType.COST_RECORDED,
            entity_id=entity_id,
            entity_type="experiment",
            description=f"Cost recorded: ${total_cost_usd:.4f} ({gpu_hours:.2f} GPU-h)",
            attributes={"total_cost_usd": total_cost_usd, "gpu_hours": gpu_hours},
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def all_events(self) -> List[TimelineEvent]:
        return sorted(self._events, key=lambda e: e.recorded_at)

    def by_type(self, event_type: str) -> List[TimelineEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def by_entity(self, entity_id: str) -> List[TimelineEvent]:
        return [e for e in self._events if e.entity_id == entity_id]

    def since(self, iso_timestamp: str) -> List[TimelineEvent]:
        return [
            e for e in self.all_events()
            if e.recorded_at >= iso_timestamp
        ]

    def between(self, start_iso: str, end_iso: str) -> List[TimelineEvent]:
        return [
            e for e in self.all_events()
            if start_iso <= e.recorded_at <= end_iso
        ]

    def recent(self, n: int = 20) -> List[TimelineEvent]:
        return self.all_events()[-n:]

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def experiment_history(self) -> List[Dict[str, Any]]:
        started = {
            e.entity_id: e for e in self.by_type(TimelineEventType.EXPERIMENT_STARTED)
        }
        completed = {
            e.entity_id: e
            for e in (
                self.by_type(TimelineEventType.EXPERIMENT_COMPLETED)
                + self.by_type(TimelineEventType.EXPERIMENT_FAILED)
            )
        }
        result = []
        for eid, start_event in started.items():
            entry: Dict[str, Any] = {
                "experiment_id": eid,
                "name": start_event.attributes.get("name", eid),
                "started_at": start_event.recorded_at,
                "researcher": start_event.researcher,
                "completed_at": None,
                "status": "running",
                "accuracy": None,
            }
            if eid in completed:
                c = completed[eid]
                entry["completed_at"] = c.recorded_at
                entry["status"] = c.attributes.get("status", "completed")
                entry["accuracy"] = c.attributes.get("accuracy")
            result.append(entry)
        return sorted(result, key=lambda x: x["started_at"])

    def champion_history(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.by_type(TimelineEventType.CHAMPION_CHANGED)]

    def dataset_history(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.by_type(TimelineEventType.DATASET_UPDATED)]

    def architecture_history(self) -> List[Dict[str, Any]]:
        return [
            e.to_dict() for e in self.by_type(TimelineEventType.ARCHITECTURE_CHANGED)
        ]

    def deployment_history(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.by_type(TimelineEventType.DEPLOYMENT_CREATED)]

    def total_events(self) -> int:
        return len(self._events)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def timeline_report(self) -> Dict[str, Any]:
        events = self.all_events()
        type_counts: Dict[str, int] = {}
        for e in events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
        return {
            "total_events": len(events),
            "event_type_distribution": type_counts,
            "earliest_event": events[0].recorded_at if events else None,
            "latest_event": events[-1].recorded_at if events else None,
            "experiment_history": self.experiment_history(),
            "champion_history": self.champion_history(),
            "dataset_history": self.dataset_history(),
            "deployment_history": self.deployment_history(),
            "architecture_history": self.architecture_history(),
        }

    def to_markdown(self) -> str:
        lines = ["# Research Timeline\n"]
        for event in self.all_events():
            ts = event.recorded_at[:19].replace("T", " ")
            lines.append(f"- **{ts}** `{event.event_type}` — {event.description}")
        return "\n".join(lines)

    def load_event(self, event: TimelineEvent) -> None:
        """Load a pre-built event (used by persistence layer)."""
        self._events.append(event)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._events]
