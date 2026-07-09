"""Training Pipeline Integration models (S20).

Defines the job spec, stage enum, per-stage results, structured events, and
the full pipeline job record used by ResearchLabPipelineController.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ─────────────────────────────────────────────────────────────────────


class PipelineStage(str, Enum):
    """Ordered stages of a Research Lab training pipeline run.

    The canonical sequence is:
        DATASET_GENERATION → FEATURE_GENERATION → TRAINING → VALIDATION
        → BACKTEST → CERTIFICATION → SHADOW_TEST

    TRAINING is the stage that requires a TRAINING_START approval gate to be
    consumed before execution begins.
    """

    DATASET_GENERATION = "DATASET_GENERATION"
    FEATURE_GENERATION = "FEATURE_GENERATION"
    TRAINING = "TRAINING"          # ← TRAINING_START gate enforced here
    VALIDATION = "VALIDATION"
    BACKTEST = "BACKTEST"
    CERTIFICATION = "CERTIFICATION"
    SHADOW_TEST = "SHADOW_TEST"


PIPELINE_STAGE_ORDER: list[PipelineStage] = [
    PipelineStage.DATASET_GENERATION,
    PipelineStage.FEATURE_GENERATION,
    PipelineStage.TRAINING,
    PipelineStage.VALIDATION,
    PipelineStage.BACKTEST,
    PipelineStage.CERTIFICATION,
    PipelineStage.SHADOW_TEST,
]


class PipelineJobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PipelineStageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"   # gate not yet approved


class PipelineEventType(str, Enum):
    JOB_QUEUED = "JOB_QUEUED"
    JOB_STARTED = "JOB_STARTED"
    JOB_COMPLETED = "JOB_COMPLETED"
    JOB_FAILED = "JOB_FAILED"
    JOB_CANCELLED = "JOB_CANCELLED"
    STAGE_STARTED = "STAGE_STARTED"
    STAGE_COMPLETED = "STAGE_COMPLETED"
    STAGE_FAILED = "STAGE_FAILED"
    GATE_BLOCKED = "GATE_BLOCKED"
    GATE_CONSUMED = "GATE_CONSUMED"


# ── Per-stage result ───────────────────────────────────────────────────────────


@dataclass
class PipelineStageResult:
    """Records the outcome of a single pipeline stage."""

    stage: str
    status: str                           # PipelineStageStatus value
    started_at: float
    completed_at: float | None = None
    duration_s: float | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    error: str = ""

    def complete(self, artifacts: dict[str, Any] | None = None,
                 logs: list[str] | None = None) -> None:
        self.status = PipelineStageStatus.COMPLETED.value
        self.completed_at = time.time()
        self.duration_s = round(self.completed_at - self.started_at, 3)
        if artifacts:
            self.artifacts.update(artifacts)
        if logs:
            self.logs.extend(logs)

    def fail(self, error: str, logs: list[str] | None = None) -> None:
        self.status = PipelineStageStatus.FAILED.value
        self.completed_at = time.time()
        self.duration_s = round(self.completed_at - self.started_at, 3)
        self.error = error
        if logs:
            self.logs.extend(logs)

    def block(self, error: str, logs: list[str] | None = None) -> None:
        """Mark a stage as BLOCKED (gate not yet approved) rather than FAILED."""
        self.status = PipelineStageStatus.BLOCKED.value
        self.completed_at = time.time()
        self.duration_s = round(self.completed_at - self.started_at, 3)
        self.error = error
        if logs:
            self.logs.extend(logs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "started_at": round(self.started_at, 3),
            "completed_at": round(self.completed_at, 3) if self.completed_at else None,
            "duration_s": self.duration_s,
            "artifacts": self.artifacts,
            "logs": self.logs,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineStageResult:
        obj = cls(
            stage=d["stage"],
            status=d["status"],
            started_at=d["started_at"],
            completed_at=d.get("completed_at"),
            duration_s=d.get("duration_s"),
            artifacts=d.get("artifacts") or {},
            logs=d.get("logs") or [],
            error=d.get("error") or "",
        )
        return obj


# ── Stage-transition event ─────────────────────────────────────────────────────


@dataclass
class PipelineStageEvent:
    """Structured event emitted at each stage transition.

    Downstream systems (shadow test automation, artifact management, alerting)
    consume these events to react to pipeline progress without coupling directly
    to the controller.
    """

    event_id: str
    job_id: str
    stage: str | None           # None for job-level events
    event_type: str             # PipelineEventType value
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        job_id: str,
        event_type: PipelineEventType,
        stage: PipelineStage | None = None,
        data: dict[str, Any] | None = None,
    ) -> PipelineStageEvent:
        return cls(
            event_id=f"evt-{str(uuid.uuid4())[:12]}",
            job_id=job_id,
            stage=stage.value if stage else None,
            event_type=event_type.value,
            timestamp=time.time(),
            data=data or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "job_id": self.job_id,
            "stage": self.stage,
            "event_type": self.event_type,
            "timestamp": round(self.timestamp, 3),
            "data": self.data,
        }


# ── Job spec (input) ──────────────────────────────────────────────────────────


@dataclass
class PipelineJobSpec:
    """Input specification for creating a pipeline job.

    Either experiment_id or recommendation_id must be provided; both can be
    set if the pipeline is triggered by a recommendation for a known experiment.
    Stages not in ``stages_to_run`` are skipped.
    """

    requested_by: str
    experiment_id: str | None = None
    recommendation_id: str | None = None
    stages_to_run: list[str] = field(default_factory=lambda: [
        s.value for s in PIPELINE_STAGE_ORDER
    ])
    stage_configs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id and not self.recommendation_id:
            raise ValueError(
                "PipelineJobSpec requires at least one of: experiment_id, recommendation_id"
            )


# ── Pipeline job record ────────────────────────────────────────────────────────


@dataclass
class PipelineJob:
    """Full pipeline job record — created from a PipelineJobSpec and updated in-place
    as the controller progresses through stages.
    """

    job_id: str
    requested_by: str
    status: str                           # PipelineJobStatus value
    stages_to_run: list[str]
    created_at: float
    updated_at: float

    experiment_id: str | None = None
    recommendation_id: str | None = None
    stage_configs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    current_stage: str | None = None
    stages_completed: list[str] = field(default_factory=list)
    stage_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)

    started_at: float | None = None
    completed_at: float | None = None
    error: str = ""

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def new(cls, spec: PipelineJobSpec) -> PipelineJob:
        now = time.time()
        return cls(
            job_id=f"pipe-{str(uuid.uuid4())[:12]}",
            requested_by=spec.requested_by,
            experiment_id=spec.experiment_id,
            recommendation_id=spec.recommendation_id,
            status=PipelineJobStatus.QUEUED.value,
            stages_to_run=list(spec.stages_to_run),
            stage_configs=dict(spec.stage_configs),
            notes=spec.notes,
            created_at=now,
            updated_at=now,
        )

    # ── Mutations ─────────────────────────────────────────────────────────────

    def _touch(self) -> None:
        self.updated_at = time.time()

    def start(self) -> None:
        self.status = PipelineJobStatus.RUNNING.value
        self.started_at = time.time()
        self._touch()

    def mark_completed(self) -> None:
        self.status = PipelineJobStatus.COMPLETED.value
        self.completed_at = time.time()
        self.current_stage = None
        self._touch()

    def mark_failed(self, error: str) -> None:
        self.status = PipelineJobStatus.FAILED.value
        self.completed_at = time.time()
        self.error = error
        self._touch()

    def mark_cancelled(self, reason: str = "") -> None:
        self.status = PipelineJobStatus.CANCELLED.value
        self.completed_at = time.time()
        self.error = reason
        self._touch()

    def record_stage_result(self, result: PipelineStageResult) -> None:
        self.stage_results[result.stage] = result.to_dict()
        if result.status == PipelineStageStatus.COMPLETED.value:
            if result.stage not in self.stages_completed:
                self.stages_completed.append(result.stage)
        if result.artifacts:
            self.artifact_refs.update(result.artifacts)
        self._touch()

    def emit_event(self, event: PipelineStageEvent) -> None:
        self.events.append(event.to_dict())
        self._touch()

    def last_events(self, n: int = 20) -> list[dict[str, Any]]:
        return self.events[-n:]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "requested_by": self.requested_by,
            "experiment_id": self.experiment_id,
            "recommendation_id": self.recommendation_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "stages_to_run": self.stages_to_run,
            "stages_completed": self.stages_completed,
            "stage_results": self.stage_results,
            "events": self.events,
            "artifact_refs": self.artifact_refs,
            "stage_configs": self.stage_configs,
            "notes": self.notes,
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "completed_at": round(self.completed_at, 3) if self.completed_at else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineJob:
        return cls(
            job_id=d["job_id"],
            requested_by=d.get("requested_by", ""),
            experiment_id=d.get("experiment_id"),
            recommendation_id=d.get("recommendation_id"),
            status=d.get("status", PipelineJobStatus.QUEUED.value),
            current_stage=d.get("current_stage"),
            stages_to_run=d.get("stages_to_run") or [],
            stages_completed=d.get("stages_completed") or [],
            stage_results=d.get("stage_results") or {},
            events=d.get("events") or [],
            artifact_refs=d.get("artifact_refs") or {},
            stage_configs=d.get("stage_configs") or {},
            notes=d.get("notes") or "",
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            error=d.get("error") or "",
        )
