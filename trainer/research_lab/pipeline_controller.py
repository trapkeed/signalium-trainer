"""ResearchLabPipelineController — Training Pipeline Integration (S20).

Orchestrates the seven-stage research pipeline:
    DATASET_GENERATION → FEATURE_GENERATION → TRAINING → VALIDATION
    → BACKTEST → CERTIFICATION → SHADOW_TEST

Gate enforcement
----------------
Before executing the TRAINING stage the controller calls
``approval_gate.require_approval(ApprovalGateType.TRAINING_START, job_id)``.
If no valid approval exists the stage is marked BLOCKED and the job fails with
an :class:`ApprovalRequiredError`.

Event emission
--------------
A :class:`PipelineStageEvent` is emitted at every stage transition (started /
completed / failed) and at every job-level transition (queued → running →
completed / failed).  Events are persisted inside the ``PipelineJob`` record so
downstream consumers (shadow testing, artifact management) can poll
``GET /pipeline/jobs/{id}`` and inspect the ``events`` list.

Stage execution
---------------
Execution is delegated to a :class:`StageExecutor` instance, which can be
replaced with a test-double for unit tests.  The bundled
:class:`DefaultStageExecutor` is a thin stub that succeeds immediately for all
stages; wiring to real GPU/RunPod training infrastructure is out of scope for
this task and deferred to later tasks.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from trainer.research_lab.approval_gate import ApprovalGate
from trainer.research_lab.approval_models import ApprovalGateType, ApprovalRequiredError
from trainer.research_lab.pipeline_models import (
    PIPELINE_STAGE_ORDER,
    PipelineEventType,
    PipelineJob,
    PipelineJobSpec,
    PipelineJobStatus,
    PipelineStage,
    PipelineStageEvent,
    PipelineStageResult,
    PipelineStageStatus,
)
from trainer.research_lab.rl_database import ResearchLabDatabase

logger = logging.getLogger(__name__)


# ── StageExecutor protocol ────────────────────────────────────────────────────


class StageExecutor(Protocol):
    """Pluggable executor for each pipeline stage.

    Implementations should return a *started* :class:`PipelineStageResult` that
    the controller will later call ``.complete()`` or ``.fail()`` on, OR raise
    an exception to signal a hard failure.

    For the current implementation, executors are expected to be synchronous and
    to fully complete the stage before returning.  The controller calls
    :meth:`execute_stage` and receives a finished result.
    """

    def execute_stage(
        self,
        stage: PipelineStage,
        job: PipelineJob,
        context: dict[str, Any],
    ) -> PipelineStageResult:
        ...


class DefaultStageExecutor:
    """Stub executor that succeeds immediately for every stage.

    In production this will be replaced by executors that dispatch real dataset
    generation jobs, training runs, validation sweeps, and backtests.  The stub
    exists so the controller can be exercised end-to-end without GPU/RunPod.
    """

    def execute_stage(
        self,
        stage: PipelineStage,
        job: PipelineJob,
        context: dict[str, Any],
    ) -> PipelineStageResult:
        result = PipelineStageResult(
            stage=stage.value,
            status=PipelineStageStatus.RUNNING.value,
            started_at=time.time(),
        )
        # Stub: immediately succeed with placeholder artifact refs.
        artifacts: dict[str, Any] = {
            f"{stage.value.lower()}_artifact": f"{stage.value.lower()}-{job.job_id}",
        }
        logs = [f"[stub] Stage {stage.value} completed for job {job.job_id}"]
        result.complete(artifacts=artifacts, logs=logs)
        return result


# ── Controller ────────────────────────────────────────────────────────────────


class ResearchLabPipelineController:
    """Orchestrates multi-stage training pipeline jobs.

    Parameters
    ----------
    db:
        Research Lab SQLite database (for job persistence).
    approval_gate:
        Approval gate used to consume TRAINING_START approvals before
        executing the TRAINING stage.
    executor:
        Stage executor.  Defaults to :class:`DefaultStageExecutor`.
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        approval_gate: ApprovalGate,
        executor: StageExecutor | None = None,
    ) -> None:
        self._db = db
        self._gate = approval_gate
        self._executor = executor or DefaultStageExecutor()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_pipeline(self, spec: PipelineJobSpec) -> PipelineJob:
        """Create a job from *spec*, run all requested stages, and return the
        completed (or failed) job.

        This method is synchronous.  If any stage fails the job transitions to
        FAILED and remaining stages are skipped.  The job record is persisted
        after every stage so it is always readable mid-run.
        """
        job = PipelineJob.new(spec)
        self._emit_and_persist(
            job,
            PipelineStageEvent.new(job.job_id, PipelineEventType.JOB_QUEUED,
                                   data={"spec": {
                                       "experiment_id": spec.experiment_id,
                                       "recommendation_id": spec.recommendation_id,
                                       "stages_to_run": spec.stages_to_run,
                                   }}),
        )
        logger.info(
            f"PipelineController: job {job.job_id} queued "
            f"(experiment={spec.experiment_id!r}, "
            f"recommendation={spec.recommendation_id!r})"
        )

        job.start()
        self._emit_and_persist(
            job,
            PipelineStageEvent.new(job.job_id, PipelineEventType.JOB_STARTED),
        )

        stages_to_run = self._resolve_stages(spec.stages_to_run)
        context: dict[str, Any] = {
            "spec": spec,
            "experiment_id": spec.experiment_id,
            "recommendation_id": spec.recommendation_id,
        }

        for stage in stages_to_run:
            if job.status == PipelineJobStatus.FAILED.value:
                break  # earlier stage failed; skip remaining

            # ── TRAINING_START gate ──────────────────────────────────────────
            if stage == PipelineStage.TRAINING:
                try:
                    consumed = self._gate.require_approval(
                        ApprovalGateType.TRAINING_START, job.job_id,
                        consumed_by=job.requested_by,
                    )
                    self._emit_and_persist(
                        job,
                        PipelineStageEvent.new(
                            job.job_id,
                            PipelineEventType.GATE_CONSUMED,
                            stage=stage,
                            data={"approval_request_id": consumed.request_id,
                                  "approved_by": consumed.resolved_by},
                        ),
                    )
                    logger.info(
                        f"PipelineController: job {job.job_id} — TRAINING_START "
                        f"approval consumed (request_id={consumed.request_id!r})"
                    )
                except ApprovalRequiredError as exc:
                    blocked_result = PipelineStageResult(
                        stage=stage.value,
                        status=PipelineStageStatus.RUNNING.value,
                        started_at=time.time(),
                    )
                    blocked_result.block(
                        error=f"TRAINING_START gate blocked: {exc}",
                        logs=["Approval required but not found. "
                              "Create and approve a TRAINING_START request via POST /approvals."],
                    )
                    job.current_stage = stage.value
                    job.record_stage_result(blocked_result)
                    self._emit_and_persist(
                        job,
                        PipelineStageEvent.new(
                            job.job_id,
                            PipelineEventType.GATE_BLOCKED,
                            stage=stage,
                            data={"reason": str(exc)},
                        ),
                    )
                    job.mark_failed(
                        f"TRAINING_START approval gate blocked for job {job.job_id}"
                    )
                    self._emit_and_persist(
                        job,
                        PipelineStageEvent.new(
                            job.job_id, PipelineEventType.JOB_FAILED,
                            data={"error": job.error}
                        ),
                    )
                    logger.warning(
                        f"PipelineController: job {job.job_id} blocked at TRAINING stage — "
                        f"no valid TRAINING_START approval"
                    )
                    return job

            # ── Execute stage ────────────────────────────────────────────────
            job.current_stage = stage.value
            self._emit_and_persist(
                job,
                PipelineStageEvent.new(job.job_id, PipelineEventType.STAGE_STARTED,
                                       stage=stage),
            )

            try:
                result = self._executor.execute_stage(stage, job, context)
            except Exception as exc:
                result = PipelineStageResult(
                    stage=stage.value,
                    status=PipelineStageStatus.RUNNING.value,
                    started_at=time.time(),
                )
                result.fail(error=f"Executor raised: {exc!r}",
                            logs=[f"[error] {exc!r}"])

            job.record_stage_result(result)

            if result.status == PipelineStageStatus.COMPLETED.value:
                context[f"{stage.value.lower()}_artifacts"] = result.artifacts
                self._emit_and_persist(
                    job,
                    PipelineStageEvent.new(
                        job.job_id, PipelineEventType.STAGE_COMPLETED,
                        stage=stage,
                        data={"artifacts": result.artifacts,
                              "duration_s": result.duration_s},
                    ),
                )
                logger.info(
                    f"PipelineController: job {job.job_id} stage {stage.value} "
                    f"completed in {result.duration_s}s"
                )
            else:
                self._emit_and_persist(
                    job,
                    PipelineStageEvent.new(
                        job.job_id, PipelineEventType.STAGE_FAILED,
                        stage=stage,
                        data={"error": result.error},
                    ),
                )
                job.mark_failed(
                    f"Stage {stage.value} failed: {result.error}"
                )
                self._emit_and_persist(
                    job,
                    PipelineStageEvent.new(
                        job.job_id, PipelineEventType.JOB_FAILED,
                        data={"failed_stage": stage.value, "error": job.error},
                    ),
                )
                logger.error(
                    f"PipelineController: job {job.job_id} failed at stage "
                    f"{stage.value}: {result.error}"
                )
                return job

        # All stages succeeded
        if job.status == PipelineJobStatus.RUNNING.value:
            job.mark_completed()
            self._emit_and_persist(
                job,
                PipelineStageEvent.new(
                    job.job_id, PipelineEventType.JOB_COMPLETED,
                    data={"stages_completed": job.stages_completed,
                          "artifact_refs": job.artifact_refs},
                ),
            )
            logger.info(
                f"PipelineController: job {job.job_id} completed "
                f"(stages={job.stages_completed!r})"
            )

        return job

    def get_job(self, job_id: str) -> PipelineJob | None:
        """Return the pipeline job with *job_id*, or ``None`` if not found."""
        row = self._db.get_pipeline_job(job_id)
        return PipelineJob.from_dict(row) if row else None

    def list_jobs(
        self,
        status: str | None = None,
        experiment_id: str | None = None,
        recommendation_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return a list of pipeline job dicts matching the given filters."""
        return self._db.list_pipeline_jobs(
            status=status,
            experiment_id=experiment_id,
            recommendation_id=recommendation_id,
            limit=limit,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _emit_and_persist(self, job: PipelineJob, event: PipelineStageEvent) -> None:
        """Append *event* to the job's event log (idempotent) and persist.

        Idempotent: if an event with the same ``event_id`` already exists in the
        job's event log it is not appended again.  This protects against duplicate
        events if the caller retries the same transition (e.g. after a transient
        persistence failure).
        """
        existing_ids = {e.get("event_id") for e in job.events}
        if event.event_id not in existing_ids:
            job.emit_event(event)
        self._db.upsert_pipeline_job(job.to_dict())

    @staticmethod
    def _resolve_stages(requested: list[str]) -> list[PipelineStage]:
        """Return the ordered subset of stages to run.

        Rules
        -----
        - Empty list → run all stages in canonical order.
        - Non-empty list → validate every name; raises ``ValueError`` on any
          unknown stage name so callers receive a clear 400 rather than a silent
          no-op.
        - Valid entries are returned in canonical pipeline order regardless of
          the order they appear in *requested*.
        """
        if not requested:
            return list(PIPELINE_STAGE_ORDER)

        valid_values = {s.value for s in PIPELINE_STAGE_ORDER}
        unknown = [name for name in requested if name not in valid_values]
        if unknown:
            raise ValueError(
                f"Unknown pipeline stage(s): {unknown!r}. "
                f"Valid stages: {sorted(valid_values)}"
            )

        requested_set = set(requested)
        return [s for s in PIPELINE_STAGE_ORDER if s.value in requested_set]
