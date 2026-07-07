"""Phase 27.6 — QDE Training Pipeline.

Manages the ordered execution of pipeline stages for a QDE version:

    Historical Data → Training → Validation → Backtesting
        → Shadow Testing → Deployment Candidate
"""

from __future__ import annotations

import time
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.qde.qde_models import (
    PIPELINE_ORDER,
    PipelineRun,
    PipelineStage,
    PipelineStepResult,
    QDEStage,
    QDEVersion,
)
from trainer.qde.qde_registry import QDERegistry


# Map pipeline stage → QDE lifecycle stage
_PIPELINE_TO_QDE_STAGE: dict[PipelineStage, QDEStage] = {
    PipelineStage.HISTORICAL_DATA: QDEStage.TRAINING,
    PipelineStage.TRAINING: QDEStage.TRAINING,
    PipelineStage.VALIDATION: QDEStage.VALIDATION,
    PipelineStage.BACKTESTING: QDEStage.BACKTESTING,
    PipelineStage.SHADOW_TESTING: QDEStage.SHADOW_TESTING,
    PipelineStage.DEPLOYMENT_CANDIDATE: QDEStage.DEPLOYMENT_CANDIDATE,
}


class QDEPipeline:
    """Orchestrates the QDE training pipeline.

    Each pipeline step is executed sequentially; failure at any step
    stops the pipeline and marks the version as FAILED.
    """

    def __init__(self, registry: QDERegistry) -> None:
        self._registry = registry
        self._runs: dict[str, PipelineRun] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, version_id: str) -> PipelineRun:
        """Start a new pipeline run for a registered QDE version."""
        version = self._registry.get(version_id)
        if version is None:
            raise KeyError(f"QDE version {version_id} not found")
        if version.stage not in (QDEStage.REGISTERED, QDEStage.FAILED):
            raise ValueError(
                f"Version {version_id} is in stage {version.stage.value}; "
                "only REGISTERED or FAILED versions may start a new pipeline run."
            )

        run = PipelineRun.create(version_id)
        self._runs[run.run_id] = run
        logger.info(f"QDEPipeline.start: run {run.run_id} for {version_id}")
        return run

    def advance(
        self,
        run_id: str,
        metrics: dict[str, float] | None = None,
        logs: list[str] | None = None,
    ) -> PipelineStepResult:
        """Mark the current pending step as PASSED and advance to the next one.

        Args:
            run_id: The active pipeline run.
            metrics: Metrics produced by this step (merged into the version).
            logs: Log lines for this step.

        Returns:
            The step that was just completed.
        """
        run = self._get_run(run_id)
        version = self._get_version(run.version_id)

        step = run.current_step()
        if step is None:
            raise RuntimeError(f"No pending steps in run {run_id}")

        now = time.time()
        if step.started_at is None:
            step.started_at = now
        step.completed_at = now
        step.status = "PASSED"
        step.metrics = metrics or {}
        step.logs = logs or []

        # Record metrics on the version
        if metrics:
            version.record_metrics(metrics)

        # Advance QDE stage
        qde_stage = _PIPELINE_TO_QDE_STAGE.get(step.stage)
        if qde_stage and version.stage != qde_stage:
            version.advance_stage(qde_stage)

        # If a next step exists, mark it PENDING (already is, but set started)
        next_step = run.current_step()
        if next_step is None:
            # All steps complete
            run.status = "COMPLETED"
            run.completed_at = time.time()
            version.advance_stage(QDEStage.DEPLOYMENT_CANDIDATE)
            logger.info(f"QDEPipeline: run {run_id} COMPLETED → DEPLOYMENT_CANDIDATE")
        else:
            next_step.started_at = time.time()
            next_step.status = "RUNNING"

        self._registry.update(version)
        return step

    def fail(
        self,
        run_id: str,
        error: str,
        logs: list[str] | None = None,
    ) -> PipelineStepResult:
        """Fail the current step and abort the pipeline."""
        run = self._get_run(run_id)
        version = self._get_version(run.version_id)

        step = run.current_step()
        if step is None:
            raise RuntimeError(f"No pending steps in run {run_id}")

        now = time.time()
        step.started_at = step.started_at or now
        step.completed_at = now
        step.status = "FAILED"
        step.error = error
        step.logs = logs or []

        run.status = "FAILED"
        run.completed_at = now
        version.advance_stage(QDEStage.FAILED)
        self._registry.update(version)
        logger.error(f"QDEPipeline: run {run_id} FAILED at {step.stage.value}: {error}")
        return step

    def skip(self, run_id: str, reason: str = "") -> PipelineStepResult:
        """Skip the current step (e.g. re-use cached artifacts)."""
        run = self._get_run(run_id)
        step = run.current_step()
        if step is None:
            raise RuntimeError(f"No pending steps in run {run_id}")

        now = time.time()
        step.started_at = step.started_at or now
        step.completed_at = now
        step.status = "SKIPPED"
        step.logs = [f"Skipped: {reason}"] if reason else ["Skipped"]

        # Start next
        next_step = run.current_step()
        if next_step:
            next_step.started_at = time.time()
            next_step.status = "RUNNING"
        else:
            run.status = "COMPLETED"
            run.completed_at = time.time()

        return step

    def get_run(self, run_id: str) -> PipelineRun | None:
        return self._runs.get(run_id)

    def runs_for_version(self, version_id: str) -> list[PipelineRun]:
        return [r for r in self._runs.values() if r.version_id == version_id]

    def all_runs(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in sorted(self._runs.values(), key=lambda r: r.started_at)]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_run(self, run_id: str) -> PipelineRun:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"Pipeline run {run_id} not found")
        if run.status in ("COMPLETED", "FAILED", "ABORTED"):
            raise RuntimeError(f"Pipeline run {run_id} is already {run.status}")
        return run

    def _get_version(self, version_id: str) -> QDEVersion:
        version = self._registry.get(version_id)
        if version is None:
            raise KeyError(f"QDE version {version_id} not found")
        return version
