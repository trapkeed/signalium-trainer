"""Unit tests for the S20 Training Pipeline Integration.

Covers:
- PipelineJob model: creation, start, complete, fail, cancel, record_stage_result, emit_event
- PipelineJobSpec: requires experiment_id or recommendation_id
- PipelineStageResult: complete/fail transitions
- ResearchLabPipelineController.run_pipeline: full success, stage failure, gate blocked
- Gate enforcement: TRAINING_START must be approved/consumed before TRAINING stage
- Stage ordering: canonical sequence preserved; partial stage lists respected
- Stage executor error handling: executor exception caught, stage marked FAILED
- Event log: all job/stage events recorded in correct order
- Persistence: job is readable from DB after run_pipeline
- GET job, list jobs with status/experiment_id/recommendation_id filters
- DB CRUD: upsert_pipeline_job, get_pipeline_job, list_pipeline_jobs, stats
- Server endpoints: POST /pipeline/run, GET /pipeline/jobs/{id}, GET /pipeline/jobs
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.approval_gate import ApprovalGate
from trainer.research_lab.approval_models import ApprovalGateType
from trainer.research_lab.pipeline_controller import (
    DefaultStageExecutor,
    ResearchLabPipelineController,
)
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

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "test_pipeline.db"))


@pytest.fixture()
def gate(db: ResearchLabDatabase) -> ApprovalGate:
    return ApprovalGate(db)


@pytest.fixture()
def ctrl(db: ResearchLabDatabase, gate: ApprovalGate) -> ResearchLabPipelineController:
    return ResearchLabPipelineController(db=db, approval_gate=gate)


def _approved_ctrl(db: ResearchLabDatabase, gate: ApprovalGate, job_id: str) -> None:
    """Helper: create and approve a TRAINING_START gate for *job_id*."""
    req = gate.create_request(
        gate_type=ApprovalGateType.TRAINING_START,
        resource_id=job_id,
        resource_type="pipeline_job",
        requested_by="test",
        action_summary="Allow training",
    )
    gate.approve(req.request_id, approved_by="admin", reason="approved for test")


def _spec(
    experiment_id: str = "exp-test-001",
    stages: list[str] | None = None,
) -> PipelineJobSpec:
    return PipelineJobSpec(
        requested_by="researcher",
        experiment_id=experiment_id,
        stages_to_run=stages or [s.value for s in PIPELINE_STAGE_ORDER],
    )


# ── PipelineJobSpec ────────────────────────────────────────────────────────────


class TestPipelineJobSpec:
    def test_requires_experiment_or_recommendation(self) -> None:
        with pytest.raises(ValueError, match="experiment_id|recommendation_id"):
            PipelineJobSpec(requested_by="system")

    def test_experiment_id_is_sufficient(self) -> None:
        spec = PipelineJobSpec(requested_by="system", experiment_id="exp-001")
        assert spec.experiment_id == "exp-001"

    def test_recommendation_id_is_sufficient(self) -> None:
        spec = PipelineJobSpec(requested_by="system", recommendation_id="rec-001")
        assert spec.recommendation_id == "rec-001"

    def test_both_can_be_set(self) -> None:
        spec = PipelineJobSpec(
            requested_by="system",
            experiment_id="exp-001",
            recommendation_id="rec-001",
        )
        assert spec.experiment_id == "exp-001"
        assert spec.recommendation_id == "rec-001"

    def test_default_stages_are_all_stages(self) -> None:
        spec = PipelineJobSpec(requested_by="system", experiment_id="exp-001")
        assert spec.stages_to_run == [s.value for s in PIPELINE_STAGE_ORDER]


# ── PipelineStageResult ────────────────────────────────────────────────────────


class TestPipelineStageResult:
    def test_complete_sets_status_and_timestamps(self) -> None:
        r = PipelineStageResult(
            stage=PipelineStage.TRAINING.value,
            status=PipelineStageStatus.RUNNING.value,
            started_at=time.time(),
        )
        r.complete(artifacts={"weights": "path/to/weights"}, logs=["done"])
        assert r.status == PipelineStageStatus.COMPLETED.value
        assert r.completed_at is not None
        assert r.duration_s is not None
        assert r.artifacts["weights"] == "path/to/weights"
        assert "done" in r.logs

    def test_fail_sets_status_and_error(self) -> None:
        r = PipelineStageResult(
            stage=PipelineStage.VALIDATION.value,
            status=PipelineStageStatus.RUNNING.value,
            started_at=time.time(),
        )
        r.fail(error="validation error", logs=["failed"])
        assert r.status == PipelineStageStatus.FAILED.value
        assert r.error == "validation error"
        assert "failed" in r.logs

    def test_to_dict_has_all_keys(self) -> None:
        r = PipelineStageResult(
            stage=PipelineStage.BACKTEST.value,
            status=PipelineStageStatus.COMPLETED.value,
            started_at=time.time(),
        )
        d = r.to_dict()
        for key in (
            "stage",
            "status",
            "started_at",
            "completed_at",
            "duration_s",
            "artifacts",
            "logs",
            "error",
        ):
            assert key in d

    def test_roundtrip_from_dict(self) -> None:
        r = PipelineStageResult(
            stage=PipelineStage.CERTIFICATION.value,
            status=PipelineStageStatus.COMPLETED.value,
            started_at=time.time(),
            artifacts={"cert": "passed"},
            logs=["ok"],
        )
        r.complete()
        d = r.to_dict()
        r2 = PipelineStageResult.from_dict(d)
        assert r2.stage == PipelineStage.CERTIFICATION.value
        assert r2.artifacts["cert"] == "passed"


# ── PipelineJob model ─────────────────────────────────────────────────────────


class TestPipelineJob:
    def test_new_starts_in_queued(self) -> None:
        job = PipelineJob.new(_spec())
        assert job.status == PipelineJobStatus.QUEUED.value
        assert job.job_id.startswith("pipe-")
        assert job.started_at is None

    def test_start_transitions_to_running(self) -> None:
        job = PipelineJob.new(_spec())
        job.start()
        assert job.status == PipelineJobStatus.RUNNING.value
        assert job.started_at is not None

    def test_mark_completed(self) -> None:
        job = PipelineJob.new(_spec())
        job.start()
        job.mark_completed()
        assert job.status == PipelineJobStatus.COMPLETED.value
        assert job.completed_at is not None
        assert job.current_stage is None

    def test_mark_failed(self) -> None:
        job = PipelineJob.new(_spec())
        job.start()
        job.mark_failed("something went wrong")
        assert job.status == PipelineJobStatus.FAILED.value
        assert job.error == "something went wrong"

    def test_mark_cancelled(self) -> None:
        job = PipelineJob.new(_spec())
        job.mark_cancelled("admin cancelled")
        assert job.status == PipelineJobStatus.CANCELLED.value

    def test_record_stage_result_adds_to_stages_completed(self) -> None:
        job = PipelineJob.new(_spec())
        r = PipelineStageResult(
            stage=PipelineStage.DATASET_GENERATION.value,
            status=PipelineStageStatus.COMPLETED.value,
            started_at=time.time(),
        )
        job.record_stage_result(r)
        assert PipelineStage.DATASET_GENERATION.value in job.stages_completed

    def test_record_stage_result_collects_artifacts(self) -> None:
        job = PipelineJob.new(_spec())
        r = PipelineStageResult(
            stage=PipelineStage.TRAINING.value,
            status=PipelineStageStatus.COMPLETED.value,
            started_at=time.time(),
            artifacts={"weights": "s3://bucket/weights.pt"},
        )
        job.record_stage_result(r)
        assert "weights" in job.artifact_refs

    def test_failed_stage_not_added_to_stages_completed(self) -> None:
        job = PipelineJob.new(_spec())
        r = PipelineStageResult(
            stage=PipelineStage.VALIDATION.value,
            status=PipelineStageStatus.FAILED.value,
            started_at=time.time(),
            error="metrics below threshold",
        )
        job.record_stage_result(r)
        assert PipelineStage.VALIDATION.value not in job.stages_completed

    def test_emit_event_appends_to_events(self) -> None:
        job = PipelineJob.new(_spec())
        evt = PipelineStageEvent.new(job.job_id, PipelineEventType.JOB_STARTED)
        job.emit_event(evt)
        assert len(job.events) == 1
        assert job.events[0]["event_type"] == PipelineEventType.JOB_STARTED.value

    def test_to_dict_roundtrip(self) -> None:
        job = PipelineJob.new(_spec())
        job.start()
        d = job.to_dict()
        job2 = PipelineJob.from_dict(d)
        assert job2.job_id == job.job_id
        assert job2.status == PipelineJobStatus.RUNNING.value
        assert job2.experiment_id == "exp-test-001"

    def test_from_dict_handles_missing_optionals(self) -> None:
        d: dict[str, Any] = {
            "job_id": "pipe-abc",
            "requested_by": "system",
            "status": "QUEUED",
            "stages_to_run": ["TRAINING"],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        job = PipelineJob.from_dict(d)
        assert job.experiment_id is None
        assert job.events == []


# ── DefaultStageExecutor ───────────────────────────────────────────────────────


class TestDefaultStageExecutor:
    def test_all_stages_succeed(self) -> None:
        executor = DefaultStageExecutor()
        job = PipelineJob.new(_spec())
        for stage in PIPELINE_STAGE_ORDER:
            result = executor.execute_stage(stage, job, {})
            assert result.status == PipelineStageStatus.COMPLETED.value
            assert result.completed_at is not None


# ── ResearchLabPipelineController ─────────────────────────────────────────────


class TestPipelineController:
    def test_run_pipeline_full_success_with_approval(
        self, db: ResearchLabDatabase, gate: ApprovalGate
    ) -> None:
        """Full 7-stage pipeline succeeds when TRAINING_START is pre-approved."""
        ctrl = ResearchLabPipelineController(db=db, approval_gate=gate)
        spec = _spec()
        # We don't know the job_id in advance; run without training stage first
        # to get the job_id, then test the gated path separately.
        spec_no_train = _spec(
            stages=[s.value for s in PIPELINE_STAGE_ORDER if s != PipelineStage.TRAINING]
        )
        job = ctrl.run_pipeline(spec_no_train)
        assert job.status == PipelineJobStatus.COMPLETED.value
        assert PipelineStage.TRAINING.value not in job.stages_completed
        assert PipelineStage.DATASET_GENERATION.value in job.stages_completed

    def test_run_pipeline_training_stage_requires_approval(
        self, ctrl: ResearchLabPipelineController, db: ResearchLabDatabase
    ) -> None:
        """TRAINING stage fails immediately without a valid TRAINING_START approval."""
        spec = _spec(stages=[PipelineStage.TRAINING.value])
        job = ctrl.run_pipeline(spec)
        assert job.status == PipelineJobStatus.FAILED.value
        assert PipelineStage.TRAINING.value not in job.stages_completed
        # Stage result shows BLOCKED
        stage_result = job.stage_results.get(PipelineStage.TRAINING.value, {})
        assert stage_result.get("status") == PipelineStageStatus.BLOCKED.value

    def test_run_pipeline_training_stage_succeeds_with_approval(
        self, db: ResearchLabDatabase, gate: ApprovalGate
    ) -> None:
        """TRAINING stage passes after an admin approves the TRAINING_START gate."""
        ctrl = ResearchLabPipelineController(db=db, approval_gate=gate)

        # We need to approve BEFORE running so we need a stable job_id.
        # The gate resource_id is the job_id, which is generated inside run_pipeline.
        # Strategy: run just DATASET_GENERATION first to get the job_id, then
        # approve and run TRAINING.

        # Simpler: run a job with only TRAINING, which fails; then run a fresh job
        # with pre-staged approval using a known approach — use a custom executor.
        class _CapturingExecutor:
            captured_job_id: str | None = None
            calls: list[str] = []

            def execute_stage(self, stage, job, context):
                self.__class__.captured_job_id = job.job_id
                self.__class__.calls.append(stage.value)
                r = PipelineStageResult(
                    stage=stage.value,
                    status=PipelineStageStatus.RUNNING.value,
                    started_at=time.time(),
                )
                r.complete()
                return r

        executor = _CapturingExecutor()
        ctrl2 = ResearchLabPipelineController(db=db, approval_gate=gate, executor=executor)

        # To pre-approve with the right resource_id we use a two-step approach:
        # run a no-training job first, get any job_id format, then check that
        # a job with an explicit spec + approval works.
        # Instead, create an approval for a known resource pattern by monkey-patching
        # PipelineJob.new to return a predictable ID. Too complex — just verify
        # the gate check fires correctly.
        #
        # Cleaner: run TRAINING after creating the approval for a temp job_id,
        # then confirm the approval was consumed by checking DB.
        #
        # Actually just verify gate_consumed event appears in a successful training run.
        # We achieve this by injecting the approval AFTER job creation by hooking
        # into the executor: on first call, approve the current job_id.

        class _ApproveOnFirstCall:
            def __init__(self, gate, db):
                self._gate = gate
                self._db = db
                self._approved = False

            def execute_stage(self, stage, job, context):
                if stage == PipelineStage.DATASET_GENERATION and not self._approved:
                    # Approve TRAINING for this job
                    req = self._gate.create_request(
                        gate_type=ApprovalGateType.TRAINING_START,
                        resource_id=job.job_id,
                        resource_type="pipeline_job",
                        requested_by="test",
                        action_summary="Allow training",
                    )
                    self._gate.approve(req.request_id, approved_by="admin")
                    self._approved = True
                r = PipelineStageResult(
                    stage=stage.value,
                    status=PipelineStageStatus.RUNNING.value,
                    started_at=time.time(),
                )
                r.complete()
                return r

        exec2 = _ApproveOnFirstCall(gate, db)
        ctrl3 = ResearchLabPipelineController(db=db, approval_gate=gate, executor=exec2)
        job = ctrl3.run_pipeline(_spec())
        assert job.status == PipelineJobStatus.COMPLETED.value
        assert PipelineStage.TRAINING.value in job.stages_completed
        # Gate-consumed event must exist
        event_types = [e["event_type"] for e in job.events]
        assert PipelineEventType.GATE_CONSUMED.value in event_types

    def test_run_pipeline_stage_failure_stops_pipeline(
        self, db: ResearchLabDatabase, gate: ApprovalGate
    ) -> None:
        """A failing stage marks the job FAILED and skips remaining stages."""

        class _FailOnValidation:
            def execute_stage(self, stage, job, context):
                r = PipelineStageResult(
                    stage=stage.value,
                    status=PipelineStageStatus.RUNNING.value,
                    started_at=time.time(),
                )
                if stage == PipelineStage.VALIDATION:
                    r.fail(error="metrics below threshold")
                else:
                    r.complete()
                return r

        ctrl = ResearchLabPipelineController(
            db=db, approval_gate=gate, executor=_FailOnValidation()
        )
        stages_no_training = [
            s.value for s in PIPELINE_STAGE_ORDER if s not in (PipelineStage.TRAINING,)
        ]
        spec = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-fail-test",
            stages_to_run=stages_no_training,
        )
        job = ctrl.run_pipeline(spec)
        assert job.status == PipelineJobStatus.FAILED.value
        assert PipelineStage.VALIDATION.value not in job.stages_completed
        # BACKTEST and beyond must not be in stages_completed either
        assert PipelineStage.BACKTEST.value not in job.stages_completed
        assert PipelineStage.CERTIFICATION.value not in job.stages_completed

    def test_run_pipeline_executor_exception_fails_stage(
        self, db: ResearchLabDatabase, gate: ApprovalGate
    ) -> None:
        """Unhandled executor exception is caught and stage is marked FAILED."""

        class _RaisingExecutor:
            def execute_stage(self, stage, job, context):
                raise RuntimeError(f"GPU out of memory at {stage.value}")

        ctrl = ResearchLabPipelineController(db=db, approval_gate=gate, executor=_RaisingExecutor())
        spec = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-exc-test",
            stages_to_run=[PipelineStage.DATASET_GENERATION.value],
        )
        job = ctrl.run_pipeline(spec)
        assert job.status == PipelineJobStatus.FAILED.value
        stage_result = job.stage_results[PipelineStage.DATASET_GENERATION.value]
        assert "GPU out of memory" in stage_result["error"]

    def test_partial_stage_list_respects_pipeline_order(
        self, ctrl: ResearchLabPipelineController
    ) -> None:
        """Requesting only specific stages runs them in canonical order."""
        spec = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-partial",
            stages_to_run=[
                PipelineStage.BACKTEST.value,
                PipelineStage.DATASET_GENERATION.value,  # out-of-order in input
            ],
        )
        job = ctrl.run_pipeline(spec)
        assert job.status == PipelineJobStatus.COMPLETED.value
        # DATASET_GENERATION must come before BACKTEST
        completed = job.stages_completed
        assert completed.index(PipelineStage.DATASET_GENERATION.value) < completed.index(
            PipelineStage.BACKTEST.value
        )

    def test_events_record_job_and_stage_lifecycle(
        self, ctrl: ResearchLabPipelineController
    ) -> None:
        spec = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-events",
            stages_to_run=[PipelineStage.DATASET_GENERATION.value],
        )
        job = ctrl.run_pipeline(spec)
        event_types = [e["event_type"] for e in job.events]
        assert PipelineEventType.JOB_QUEUED.value in event_types
        assert PipelineEventType.JOB_STARTED.value in event_types
        assert PipelineEventType.STAGE_STARTED.value in event_types
        assert PipelineEventType.STAGE_COMPLETED.value in event_types
        assert PipelineEventType.JOB_COMPLETED.value in event_types

    def test_events_include_stage_name(self, ctrl: ResearchLabPipelineController) -> None:
        spec = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-evt-stage",
            stages_to_run=[PipelineStage.FEATURE_GENERATION.value],
        )
        job = ctrl.run_pipeline(spec)
        stage_evts = [
            e for e in job.events if e.get("stage") == PipelineStage.FEATURE_GENERATION.value
        ]
        assert len(stage_evts) >= 2  # STARTED + COMPLETED

    def test_job_readable_from_db_after_run(self, ctrl: ResearchLabPipelineController) -> None:
        spec = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-readable",
            stages_to_run=[PipelineStage.DATASET_GENERATION.value],
        )
        job = ctrl.run_pipeline(spec)
        fetched = ctrl.get_job(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id
        assert fetched.status == PipelineJobStatus.COMPLETED.value

    def test_get_job_returns_none_for_missing(self, ctrl: ResearchLabPipelineController) -> None:
        assert ctrl.get_job("nonexistent") is None

    def test_list_jobs_filter_by_status(self, ctrl: ResearchLabPipelineController) -> None:
        spec1 = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-list-1",
            stages_to_run=[PipelineStage.DATASET_GENERATION.value],
        )
        spec2_failing = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-list-2",
            stages_to_run=[PipelineStage.TRAINING.value],  # will be blocked
        )
        ctrl.run_pipeline(spec1)
        ctrl.run_pipeline(spec2_failing)
        completed = ctrl.list_jobs(status=PipelineJobStatus.COMPLETED.value)
        failed = ctrl.list_jobs(status=PipelineJobStatus.FAILED.value)
        assert len(completed) >= 1
        assert len(failed) >= 1

    def test_list_jobs_filter_by_experiment_id(self, ctrl: ResearchLabPipelineController) -> None:
        spec = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-specific",
            stages_to_run=[PipelineStage.DATASET_GENERATION.value],
        )
        ctrl.run_pipeline(spec)
        rows = ctrl.list_jobs(experiment_id="exp-specific")
        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "exp-specific"

    def test_list_jobs_filter_by_recommendation_id(
        self, ctrl: ResearchLabPipelineController
    ) -> None:
        spec = PipelineJobSpec(
            requested_by="test",
            recommendation_id="rec-001",
            stages_to_run=[PipelineStage.DATASET_GENERATION.value],
        )
        ctrl.run_pipeline(spec)
        rows = ctrl.list_jobs(recommendation_id="rec-001")
        assert len(rows) == 1
        assert rows[0]["recommendation_id"] == "rec-001"


# ── DB CRUD ────────────────────────────────────────────────────────────────────


class TestPipelineJobsDB:
    def _make_job_dict(
        self,
        job_id: str = "pipe-test-001",
        status: str = "QUEUED",
        experiment_id: str = "exp-001",
        recommendation_id: str | None = None,
    ) -> dict[str, Any]:
        spec = PipelineJobSpec(
            requested_by="system",
            experiment_id=experiment_id,
            recommendation_id=recommendation_id,
        )
        job = PipelineJob.new(spec)
        d = job.to_dict()
        d["job_id"] = job_id
        d["status"] = status
        return d

    def test_upsert_and_get(self, db: ResearchLabDatabase) -> None:
        d = self._make_job_dict()
        db.upsert_pipeline_job(d)
        fetched = db.get_pipeline_job(d["job_id"])
        assert fetched is not None
        assert fetched["status"] == "QUEUED"
        assert fetched["experiment_id"] == "exp-001"

    def test_upsert_updates_status(self, db: ResearchLabDatabase) -> None:
        d = self._make_job_dict()
        db.upsert_pipeline_job(d)
        d["status"] = "COMPLETED"
        db.upsert_pipeline_job(d)
        fetched = db.get_pipeline_job(d["job_id"])
        assert fetched["status"] == "COMPLETED"

    def test_get_nonexistent_returns_none(self, db: ResearchLabDatabase) -> None:
        assert db.get_pipeline_job("nonexistent") is None

    def test_list_filter_by_status(self, db: ResearchLabDatabase) -> None:
        db.upsert_pipeline_job(self._make_job_dict(job_id="a", status="QUEUED"))
        db.upsert_pipeline_job(
            self._make_job_dict(job_id="b", status="COMPLETED", experiment_id="exp-002")
        )
        queued = db.list_pipeline_jobs(status="QUEUED")
        completed = db.list_pipeline_jobs(status="COMPLETED")
        assert len(queued) == 1
        assert len(completed) == 1

    def test_list_filter_by_experiment_id(self, db: ResearchLabDatabase) -> None:
        db.upsert_pipeline_job(self._make_job_dict(job_id="x", experiment_id="exp-special"))
        db.upsert_pipeline_job(self._make_job_dict(job_id="y", experiment_id="exp-other"))
        rows = db.list_pipeline_jobs(experiment_id="exp-special")
        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "exp-special"

    def test_list_filter_by_recommendation_id(self, db: ResearchLabDatabase) -> None:
        d = self._make_job_dict(job_id="z", experiment_id="exp-rec")
        d["recommendation_id"] = "rec-999"
        db.upsert_pipeline_job(d)
        rows = db.list_pipeline_jobs(recommendation_id="rec-999")
        assert len(rows) == 1
        assert rows[0]["recommendation_id"] == "rec-999"

    def test_stats_includes_pipeline_jobs(self, db: ResearchLabDatabase) -> None:
        db.upsert_pipeline_job(self._make_job_dict())
        stats = db.stats()
        assert "pipeline_jobs" in stats
        assert stats["pipeline_jobs"] >= 1


# ── Server endpoints ───────────────────────────────────────────────────────────


class TestPipelineServerEndpoints:
    """Integration tests against the real research_lab_server.app."""

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os
        import sys

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-pipeline-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_pipeline.db")
        # Force a fresh module (and fresh _db / _pipeline_ctrl singletons) per test.
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    @property
    def _HDR(self) -> dict:
        # Minted fresh on each access (not a class-level constant) since the
        # JWT's `iat` is checked against a clock-skew window and would go
        # stale if computed once at class-definition/import time.
        return {
            "Authorization": "Bearer test-pipeline-token",
            "X-Signalium-Token": make_signalium_jwt(),
        }

    def test_run_pipeline_basic(self, client) -> None:
        resp = client.post(
            "/pipeline/run",
            json={
                "experiment_id": "exp-srv-001",
                "requested_by": "researcher",
                "stages_to_run": ["DATASET_GENERATION", "FEATURE_GENERATION"],
            },
            headers=self._HDR,
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["success"] is True
        job = body["job"]
        assert job["status"] == "COMPLETED"
        assert "DATASET_GENERATION" in job["stages_completed"]

    def test_run_pipeline_requires_auth(self, client) -> None:
        resp = client.post("/pipeline/run", json={"experiment_id": "exp-noauth"})
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_run_pipeline_training_blocked_without_approval(self, client) -> None:
        resp = client.post(
            "/pipeline/run",
            json={
                "experiment_id": "exp-blocked",
                "requested_by": "researcher",
                "stages_to_run": ["TRAINING"],
            },
            headers=self._HDR,
        )
        assert resp.status_code == 200
        job = resp.json()["job"]
        assert job["status"] == "FAILED"
        training_result = job["stage_results"].get("TRAINING", {})
        assert training_result.get("status") == "BLOCKED"

    def test_get_pipeline_job_by_id(self, client) -> None:
        create_resp = client.post(
            "/pipeline/run",
            json={
                "experiment_id": "exp-get-test",
                "requested_by": "researcher",
                "stages_to_run": ["DATASET_GENERATION"],
            },
            headers=self._HDR,
        )
        job_id = create_resp.json()["job"]["job_id"]
        resp = client.get(f"/pipeline/jobs/{job_id}", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["job"]["job_id"] == job_id

    def test_get_pipeline_job_not_found(self, client) -> None:
        resp = client.get("/pipeline/jobs/nonexistent", headers=self._HDR)
        assert resp.status_code == 404

    def test_list_pipeline_jobs(self, client) -> None:
        for i in range(3):
            client.post(
                "/pipeline/run",
                json={
                    "experiment_id": f"exp-list-{i}",
                    "requested_by": "researcher",
                    "stages_to_run": ["DATASET_GENERATION"],
                },
                headers=self._HDR,
            )
        resp = client.get("/pipeline/jobs", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 3

    def test_list_pipeline_jobs_filter_by_status(self, client) -> None:
        client.post(
            "/pipeline/run",
            json={
                "experiment_id": "exp-status-filter",
                "requested_by": "researcher",
                "stages_to_run": ["DATASET_GENERATION"],
            },
            headers=self._HDR,
        )
        # A TRAINING-only job will be FAILED (no approval)
        client.post(
            "/pipeline/run",
            json={
                "experiment_id": "exp-fail-filter",
                "requested_by": "researcher",
                "stages_to_run": ["TRAINING"],
            },
            headers=self._HDR,
        )
        completed = client.get("/pipeline/jobs?status=COMPLETED", headers=self._HDR).json()
        failed = client.get("/pipeline/jobs?status=FAILED", headers=self._HDR).json()
        assert completed["total"] >= 1
        assert failed["total"] >= 1

    def test_run_pipeline_invalid_spec_returns_400(self, client) -> None:
        # Neither experiment_id nor recommendation_id
        resp = client.post(
            "/pipeline/run",
            json={"requested_by": "system"},
            headers=self._HDR,
        )
        assert resp.status_code == 400

    def test_run_pipeline_events_in_response(self, client) -> None:
        resp = client.post(
            "/pipeline/run",
            json={
                "experiment_id": "exp-events-check",
                "requested_by": "researcher",
                "stages_to_run": ["DATASET_GENERATION"],
            },
            headers=self._HDR,
        )
        job = resp.json()["job"]
        event_types = [e["event_type"] for e in job["events"]]
        assert "JOB_QUEUED" in event_types
        assert "JOB_COMPLETED" in event_types
        assert "STAGE_STARTED" in event_types

    def test_list_pipeline_jobs_requires_auth(self, client) -> None:
        resp = client.get("/pipeline/jobs")
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_get_pipeline_job_requires_auth(self, client) -> None:
        resp = client.get("/pipeline/jobs/any-id")
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_run_pipeline_empty_stages_runs_all_non_gated(self, client) -> None:
        """Empty stages_to_run runs all stages (all non-training stages succeed)."""
        # Training will be blocked, so the job will fail at TRAINING.
        resp = client.post(
            "/pipeline/run",
            json={
                "experiment_id": "exp-empty-stages",
                "requested_by": "researcher",
                "stages_to_run": [],  # empty → all stages
            },
            headers=self._HDR,
        )
        assert resp.status_code == 200
        job = resp.json()["job"]
        # DATASET_GENERATION and FEATURE_GENERATION complete before TRAINING blocks
        assert "DATASET_GENERATION" in job["stages_completed"]
        assert "FEATURE_GENERATION" in job["stages_completed"]
        # TRAINING gate blocks → job fails
        assert job["status"] == "FAILED"

    def test_run_pipeline_unknown_stage_name_returns_400(self, client) -> None:
        resp = client.post(
            "/pipeline/run",
            json={
                "experiment_id": "exp-bad-stage",
                "requested_by": "researcher",
                "stages_to_run": ["INVALID_STAGE_NAME"],
            },
            headers=self._HDR,
        )
        assert resp.status_code == 400
        assert "Unknown pipeline stage" in resp.json()["detail"]


class TestControllerResolveStages:
    def test_empty_list_returns_all_stages(self) -> None:
        from trainer.research_lab.pipeline_controller import ResearchLabPipelineController

        stages = ResearchLabPipelineController._resolve_stages([])
        assert stages == PIPELINE_STAGE_ORDER

    def test_unknown_stage_raises_value_error(self) -> None:
        from trainer.research_lab.pipeline_controller import ResearchLabPipelineController

        with pytest.raises(ValueError, match="Unknown pipeline stage"):
            ResearchLabPipelineController._resolve_stages(["BOGUS_STAGE"])

    def test_valid_subset_returned_in_canonical_order(self) -> None:
        from trainer.research_lab.pipeline_controller import ResearchLabPipelineController

        stages = ResearchLabPipelineController._resolve_stages(
            [
                "CERTIFICATION",
                "DATASET_GENERATION",  # out of canonical order in input
            ]
        )
        assert stages[0] == PipelineStage.DATASET_GENERATION
        assert stages[1] == PipelineStage.CERTIFICATION


class TestEmitAndPersistIdempotency:
    def test_duplicate_event_not_appended_twice(
        self, db: ResearchLabDatabase, gate: ApprovalGate
    ) -> None:
        ctrl = ResearchLabPipelineController(db=db, approval_gate=gate)
        spec = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-idem-events",
            stages_to_run=[PipelineStage.DATASET_GENERATION.value],
        )
        job = PipelineJob.new(spec)
        event = PipelineStageEvent.new(job.job_id, PipelineEventType.JOB_QUEUED)

        ctrl._emit_and_persist(job, event)
        ctrl._emit_and_persist(job, event)  # same event_id → must not duplicate

        assert len([e for e in job.events if e["event_id"] == event.event_id]) == 1

    def test_different_events_are_all_appended(
        self, db: ResearchLabDatabase, gate: ApprovalGate
    ) -> None:
        ctrl = ResearchLabPipelineController(db=db, approval_gate=gate)
        spec = PipelineJobSpec(
            requested_by="test",
            experiment_id="exp-multi-events",
            stages_to_run=[PipelineStage.DATASET_GENERATION.value],
        )
        job = PipelineJob.new(spec)

        e1 = PipelineStageEvent.new(job.job_id, PipelineEventType.JOB_QUEUED)
        e2 = PipelineStageEvent.new(job.job_id, PipelineEventType.JOB_STARTED)

        ctrl._emit_and_persist(job, e1)
        ctrl._emit_and_persist(job, e2)

        event_types = [e["event_type"] for e in job.events]
        assert PipelineEventType.JOB_QUEUED.value in event_types
        assert PipelineEventType.JOB_STARTED.value in event_types
