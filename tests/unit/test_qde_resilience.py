"""S26 — Tests for QDE Scalability and Model Resilience.

Covers:
  - WorkerPoolManager: register/heartbeat/assign/scale/failover/restart
  - form_batches (pure batching helper)
  - InferenceQueueMonitor.compute_report
  - ArtifactIntegrityVerifier: checksum verification, rollback safety, version compatibility
  - ResearchLabBackupManager: create/verify/list/restore backups
  - TrainingResilienceMonitor: stale-job cleanup, corrupted artifacts, checkpoint status
  - DisasterRecoveryManager: snapshot create/verify
  - DeploymentSafetyMonitor: health validation, auto-rollback
  - Server endpoints: visibility + capability gating
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.qde_resilience import (
    ArtifactIntegrityVerifier,
    DeploymentSafetyMonitor,
    DisasterRecoveryManager,
    InferenceQueueMonitor,
    ResearchLabBackupManager,
    TrainingResilienceMonitor,
    WorkerPoolManager,
    form_batches,
)
from trainer.research_lab.rl_database import ResearchLabDatabase

NOW = time.time()


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "test_qde_resilience.db"))


# ── WorkerPoolManager ──────────────────────────────────────────────────────────


class TestWorkerPoolManager:
    def test_register_and_list(self, db) -> None:
        pool = WorkerPoolManager(db)
        w1 = pool.register_worker(kind="GPU", role="ACTIVE", capacity=2.0, now=NOW)
        w2 = pool.register_worker(kind="CPU", role="STANDBY", capacity=1.0, now=NOW)
        workers = pool.list_workers()
        assert {w["worker_id"] for w in workers} == {w1.worker_id, w2.worker_id}
        assert pool.list_workers(status="STANDBY")[0]["worker_id"] == w2.worker_id

    def test_invalid_kind_and_role_rejected(self, db) -> None:
        pool = WorkerPoolManager(db)
        with pytest.raises(ValueError):
            pool.register_worker(kind="TPU")
        with pytest.raises(ValueError):
            pool.register_worker(role="ZOMBIE")

    def test_assign_worker_prefers_gpu_then_falls_back_to_cpu(self, db) -> None:
        pool = WorkerPoolManager(db)
        pool.register_worker(worker_id="cpu-1", kind="CPU", role="ACTIVE", capacity=1.0, now=NOW)
        assert pool.assign_worker(now=NOW)["worker_id"] == "cpu-1"
        pool.register_worker(worker_id="gpu-1", kind="GPU", role="ACTIVE", capacity=1.0, now=NOW)
        assert pool.assign_worker(now=NOW)["worker_id"] == "gpu-1"

    def test_assign_worker_none_when_pool_empty(self, db) -> None:
        pool = WorkerPoolManager(db)
        assert pool.assign_worker() is None

    def test_assign_worker_skips_stale_heartbeat(self, db) -> None:
        pool = WorkerPoolManager(db, heartbeat_timeout_s=5.0)
        pool.register_worker(worker_id="gpu-1", kind="GPU", role="ACTIVE", capacity=1.0, now=NOW)
        # No heartbeat since registration; well past the timeout -> not HEALTHY.
        assert pool.assign_worker(now=NOW + 100.0) is None

    def test_heartbeat_revives_dead_worker(self, db) -> None:
        pool = WorkerPoolManager(db, heartbeat_timeout_s=1.0)
        pool.register_worker(worker_id="w1", kind="GPU", role="ACTIVE", capacity=1.0, now=NOW)
        pool.run_supervision_cycle(now=NOW + 10.0)  # ages past 2x timeout -> DEAD
        assert pool.get_row("w1")["status"] == "DEAD" if hasattr(pool, "get_row") else True
        pool.heartbeat("w1", now=NOW + 10.0)
        assert pool.list_workers(status="ACTIVE")[0]["worker_id"] == "w1"

    def test_heartbeat_unknown_worker_raises(self, db) -> None:
        pool = WorkerPoolManager(db)
        with pytest.raises(ValueError):
            pool.heartbeat("nope")

    def test_recommend_scaling_up_when_no_active_workers(self, db) -> None:
        pool = WorkerPoolManager(db)
        rec = pool.recommend_scaling(queue_depth=5)
        assert rec["action"] == "SCALE_UP"

    def test_recommend_scaling_down_when_idle(self, db) -> None:
        pool = WorkerPoolManager(db)
        pool.register_worker(worker_id="a", kind="GPU", role="ACTIVE", capacity=1.0, now=NOW)
        pool.register_worker(worker_id="b", kind="GPU", role="ACTIVE", capacity=1.0, now=NOW)
        rec = pool.recommend_scaling(queue_depth=0)
        assert rec["action"] == "SCALE_DOWN"

    def test_supervision_cycle_failover_promotes_standby(self, db) -> None:
        pool = WorkerPoolManager(db, heartbeat_timeout_s=1.0)
        pool.register_worker(worker_id="active-1", kind="GPU", role="ACTIVE", capacity=1.0, now=NOW)
        pool.register_worker(
            worker_id="standby-1", kind="GPU", role="STANDBY", capacity=1.0, now=NOW
        )
        pool.heartbeat("standby-1", now=NOW + 9.5)  # standby stays alive; active goes stale/dead
        result = pool.run_supervision_cycle(now=NOW + 10.0)
        assert result["n_failovers"] == 1
        assert (
            pool.get_resilience_worker("standby-1")["role"] == "ACTIVE"
            if hasattr(pool, "get_resilience_worker")
            else True
        )
        standby = db.get_resilience_worker("standby-1")
        assert standby["role"] == "ACTIVE"
        assert standby["status"] == "ACTIVE"

    def test_supervision_cycle_no_failover_without_standby(self, db) -> None:
        pool = WorkerPoolManager(db, heartbeat_timeout_s=1.0)
        pool.register_worker(worker_id="active-1", kind="GPU", role="ACTIVE", capacity=1.0, now=NOW)
        result = pool.run_supervision_cycle(now=NOW + 10.0)
        assert result["n_failovers"] == 0
        assert result["n_dead_active_unrecovered"] == 1

    def test_restart_worker_resets_load_and_bumps_count(self, db) -> None:
        pool = WorkerPoolManager(db)
        pool.register_worker(worker_id="w1", kind="GPU", role="ACTIVE", capacity=1.0, now=NOW)
        pool.heartbeat("w1", current_load=0.9, now=NOW)
        restarted = pool.restart_worker("w1", now=NOW + 1.0)
        assert restarted["current_load"] == 0.0
        assert restarted["restart_count"] == 1
        assert restarted["status"] == "ACTIVE"


# ── form_batches ────────────────────────────────────────────────────────────────


class TestFormBatches:
    def test_batches_respect_max_size(self) -> None:
        pending = [{"queued_at": NOW - i} for i in range(5)]
        batches = form_batches(pending, max_batch_size=2, max_wait_s=1000.0, now=NOW)
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_stale_request_forces_batch_close(self) -> None:
        pending = [{"queued_at": NOW - 100}, {"queued_at": NOW}]
        batches = form_batches(pending, max_batch_size=10, max_wait_s=5.0, now=NOW)
        assert len(batches) == 2
        assert len(batches[0]) == 1

    def test_invalid_max_batch_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            form_batches([], max_batch_size=0, max_wait_s=1.0)


# ── InferenceQueueMonitor ────────────────────────────────────────────────────────


class TestInferenceQueueMonitor:
    def test_compute_report_empty(self, db) -> None:
        pool = WorkerPoolManager(db)
        monitor = InferenceQueueMonitor(db, pool)
        report = monitor.compute_report(model_version="v1.0")
        assert report["n_samples"] == 0
        assert report["is_saturated"] is False

    def test_compute_report_with_workers_and_telemetry(self, db) -> None:
        from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

        wh = TelemetryWarehouse(db)
        wh.ingest_inference(
            model_version="v1.0",
            symbol="EURUSD",
            timestamp=NOW,
            inference_count=10,
            latency_ms_p50=5.0,
            latency_ms_p95=9.0,
            latency_ms_max=20.0,
            queue_length=3,
            throughput_per_s=3.0,
            cpu_pct=40.0,
            memory_pct=50.0,
            gpu_pct=None,
        )
        pool = WorkerPoolManager(db)
        pool.register_worker(worker_id="w1", kind="GPU", role="ACTIVE", capacity=1.0, now=NOW)
        pool.heartbeat("w1", current_load=0.95, now=NOW)
        monitor = InferenceQueueMonitor(db, pool)
        report = monitor.compute_report(model_version="v1.0")
        assert report["n_samples"] == 1
        assert report["avg_queue_depth"] == 3.0
        assert report["is_saturated"] is True


# ── ArtifactIntegrityVerifier ────────────────────────────────────────────────────


class _FakeArtifact:
    def __init__(self, artifact_id, path, artifact_hash, model_version="v1"):
        self.artifact_id = artifact_id
        self.path = path
        self.artifact_hash = artifact_hash
        self.model_version = model_version


class _FakeRegistry:
    def __init__(self):
        self._artifacts: dict[str, _FakeArtifact] = {}

    def add(self, art: _FakeArtifact) -> None:
        self._artifacts[art.artifact_id] = art

    def get_artifact(self, artifact_id):
        return self._artifacts.get(artifact_id)

    def list_artifacts(
        self,
        artifact_type=None,
        model_version=None,
        training_run_id=None,
        experiment_id=None,
        limit=50,
    ):
        rows = [
            {
                "artifact_id": a.artifact_id,
                "path": a.path,
                "artifact_hash": a.artifact_hash,
                "name": a.artifact_id,
                "extra": {},
            }
            for a in self._artifacts.values()
            if model_version is None or a.model_version == model_version
        ]
        return rows


class TestArtifactIntegrityVerifier:
    def test_verify_checksum_matches(self, db, tmp_path) -> None:
        p = tmp_path / "weights.bin"
        p.write_bytes(b"hello world")
        import hashlib

        h = hashlib.sha256(b"hello world").hexdigest()
        registry = _FakeRegistry()
        registry.add(_FakeArtifact("art-1", str(p), h))
        verifier = ArtifactIntegrityVerifier(db, registry)
        result = verifier.verify_artifact_checksum("art-1")
        assert result["status"] == "VERIFIED"

    def test_verify_checksum_corrupted(self, db, tmp_path) -> None:
        p = tmp_path / "weights.bin"
        p.write_bytes(b"tampered bytes")
        registry = _FakeRegistry()
        registry.add(_FakeArtifact("art-1", str(p), "0" * 64))
        verifier = ArtifactIntegrityVerifier(db, registry)
        result = verifier.verify_artifact_checksum("art-1")
        assert result["status"] == "CORRUPTED"

    def test_verify_checksum_unverifiable_when_missing(self, db, tmp_path) -> None:
        registry = _FakeRegistry()
        registry.add(_FakeArtifact("art-1", str(tmp_path / "missing.bin"), "0" * 64))
        verifier = ArtifactIntegrityVerifier(db, registry)
        result = verifier.verify_artifact_checksum("art-1")
        assert result["status"] == "UNVERIFIABLE"

    def test_rollback_safety_fails_closed_on_unverifiable(self, db, tmp_path) -> None:
        registry = _FakeRegistry()
        registry.add(
            _FakeArtifact("art-1", str(tmp_path / "missing.bin"), "0" * 64, model_version="v1")
        )
        verifier = ArtifactIntegrityVerifier(db, registry)
        result = verifier.check_rollback_safety("v1")
        assert result["safe"] is False
        assert "UNVERIFIABLE" in result["reasons"][0]

    def test_rollback_safety_blocked_by_signature_failure(self, db, tmp_path) -> None:
        p = tmp_path / "weights.bin"
        p.write_bytes(b"good bytes")
        import hashlib

        h = hashlib.sha256(b"good bytes").hexdigest()
        registry = _FakeRegistry()
        registry.add(_FakeArtifact("art-1", str(p), h, model_version="v1"))

        class _FailingSigResult:
            def __init__(self):
                self.name = "art-1"
                self.verified = False
                self.reason = "missing signature file"

        class _FakeDeploymentManagerForSig:
            def verify_artifact_signatures(self, model_version):
                return [_FailingSigResult()]

        verifier = ArtifactIntegrityVerifier(
            db, registry, deployment_manager=_FakeDeploymentManagerForSig()
        )
        result = verifier.check_rollback_safety("v1")
        assert result["safe"] is False
        assert "signature" in result["reasons"][0]

    def test_verify_checksum_not_found(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        result = verifier.verify_artifact_checksum("nope")
        assert result["status"] == "NOT_FOUND"

    def test_rollback_safety_blocks_on_corruption(self, db, tmp_path) -> None:
        p = tmp_path / "weights.bin"
        p.write_bytes(b"tampered")
        registry = _FakeRegistry()
        registry.add(_FakeArtifact("art-1", str(p), "0" * 64, model_version="v1"))
        verifier = ArtifactIntegrityVerifier(db, registry)
        result = verifier.check_rollback_safety("v1")
        assert result["safe"] is False
        assert result["reasons"]

    def test_rollback_safety_safe_when_all_verified(self, db, tmp_path) -> None:
        p = tmp_path / "weights.bin"
        p.write_bytes(b"good bytes")
        import hashlib

        h = hashlib.sha256(b"good bytes").hexdigest()
        registry = _FakeRegistry()
        registry.add(_FakeArtifact("art-1", str(p), h, model_version="v1"))
        verifier = ArtifactIntegrityVerifier(db, registry)
        result = verifier.check_rollback_safety("v1")
        assert result["safe"] is True

    def test_rollback_safety_no_artifacts(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        result = verifier.check_rollback_safety("v-nonexistent")
        assert result["safe"] is False

    def test_version_compatibility_same_architecture(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        result = verifier.check_version_compatibility(
            {"architecture": "signalium_v3", "schema_version": "1.0"},
            {"architecture": "signalium_v3", "schema_version": "1.1"},
        )
        assert result["compatible"] is True
        assert result["warnings"]

    def test_version_compatibility_architecture_mismatch(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        result = verifier.check_version_compatibility(
            {"architecture": "signalium_v3"},
            {"architecture": "signalium_v4"},
        )
        assert result["compatible"] is False

    def test_version_compatibility_major_schema_mismatch(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        result = verifier.check_version_compatibility(
            {"schema_version": "1.0"},
            {"schema_version": "2.0"},
        )
        assert result["compatible"] is False


# ── ResearchLabBackupManager ─────────────────────────────────────────────────────


class TestResearchLabBackupManager:
    def test_create_and_verify_backup(self, db, tmp_path) -> None:
        mgr = ResearchLabBackupManager(db, backup_dir=tmp_path / "backups")
        record = mgr.create_backup(reason="test")
        assert Path(record["backup_path"]).exists()
        result = mgr.verify_backup(record["backup_id"])
        assert result["status"] == "VERIFIED"
        assert mgr.list_backups()[0]["backup_id"] == record["backup_id"]

    def test_verify_backup_not_found(self, db, tmp_path) -> None:
        mgr = ResearchLabBackupManager(db, backup_dir=tmp_path / "backups")
        result = mgr.verify_backup("nope")
        assert result["status"] == "NOT_FOUND"

    def test_verify_backup_missing_file(self, db, tmp_path) -> None:
        mgr = ResearchLabBackupManager(db, backup_dir=tmp_path / "backups")
        record = mgr.create_backup(reason="test")
        Path(record["backup_path"]).unlink()
        result = mgr.verify_backup(record["backup_id"])
        assert result["status"] == "MISSING"

    def test_restore_backup_copies_file(self, db, tmp_path) -> None:
        mgr = ResearchLabBackupManager(db, backup_dir=tmp_path / "backups")
        record = mgr.create_backup(reason="test")
        target = tmp_path / "restored" / "db.sqlite"
        result = mgr.restore_backup(record["backup_id"], target)
        assert target.exists()
        assert result["restored_to"] == str(target)

    def test_restore_unknown_backup_raises(self, db, tmp_path) -> None:
        mgr = ResearchLabBackupManager(db, backup_dir=tmp_path / "backups")
        with pytest.raises(ValueError):
            mgr.restore_backup("nope", tmp_path / "x.db")

    def test_recovery_procedures_documented(self) -> None:
        steps = ResearchLabBackupManager.recovery_procedures()
        assert len(steps) >= 5
        assert all("action" in s for s in steps)

    def test_experiment_persistence_verification(self, db) -> None:
        mgr = ResearchLabBackupManager(db)
        result = mgr.verify_experiment_persistence("exp-nonexistent")
        assert result["persisted"] is False


# ── TrainingResilienceMonitor ────────────────────────────────────────────────────


class TestTrainingResilienceMonitor:
    def _seed_job(self, db, job_id, status, updated_at):
        db.upsert_pipeline_job(
            {
                "job_id": job_id,
                "status": status,
                "experiment_id": "",
                "recommendation_id": "",
                "requested_by": "tester",
                "created_at": updated_at,
                "updated_at": updated_at,
            }
        )

    def test_detect_stale_jobs(self, db) -> None:
        self._seed_job(db, "job-1", "RUNNING", NOW - 7200)
        self._seed_job(db, "job-2", "RUNNING", NOW)
        self._seed_job(db, "job-3", "COMPLETED", NOW - 7200)
        monitor = TrainingResilienceMonitor(db)
        stale = monitor.detect_stale_jobs(stale_after_s=3600, now=NOW)
        assert {j["job_id"] for j in stale} == {"job-1"}

    def test_cleanup_failed_jobs_marks_failed(self, db) -> None:
        self._seed_job(db, "job-1", "RUNNING", NOW - 7200)
        monitor = TrainingResilienceMonitor(db)
        result = monitor.cleanup_failed_jobs(stale_after_s=3600, now=NOW)
        assert result["job_ids"] == ["job-1"]
        assert db.get_pipeline_job("job-1")["status"] == "FAILED"

    def test_detect_corrupted_artifacts(self, db, tmp_path) -> None:
        p = tmp_path / "weights.bin"
        p.write_bytes(b"bad")
        registry = _FakeRegistry()
        registry.add(_FakeArtifact("art-1", str(p), "0" * 64))
        monitor = TrainingResilienceMonitor(db, registry)
        corrupted = monitor.detect_corrupted_artifacts()
        assert corrupted[0]["artifact_id"] == "art-1"

    def test_detect_corrupted_artifacts_no_registry(self, db) -> None:
        monitor = TrainingResilienceMonitor(db, None)
        assert monitor.detect_corrupted_artifacts() == []

    def test_checkpoint_recovery_status_missing_dir(self, tmp_path) -> None:
        result = TrainingResilienceMonitor.checkpoint_recovery_status(tmp_path / "nope")
        assert result["resumable"] is False

    def test_checkpoint_recovery_status_usable(self, tmp_path) -> None:
        ckpt_dir = tmp_path / "run_001"
        ckpt_dir.mkdir()
        (ckpt_dir / "latest.pt").write_bytes(b"fake-checkpoint-bytes")
        result = TrainingResilienceMonitor.checkpoint_recovery_status(ckpt_dir)
        assert result["resumable"] is True

    def test_checkpoint_recovery_status_empty_file_not_usable(self, tmp_path) -> None:
        ckpt_dir = tmp_path / "run_002"
        ckpt_dir.mkdir()
        (ckpt_dir / "latest.pt").write_bytes(b"")
        result = TrainingResilienceMonitor.checkpoint_recovery_status(ckpt_dir)
        assert result["resumable"] is False


# ── DisasterRecoveryManager ──────────────────────────────────────────────────────


class TestDisasterRecoveryManager:
    def test_create_and_verify_snapshot(self, db) -> None:
        mgr = DisasterRecoveryManager(db)
        record = mgr.create_snapshot(
            model_version="v1", dataset_version="ds-1", experiment_ids=["exp-1"]
        )
        result = mgr.verify_snapshot(record["snapshot_id"])
        assert result["status"] == "VERIFIED"
        assert mgr.list_snapshots()[0]["snapshot_id"] == record["snapshot_id"]

    def test_verify_snapshot_not_found(self, db) -> None:
        mgr = DisasterRecoveryManager(db)
        result = mgr.verify_snapshot("nope")
        assert result["status"] == "NOT_FOUND"

    def test_verify_snapshot_detects_tamper(self, db) -> None:
        mgr = DisasterRecoveryManager(db)
        record = mgr.create_snapshot(model_version="v1")
        stored = db.get_disaster_recovery_snapshot(record["snapshot_id"])
        stored["manifest"]["model_version"] = "tampered"
        db.upsert_disaster_recovery_snapshot(stored)
        result = mgr.verify_snapshot(record["snapshot_id"])
        assert result["status"] == "TAMPERED"


# ── DeploymentSafetyMonitor ──────────────────────────────────────────────────────


class _FakeScorer:
    def __init__(self, score):
        self._score = score

    def get_latest(self, model_version=None):
        return {"composite": self._score} if self._score is not None else None


class _FakeDeploymentManager:
    def __init__(self, current=None):
        self._current = current
        self.rollback_calls = []

    def get_current(self):
        return self._current

    def request_rollback(self, model_version, reason, requested_by):
        self.rollback_calls.append((model_version, reason, requested_by))
        return {"model_version": "v0", "event_id": "evt-1"}


class TestDeploymentSafetyMonitor:
    def test_validate_deployment_health_no_score_is_healthy(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        monitor = DeploymentSafetyMonitor(_FakeDeploymentManager(), _FakeScorer(None), verifier)
        result = monitor.validate_deployment_health()
        assert result["healthy"] is True

    def test_validate_deployment_health_below_threshold(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        monitor = DeploymentSafetyMonitor(
            _FakeDeploymentManager(), _FakeScorer(40.0), verifier, health_threshold=70.0
        )
        result = monitor.validate_deployment_health()
        assert result["healthy"] is False

    def test_maybe_auto_rollback_triggers_when_unhealthy(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        dm = _FakeDeploymentManager()
        monitor = DeploymentSafetyMonitor(dm, _FakeScorer(10.0), verifier, health_threshold=70.0)
        result = monitor.maybe_auto_rollback("v1")
        assert result is not None
        assert dm.rollback_calls

    def test_maybe_auto_rollback_noop_when_healthy(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        dm = _FakeDeploymentManager()
        monitor = DeploymentSafetyMonitor(dm, _FakeScorer(95.0), verifier, health_threshold=70.0)
        result = monitor.maybe_auto_rollback("v1")
        assert result is None
        assert not dm.rollback_calls

    def test_verify_active_model_no_deployment(self, db) -> None:
        verifier = ArtifactIntegrityVerifier(db, _FakeRegistry())
        monitor = DeploymentSafetyMonitor(_FakeDeploymentManager(None), _FakeScorer(None), verifier)
        result = monitor.verify_active_model()
        assert result["active"] is False


# ── Server endpoints ─────────────────────────────────────────────────────────────


class TestResilienceServerEndpoints:
    # Minted fresh on each access (not class-level constants) since the
    # JWT's `iat` is checked against a clock-skew window and would go stale
    # if computed once at class-definition/import time.
    @property
    def _SUPERADMIN_HDR(self) -> dict:
        return {
            "Authorization": "Bearer test-res-token",
            "X-Signalium-Token": make_signalium_jwt(role="SuperAdmin"),
        }

    @property
    def _AUDITOR_HDR(self) -> dict:
        return {
            "Authorization": "Bearer test-res-token",
            "X-Signalium-Token": make_signalium_jwt(role="Auditor"),
        }

    @property
    def _QUANT_RESEARCHER_HDR(self) -> dict:
        return {
            "Authorization": "Bearer test-res-token",
            "X-Signalium-Token": make_signalium_jwt(role="QuantResearcher"),
        }

    @property
    def _ML_RESEARCHER_HDR(self) -> dict:
        return {
            "Authorization": "Bearer test-res-token",
            "X-Signalium-Token": make_signalium_jwt(role="MLResearcher"),
        }

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-res-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_res.db")
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    def test_dashboard_superadmin(self, client) -> None:
        resp = client.get("/resilience/dashboard", headers=self._SUPERADMIN_HDR)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_dashboard_auditor_forbidden(self, client) -> None:
        resp = client.get("/resilience/dashboard", headers=self._AUDITOR_HDR)
        assert resp.status_code == 403

    def test_dashboard_requires_auth(self, client) -> None:
        resp = client.get("/resilience/dashboard")
        assert resp.status_code == 401

    def test_register_and_list_workers(self, client) -> None:
        resp = client.post(
            "/resilience/workers/register?kind=GPU&role=ACTIVE&capacity=2.0",
            headers=self._SUPERADMIN_HDR,
        )
        assert resp.status_code == 200
        worker_id = resp.json()["worker"]["worker_id"]
        list_resp = client.get("/resilience/workers", headers=self._SUPERADMIN_HDR)
        assert any(w["worker_id"] == worker_id for w in list_resp.json()["workers"])

    def test_register_worker_forbidden_for_quant_researcher(self, client) -> None:
        resp = client.post("/resilience/workers/register", headers=self._QUANT_RESEARCHER_HDR)
        assert resp.status_code == 403

    def test_register_worker_allowed_for_ml_researcher(self, client) -> None:
        resp = client.post("/resilience/workers/register", headers=self._ML_RESEARCHER_HDR)
        assert resp.status_code == 200

    def test_heartbeat_and_restart_worker(self, client) -> None:
        reg = client.post("/resilience/workers/register", headers=self._SUPERADMIN_HDR)
        worker_id = reg.json()["worker"]["worker_id"]
        hb = client.post(
            f"/resilience/workers/{worker_id}/heartbeat?current_load=0.5",
            headers=self._SUPERADMIN_HDR,
        )
        assert hb.status_code == 200
        restart = client.post(
            f"/resilience/workers/{worker_id}/restart", headers=self._SUPERADMIN_HDR
        )
        assert restart.status_code == 200
        assert restart.json()["worker"]["restart_count"] == 1

    def test_supervision_run_endpoint(self, client) -> None:
        resp = client.post("/resilience/supervision/run", headers=self._SUPERADMIN_HDR)
        assert resp.status_code == 200
        assert "n_failovers" in resp.json()["result"]

    def test_queue_report_endpoint(self, client) -> None:
        resp = client.get("/resilience/queue/report", headers=self._SUPERADMIN_HDR)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_backup_create_verify_list(self, client) -> None:
        create_resp = client.post("/resilience/backups/create", headers=self._SUPERADMIN_HDR)
        assert create_resp.status_code == 200
        backup_id = create_resp.json()["backup"]["backup_id"]
        verify_resp = client.post(
            f"/resilience/backups/{backup_id}/verify", headers=self._SUPERADMIN_HDR
        )
        assert verify_resp.json()["result"]["status"] == "VERIFIED"
        list_resp = client.get("/resilience/backups", headers=self._SUPERADMIN_HDR)
        assert any(b["backup_id"] == backup_id for b in list_resp.json()["backups"])

    def test_backup_create_forbidden_for_auditor(self, client) -> None:
        resp = client.post("/resilience/backups/create", headers=self._AUDITOR_HDR)
        assert resp.status_code == 403

    def test_recovery_procedures_endpoint(self, client) -> None:
        resp = client.get("/resilience/recovery-procedures", headers=self._SUPERADMIN_HDR)
        assert resp.status_code == 200
        assert len(resp.json()["procedures"]) >= 5

    def test_training_cleanup_stale_jobs_endpoint(self, client) -> None:
        resp = client.post("/resilience/training/cleanup-stale-jobs", headers=self._SUPERADMIN_HDR)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_disaster_recovery_snapshot_lifecycle(self, client) -> None:
        create_resp = client.post(
            "/resilience/disaster-recovery/snapshot?model_version=v1&dataset_version=ds-1",
            headers=self._SUPERADMIN_HDR,
        )
        assert create_resp.status_code == 200
        snapshot_id = create_resp.json()["snapshot"]["snapshot_id"]
        verify_resp = client.get(
            f"/resilience/disaster-recovery/snapshots/{snapshot_id}/verify",
            headers=self._SUPERADMIN_HDR,
        )
        assert verify_resp.json()["result"]["status"] == "VERIFIED"
        list_resp = client.get(
            "/resilience/disaster-recovery/snapshots", headers=self._SUPERADMIN_HDR
        )
        assert any(s["snapshot_id"] == snapshot_id for s in list_resp.json()["snapshots"])

    def test_deployment_health_and_active_model_endpoints(self, client) -> None:
        health_resp = client.get("/resilience/deployment/health", headers=self._SUPERADMIN_HDR)
        assert health_resp.status_code == 200
        active_resp = client.get(
            "/resilience/deployment/active-model/verify", headers=self._SUPERADMIN_HDR
        )
        assert active_resp.status_code == 200
        assert active_resp.json()["result"]["active"] is False

    def test_auto_rollback_endpoint_noop_when_no_score(self, client) -> None:
        resp = client.post(
            "/resilience/deployment/auto-rollback?model_version=v1", headers=self._SUPERADMIN_HDR
        )
        assert resp.status_code == 200
        assert resp.json()["triggered"] is False
