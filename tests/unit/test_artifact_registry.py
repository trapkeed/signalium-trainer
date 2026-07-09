"""S20 — Tests for Dataset Versioning + Artifact Management.

Covers:
  - ArtifactRecord / TrainingRunRecord model validation
  - ArtifactRegistry: registration, duplicate detection, hash mismatch
  - DB CRUD: training_runs + artifacts tables
  - Server endpoints: POST /artifacts/register, GET /artifacts, GET /artifacts/{id},
                      POST /artifacts/runs, GET /artifacts/runs, GET /artifacts/runs/{id}
  - MLOpsExporter artifact_registry hook
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.artifact_models import (
    ARTIFACT_TYPES,
    ArtifactRecord,
    ArtifactType,
    TrainingRunRecord,
    compute_bytes_hash,
    validate_artifact_hash,
    validate_artifact_type,
)
from trainer.research_lab.artifact_registry import (
    ArtifactHashMismatchError,
    ArtifactRegistry,
    DuplicateArtifactError,
)
from trainer.research_lab.artifact_store import ArtifactImmutabilityError, ArtifactStore
from trainer.research_lab.rl_database import ResearchLabDatabase

# ── Fixtures ──────────────────────────────────────────────────────────────────

_GOOD_HASH = "a" * 64  # valid 64-char hex SHA-256


def _make_hash(content: str = "test-content") -> str:
    return hashlib.sha256(content.encode()).hexdigest()


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "test_artifacts.db"))


@pytest.fixture()
def registry(db: ResearchLabDatabase) -> ArtifactRegistry:
    return ArtifactRegistry(db)


@pytest.fixture()
def registry_with_store(db: ResearchLabDatabase, tmp_path: Path) -> ArtifactRegistry:
    return ArtifactRegistry(db, store=ArtifactStore(root=tmp_path / "artifact_store"))


@pytest.fixture()
def good_run(registry: ArtifactRegistry) -> TrainingRunRecord:
    run = TrainingRunRecord.new(
        experiment_id="exp-abc123",
        dataset_version="ds-v3.1",
        feature_version="feat-v2.0",
        code_commit="a" * 40,
        config_hash=_make_hash("config"),
        artifact_hash=_make_hash("weights"),
        requested_by="researcher",
    )
    registry.register_training_run(run)
    return run


# ── Model validation ──────────────────────────────────────────────────────────


class TestArtifactModels:
    def test_valid_hash_passes(self) -> None:
        validate_artifact_hash(_GOOD_HASH)

    def test_short_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="64-character"):
            validate_artifact_hash("abc123")

    def test_non_hex_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="hex"):
            validate_artifact_hash("z" * 64)

    def test_valid_artifact_type_passes(self) -> None:
        for t in ARTIFACT_TYPES:
            validate_artifact_type(t)

    def test_unknown_artifact_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown artifact_type"):
            validate_artifact_type("INVISIBLE_TYPE")

    def test_artifact_record_new_sets_id(self) -> None:
        rec = ArtifactRecord.new(
            artifact_type=ArtifactType.WEIGHTS.value,
            name="weights.pt",
            path="/exports/weights.pt",
            artifact_hash=_GOOD_HASH,
        )
        assert rec.artifact_id.startswith("art-")

    def test_artifact_record_roundtrip(self) -> None:
        rec = ArtifactRecord.new(
            artifact_type=ArtifactType.ONNX.value,
            name="model.onnx",
            path="/exports/model.onnx",
            artifact_hash=_GOOD_HASH,
            model_version="v3.0",
            experiment_id="exp-xyz",
            tags=["production", "onnx"],
        )
        restored = ArtifactRecord.from_dict(rec.to_dict())
        assert restored.artifact_id == rec.artifact_id
        assert restored.artifact_type == rec.artifact_type
        assert restored.tags == ["production", "onnx"]

    def test_training_run_record_new(self) -> None:
        run = TrainingRunRecord.new(
            experiment_id="exp-001",
            dataset_version="ds-v1",
            feature_version="feat-v1",
            code_commit="abc" * 14,
            config_hash=_GOOD_HASH,
            artifact_hash=_GOOD_HASH,
        )
        assert run.run_id.startswith("run-")
        assert run.experiment_id == "exp-001"

    def test_training_run_bad_artifact_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="64-character"):
            TrainingRunRecord.new(
                experiment_id="exp-001",
                dataset_version="ds-v1",
                feature_version="feat-v1",
                code_commit="abc",
                config_hash=_GOOD_HASH,
                artifact_hash="too-short",
            )

    def test_training_run_roundtrip(self) -> None:
        run = TrainingRunRecord.new(
            experiment_id="exp-abc",
            dataset_version="ds-v2",
            feature_version="feat-v1",
            code_commit="c" * 40,
            config_hash=_GOOD_HASH,
            artifact_hash=_make_hash("run-content"),
            mlflow_run_id="mlf-run-0001",
            pipeline_job_id="pipe-job-001",
        )
        restored = TrainingRunRecord.from_dict(run.to_dict())
        assert restored.run_id == run.run_id
        assert restored.mlflow_run_id == "mlf-run-0001"
        assert restored.pipeline_job_id == "pipe-job-001"

    def test_compute_bytes_hash(self) -> None:
        data = b"hello artifact"
        h = compute_bytes_hash(data)
        assert len(h) == 64
        assert h == hashlib.sha256(data).hexdigest()

    def test_artifact_type_enum_values(self) -> None:
        assert set(ARTIFACT_TYPES) == {
            "WEIGHTS",
            "CONFIG",
            "CALIBRATION",
            "ONNX",
            "TORCHSCRIPT",
            "DEPLOYMENT_MANIFEST",
        }


# ── ArtifactRegistry ──────────────────────────────────────────────────────────


class TestArtifactRegistry:
    def _make_artifact(
        self, artifact_type: str = "WEIGHTS", content: str = "data"
    ) -> ArtifactRecord:
        return ArtifactRecord.new(
            artifact_type=artifact_type,
            name=f"{artifact_type.lower()}.bin",
            path=f"/exports/{artifact_type.lower()}.bin",
            artifact_hash=_make_hash(content),
        )

    def test_register_and_get(self, registry: ArtifactRegistry) -> None:
        art = self._make_artifact()
        registry.register_artifact(art)
        fetched = registry.get_artifact(art.artifact_id)
        assert fetched is not None
        assert fetched.artifact_id == art.artifact_id
        assert fetched.artifact_type == "WEIGHTS"

    def test_duplicate_rejected(self, registry: ArtifactRegistry) -> None:
        art1 = self._make_artifact(content="same")
        art2 = self._make_artifact(content="same")  # same hash
        registry.register_artifact(art1)
        with pytest.raises(DuplicateArtifactError):
            registry.register_artifact(art2)

    def test_different_type_same_hash_allowed(self, registry: ArtifactRegistry) -> None:
        """Same hash but different type is a distinct artifact."""
        h = _make_hash("shared-bytes")
        art1 = ArtifactRecord.new("WEIGHTS", "weights.pt", "/w.pt", h)
        art2 = ArtifactRecord.new("ONNX", "model.onnx", "/m.onnx", h)
        registry.register_artifact(art1)
        registry.register_artifact(art2)  # must not raise

    def test_all_six_artifact_types_registrable(self, registry: ArtifactRegistry) -> None:
        for i, atype in enumerate(ARTIFACT_TYPES):
            art = ArtifactRecord.new(atype, f"{atype}.bin", f"/{i}.bin", _make_hash(f"{atype}-{i}"))
            registry.register_artifact(art)
        assert len(registry.list_artifacts()) == len(ARTIFACT_TYPES)

    def test_list_filter_by_type(self, registry: ArtifactRegistry) -> None:
        registry.register_artifact(self._make_artifact("WEIGHTS", "w1"))
        registry.register_artifact(self._make_artifact("ONNX", "o1"))
        weights = registry.list_artifacts(artifact_type="WEIGHTS")
        assert len(weights) == 1
        assert weights[0]["artifact_type"] == "WEIGHTS"

    def test_list_filter_by_model_version(self, registry: ArtifactRegistry) -> None:
        art = ArtifactRecord.new(
            "CONFIG", "cfg.json", "/cfg.json", _make_hash("cfg"), model_version="v3"
        )
        registry.register_artifact(art)
        results = registry.list_artifacts(model_version="v3")
        assert len(results) == 1
        assert results[0]["model_version"] == "v3"

    def test_list_filter_by_training_run(
        self, registry: ArtifactRegistry, good_run: TrainingRunRecord
    ) -> None:
        art = ArtifactRecord.new(
            "WEIGHTS", "w.pt", "/w.pt", _make_hash("weights"), training_run_id=good_run.run_id
        )
        registry.register_artifact(art)
        arts = registry.artifacts_for_run(good_run.run_id)
        assert len(arts) == 1
        assert arts[0]["training_run_id"] == good_run.run_id

    def test_provenance_includes_training_run(
        self, registry: ArtifactRegistry, good_run: TrainingRunRecord
    ) -> None:
        art = ArtifactRecord.new(
            "WEIGHTS",
            "w.pt",
            "/w.pt",
            _make_hash("weights"),
            training_run_id=good_run.run_id,
            experiment_id=good_run.experiment_id,
        )
        registry.register_artifact(art)
        prov = registry.provenance(art.artifact_id)
        assert prov is not None
        assert prov["artifact"]["artifact_id"] == art.artifact_id
        assert prov["training_run"] is not None
        assert prov["training_run"]["run_id"] == good_run.run_id

    def test_provenance_no_run_returns_none_for_run(self, registry: ArtifactRegistry) -> None:
        art = self._make_artifact(content="standalone")
        registry.register_artifact(art)
        prov = registry.provenance(art.artifact_id)
        assert prov is not None
        assert prov["training_run"] is None

    def test_provenance_unknown_id_returns_none(self, registry: ArtifactRegistry) -> None:
        assert registry.provenance("art-nonexistent") is None

    def test_get_artifact_unknown_returns_none(self, registry: ArtifactRegistry) -> None:
        assert registry.get_artifact("art-not-exist") is None

    def test_verify_file_hash_match(self, registry: ArtifactRegistry, tmp_path: Path) -> None:
        content = b"model weights content"
        fpath = tmp_path / "weights.pt"
        fpath.write_bytes(content)
        art = ArtifactRecord.new(
            "WEIGHTS",
            "weights.pt",
            str(fpath),
            artifact_hash=hashlib.sha256(content).hexdigest(),
        )
        registry.register_artifact(art, verify_file=True)  # must not raise

    def test_verify_file_hash_mismatch_raises(
        self, registry: ArtifactRegistry, tmp_path: Path
    ) -> None:
        content = b"actual content"
        fpath = tmp_path / "weights.pt"
        fpath.write_bytes(content)
        art = ArtifactRecord.new(
            "WEIGHTS",
            "weights.pt",
            str(fpath),
            artifact_hash=_make_hash("different content"),
        )
        with pytest.raises(ArtifactHashMismatchError):
            registry.register_artifact(art, verify_file=True)

    def test_verify_file_missing_file_skipped(self, registry: ArtifactRegistry) -> None:
        """If path does not exist, file verification is skipped silently."""
        art = ArtifactRecord.new("TORCHSCRIPT", "model.pt", "/nonexistent/path.pt", _GOOD_HASH)
        registry.register_artifact(art, verify_file=True)  # must not raise

    def test_hash_bytes_utility(self) -> None:
        data = b"test data"
        h = ArtifactRegistry.hash_bytes(data)
        assert h == hashlib.sha256(data).hexdigest()

    # ── Training run methods ──────────────────────────────────────────────────

    def test_register_and_get_training_run(self, registry: ArtifactRegistry) -> None:
        run = TrainingRunRecord.new(
            experiment_id="exp-r1",
            dataset_version="ds-v1",
            feature_version="feat-v1",
            code_commit="a" * 40,
            config_hash=_GOOD_HASH,
            artifact_hash=_make_hash("r1"),
        )
        registry.register_training_run(run)
        fetched = registry.get_training_run(run.run_id)
        assert fetched is not None
        assert fetched.run_id == run.run_id
        assert fetched.dataset_version == "ds-v1"

    def test_list_training_runs_filter_by_experiment(self, registry: ArtifactRegistry) -> None:
        for i in range(3):
            run = TrainingRunRecord.new(
                experiment_id="exp-filter" if i < 2 else "exp-other",
                dataset_version="ds",
                feature_version="feat",
                code_commit="a" * 40,
                config_hash=_GOOD_HASH,
                artifact_hash=_make_hash(f"run-{i}"),
            )
            registry.register_training_run(run)
        result = registry.list_training_runs(experiment_id="exp-filter")
        assert len(result) == 2


# ── DB CRUD ───────────────────────────────────────────────────────────────────


class TestArtifactDB:
    def test_upsert_and_get_artifact(self, db: ResearchLabDatabase) -> None:
        rec = ArtifactRecord.new(
            "CALIBRATION",
            "cal.json",
            "/cal.json",
            _make_hash("cal"),
        )
        db.upsert_artifact(rec.to_dict())
        row = db.get_artifact(rec.artifact_id)
        assert row is not None
        assert row["artifact_type"] == "CALIBRATION"

    def test_find_by_type_hash(self, db: ResearchLabDatabase) -> None:
        rec = ArtifactRecord.new("CONFIG", "cfg.json", "/cfg.json", _make_hash("cfg"))
        db.upsert_artifact(rec.to_dict())
        found = db.find_artifact_by_type_hash("CONFIG", _make_hash("cfg"))
        assert found is not None
        assert found["artifact_id"] == rec.artifact_id

    def test_find_by_type_hash_wrong_type_returns_none(self, db: ResearchLabDatabase) -> None:
        rec = ArtifactRecord.new("CONFIG", "cfg.json", "/cfg.json", _make_hash("cfg"))
        db.upsert_artifact(rec.to_dict())
        assert db.find_artifact_by_type_hash("WEIGHTS", _make_hash("cfg")) is None

    def test_list_artifacts_all(self, db: ResearchLabDatabase) -> None:
        for i, atype in enumerate(ARTIFACT_TYPES):
            rec = ArtifactRecord.new(atype, f"f{i}", f"/f{i}", _make_hash(f"content-{i}"))
            db.upsert_artifact(rec.to_dict())
        rows = db.list_artifacts()
        assert len(rows) == len(ARTIFACT_TYPES)

    def test_list_artifacts_limit(self, db: ResearchLabDatabase) -> None:
        for i in range(5):
            rec = ArtifactRecord.new("WEIGHTS", f"w{i}.pt", f"/w{i}.pt", _make_hash(f"w{i}"))
            db.upsert_artifact(rec.to_dict())
        rows = db.list_artifacts(limit=3)
        assert len(rows) == 3

    def test_stats_includes_artifacts_and_runs(self, db: ResearchLabDatabase) -> None:
        run = TrainingRunRecord.new(
            experiment_id="e1",
            dataset_version="ds",
            feature_version="feat",
            code_commit="a" * 40,
            config_hash=_GOOD_HASH,
            artifact_hash=_make_hash("r"),
        )
        db.upsert_training_run(run.to_dict())
        art = ArtifactRecord.new("WEIGHTS", "w.pt", "/w.pt", _make_hash("weights"))
        db.upsert_artifact(art.to_dict())
        stats = db.stats()
        assert stats.get("training_runs", 0) == 1
        assert stats.get("artifacts", 0) == 1


# ── MLOpsExporter hook ────────────────────────────────────────────────────────


class TestMLOpsExporterArtifactHook:
    def test_save_without_registry_no_error(self, tmp_path: Path, db: ResearchLabDatabase) -> None:
        """save() works with no artifact_registry (legacy callers)."""

        exporter = self._make_exporter(db, artifact_registry=None)
        data = {"export_type": "model_registry", "models": []}
        exporter.save(data, str(tmp_path / "export.json"))

    def test_save_with_registry_registers_manifest(
        self, tmp_path: Path, db: ResearchLabDatabase
    ) -> None:
        reg = ArtifactRegistry(db)
        exporter = self._make_exporter(db, artifact_registry=reg)
        data = {"export_type": "model_registry", "models": []}
        exporter.save(data, str(tmp_path / "export.json"))
        arts = reg.list_artifacts(artifact_type="DEPLOYMENT_MANIFEST")
        assert len(arts) == 1
        assert arts[0]["name"] == "export.json"
        assert arts[0]["notes"] == "MLOps export: model_registry"

    def test_save_duplicate_export_does_not_raise(
        self, tmp_path: Path, db: ResearchLabDatabase
    ) -> None:
        """Saving the same export twice logs a warning but does not crash."""
        reg = ArtifactRegistry(db)
        exporter = self._make_exporter(db, artifact_registry=reg)
        data = {"export_type": "audit_report"}
        path = str(tmp_path / "audit.json")
        exporter.save(data, path)
        exporter.save(data, path)  # duplicate hash → warning, not crash

    def _make_exporter(self, db, artifact_registry=None):
        from unittest.mock import MagicMock

        from trainer.mlops.alerting import AlertingSystem
        from trainer.mlops.audit import AuditSystem
        from trainer.mlops.export import MLOpsExporter

        mock_db = MagicMock()
        mock_registry = MagicMock()
        mock_registry.all_models.return_value = []
        mock_registry.summary.return_value = {}
        mock_registry.champion.return_value = None
        mock_audit = MagicMock(spec=AuditSystem)
        mock_audit.get_report.return_value = {}
        mock_alerting = MagicMock(spec=AlertingSystem)
        mock_alerting.stats.return_value = {}
        return MLOpsExporter(
            db=mock_db,
            registry=mock_registry,
            audit=mock_audit,
            alerting=mock_alerting,
            artifact_registry=artifact_registry,
        )


# ── Server endpoints ──────────────────────────────────────────────────────────


class TestArtifactServerEndpoints:
    _HDR = {
        "Authorization": "Bearer test-artifact-token",
        "X-Signalium-Token": make_signalium_jwt(),
    }

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-artifact-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_artifacts.db")
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    def _good_payload(self, content: str = "weights", atype: str = "WEIGHTS") -> dict:
        return {
            "artifact_type": atype,
            "name": f"{atype.lower()}.bin",
            "path": f"/exports/{atype.lower()}.bin",
            "artifact_hash": _make_hash(content),
            "model_version": "v3.0",
            "experiment_id": "exp-test-001",
        }

    def test_register_artifact_success(self, client) -> None:
        resp = client.post("/artifacts/register", json=self._good_payload(), headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["artifact"]["artifact_id"].startswith("art-")
        assert body["artifact"]["artifact_type"] == "WEIGHTS"

    def test_register_artifact_invalid_type(self, client) -> None:
        payload = self._good_payload()
        payload["artifact_type"] = "INVALID_TYPE"
        resp = client.post("/artifacts/register", json=payload, headers=self._HDR)
        assert resp.status_code == 400
        assert "Unknown artifact_type" in resp.json()["detail"]

    def test_register_artifact_invalid_hash(self, client) -> None:
        payload = self._good_payload()
        payload["artifact_hash"] = "short"
        resp = client.post("/artifacts/register", json=payload, headers=self._HDR)
        assert resp.status_code == 400
        assert "64-character" in resp.json()["detail"]

    def test_register_duplicate_returns_400(self, client) -> None:
        payload = self._good_payload(content="unique1")
        client.post("/artifacts/register", json=payload, headers=self._HDR)
        resp = client.post("/artifacts/register", json=payload, headers=self._HDR)
        assert resp.status_code == 400
        assert "already registered" in resp.json()["detail"]

    def test_list_artifacts_empty(self, client) -> None:
        resp = client.get("/artifacts", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_artifacts_with_filter(self, client) -> None:
        client.post(
            "/artifacts/register", json=self._good_payload("w1", "WEIGHTS"), headers=self._HDR
        )
        client.post("/artifacts/register", json=self._good_payload("o1", "ONNX"), headers=self._HDR)
        resp = client.get("/artifacts?artifact_type=WEIGHTS", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        assert resp.json()["artifacts"][0]["artifact_type"] == "WEIGHTS"

    def test_get_artifact_by_id(self, client) -> None:
        reg_resp = client.post(
            "/artifacts/register", json=self._good_payload("getme"), headers=self._HDR
        )
        art_id = reg_resp.json()["artifact"]["artifact_id"]
        resp = client.get(f"/artifacts/{art_id}", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["artifact"]["artifact_id"] == art_id

    def test_get_artifact_not_found(self, client) -> None:
        resp = client.get("/artifacts/art-nonexistent", headers=self._HDR)
        assert resp.status_code == 404

    def test_artifact_types_endpoint(self, client) -> None:
        resp = client.get("/artifacts/types", headers=self._HDR)
        assert resp.status_code == 200
        assert set(resp.json()["artifact_types"]) == set(ARTIFACT_TYPES)

    def test_register_artifact_requires_auth(self, client) -> None:
        resp = client.post("/artifacts/register", json=self._good_payload())
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_list_artifacts_requires_auth(self, client) -> None:
        resp = client.get("/artifacts")
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_get_artifact_requires_auth(self, client) -> None:
        resp = client.get("/artifacts/art-anything")
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_register_training_run_success(self, client) -> None:
        payload = {
            "experiment_id": "exp-srv-001",
            "dataset_version": "ds-v3",
            "feature_version": "feat-v2",
            "code_commit": "a" * 40,
            "config_hash": _GOOD_HASH,
            "artifact_hash": _make_hash("main-weights"),
            "model_version": "v3.0",
            "mlflow_run_id": "mlf-999",
        }
        resp = client.post("/artifacts/runs", json=payload, headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["training_run"]["run_id"].startswith("run-")

    def test_register_training_run_bad_hash(self, client) -> None:
        payload = {
            "experiment_id": "exp-srv-001",
            "dataset_version": "ds",
            "feature_version": "feat",
            "code_commit": "abc",
            "config_hash": _GOOD_HASH,
            "artifact_hash": "bad",
        }
        resp = client.post("/artifacts/runs", json=payload, headers=self._HDR)
        assert resp.status_code == 400

    def test_list_training_runs(self, client) -> None:
        payload = {
            "experiment_id": "exp-list-test",
            "dataset_version": "ds-v1",
            "feature_version": "feat-v1",
            "code_commit": "b" * 40,
            "config_hash": _GOOD_HASH,
            "artifact_hash": _make_hash("r1"),
        }
        client.post("/artifacts/runs", json=payload, headers=self._HDR)
        resp = client.get("/artifacts/runs?experiment_id=exp-list-test", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_get_training_run_with_artifacts(self, client) -> None:
        run_resp = client.post(
            "/artifacts/runs",
            json={
                "experiment_id": "exp-full",
                "dataset_version": "ds-v1",
                "feature_version": "feat-v1",
                "code_commit": "c" * 40,
                "config_hash": _GOOD_HASH,
                "artifact_hash": _make_hash("main-w"),
            },
            headers=self._HDR,
        )
        run_id = run_resp.json()["training_run"]["run_id"]

        art_payload = self._good_payload("weights-for-run")
        art_payload["training_run_id"] = run_id
        client.post("/artifacts/register", json=art_payload, headers=self._HDR)

        resp = client.get(f"/artifacts/runs/{run_id}", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["training_run"]["run_id"] == run_id
        assert len(body["artifacts"]) == 1

    def test_get_training_run_not_found(self, client) -> None:
        resp = client.get("/artifacts/runs/run-nonexistent", headers=self._HDR)
        assert resp.status_code == 404

    def test_register_artifact_verify_file_hash_mismatch_returns_400(
        self, client, tmp_path: Path
    ) -> None:
        """verify_file=true with a mismatched hash must return 400."""
        content = b"real model content"
        fpath = tmp_path / "mismatch.pt"
        fpath.write_bytes(content)
        payload = {
            "artifact_type": "WEIGHTS",
            "name": "mismatch.pt",
            "path": str(fpath),
            "artifact_hash": _make_hash("wrong content"),  # deliberately wrong
            "verify_file": True,
        }
        resp = client.post("/artifacts/register", json=payload, headers=self._HDR)
        assert resp.status_code == 400
        assert "mismatch" in resp.json()["detail"].lower()

    def test_list_training_runs_requires_auth(self, client) -> None:
        resp = client.get("/artifacts/runs")
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_get_training_run_requires_auth(self, client) -> None:
        resp = client.get("/artifacts/runs/run-any")
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_register_training_run_requires_auth(self, client) -> None:
        resp = client.post("/artifacts/runs", json={})
        assert resp.status_code == 401  # no token → 401 Unauthorized


class TestArtifactDBUniqueIndexEnforcement:
    """Defense-in-depth: DB-level unique index on (artifact_type, artifact_hash)."""

    def test_db_unique_index_rejects_duplicate(self, db: ResearchLabDatabase) -> None:
        import sqlite3

        rec1 = ArtifactRecord.new("WEIGHTS", "w1.pt", "/w1.pt", _make_hash("shared"))
        rec2 = ArtifactRecord.new("WEIGHTS", "w2.pt", "/w2.pt", _make_hash("shared"))
        db.upsert_artifact(rec1.to_dict())
        with pytest.raises((sqlite3.IntegrityError, sqlite3.DatabaseError)):
            db.upsert_artifact(rec2.to_dict())

    def test_db_unique_index_allows_different_type(self, db: ResearchLabDatabase) -> None:
        """Same hash for different artifact types is a different row."""
        h = _make_hash("shared-bytes")
        rec1 = ArtifactRecord.new("WEIGHTS", "w.pt", "/w.pt", h)
        rec2 = ArtifactRecord.new("ONNX", "m.onnx", "/m.onnx", h)
        db.upsert_artifact(rec1.to_dict())
        db.upsert_artifact(rec2.to_dict())  # must not raise
        assert db.find_artifact_by_type_hash("WEIGHTS", h) is not None
        assert db.find_artifact_by_type_hash("ONNX", h) is not None


class TestArtifactRegistryStoreIntegration:
    """S21A — write_and_register / resolve_path against the immutable store."""

    def test_write_and_register_writes_and_returns_record(
        self, registry_with_store: ArtifactRegistry
    ) -> None:
        record = registry_with_store.write_and_register(
            artifact_type=ArtifactType.WEIGHTS.value,
            version="v1",
            filename="weights.pt",
            data=b"weights-bytes",
        )
        assert record.artifact_type == "WEIGHTS"
        assert record.model_version == "v1"
        assert Path(record.path).read_bytes() == b"weights-bytes"
        assert record.artifact_hash == hashlib.sha256(b"weights-bytes").hexdigest()

    def test_write_and_register_path_resolves_under_store_root(
        self, registry_with_store: ArtifactRegistry
    ) -> None:
        record = registry_with_store.write_and_register(
            artifact_type=ArtifactType.CONFIG.value,
            version="v1",
            filename="config.json",
            data=b"{}",
        )
        assert Path(record.path) == registry_with_store.store.path_for("v1", "config.json")

    def test_write_and_register_rejects_overwrite(
        self, registry_with_store: ArtifactRegistry
    ) -> None:
        registry_with_store.write_and_register(
            artifact_type=ArtifactType.WEIGHTS.value,
            version="v1",
            filename="weights.pt",
            data=b"first",
        )
        with pytest.raises(ArtifactImmutabilityError):
            registry_with_store.write_and_register(
                artifact_type=ArtifactType.WEIGHTS.value,
                version="v1",
                filename="weights.pt",
                data=b"second",
            )

    def test_resolve_path_matches_store(self, registry_with_store: ArtifactRegistry) -> None:
        expected = registry_with_store.store.path_for("v3", "model.onnx")
        assert registry_with_store.resolve_path("v3", "model.onnx") == expected

    def test_default_store_used_when_none_supplied(self, registry: ArtifactRegistry) -> None:
        assert registry.store is not None
        assert registry.store.root is not None
