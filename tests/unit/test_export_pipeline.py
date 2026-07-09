"""Unit tests for the S21A unified QDE export pipeline.

These tests do not require ``torch`` — they exercise the manifest/checksum/
registry-glue logic via a pre-supplied ``weights_path`` and skip TorchScript/
ONNX export (which requires a real model + torch). A separate torch-gated
integration test covers the full weights+TorchScript+ONNX path end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer.exporters.pipeline import (
    REQUIRED_MANIFEST_FIELDS,
    QDEExportPipeline,
    QDEExportPipelineError,
)
from trainer.research_lab.artifact_registry import ArtifactRegistry
from trainer.research_lab.rl_database import ResearchLabDatabase


@pytest.fixture()
def registry(tmp_path: Path) -> ArtifactRegistry:
    db = ResearchLabDatabase(str(tmp_path / "research_lab.db"))
    return ArtifactRegistry(db)


@pytest.fixture()
def pre_existing_weights(tmp_path: Path) -> Path:
    weights = tmp_path / "pretrained_weights.pt"
    weights.write_bytes(b"fake-weights-bytes-for-testing")
    return weights


class TestExportVersionRequiresWeightsSource:
    def test_raises_without_model_or_weights_path(self, tmp_path: Path) -> None:
        pipeline = QDEExportPipeline()
        with pytest.raises(QDEExportPipelineError):
            pipeline.export_version(model_version="v1", output_dir=tmp_path)

    def test_raises_if_weights_path_missing(self, tmp_path: Path) -> None:
        pipeline = QDEExportPipeline()
        with pytest.raises(QDEExportPipelineError):
            pipeline.export_version(
                model_version="v1",
                output_dir=tmp_path,
                weights_path=tmp_path / "does_not_exist.pt",
            )


class TestExportVersionManifest:
    def test_produces_all_required_files(
        self, tmp_path: Path, pre_existing_weights: Path
    ) -> None:
        pipeline = QDEExportPipeline()
        result = pipeline.export_version(
            model_version="v3.1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
            dataset_version="ds-v3.1",
            feature_version="feat-v2.0",
            training_commit="abc1234",
            calibration_version="cal-v1",
            certification_status="CERTIFIED",
            promotion_status="SHADOW",
            supported_symbols=["EURUSD", "BTCUSD"],
            supported_tasks=["classification"],
        )

        version_dir = Path(result["output_dir"])
        assert version_dir == tmp_path / "v3.1"
        for filename in ("config.json", "calibration.json", "metadata.json", "qde_manifest.json"):
            assert (version_dir / filename).exists()

        manifest = result["manifest"]
        for field_name in REQUIRED_MANIFEST_FIELDS:
            assert field_name in manifest, f"manifest missing required field: {field_name}"

        assert manifest["model_version"] == "v3.1"
        assert manifest["dataset_version"] == "ds-v3.1"
        assert manifest["feature_version"] == "feat-v2.0"
        assert manifest["training_commit"] == "abc1234"
        assert manifest["calibration_version"] == "cal-v1"
        assert manifest["certification_status"] == "CERTIFIED"
        assert manifest["promotion_status"] == "SHADOW"
        assert manifest["supported_symbols"] == ["EURUSD", "BTCUSD"]
        assert manifest["supported_tasks"] == ["classification"]
        assert manifest["artifact_hash"]  # sha256 of weights, non-empty
        assert len(manifest["artifact_hash"]) == 64
        assert len(result["manifest_sha256"]) == 64

    def test_manifest_file_matches_returned_dict(
        self, tmp_path: Path, pre_existing_weights: Path
    ) -> None:
        pipeline = QDEExportPipeline()
        result = pipeline.export_version(
            model_version="v1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
        )
        on_disk = json.loads(Path(result["manifest_path"]).read_text())
        assert on_disk == result["manifest"]

    def test_every_exported_file_has_checksum_recorded(
        self, tmp_path: Path, pre_existing_weights: Path
    ) -> None:
        pipeline = QDEExportPipeline()
        result = pipeline.export_version(
            model_version="v1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
        )
        files_by_name = {f["name"]: f for f in result["files"]}
        # weights, config, calibration, metadata, manifest (torchscript/onnx skipped: no model)
        assert set(files_by_name) == {
            "pretrained_weights.pt",
            "config.json",
            "calibration.json",
            "metadata.json",
            "qde_manifest.json",
        }
        for entry in files_by_name.values():
            assert len(entry["sha256"]) == 64
            assert entry["size_bytes"] > 0

    def test_checksums_are_actual_sha256_of_file_contents(
        self, tmp_path: Path, pre_existing_weights: Path
    ) -> None:
        import hashlib

        pipeline = QDEExportPipeline()
        result = pipeline.export_version(
            model_version="v1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
        )
        for entry in result["files"]:
            if entry["tag"] == "manifest":
                # The manifest's own checksum is necessarily computed just
                # before its self-referential entry is appended (a file
                # cannot hash its own final byte-for-byte content), so it is
                # verified separately below rather than against the file's
                # current bytes.
                continue
            actual = hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
            assert actual == entry["sha256"], f"checksum mismatch for {entry['name']}"

    def test_manifest_sha256_matches_actual_on_disk_bytes(
        self, tmp_path: Path, pre_existing_weights: Path
    ) -> None:
        """qde_manifest.json never lists/hashes itself (self-reference is
        impossible), so `result["manifest_sha256"]` must be the exact SHA-256
        of the file's real, final on-disk bytes — not a pre-image hash."""
        import hashlib

        pipeline = QDEExportPipeline()
        result = pipeline.export_version(
            model_version="v1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
        )
        actual = hashlib.sha256(Path(result["manifest_path"]).read_bytes()).hexdigest()
        assert actual == result["manifest_sha256"]
        on_disk = json.loads(Path(result["manifest_path"]).read_text())
        assert on_disk["files"][-1]["tag"] != "manifest"


class TestExportVersionCalibrationAndConfig:
    def test_calibration_report_to_dict_is_used(
        self, tmp_path: Path, pre_existing_weights: Path
    ) -> None:
        class FakeCalibrationReport:
            def to_dict(self) -> dict:
                return {"ece": 0.01, "mce": 0.05}

        pipeline = QDEExportPipeline()
        result = pipeline.export_version(
            model_version="v1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
            calibration_report=FakeCalibrationReport(),
        )
        cal_path = Path(result["output_dir"]) / "calibration.json"
        data = json.loads(cal_path.read_text())
        assert data == {"ece": 0.01, "mce": 0.05}

    def test_dataclass_config_is_serialised(
        self, tmp_path: Path, pre_existing_weights: Path
    ) -> None:
        from trainer.config.schema import TrainerConfig

        pipeline = QDEExportPipeline()
        result = pipeline.export_version(
            model_version="v1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
            config=TrainerConfig(),
        )
        config_path = Path(result["output_dir"]) / "config.json"
        data = json.loads(config_path.read_text())
        assert "export" in data
        assert data["export"]["export_torchscript"] is True


class TestExportVersionRegistry:
    def test_registers_every_artifact(
        self, tmp_path: Path, pre_existing_weights: Path, registry: ArtifactRegistry
    ) -> None:
        pipeline = QDEExportPipeline(artifact_registry=registry)
        result = pipeline.export_version(
            model_version="v1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
            training_run_id="run-abc123",
            experiment_id="exp-1",
        )
        # metadata.json is deliberately not registered — it has no
        # corresponding ArtifactType among the six canonical types.
        assert set(result["artifacts"]) == {"weights", "config", "calibration", "manifest"}

        weights_record = registry.get_artifact(result["artifacts"]["weights"])
        assert weights_record is not None
        assert weights_record.artifact_type == "WEIGHTS"
        assert weights_record.model_version == "v1"
        assert weights_record.training_run_id == "run-abc123"

        manifest_record = registry.get_artifact(result["artifacts"]["manifest"])
        assert manifest_record is not None
        assert manifest_record.artifact_type == "DEPLOYMENT_MANIFEST"

    def test_reexport_with_identical_content_does_not_raise(
        self, tmp_path: Path, pre_existing_weights: Path, registry: ArtifactRegistry
    ) -> None:
        pipeline = QDEExportPipeline(artifact_registry=registry)
        kwargs = dict(
            model_version="v1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
        )
        first = pipeline.export_version(**kwargs)
        # Second export of identical weights bytes hits DuplicateArtifactError
        # internally for the (type, hash) pair — must not raise, just skip
        # registering (returns no artifact_id for that entry).
        second = pipeline.export_version(**kwargs)
        assert first["manifest"]["artifact_hash"] == second["manifest"]["artifact_hash"]

    def test_without_registry_returns_empty_artifacts_map(
        self, tmp_path: Path, pre_existing_weights: Path
    ) -> None:
        pipeline = QDEExportPipeline()
        result = pipeline.export_version(
            model_version="v1",
            output_dir=tmp_path,
            weights_path=pre_existing_weights,
            export_formats=(),
        )
        assert result["artifacts"] == {}
