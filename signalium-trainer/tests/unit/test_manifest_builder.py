"""Unit tests for ManifestBuilder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer.exporters.manifest import ManifestBuilder


@pytest.fixture()
def artifact_file(tmp_path: Path) -> Path:
    f = tmp_path / "model.onnx"
    f.write_bytes(b"fake model bytes")
    return f


def test_build_contains_required_keys(artifact_file: Path) -> None:
    builder = ManifestBuilder("m1", "1.0.0", "onnx", "MyModel")
    builder.add_file(artifact_file)
    manifest = builder.build()
    assert manifest["model_id"] == "m1"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["files"]) == 1
    assert "sha256" in manifest["files"][0]


def test_write_creates_json(tmp_path: Path, artifact_file: Path) -> None:
    builder = ManifestBuilder("m2", "2.0.0", "torchscript", "AnotherModel")
    builder.add_file(artifact_file)
    path = builder.write(tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["model_id"] == "m2"


def test_set_metrics_included(artifact_file: Path) -> None:
    builder = ManifestBuilder("m3", "1.0.0", "onnx", "M")
    builder.add_file(artifact_file)
    builder.set_metrics({"accuracy": 0.95, "f1": 0.93})
    manifest = builder.build()
    assert manifest["metrics"]["accuracy"] == pytest.approx(0.95)


def test_sha256_computed(artifact_file: Path) -> None:
    builder = ManifestBuilder("m4", "1.0.0", "onnx", "M")
    builder.add_file(artifact_file)
    manifest = builder.build()
    assert len(manifest["files"][0]["sha256"]) == 64
