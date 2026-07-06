"""Unit tests for DatasetValidator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trainer.datasets.validation import DatasetValidator


@pytest.fixture()
def valid_dataset(tmp_path: Path) -> Path:
    """Create a minimal valid dataset directory."""
    for split in ("train", "val", "test"):
        df = pd.DataFrame({"feat": np.random.rand(10), "label": np.random.randint(0, 2, 10)})
        df.to_parquet(tmp_path / f"{split}.parquet", index=False)
    return tmp_path


def test_passes_valid_dataset(valid_dataset: Path) -> None:
    validator = DatasetValidator(valid_dataset, verify_checksums=False, fail_on_error=True)
    result = validator.validate()
    assert result.passed
    assert not result.errors


def test_fails_missing_directory() -> None:
    validator = DatasetValidator(
        Path("/nonexistent/path"), verify_checksums=False, fail_on_error=False
    )
    result = validator.validate()
    assert not result.passed
    assert any("does not exist" in e for e in result.errors)


def test_fails_missing_split(tmp_path: Path) -> None:
    # Only create train, skip val and test
    df = pd.DataFrame({"feat": [1.0], "label": [0]})
    df.to_parquet(tmp_path / "train.parquet", index=False)

    validator = DatasetValidator(tmp_path, verify_checksums=False, fail_on_error=False)
    result = validator.validate()
    assert not result.passed
    assert any("val.parquet" in e for e in result.errors)


def test_warns_on_missing_manifest(valid_dataset: Path) -> None:
    validator = DatasetValidator(valid_dataset, verify_checksums=False, fail_on_error=False)
    result = validator.validate()
    assert any("Manifest" in w for w in result.warnings)


def test_raises_when_fail_on_error(tmp_path: Path) -> None:
    validator = DatasetValidator(
        tmp_path / "nonexistent", verify_checksums=False, fail_on_error=True
    )
    with pytest.raises(ValueError, match="validation failed"):
        validator.validate()
