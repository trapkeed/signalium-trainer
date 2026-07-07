"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.mocks.dataset import MockDataset, make_parquet_dataset


@pytest.fixture(scope="session")
def sample_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped fixture with a synthetic Parquet dataset."""
    tmp = tmp_path_factory.mktemp("datasets")
    return make_parquet_dataset(tmp, n_samples=100, feature_dim=6, num_classes=3)


@pytest.fixture()
def mock_dataset() -> MockDataset:
    """Small in-memory dataset for fast unit tests."""
    return MockDataset(n_samples=64, feature_dim=8, num_classes=3, seed=0)
