"""Integration tests for ParquetDataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.mocks.dataset import make_parquet_dataset
from trainer.datasets.parquet import ParquetDataset


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    return make_parquet_dataset(tmp_path, n_samples=50, feature_dim=4, num_classes=2)


def test_loads_train_split(dataset_dir: Path) -> None:
    ds = ParquetDataset(dataset_dir, split="train")
    assert len(ds) == 50
    assert ds.feature_dim == 4
    assert ds.num_classes == 2


def test_getitem_returns_tensors(dataset_dir: Path) -> None:
    ds = ParquetDataset(dataset_dir, split="train")
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (4,)


def test_all_splits_accessible(dataset_dir: Path) -> None:
    for split in ("train", "val", "test"):
        ds = ParquetDataset(dataset_dir, split=split)
        assert len(ds) > 0


def test_missing_parquet_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ParquetDataset(tmp_path, split="train")


def test_feature_columns_subset(dataset_dir: Path) -> None:
    ds = ParquetDataset(dataset_dir, split="train", feature_columns=["feat_0", "feat_1"])
    assert ds.feature_dim == 2
    x, _ = ds[0]
    assert x.shape == (2,)
