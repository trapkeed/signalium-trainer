"""Unit tests for DataLoaderFactory."""

from __future__ import annotations

import pytest

from trainer.config.schema import DataLoaderConfig
from trainer.dataloaders.factory import DataLoaderFactory
from tests.mocks.dataset import MockDataset


@pytest.fixture()
def config() -> DataLoaderConfig:
    return DataLoaderConfig(
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        train_split=0.6,
        val_split=0.2,
        test_split=0.2,
    )


def test_from_dataset_creates_three_loaders(config: DataLoaderConfig) -> None:
    dataset = MockDataset(n_samples=100)
    factory = DataLoaderFactory(config, seed=42)
    loaders = factory.from_dataset(dataset)
    assert loaders.train is not None
    assert loaders.val is not None
    assert loaders.test is not None


def test_splits_sum_to_total(config: DataLoaderConfig) -> None:
    dataset = MockDataset(n_samples=100)
    factory = DataLoaderFactory(config, seed=42)
    loaders = factory.from_dataset(dataset)
    total = len(loaders.train.dataset) + len(loaders.val.dataset) + len(loaders.test.dataset)
    assert total == 100


def test_train_loader_returns_batches(config: DataLoaderConfig) -> None:
    dataset = MockDataset(n_samples=50)
    factory = DataLoaderFactory(config, seed=0)
    loaders = factory.from_dataset(dataset)
    x, y = next(iter(loaders.train))
    assert x.shape[0] <= config.batch_size
    assert y.shape[0] <= config.batch_size
