"""PyTorch DataLoader factory with reproducible splitting and prefetching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from trainer.config.schema import DataLoaderConfig


@dataclass
class SplitDataLoaders:
    """Container for train / val / test DataLoaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


class DataLoaderFactory:
    """Builds reproducible, performance-optimised DataLoaders.

    Args:
        config: DataLoaderConfig with batch size, workers, split ratios, etc.
        seed: Random seed for reproducible train/val/test splitting.
    """

    def __init__(self, config: DataLoaderConfig, seed: int = 42) -> None:
        self._config = config
        self._seed = seed

    def from_dataset(self, dataset: Dataset) -> SplitDataLoaders:
        """Split a full dataset and create train / val / test DataLoaders.

        Args:
            dataset: The full un-split dataset.

        Returns:
            SplitDataLoaders with train, val, and test loaders.
        """
        total = len(dataset)  # type: ignore[arg-type]
        n_train = int(total * self._config.train_split)
        n_val = int(total * self._config.val_split)
        n_test = total - n_train - n_val

        generator = torch.Generator().manual_seed(self._seed)
        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )
        return SplitDataLoaders(
            train=self._build(train_ds, shuffle=True),
            val=self._build(val_ds, shuffle=False),
            test=self._build(test_ds, shuffle=False),
        )

    def from_split_datasets(
        self,
        train: Dataset,
        val: Dataset,
        test: Dataset,
    ) -> SplitDataLoaders:
        """Wrap pre-split datasets into DataLoaders.

        Args:
            train: Training dataset.
            val: Validation dataset.
            test: Test dataset.

        Returns:
            SplitDataLoaders.
        """
        return SplitDataLoaders(
            train=self._build(train, shuffle=True),
            val=self._build(val, shuffle=False),
            test=self._build(test, shuffle=False),
        )

    def _build(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Construct a single DataLoader with pinned memory and prefetching."""
        cfg = self._config
        pin_memory = cfg.pin_memory and torch.cuda.is_available()

        kwargs: dict = {
            "batch_size": cfg.batch_size,
            "shuffle": shuffle,
            "num_workers": cfg.num_workers,
            "pin_memory": pin_memory,
            "drop_last": cfg.drop_last,
            "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
        }
        if cfg.num_workers > 0:
            kwargs["prefetch_factor"] = cfg.prefetch_factor

        generator = torch.Generator().manual_seed(self._seed)
        kwargs["generator"] = generator

        return DataLoader(dataset, **kwargs)
