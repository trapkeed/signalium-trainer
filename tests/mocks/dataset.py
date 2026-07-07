"""Mock dataset helpers for testing without real Parquet files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trainer.datasets.base import BaseDataset


class MockDataset(BaseDataset):
    """In-memory dataset with random features and labels for unit testing.

    Args:
        n_samples: Number of synthetic samples.
        feature_dim: Dimensionality of each feature vector.
        num_classes: Number of label classes.
        split: Dataset split identifier.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 100,
        feature_dim: int = 16,
        num_classes: int = 3,
        split: str = "train",
        seed: int = 0,
    ) -> None:
        super().__init__(root=Path("/tmp/mock"), split=split)
        rng = np.random.default_rng(seed)
        self._x = torch.tensor(rng.standard_normal((n_samples, feature_dim)), dtype=torch.float32)
        self._y = torch.tensor(rng.integers(0, num_classes, n_samples), dtype=torch.long)
        self._feature_dim = feature_dim
        self._num_classes = num_classes

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[index], self._y[index]


def make_parquet_dataset(
    tmp_path: Path,
    n_samples: int = 200,
    feature_dim: int = 8,
    num_classes: int = 2,
    splits: tuple[str, ...] = ("train", "val", "test"),
    seed: int = 42,
) -> Path:
    """Write synthetic Parquet files for each split into tmp_path.

    Args:
        tmp_path: Directory to write Parquet files into.
        n_samples: Number of rows per split file.
        feature_dim: Number of feature columns.
        num_classes: Number of distinct label values.
        splits: Which split files to create.
        seed: Random seed.

    Returns:
        tmp_path (for chaining).
    """
    rng = np.random.default_rng(seed)
    feature_cols = [f"feat_{i}" for i in range(feature_dim)]
    for split in splits:
        features = rng.standard_normal((n_samples, feature_dim))
        labels = rng.integers(0, num_classes, n_samples)
        df = pd.DataFrame(features, columns=feature_cols)
        df["label"] = labels
        df.to_parquet(tmp_path / f"{split}.parquet", index=False)
    return tmp_path
