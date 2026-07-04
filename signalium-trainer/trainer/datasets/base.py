"""Abstract base class for all Signalium datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base for all Signalium PyTorch datasets.

    Subclasses implement :meth:`__len__` and :meth:`__getitem__`.
    The contract enforces that every dataset knows its root directory
    and exposes feature/label metadata used by trainers.

    Args:
        root: Root directory where the dataset files live.
        split: Dataset split — "train", "val", or "test".
        transform: Optional callable applied to each sample.
    """

    VALID_SPLITS = frozenset({"train", "val", "test"})

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: Any = None,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")
        self._root = Path(root)
        self._split = split
        self._transform = transform

    @property
    def root(self) -> Path:
        """Root directory of the dataset."""
        return self._root

    @property
    def split(self) -> str:
        """Active split — train / val / test."""
        return self._split

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of a single input feature vector."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes (or 1 for regression)."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in this split."""

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, label) tensors for the given index."""

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root={self._root}, split={self._split}, size={len(self)})"
        )
