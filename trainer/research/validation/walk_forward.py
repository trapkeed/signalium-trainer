"""WalkForwardValidator — time-series aware walk-forward cross-validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass
class ValidationFold:
    """A single train/validation split with metadata.

    Attributes:
        fold_index: 0-indexed fold number.
        train_start: Start index of the training window.
        train_end: Exclusive end index of the training window.
        val_start: Start index of the validation window.
        val_end: Exclusive end index of the validation window.
        x_train: Training feature tensor.
        y_train: Training label tensor.
        x_val: Validation feature tensor.
        y_val: Validation label tensor.
    """

    fold_index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor

    @property
    def train_size(self) -> int:
        """Number of training samples in this fold."""
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        """Number of validation samples in this fold."""
        return self.val_end - self.val_start


class WalkForwardValidator:
    """Generates time-series aware walk-forward cross-validation folds.

    Walk-forward validation respects temporal ordering: the training window
    always precedes the validation window. Each fold expands the training
    set by ``step_size`` samples while the validation window slides forward.

    Timeline::

        Fold 0:  |<--- train (min_train_size) --->|<- val ->|
        Fold 1:  |<--- train (+ step_size) ------->|<- val ->|
        Fold 2:  |<--- train (+ 2*step_size) ------>|<- val ->|

    Args:
        min_train_size: Minimum number of training samples for the first fold.
        val_size: Number of samples in each validation window.
        step_size: Number of samples to advance the training window per fold.
        max_folds: Maximum number of folds to generate (0 = unlimited).

    Raises:
        ValueError: If parameters are inconsistent.
    """

    def __init__(
        self,
        min_train_size: int,
        val_size: int,
        step_size: int,
        max_folds: int = 0,
    ) -> None:
        if min_train_size <= 0:
            raise ValueError("min_train_size must be > 0.")
        if val_size <= 0:
            raise ValueError("val_size must be > 0.")
        if step_size <= 0:
            raise ValueError("step_size must be > 0.")
        self._min_train = min_train_size
        self._val_size = val_size
        self._step = step_size
        self._max_folds = max_folds

    def split(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Iterator[ValidationFold]:
        """Generate walk-forward folds from a dataset.

        Args:
            x: Feature tensor of shape (n_samples, ...).
            y: Label tensor of shape (n_samples,).

        Yields:
            ValidationFold instances for each fold.

        Raises:
            ValueError: If the dataset is too small for even one fold.
        """
        n = x.shape[0]
        min_required = self._min_train + self._val_size
        if n < min_required:
            raise ValueError(
                f"Dataset too small: need {min_required} samples "
                f"(min_train={self._min_train} + val={self._val_size}), "
                f"got {n}."
            )

        fold_idx = 0
        train_end = self._min_train

        while train_end + self._val_size <= n:
            if self._max_folds > 0 and fold_idx >= self._max_folds:
                break

            val_start = train_end
            val_end = val_start + self._val_size

            yield ValidationFold(
                fold_index=fold_idx,
                train_start=0,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                x_train=x[:train_end],
                y_train=y[:train_end],
                x_val=x[val_start:val_end],
                y_val=y[val_start:val_end],
            )

            fold_idx += 1
            train_end += self._step

    def num_folds(self, n_samples: int) -> int:
        """Compute the number of folds for a dataset of size n_samples.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            Number of folds that would be generated.
        """
        if n_samples < self._min_train + self._val_size:
            return 0
        count = 0
        train_end = self._min_train
        while train_end + self._val_size <= n_samples:
            if self._max_folds > 0 and count >= self._max_folds:
                break
            count += 1
            train_end += self._step
        return count
