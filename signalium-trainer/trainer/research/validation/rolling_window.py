"""RollingWindowValidator — fixed-size sliding window cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class RollingFold:
    """A single fold from rolling-window validation.

    Attributes:
        fold_index: 0-indexed fold number.
        window_start: Start index of the entire window.
        train_end: Exclusive end of the training portion.
        val_end: Exclusive end of the validation portion.
        x_train: Training feature tensor.
        y_train: Training label tensor.
        x_val: Validation feature tensor.
        y_val: Validation label tensor.
    """

    fold_index: int
    window_start: int
    train_end: int
    val_end: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor

    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return self.train_end - self.window_start

    @property
    def val_size(self) -> int:
        """Number of validation samples."""
        return self.val_end - self.train_end


class RollingWindowValidator:
    """Generates fixed-size sliding window folds for time-series validation.

    Unlike WalkForward (where the training window grows), RollingWindow
    keeps the total window size fixed and slides it forward by ``step_size``
    on each fold. This simulates a production model retrained on the most
    recent ``train_size`` bars.

    Timeline::

        Fold 0:  [<--train--|val--|          ]
        Fold 1:  [          |<--train--|val--|]

    Args:
        train_size: Number of training samples per fold.
        val_size: Number of validation samples per fold.
        step_size: Number of samples to advance per fold.
        max_folds: Maximum number of folds to generate (0 = unlimited).

    Raises:
        ValueError: If parameters are inconsistent.
    """

    def __init__(
        self,
        train_size: int,
        val_size: int,
        step_size: int,
        max_folds: int = 0,
    ) -> None:
        if train_size <= 0:
            raise ValueError("train_size must be > 0.")
        if val_size <= 0:
            raise ValueError("val_size must be > 0.")
        if step_size <= 0:
            raise ValueError("step_size must be > 0.")
        self._train = train_size
        self._val = val_size
        self._step = step_size
        self._max_folds = max_folds
        self._window = train_size + val_size

    def split(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Iterator[RollingFold]:
        """Generate rolling-window folds from a dataset.

        Args:
            x: Feature tensor of shape (n_samples, ...).
            y: Label tensor of shape (n_samples,).

        Yields:
            RollingFold instances.

        Raises:
            ValueError: If the dataset is too small for even one fold.
        """
        n = x.shape[0]
        if n < self._window:
            raise ValueError(
                f"Dataset has {n} samples but the rolling window needs "
                f"{self._window} ({self._train} train + {self._val} val)."
            )

        fold_idx = 0
        start = 0

        while start + self._window <= n:
            if self._max_folds > 0 and fold_idx >= self._max_folds:
                break

            train_end = start + self._train
            val_end = train_end + self._val

            yield RollingFold(
                fold_index=fold_idx,
                window_start=start,
                train_end=train_end,
                val_end=val_end,
                x_train=x[start:train_end],
                y_train=y[start:train_end],
                x_val=x[train_end:val_end],
                y_val=y[train_end:val_end],
            )

            fold_idx += 1
            start += self._step

    def num_folds(self, n_samples: int) -> int:
        """Compute the number of folds for a dataset of size n_samples.

        Args:
            n_samples: Total number of samples.

        Returns:
            Number of folds that would be generated.
        """
        if n_samples < self._window:
            return 0
        count = 0
        start = 0
        while start + self._window <= n_samples:
            if self._max_folds > 0 and count >= self._max_folds:
                break
            count += 1
            start += self._step
        return count
