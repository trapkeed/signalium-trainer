"""Phase 15 — Additional walk-forward validation window strategies.

Complements the existing WalkForwardValidator (expanding) and
RollingWindowValidator (sliding/fixed) with:

  SlidingWindowValidator   — alias confirming the rolling approach
  ExpandingWindowValidator — clean alias for walk-forward (expanding train window)
  AnchoredWalkForward      — training window always starts from t=0
  NestedValidator          — outer train / inner val+test (purged gap)

All validators yield typed ``*Fold`` dataclass instances and never shuffle data.
Chronological ordering is always preserved.

Design invariants:
  - Never modify the input tensors.
  - Respect gap_size: a purge gap between train/val to prevent leakage.
  - Yield 0 folds (not raise) when data is too short.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch


# ---------------------------------------------------------------------------
# Common fold container (extends the existing ones with gap info)
# ---------------------------------------------------------------------------


@dataclass
class WindowFold:
    """A complete train / validation / test split for one walk-forward step.

    Attributes:
        fold_index: 0-indexed fold number.
        train_start: Dataset start index of the training slice.
        train_end: Exclusive end index of the training slice.
        val_start: Start index of the validation slice.
        val_end: Exclusive end index of the validation slice.
        test_start: Start index of the optional hold-out test slice.
        test_end: Exclusive end index (0 when no test set).
        x_train: Training features.
        y_train: Training labels.
        x_val: Validation features.
        y_val: Validation labels.
        x_test: Test features (empty tensor when no test set).
        y_test: Test labels (empty tensor when no test set).
        strategy: Identifier of the validation strategy that produced this fold.
    """

    fold_index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int = 0
    test_end: int = 0
    x_train: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    y_train: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    x_val: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    y_val: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    x_test: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    y_test: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    strategy: str = "unknown"

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        return self.val_end - self.val_start

    @property
    def test_size(self) -> int:
        return max(0, self.test_end - self.test_start)

    @property
    def has_test(self) -> bool:
        return self.test_end > self.test_start

    def summary(self) -> dict:
        return {
            "fold": self.fold_index,
            "strategy": self.strategy,
            "train": (self.train_start, self.train_end, self.train_size),
            "val": (self.val_start, self.val_end, self.val_size),
            "test": (self.test_start, self.test_end, self.test_size) if self.has_test else None,
        }


# ---------------------------------------------------------------------------
# Sliding / Rolling Window Validator
# ---------------------------------------------------------------------------


class SlidingWindowValidator:
    """Fixed-size sliding window validation (alias for RollingWindowValidator).

    All splits have fixed ``train_size`` samples.  The window slides forward
    by ``step_size`` samples on each fold — simulating re-training on the
    most recent N bars.

    Args:
        train_size: Fixed number of training samples per fold.
        val_size: Fixed number of validation samples per fold.
        test_size: Optional hold-out test window per fold (0 = disabled).
        step_size: Samples to advance per fold.
        gap_size: Purge gap between train end and val start (leakage buffer).
        max_folds: Hard cap on fold count (0 = unlimited).
    """

    def __init__(
        self,
        train_size: int,
        val_size: int,
        test_size: int = 0,
        step_size: int = 1,
        gap_size: int = 0,
        max_folds: int = 0,
    ) -> None:
        if train_size <= 0:
            raise ValueError("train_size must be > 0")
        if val_size <= 0:
            raise ValueError("val_size must be > 0")
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        self._train = train_size
        self._val = val_size
        self._test = test_size
        self._step = step_size
        self._gap = gap_size
        self._max_folds = max_folds
        self._total_window = train_size + gap_size + val_size + test_size

    def split(self, x: torch.Tensor, y: torch.Tensor) -> Iterator[WindowFold]:
        """Generate sliding-window folds in chronological order."""
        n = x.shape[0]
        fold_idx = 0
        start = 0

        while start + self._total_window <= n:
            if self._max_folds > 0 and fold_idx >= self._max_folds:
                break

            train_end = start + self._train
            val_start = train_end + self._gap
            val_end = val_start + self._val
            test_start = val_end
            test_end = test_start + self._test

            yield WindowFold(
                fold_index=fold_idx,
                train_start=start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start if self._test > 0 else 0,
                test_end=test_end if self._test > 0 else 0,
                x_train=x[start:train_end],
                y_train=y[start:train_end],
                x_val=x[val_start:val_end],
                y_val=y[val_start:val_end],
                x_test=x[test_start:test_end] if self._test > 0 else torch.empty(0),
                y_test=y[test_start:test_end] if self._test > 0 else torch.empty(0),
                strategy="sliding_window",
            )
            fold_idx += 1
            start += self._step

    def num_folds(self, n_samples: int) -> int:
        if n_samples < self._total_window:
            return 0
        count = 0
        start = 0
        while start + self._total_window <= n_samples:
            if self._max_folds > 0 and count >= self._max_folds:
                break
            count += 1
            start += self._step
        return count


# ---------------------------------------------------------------------------
# Expanding Window Validator
# ---------------------------------------------------------------------------


class ExpandingWindowValidator:
    """Expanding training window (canonical walk-forward validation).

    Training window grows by ``step_size`` on each fold while the
    validation (and optional test) windows slide forward.

    Args:
        min_train_size: Minimum training samples for the first fold.
        val_size: Fixed validation window size.
        test_size: Optional hold-out test window per fold (0 = disabled).
        step_size: Samples added to training window per fold.
        gap_size: Purge gap between train end and val start.
        max_folds: Hard cap on fold count (0 = unlimited).
    """

    def __init__(
        self,
        min_train_size: int,
        val_size: int,
        test_size: int = 0,
        step_size: int = 1,
        gap_size: int = 0,
        max_folds: int = 0,
    ) -> None:
        if min_train_size <= 0:
            raise ValueError("min_train_size must be > 0")
        if val_size <= 0:
            raise ValueError("val_size must be > 0")
        self._min_train = min_train_size
        self._val = val_size
        self._test = test_size
        self._step = step_size
        self._gap = gap_size
        self._max_folds = max_folds

    def split(self, x: torch.Tensor, y: torch.Tensor) -> Iterator[WindowFold]:
        """Generate expanding-window folds in chronological order."""
        n = x.shape[0]
        fold_idx = 0
        train_end = self._min_train
        needed = self._gap + self._val + self._test

        while train_end + needed <= n:
            if self._max_folds > 0 and fold_idx >= self._max_folds:
                break

            val_start = train_end + self._gap
            val_end = val_start + self._val
            test_start = val_end
            test_end = test_start + self._test

            if test_end > n:
                break

            yield WindowFold(
                fold_index=fold_idx,
                train_start=0,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start if self._test > 0 else 0,
                test_end=test_end if self._test > 0 else 0,
                x_train=x[:train_end],
                y_train=y[:train_end],
                x_val=x[val_start:val_end],
                y_val=y[val_start:val_end],
                x_test=x[test_start:test_end] if self._test > 0 else torch.empty(0),
                y_test=y[test_start:test_end] if self._test > 0 else torch.empty(0),
                strategy="expanding_window",
            )
            fold_idx += 1
            train_end += self._step

    def num_folds(self, n_samples: int) -> int:
        needed = self._gap + self._val + self._test
        count = 0
        train_end = self._min_train
        while train_end + needed <= n_samples:
            if self._max_folds > 0 and count >= self._max_folds:
                break
            count += 1
            train_end += self._step
        return count


# ---------------------------------------------------------------------------
# Anchored Walk-Forward Validator
# ---------------------------------------------------------------------------


class AnchoredWalkForward:
    """Anchored walk-forward: training always starts from the first sample.

    Unlike expanding-window which slides the start as well, the Anchored
    strategy keeps t=0 as the immutable training anchor.  The validation
    window slides forward.  On each fold, the full history back to the
    origin is available for training.

    Args:
        anchor_start: Absolute start index of training (default 0).
        initial_train_size: Minimum training samples before the first fold.
        val_size: Validation window size.
        test_size: Optional hold-out test window (0 = disabled).
        step_size: Samples to advance the val window per fold.
        gap_size: Purge gap between train end and val start.
        max_folds: Hard cap on fold count.
    """

    def __init__(
        self,
        initial_train_size: int,
        val_size: int,
        test_size: int = 0,
        step_size: int = 1,
        gap_size: int = 0,
        anchor_start: int = 0,
        max_folds: int = 0,
    ) -> None:
        self._anchor = anchor_start
        self._init_train = initial_train_size
        self._val = val_size
        self._test = test_size
        self._step = step_size
        self._gap = gap_size
        self._max_folds = max_folds

    def split(self, x: torch.Tensor, y: torch.Tensor) -> Iterator[WindowFold]:
        """Generate anchored folds in chronological order."""
        n = x.shape[0]
        fold_idx = 0
        # The training end grows; val slides forward too
        train_end = self._anchor + self._init_train
        needed = self._gap + self._val + self._test

        while train_end + needed <= n:
            if self._max_folds > 0 and fold_idx >= self._max_folds:
                break

            val_start = train_end + self._gap
            val_end = val_start + self._val
            test_start = val_end
            test_end = test_start + self._test

            if test_end > n:
                break

            yield WindowFold(
                fold_index=fold_idx,
                train_start=self._anchor,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start if self._test > 0 else 0,
                test_end=test_end if self._test > 0 else 0,
                x_train=x[self._anchor:train_end],
                y_train=y[self._anchor:train_end],
                x_val=x[val_start:val_end],
                y_val=y[val_start:val_end],
                x_test=x[test_start:test_end] if self._test > 0 else torch.empty(0),
                y_test=y[test_start:test_end] if self._test > 0 else torch.empty(0),
                strategy="anchored_walk_forward",
            )
            fold_idx += 1
            train_end += self._step


# ---------------------------------------------------------------------------
# Nested Validator
# ---------------------------------------------------------------------------


@dataclass
class NestedFold:
    """A nested (outer + inner) cross-validation fold.

    Outer split: train vs (val + test)
    Inner split: within the outer training window for hyper-parameter search.

    Attributes:
        outer_fold_index: Outer fold number.
        outer_train_start/end: Outer training indices.
        outer_val_start/end: Outer validation indices.
        inner_folds: List of WindowFold from the inner validator.
        x_outer_train, y_outer_train, x_outer_val, y_outer_val: Tensors.
    """

    outer_fold_index: int
    outer_train_start: int
    outer_train_end: int
    outer_val_start: int
    outer_val_end: int
    inner_folds: list[WindowFold]
    x_outer_train: torch.Tensor
    y_outer_train: torch.Tensor
    x_outer_val: torch.Tensor
    y_outer_val: torch.Tensor


class NestedValidator:
    """Nested cross-validation for time-series data.

    Uses an outer expanding-window loop and an inner sliding-window loop
    within the outer training split.  Useful for hyper-parameter search
    without look-ahead leakage.

    Args:
        outer_min_train: Minimum outer training samples.
        outer_val_size: Outer validation window size.
        outer_step: Outer step size.
        inner_train_size: Inner training window size.
        inner_val_size: Inner validation window size.
        inner_step: Inner step size.
        gap_size: Purge gap applied to both outer and inner splits.
        max_outer_folds: Cap on outer folds.
    """

    def __init__(
        self,
        outer_min_train: int,
        outer_val_size: int,
        outer_step: int,
        inner_train_size: int,
        inner_val_size: int,
        inner_step: int,
        gap_size: int = 0,
        max_outer_folds: int = 0,
    ) -> None:
        self._outer = ExpandingWindowValidator(
            min_train_size=outer_min_train,
            val_size=outer_val_size,
            step_size=outer_step,
            gap_size=gap_size,
            max_folds=max_outer_folds,
        )
        self._inner_train = inner_train_size
        self._inner_val = inner_val_size
        self._inner_step = inner_step
        self._gap = gap_size

    def split(self, x: torch.Tensor, y: torch.Tensor) -> Iterator[NestedFold]:
        """Generate nested folds."""
        for outer in self._outer.split(x, y):
            inner_validator = SlidingWindowValidator(
                train_size=self._inner_train,
                val_size=self._inner_val,
                step_size=self._inner_step,
                gap_size=self._gap,
            )
            inner_folds = list(inner_validator.split(outer.x_train, outer.y_train))
            yield NestedFold(
                outer_fold_index=outer.fold_index,
                outer_train_start=outer.train_start,
                outer_train_end=outer.train_end,
                outer_val_start=outer.val_start,
                outer_val_end=outer.val_end,
                inner_folds=inner_folds,
                x_outer_train=outer.x_train,
                y_outer_train=outer.y_train,
                x_outer_val=outer.x_val,
                y_outer_val=outer.y_val,
            )
