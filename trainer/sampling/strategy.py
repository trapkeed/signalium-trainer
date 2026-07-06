"""Class imbalance mitigation strategies.

Supports:
- Class weights for loss functions
- WeightedRandomSampler for DataLoader-level balancing
- Oversampling (repeat minority class indices)
- Undersampling (downsample majority class indices)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from trainer.models.quant_v1.config import BalancingConfig, BalancingStrategy


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
    mode: str = "inverse_freq",
) -> torch.Tensor:
    """Compute per-class weights inversely proportional to class frequency.

    Args:
        labels: Sequence of integer class labels.
        num_classes: Total number of classes.
        mode: "inverse_freq" (1/count) or "effective_num" (1/(1-beta^n), beta=0.9999).

    Returns:
        Float tensor of shape (num_classes,).
    """
    counts = Counter(labels)
    total = len(labels)

    if mode == "effective_num":
        beta = 0.9999
        weights = []
        for c in range(num_classes):
            n = counts.get(c, 0)
            if n == 0:
                weights.append(1.0)
            else:
                weights.append((1.0 - beta) / (1.0 - beta**n))
    else:
        weights = []
        for c in range(num_classes):
            n = counts.get(c, 0)
            weights.append(total / (num_classes * n) if n > 0 else 1.0)

    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * num_classes  # Normalise so sum = num_classes


def make_weighted_sampler(
    labels: Sequence[int],
    num_classes: int,
    num_samples: int | None = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for balanced mini-batch sampling.

    Each sample is assigned a weight equal to the inverse class frequency,
    so rare classes appear as often as common classes in each batch.

    Args:
        labels: Sequence of integer class labels for every sample in the dataset.
        num_classes: Total number of classes.
        num_samples: Number of samples per epoch (defaults to len(labels)).
        replacement: Whether to sample with replacement.

    Returns:
        WeightedRandomSampler ready to be passed to DataLoader(sampler=...).
    """
    class_weights = compute_class_weights(labels, num_classes)
    sample_weights = torch.tensor(
        [class_weights[label].item() for label in labels], dtype=torch.float32
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples or len(labels),
        replacement=replacement,
    )


def oversample_dataset(
    indices: list[int],
    labels: Sequence[int],
    target_ratio: float = 1.0,
) -> list[int]:
    """Oversample minority class indices to reduce imbalance.

    Repeats minority class samples until each class reaches at least
    ``target_ratio * majority_count`` samples.

    Args:
        indices: Original dataset sample indices.
        labels: Class label for each sample (must have same length as indices).
        target_ratio: Target minority-to-majority ratio (1.0 = fully balanced).

    Returns:
        New list of indices with minority classes oversampled.
    """
    label_arr = np.array(list(labels))
    index_arr = np.array(indices)

    counts = Counter(label_arr.tolist())
    max_count = max(counts.values())
    target_count = int(max_count * target_ratio)

    result: list[int] = list(indices)
    for cls, count in counts.items():
        if count >= target_count:
            continue
        cls_indices = index_arr[label_arr == cls].tolist()
        needed = target_count - count
        extras = (cls_indices * math.ceil(needed / count))[:needed]
        result.extend(extras)

    return result


def undersample_dataset(
    indices: list[int],
    labels: Sequence[int],
    target_ratio: float = 1.0,
    seed: int = 42,
) -> list[int]:
    """Undersample majority class indices to reduce imbalance.

    Randomly removes majority class samples until each class has at most
    ``target_ratio * minority_count`` samples.

    Args:
        indices: Original dataset sample indices.
        labels: Class label for each sample.
        target_ratio: Target majority-to-minority ratio (1.0 = fully balanced).
        seed: Random seed for reproducible subsampling.

    Returns:
        New list of indices with majority classes downsampled.
    """
    rng = np.random.default_rng(seed)
    label_arr = np.array(list(labels))
    index_arr = np.array(indices)

    counts = Counter(label_arr.tolist())
    min_count = min(counts.values())
    target_count = int(min_count * target_ratio)

    result: list[int] = []
    for cls in counts:
        cls_indices = index_arr[label_arr == cls].tolist()
        if len(cls_indices) <= target_count:
            result.extend(cls_indices)
        else:
            chosen = rng.choice(cls_indices, size=target_count, replace=False).tolist()
            result.extend(chosen)

    return result


class BalancingStrategyBuilder:
    """Applies the configured balancing strategy to a dataset.

    Args:
        config: BalancingConfig specifying the strategy.
        num_classes: Total number of classes.
    """

    def __init__(self, config: BalancingConfig, num_classes: int) -> None:
        self._config = config
        self._num_classes = num_classes

    def get_sampler(
        self,
        labels: Sequence[int],
    ) -> WeightedRandomSampler | None:
        """Return a WeightedRandomSampler if strategy is WEIGHTED_SAMPLER, else None.

        Args:
            labels: Class labels for every sample in the dataset.

        Returns:
            WeightedRandomSampler or None.
        """
        if self._config.strategy == BalancingStrategy.WEIGHTED_SAMPLER:
            return make_weighted_sampler(labels, self._num_classes)
        return None

    def get_class_weights(
        self,
        labels: Sequence[int],
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        """Return class weights tensor if strategy is CLASS_WEIGHTS, else None.

        Args:
            labels: Class labels for every sample in the dataset.
            device: Optional torch device.

        Returns:
            Float tensor of shape (num_classes,) or None.
        """
        if self._config.strategy == BalancingStrategy.CLASS_WEIGHTS:
            weights = compute_class_weights(labels, self._num_classes)
            if device:
                weights = weights.to(device)
            return weights
        return None

    def apply_index_strategy(self, indices: list[int], labels: Sequence[int]) -> list[int]:
        """Apply oversample or undersample to dataset indices.

        Args:
            indices: Original sample indices.
            labels: Class labels.

        Returns:
            Modified index list (or original if strategy does not apply).
        """
        if self._config.strategy == BalancingStrategy.OVERSAMPLE:
            return oversample_dataset(indices, labels, self._config.oversample_ratio)
        if self._config.strategy == BalancingStrategy.UNDERSAMPLE:
            return undersample_dataset(indices, labels, self._config.undersample_ratio)
        return indices
