"""Unit tests for class imbalance mitigation strategies."""

from __future__ import annotations

from collections import Counter

import pytest
pytestmark = pytest.mark.requires_torch

from torch.utils.data import WeightedRandomSampler

from trainer.models.quant_v1.config import BalancingConfig, BalancingStrategy
from trainer.sampling.strategy import (
    BalancingStrategyBuilder,
    compute_class_weights,
    make_weighted_sampler,
    oversample_dataset,
    undersample_dataset,
)

NUM_CLASSES = 3
# Imbalanced labels: class 0 dominant, class 2 rare
IMBALANCED_LABELS = [0] * 70 + [1] * 20 + [2] * 10


class TestComputeClassWeights:
    def test_output_length(self) -> None:
        w = compute_class_weights(IMBALANCED_LABELS, NUM_CLASSES)
        assert len(w) == NUM_CLASSES

    def test_rare_class_higher_weight(self) -> None:
        w = compute_class_weights(IMBALANCED_LABELS, NUM_CLASSES)
        # Class 2 (rare) should have higher weight than class 0 (dominant)
        assert w[2] > w[0]

    def test_weights_positive(self) -> None:
        w = compute_class_weights(IMBALANCED_LABELS, NUM_CLASSES)
        assert (w > 0).all()

    def test_effective_num_mode(self) -> None:
        w = compute_class_weights(IMBALANCED_LABELS, NUM_CLASSES, mode="effective_num")
        assert len(w) == NUM_CLASSES
        assert (w > 0).all()


class TestMakeWeightedSampler:
    def test_returns_weighted_random_sampler(self) -> None:
        sampler = make_weighted_sampler(IMBALANCED_LABELS, NUM_CLASSES)
        assert isinstance(sampler, WeightedRandomSampler)

    def test_sampler_length(self) -> None:
        sampler = make_weighted_sampler(IMBALANCED_LABELS, NUM_CLASSES)
        assert len(sampler) == len(IMBALANCED_LABELS)

    def test_custom_num_samples(self) -> None:
        sampler = make_weighted_sampler(IMBALANCED_LABELS, NUM_CLASSES, num_samples=200)
        assert len(sampler) == 200


class TestOversampling:
    def test_result_longer_than_input(self) -> None:
        indices = list(range(len(IMBALANCED_LABELS)))
        result = oversample_dataset(indices, IMBALANCED_LABELS, target_ratio=1.0)
        assert len(result) > len(indices)

    def test_all_classes_represented(self) -> None:
        indices = list(range(len(IMBALANCED_LABELS)))
        result = oversample_dataset(indices, IMBALANCED_LABELS, target_ratio=1.0)
        # Map indices back to labels
        label_map = dict(zip(range(len(IMBALANCED_LABELS)), IMBALANCED_LABELS))
        result_labels = [label_map[i] for i in result if i in label_map]
        result_counts = Counter(result_labels)
        assert len(result_counts) == NUM_CLASSES


class TestUndersampling:
    def test_result_shorter_than_input(self) -> None:
        indices = list(range(len(IMBALANCED_LABELS)))
        result = undersample_dataset(indices, IMBALANCED_LABELS, target_ratio=1.0, seed=0)
        assert len(result) < len(indices)

    def test_majority_class_reduced(self) -> None:
        indices = list(range(len(IMBALANCED_LABELS)))
        result = undersample_dataset(indices, IMBALANCED_LABELS, target_ratio=1.0, seed=0)
        label_map = dict(zip(range(len(IMBALANCED_LABELS)), IMBALANCED_LABELS))
        result_labels = [label_map[i] for i in result if i in label_map]
        result_counts = Counter(result_labels)
        original_counts = Counter(IMBALANCED_LABELS)
        # Majority class should be reduced
        assert result_counts.get(0, 0) < original_counts[0]


class TestBalancingStrategyBuilder:
    def test_class_weights_strategy(self) -> None:
        cfg = BalancingConfig(strategy=BalancingStrategy.CLASS_WEIGHTS)
        builder = BalancingStrategyBuilder(cfg, NUM_CLASSES)
        weights = builder.get_class_weights(IMBALANCED_LABELS)
        assert weights is not None
        assert len(weights) == NUM_CLASSES

    def test_weighted_sampler_strategy(self) -> None:
        cfg = BalancingConfig(strategy=BalancingStrategy.WEIGHTED_SAMPLER)
        builder = BalancingStrategyBuilder(cfg, NUM_CLASSES)
        sampler = builder.get_sampler(IMBALANCED_LABELS)
        assert sampler is not None

    def test_none_strategy_returns_none(self) -> None:
        cfg = BalancingConfig(strategy=BalancingStrategy.NONE)
        builder = BalancingStrategyBuilder(cfg, NUM_CLASSES)
        assert builder.get_sampler(IMBALANCED_LABELS) is None
        assert builder.get_class_weights(IMBALANCED_LABELS) is None

    def test_oversample_strategy(self) -> None:
        cfg = BalancingConfig(strategy=BalancingStrategy.OVERSAMPLE)
        builder = BalancingStrategyBuilder(cfg, NUM_CLASSES)
        indices = list(range(len(IMBALANCED_LABELS)))
        result = builder.apply_index_strategy(indices, IMBALANCED_LABELS)
        assert len(result) > len(indices)

    def test_undersample_strategy(self) -> None:
        cfg = BalancingConfig(strategy=BalancingStrategy.UNDERSAMPLE)
        builder = BalancingStrategyBuilder(cfg, NUM_CLASSES)
        indices = list(range(len(IMBALANCED_LABELS)))
        result = builder.apply_index_strategy(indices, IMBALANCED_LABELS)
        assert len(result) < len(indices)
