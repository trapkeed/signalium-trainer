"""Class imbalance handling — samplers and dataset rebalancing strategies."""

from trainer.sampling.strategy import (
    BalancingStrategyBuilder,
    compute_class_weights,
    make_weighted_sampler,
    oversample_dataset,
    undersample_dataset,
)

__all__ = [
    "BalancingStrategyBuilder",
    "compute_class_weights",
    "make_weighted_sampler",
    "oversample_dataset",
    "undersample_dataset",
]
