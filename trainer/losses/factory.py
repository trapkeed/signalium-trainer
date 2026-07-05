"""Loss function factory — builds the correct loss from LossConfig."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from trainer.models.quant_v1.config import LossConfig, LossType


def build_loss(config: LossConfig, device: Optional[torch.device] = None) -> nn.Module:
    """Instantiate a loss function from a LossConfig.

    Adding a new loss requires only:
    1. Implementing nn.Module in trainer/losses/
    2. Adding a branch here
    3. Adding the new type to LossType enum

    Args:
        config: LossConfig specifying type and hyper-parameters.
        device: Optional torch device for class weight tensors.

    Returns:
        Configured loss nn.Module.
    """
    dev = device or torch.device("cpu")

    class_weights: Optional[torch.Tensor] = None
    if config.class_weights is not None:
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32, device=dev)

    if config.type == LossType.CROSS_ENTROPY:
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing,
        )

    if config.type == LossType.WEIGHTED_CROSS_ENTROPY:
        from trainer.losses.weighted_ce import WeightedCrossEntropyLoss
        return WeightedCrossEntropyLoss(
            class_weights=class_weights,
            label_smoothing=config.label_smoothing,
        )

    if config.type == LossType.FOCAL:
        from trainer.losses.focal import FocalLoss

        alpha: Optional[torch.Tensor] = None
        if config.focal_alpha is not None:
            alpha = torch.tensor(config.focal_alpha, dtype=torch.float32, device=dev)

        return FocalLoss(
            gamma=config.focal_gamma,
            alpha=alpha,
            label_smoothing=config.label_smoothing,
        )

    raise ValueError(
        f"Unknown loss type: '{config.type}'. "
        f"Available: {[t.value for t in LossType]}"
    )
