"""Weighted Cross Entropy Loss — per-class weight support for imbalanced datasets."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.framework.abstractions.loss import BaseLoss


class WeightedCrossEntropyLoss(BaseLoss):
    """Cross entropy loss with configurable per-class weights.

    Equivalent to nn.CrossEntropyLoss(weight=...) but with a cleaner
    interface for dynamic weight updates and label smoothing.

    Args:
        class_weights: Per-class weights of shape (num_classes,).
                       If None, all classes are weighted equally.
        reduction: "mean" | "sum" | "none".
        label_smoothing: Label smoothing factor in [0, 1).
        ignore_index: Class index to ignore in loss computation (-1 to disable).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights: Optional[torch.Tensor] = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: Raw class logits of shape (batch, num_classes).
            targets: Ground truth class indices of shape (batch,).

        Returns:
            Scalar loss (or per-sample if reduction='none').
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

    def update_weights(self, class_weights: torch.Tensor) -> None:
        """Update per-class weights dynamically (e.g. from dataset statistics).

        Args:
            class_weights: New weight tensor of shape (num_classes,).
        """
        device = next(self.parameters(), torch.zeros(1)).device
        self.register_buffer("class_weights", class_weights.float().to(device))
