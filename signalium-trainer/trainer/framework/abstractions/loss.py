"""BaseLoss — framework-level abstract loss function."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """Abstract base for all loss functions in the Signalium Model Framework.

    All pluggable loss functions must inherit from this class. It extends
    ``nn.Module`` so concrete losses are fully compatible with PyTorch
    optimisers, device placement, and state serialisation.

    The standardised signature::

        loss = criterion(logits, targets)

    where ``logits`` are raw (pre-softmax) class scores and ``targets``
    are integer class indices, ensures loss functions are interchangeable
    without modifying the training engine.

    Concrete implementations must implement :meth:`forward`.
    """

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss.

        Args:
            logits: Raw class logit tensor of shape (batch, num_classes).
            targets: Integer class index tensor of shape (batch,).

        Returns:
            Scalar loss tensor (or per-sample if reduction='none').
        """
