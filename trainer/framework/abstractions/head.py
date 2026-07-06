"""BaseClassificationHead, BaseConfidenceHead, BaseRegressionHead — framework-level ABCs."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseClassificationHead(ABC, nn.Module):
    """Abstract base for all classification heads in the Signalium Model Framework.

    A classification head maps a context vector (batch, context_dim) to
    class logits (batch, num_classes). Softmax is applied downstream in
    the OutputLayer — this head returns raw logits.

    Concrete implementations must implement :meth:`forward`.
    """

    @abstractmethod
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute class logits from a context vector.

        Args:
            context: Context tensor of shape (batch, context_dim).

        Returns:
            Logit tensor of shape (batch, num_classes).
        """


class BaseConfidenceHead(ABC, nn.Module):
    """Abstract base for all confidence heads in the Signalium Model Framework.

    A confidence head produces a scalar certainty estimate per sample
    (batch, 1), in the range [0, 1]. It runs in parallel with the
    classification head, sharing the same context vector.

    Concrete implementations must implement :meth:`forward`.
    """

    @abstractmethod
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute per-sample confidence scores.

        Args:
            context: Context tensor of shape (batch, context_dim).

        Returns:
            Confidence tensor of shape (batch, 1), values in [0, 1].
        """


class BaseRegressionHead(nn.Module):
    """Abstract base for all regression heads in the Signalium Model Framework.

    A regression head maps a context vector (batch, context_dim) to a
    scalar quantity (batch, 1). Output range and units are head-specific.

    Concrete implementations must implement :meth:`forward`.
    """

    @abstractmethod
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute a scalar regression output.

        Args:
            context: Context tensor of shape (batch, context_dim).

        Returns:
            Scalar output of shape (batch, 1).
        """
