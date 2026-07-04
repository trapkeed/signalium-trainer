"""BaseFeatureEncoder — framework-level abstract feature encoder."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseFeatureEncoder(nn.Module):
    """Abstract base for all feature encoders in the Signalium Model Framework.

    A feature encoder transforms raw input tensors (batch, seq_len, input_dim)
    into a richer representation (batch, seq_len, output_dim) suitable for a
    downstream sequence encoder.

    Concrete encoders must implement :meth:`forward` and expose
    :attr:`output_dim` so downstream components can auto-configure.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the encoded output features."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw input features.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Encoded tensor of shape (batch, seq_len, output_dim).
        """
