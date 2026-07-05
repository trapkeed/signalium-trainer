"""BaseSequenceEncoder — framework-level abstract temporal encoder."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseSequenceEncoder(nn.Module):
    """Abstract base for all temporal sequence encoders in the Signalium Model Framework.

    A sequence encoder takes a sequence of encoded feature vectors
    (batch, seq_len, input_dim) and compresses it into a single context
    vector (batch, output_dim) that summarises the temporal pattern.

    This is the framework-level contract that makes backbone implementations
    (LSTM, GRU, Transformer, TCN, Hybrid) interchangeable without touching
    the training pipeline.

    Concrete implementations must:
    - Accept (batch, seq_len, input_dim) tensors
    - Return (batch, output_dim) context vectors
    - Expose the :attr:`output_dim` property
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the output context vector."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of feature vectors into a single context vector.

        Args:
            x: Encoded input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Context tensor of shape (batch, output_dim).
        """
