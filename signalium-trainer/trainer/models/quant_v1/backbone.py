"""Sequence encoder backbone — abstract interface + concrete LSTM and GRU implementations.

Future backbones (Transformer, Temporal CNN, Hybrid) implement SequenceEncoder
without any changes to the training pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from trainer.models.quant_v1.config import BackboneConfig, BackboneType


class SequenceEncoder(nn.Module, ABC):
    """Abstract base for all temporal sequence encoders.

    Concrete implementations must:
    - Accept (batch, seq_len, input_dim) input
    - Return (batch, output_dim) context vector (last timestep or pooled)
    - Expose `output_dim` property
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the output context vector."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of feature vectors into a single context vector.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Context tensor of shape (batch, output_dim).
        """


class LSTMBackbone(SequenceEncoder):
    """Multi-layer LSTM sequence encoder.

    Takes a sequence of feature vectors and returns the hidden state
    from the final timestep of the last LSTM layer as the context vector.

    Args:
        input_dim: Number of input features (from InputEncoder output).
        config: BackboneConfig with hidden_size, num_layers, dropout, bidirectional.
    """

    def __init__(self, input_dim: int, config: BackboneConfig) -> None:
        super().__init__()
        self._config = config
        self._bidirectional = config.bidirectional
        self._directions = 2 if config.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        self.dropout = nn.Dropout(p=config.dropout)

    @property
    def output_dim(self) -> int:
        return self._config.hidden_size * self._directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence through LSTM layers.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Context tensor of shape (batch, hidden_size * directions).
        """
        output, (h_n, _) = self.lstm(x)

        if self._bidirectional:
            # Concatenate forward and backward final hidden states
            fwd = h_n[-2]   # Last layer forward
            bwd = h_n[-1]   # Last layer backward
            context = torch.cat([fwd, bwd], dim=-1)
        else:
            context = h_n[-1]  # Last layer, last timestep

        return self.dropout(context)


class GRUBackbone(SequenceEncoder):
    """Multi-layer GRU sequence encoder.

    Drop-in replacement for LSTMBackbone — same interface, GRU internals.

    Args:
        input_dim: Number of input features.
        config: BackboneConfig.
    """

    def __init__(self, input_dim: int, config: BackboneConfig) -> None:
        super().__init__()
        self._config = config
        self._directions = 2 if config.bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        self.dropout = nn.Dropout(p=config.dropout)

    @property
    def output_dim(self) -> int:
        return self._config.hidden_size * self._directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence through GRU layers.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Context tensor of shape (batch, hidden_size * directions).
        """
        _, h_n = self.gru(x)

        if self._config.bidirectional:
            context = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            context = h_n[-1]

        return self.dropout(context)


def build_backbone(input_dim: int, config: BackboneConfig) -> SequenceEncoder:
    """Factory that instantiates the correct backbone from config.

    Adding a new backbone requires only:
    1. Implementing SequenceEncoder
    2. Adding a branch here

    The training pipeline never changes.

    Args:
        input_dim: Feature dimensionality from the encoder.
        config: BackboneConfig specifying the backbone type.

    Returns:
        Instantiated SequenceEncoder.
    """
    if config.type == BackboneType.LSTM:
        return LSTMBackbone(input_dim=input_dim, config=config)
    if config.type == BackboneType.GRU:
        return GRUBackbone(input_dim=input_dim, config=config)
    raise ValueError(
        f"Unknown backbone type: '{config.type}'. "
        f"Available: {[t.value for t in BackboneType]}"
    )
