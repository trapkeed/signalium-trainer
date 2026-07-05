"""Sequence encoder backbone — Phase 14.9B.

Implements interchangeable backbone architectures:

  LSTM               — multi-layer bidirectional LSTM (original)
  GRU                — multi-layer bidirectional GRU (original)
  Transformer        — positional encoding + multi-head self-attention
  TemporalCNN (TCN)  — dilated causal 1-D convolutions
  HybridLSTMTransformer — LSTM layers followed by Transformer encoder

All backbones:
  - Extend ``SequenceEncoder`` (→ ``BaseSequenceEncoder``).
  - Accept ``(batch, seq_len, input_dim)`` tensors.
  - Return a ``(batch, hidden_size)`` context vector.
  - Are registered via ``build_backbone()`` — the training pipeline never changes.

Adding a new backbone requires only:
  1. Implementing ``SequenceEncoder``.
  2. Adding a branch in ``build_backbone()``.
  3. Adding the type to ``BackboneType``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as gradient_checkpoint

from trainer.framework.abstractions.sequence_encoder import BaseSequenceEncoder
from trainer.models.quant_v1.config import BackboneConfig, BackboneType


class SequenceEncoder(BaseSequenceEncoder):
    """Thin alias for ``BaseSequenceEncoder`` scoped to the quant_v1 package.

    All quant_v1 backbone implementations inherit from this class, which in
    turn inherits from the framework-level ``BaseSequenceEncoder``.  This
    preserves the existing quant_v1 import surface while connecting the
    backbone to the framework's abstract contract.
    """


# ---------------------------------------------------------------------------
# LSTM Backbone (original — unchanged)
# ---------------------------------------------------------------------------


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
            fwd = h_n[-2]
            bwd = h_n[-1]
            context = torch.cat([fwd, bwd], dim=-1)
        else:
            context = h_n[-1]

        return self.dropout(context)


# ---------------------------------------------------------------------------
# GRU Backbone (original — unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding (shared by Transformer-based backbones)
# ---------------------------------------------------------------------------


class _SinusoidalPositionalEncoding(nn.Module):
    """Add sinusoidal positional encoding to a sequence tensor.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x.

        Args:
            x: Shape (batch, seq_len, d_model).

        Returns:
            Shape (batch, seq_len, d_model).
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Backbone
# ---------------------------------------------------------------------------


class TransformerBackbone(SequenceEncoder):
    """Multi-head self-attention sequence encoder.

    Architecture:
      1. Linear projection: input_dim → hidden_size
      2. Sinusoidal positional encoding
      3. N × TransformerEncoderLayer (MHSA + FFN + LayerNorm)
      4. Sequence pooling: "last" timestep or "mean" over all timesteps
      5. Output: (batch, hidden_size)

    Supports gradient checkpointing per encoder layer to trade compute
    for activation memory on long sequences.

    Args:
        input_dim: Feature dimensionality from the input encoder.
        config: BackboneConfig — uses num_heads, ff_dim, num_layers,
                max_seq_len, pooling, dropout, hidden_size,
                gradient_checkpointing.
    """

    def __init__(self, input_dim: int, config: BackboneConfig) -> None:
        super().__init__()
        self._config = config
        self._use_checkpoint = config.gradient_checkpointing

        d_model = config.hidden_size
        self._d_model = d_model

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        max_len = max(config.max_seq_len, config.window_length, 512)
        self.pos_enc = _SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=config.dropout,
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_out = nn.Dropout(p=config.dropout)

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence through Transformer.

        Args:
            x: Shape (batch, seq_len, input_dim).

        Returns:
            Shape (batch, hidden_size).
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)

        if self._use_checkpoint and self.training:
            x = gradient_checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)

        x = self.layer_norm(x)

        if self._config.pooling == "mean":
            context = x.mean(dim=1)
        else:
            context = x[:, -1, :]

        return self.dropout_out(context)


# ---------------------------------------------------------------------------
# Temporal CNN (TCN) Backbone
# ---------------------------------------------------------------------------


class _CausalTCNBlock(nn.Module):
    """Single dilated causal 1-D convolution block.

    Applies: causal-padding → Conv1d → BatchNorm → GELU → Dropout
    with an optional residual projection.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel width.
        dilation: Dilation factor for the convolution.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self._padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal dilated convolution + residual.

        Args:
            x: Shape (batch, channels, seq_len).

        Returns:
            Shape (batch, out_channels, seq_len).
        """
        residual = self.residual(x)
        out = self.conv(x)
        if self._padding > 0:
            out = out[:, :, : -self._padding]
        out = self.activation(self.norm(out))
        out = self.dropout(out)
        return out + residual


class TemporalCNNBackbone(SequenceEncoder):
    """Dilated causal Temporal CNN sequence encoder.

    Architecture:
      1. Input projection: input_dim → tcn_channels[0]
      2. N × CausalTCNBlock (exponentially growing dilation: 1, 2, 4, ...)
      3. Global average pooling over time: (batch, out_channels, T) → (batch, out_channels)
      4. Linear projection: out_channels → hidden_size
      5. Output: (batch, hidden_size)

    Args:
        input_dim: Input feature dimension.
        config: BackboneConfig — uses tcn_channels, tcn_kernel_size,
                dropout, hidden_size, gradient_checkpointing.
    """

    def __init__(self, input_dim: int, config: BackboneConfig) -> None:
        super().__init__()
        self._config = config

        channels: list[int] = list(config.tcn_channels) if config.tcn_channels else [64, 128, 256]

        blocks: list[nn.Module] = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            blocks.append(
                _CausalTCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=config.tcn_kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                )
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*blocks)
        self.proj = nn.Linear(in_ch, config.hidden_size)
        self.dropout_out = nn.Dropout(p=config.dropout)

    @property
    def output_dim(self) -> int:
        return self._config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence through TCN.

        Args:
            x: Shape (batch, seq_len, input_dim).

        Returns:
            Shape (batch, hidden_size).
        """
        x = x.transpose(1, 2)
        x = self.tcn(x)
        context = x.mean(dim=2)
        context = self.proj(context)
        return self.dropout_out(context)


# ---------------------------------------------------------------------------
# Hybrid LSTM + Transformer Backbone
# ---------------------------------------------------------------------------


class HybridLSTMTransformerBackbone(SequenceEncoder):
    """Stacked LSTM → Transformer encoder.

    Combines the sequential inductive bias of LSTM with the global
    attention of the Transformer.

    Architecture:
      1. LSTM (rnn_layers) — captures local temporal patterns
      2. Positional encoding (added on top of LSTM output sequence)
      3. Transformer encoder (transformer_layers) — captures long-range
         dependencies across the full sequence
      4. Last-timestep pooling → (batch, hidden_size)

    Args:
        input_dim: Feature dimensionality from the input encoder.
        config: BackboneConfig — uses rnn_layers, hidden_size, num_heads,
                ff_dim, transformer_layers, max_seq_len, dropout,
                gradient_checkpointing.
    """

    def __init__(self, input_dim: int, config: BackboneConfig) -> None:
        super().__init__()
        self._config = config
        self._use_checkpoint = config.gradient_checkpointing

        d_model = config.hidden_size

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=config.rnn_layers,
            batch_first=True,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
        )

        max_len = max(config.max_seq_len, config.window_length, 512)
        self.pos_enc = _SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_len, dropout=config.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_layers
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_out = nn.Dropout(p=config.dropout)

    @property
    def output_dim(self) -> int:
        return self._config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence through LSTM then Transformer.

        Args:
            x: Shape (batch, seq_len, input_dim).

        Returns:
            Shape (batch, hidden_size).
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.pos_enc(lstm_out)

        if self._use_checkpoint and self.training:
            out = gradient_checkpoint.checkpoint(self.transformer, lstm_out, use_reentrant=False)
        else:
            out = self.transformer(lstm_out)

        out = self.layer_norm(out)
        context = out[:, -1, :]
        return self.dropout_out(context)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_backbone(input_dim: int, config: BackboneConfig) -> SequenceEncoder:
    """Instantiate the correct backbone from configuration.

    Adding a new backbone requires only:
    1. Implementing ``SequenceEncoder``
    2. Adding a branch here
    3. Adding the type to ``BackboneType``

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
    if config.type == BackboneType.TRANSFORMER:
        return TransformerBackbone(input_dim=input_dim, config=config)
    if config.type == BackboneType.TEMPORAL_CNN:
        return TemporalCNNBackbone(input_dim=input_dim, config=config)
    if config.type == BackboneType.HYBRID_LSTM_TRANSFORMER:
        return HybridLSTMTransformerBackbone(input_dim=input_dim, config=config)
    raise ValueError(
        f"Unknown backbone type: '{config.type}'. "
        f"Available: {[t.value for t in BackboneType]}"
    )
