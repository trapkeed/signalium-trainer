"""Stream encoder modules for Phase 14.9A Hybrid Representation Learning.

Implements:
  - RawMarketStream     (stream 1) — OHLCV + Spread + Timestamp via RNN
  - MultiTimeframeStream (stream 2) — per-timeframe RNN encoders + pooling
  - EngineeredFeatureStream (stream 3) — MLP over pre-computed features

All streams output a fixed-dimension vector ``(batch, out_dim)`` that the
``AdaptiveFusionLayer`` ingests.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.models.quant_v1.hybrid.config import (
    EngineeredFeatureStreamConfig,
    MultiTimeframeStreamConfig,
    RawMarketStreamConfig,
    SequenceBackboneType,
)

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


def _build_rnn(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    backbone_type: SequenceBackboneType,
) -> nn.Module:
    """Build a multi-layer LSTM or GRU, batch_first=True."""
    kwargs = dict(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )
    if backbone_type == SequenceBackboneType.LSTM:
        return nn.LSTM(**kwargs)
    return nn.GRU(**kwargs)


def _rnn_last_hidden(rnn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run an RNN and return the last-layer final hidden state ``(batch, hidden)``."""
    out = rnn(x)
    hidden = out[1]  # (h_n, c_n) for LSTM, h_n for GRU
    if isinstance(hidden, tuple):
        hidden = hidden[0]  # h_n of shape (num_layers, batch, hidden)
    return hidden[-1]  # last layer


class _MLP(nn.Sequential):
    """Simple LayerNorm-optional MLP block used by feature streams."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
    ) -> None:
        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(in_dim))
        dim = in_dim
        for _ in range(max(1, num_layers - 1)):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        super().__init__(*layers)


# ---------------------------------------------------------------------------
# Stream 1 — Raw Market Sequence
# ---------------------------------------------------------------------------


class RawMarketStream(nn.Module):
    """Encodes raw OHLCV + Spread (+ optional Timestamp) via an RNN.

    Input: ``(batch, seq_len, input_dim)``
    Output: ``(batch, out_dim)``

    The RNN final hidden state is projected to ``out_dim``.
    """

    def __init__(self, config: RawMarketStreamConfig) -> None:
        super().__init__()
        self._config = config
        self._enabled = config.enabled

        if not self._enabled:
            return

        if config.use_layer_norm:
            self.norm = nn.LayerNorm(config.input_dim)
        else:
            self.norm = nn.Identity()

        self.rnn = _build_rnn(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            backbone_type=config.backbone_type,
        )
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.out_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    @property
    def out_dim(self) -> int:
        return self._config.out_dim

    @property
    def enabled(self) -> bool:
        return self._enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``(batch, seq_len, input_dim)``
        Returns:
            ``(batch, out_dim)``
        """
        x = self.norm(x)
        h = _rnn_last_hidden(self.rnn, x)
        return self.proj(h)


# ---------------------------------------------------------------------------
# Stream 2 — Multi-Timeframe
# ---------------------------------------------------------------------------


class _TimeframeEncoder(nn.Module):
    """Single-timeframe LSTM encoder.

    Input: ``(batch, tf_seq_len, feat_dim)``
    Output: ``(batch, out_dim)``
    """

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        _, (h_n, _) = self.rnn(x)
        return self.proj(h_n[-1])


class _TimeframeAttentionPool(nn.Module):
    """Attention pooling over a stack of timeframe vectors.

    Input: ``(batch, num_tf, tf_out_dim)``
    Output: ``(batch, tf_out_dim)`` — attention-weighted sum.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score(x)  # (batch, num_tf, 1)
        weights = torch.softmax(scores, dim=1)
        return (weights * x).sum(dim=1)


class MultiTimeframeStream(nn.Module):
    """Encodes multiple timeframe tensors with independent per-timeframe RNNs.

    Input: a dict ``{timeframe_name: (batch, tf_seq_len, feat_dim)}``
    Output: ``(batch, out_dim)``

    Only enabled timeframes that are present in the input dict are processed.
    Outputs are pooled (mean or attention) then projected.
    """

    def __init__(self, config: MultiTimeframeStreamConfig) -> None:
        super().__init__()
        self._config = config
        self._enabled = config.enabled

        if not self._enabled:
            return

        self.tf_encoders = nn.ModuleDict()
        for tf_name, tf_cfg in config.timeframes.items():
            if tf_cfg.enabled:
                self.tf_encoders[tf_name] = _TimeframeEncoder(
                    input_dim=tf_cfg.input_dim,
                    hidden_dim=tf_cfg.hidden_dim,
                    out_dim=tf_cfg.out_dim,
                )

        # Infer per-timeframe output dim (all must be equal for pooling)
        enabled = [cfg for cfg in config.timeframes.values() if cfg.enabled]
        tf_embed_dim = enabled[0].out_dim if enabled else 64

        self._pooling = config.pooling
        if config.pooling == "attention":
            self.attn_pool = _TimeframeAttentionPool(embed_dim=tf_embed_dim)

        self.proj = nn.Sequential(
            nn.LayerNorm(tf_embed_dim),
            nn.Linear(tf_embed_dim, config.out_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    @property
    def out_dim(self) -> int:
        return self._config.out_dim

    @property
    def enabled(self) -> bool:
        return self._enabled

    def forward(
        self,
        timeframe_inputs: dict[str, torch.Tensor],
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Args:
            timeframe_inputs: dict of {tf_name: (batch, seq_len, feat_dim)}
            batch_size: fallback if dict is empty

        Returns:
            ``(batch, out_dim)``
        """
        encoded: list[torch.Tensor] = []
        for tf_name, encoder in self.tf_encoders.items():
            if tf_name in timeframe_inputs:
                encoded.append(encoder(timeframe_inputs[tf_name]))  # (batch, tf_out)

        if not encoded:
            # No timeframe data — return zero tensor
            bs = batch_size or 1
            dev = next(self.parameters()).device
            zero = torch.zeros(bs, self._config.out_dim, device=dev)
            return zero

        stacked = torch.stack(encoded, dim=1)  # (batch, num_tf, tf_out_dim)

        if self._pooling == "attention" and hasattr(self, "attn_pool"):
            pooled = self.attn_pool(stacked)
        else:
            pooled = stacked.mean(dim=1)  # (batch, tf_out_dim)

        return self.proj(pooled)


# ---------------------------------------------------------------------------
# Stream 3 — Engineered Features
# ---------------------------------------------------------------------------


class EngineeredFeatureStream(nn.Module):
    """Encodes pre-computed engineered features via an independent-normalised MLP.

    Retains ALL engineered features — trend, momentum, volatility, liquidity,
    market structure, sessions, market memory variants — and projects them into
    a fixed output dimension.

    Input: ``(batch, input_dim)``
    Output: ``(batch, out_dim)``
    """

    def __init__(self, config: EngineeredFeatureStreamConfig) -> None:
        super().__init__()
        self._config = config
        self._enabled = config.enabled

        if not self._enabled:
            return

        self.mlp = _MLP(
            in_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm,
        )

    @property
    def out_dim(self) -> int:
        return self._config.out_dim

    @property
    def enabled(self) -> bool:
        return self._enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``(batch, input_dim)``
        Returns:
            ``(batch, out_dim)``
        """
        return self.mlp(x)
