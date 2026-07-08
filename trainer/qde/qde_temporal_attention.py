"""Phase S9 — Temporal attention architecture.

Learns how the six timeframes (D1, H4, H1, M15, M5, M1) relate to each
other; nothing here branches on a fixed condition like "if H1 trend is up".
All cross-timeframe reasoning happens through trainable attention weights
and linear heads, which is what "no hard-coded rules" requires — the
network is free to learn, for a given instrument/regime, that (say) M5
momentum matters more than H1 structure, or the reverse, purely from data.

Architecture
------------
1. `PerTimeframeEncoder` — one encoder per timeframe (independent weights,
   shared feature-contract input dim from
   `qde_timeframe_models.TIMEFRAME_FEATURE_DIM`), each producing a fixed
   `embed_dim` vector from that timeframe's recent bar sequence.
2. `TimeframeIdentityEmbedding` — a learned embedding per timeframe slot
   (0=D1 .. 5=M1), added to each encoder output before attention. This
   lets the attention mechanism tell timeframes apart; it is a trainable
   parameter, not a hardcoded priority order.
3. `CrossTimeframeAttention` — `nn.MultiheadAttention` self-attention over
   the 6 timeframe-embedding sequence, producing a fused representation
   plus the attention weight matrix (kept for interpretability/telemetry
   only — never fed back into a rule).
4. Four learned output heads over the fused latent:
   - `alignment_head`      -> sigmoid  (does cross-TF alignment matter now)
   - `countertrend_head`   -> sigmoid  (countertrend opportunity strength)
   - `continuation_head`   -> sigmoid  (continuation opportunity strength)
   - `reversal_vs_continuation_head` -> 2-way softmax
     (reversal_probability, continuation_probability)

`TemporalAttentionEncoder` wires all of this into one `nn.Module` whose
`forward()` takes a batch of `MultiTimeframeSnapshot.as_matrix()`-shaped
tensors and returns a `TemporalAttentionOutput`.

Note: this module requires PyTorch. Per `.agents/memory`, `torch` is
currently unimportable in this sandbox (missing `libtorch_global_deps.so`)
— a pre-existing, unrelated environment issue. The file is syntax-checked
(`py_compile`) and reviewed statically here; it follows the same
`nn.Module` conventions as `trainer/models/quant_v1/hybrid/streams.py` and
is expected to run unmodified in an environment with a working torch
install.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trainer.qde.qde_timeframe_models import TIMEFRAME_FEATURE_DIM
from trainer.qde.qde_timeframe_types import ALL_TIMEFRAMES

NUM_TIMEFRAMES = len(ALL_TIMEFRAMES)

DEFAULT_TF_HIDDEN_DIM = 32
DEFAULT_EMBED_DIM = 64
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_DROPOUT = 0.1


@dataclass
class TemporalAttentionOutput:
    """Structured forward() output — tensors, one row per batch element."""

    fused_latent: torch.Tensor  # (batch, embed_dim)
    attention_weights: torch.Tensor  # (batch, num_heads, num_tf, num_tf)
    alignment_score: torch.Tensor  # (batch,) in [0, 1]
    countertrend_opportunity_score: torch.Tensor  # (batch,) in [0, 1]
    continuation_opportunity_score: torch.Tensor  # (batch,) in [0, 1]
    reversal_probability: torch.Tensor  # (batch,) in [0, 1]
    continuation_probability: torch.Tensor  # (batch,) in [0, 1], = 1 - reversal_probability


class PerTimeframeEncoder(nn.Module):
    """Independent-weight LSTM encoder for one timeframe's bar sequence.

    Input: `(batch, seq_len, TIMEFRAME_FEATURE_DIM)`
    Output: `(batch, embed_dim)`
    """

    def __init__(
        self, embed_dim: int = DEFAULT_EMBED_DIM, hidden_dim: int = DEFAULT_TF_HIDDEN_DIM
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(TIMEFRAME_FEATURE_DIM)
        self.rnn = nn.LSTM(
            input_size=TIMEFRAME_FEATURE_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        _, (h_n, _) = self.rnn(x)
        return self.proj(h_n[-1])


class TimeframeIdentityEmbedding(nn.Module):
    """Learned per-timeframe-slot embedding, added to encoder outputs so
    attention can distinguish "this is the D1 embedding" from "this is the
    M1 embedding" without any fixed ordering rule being imposed on how
    they should relate.
    """

    def __init__(
        self, embed_dim: int = DEFAULT_EMBED_DIM, num_timeframes: int = NUM_TIMEFRAMES
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_timeframes, embed_dim)

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        """`stacked`: (batch, num_tf, embed_dim) -> same shape, with the
        identity embedding added per timeframe slot.
        """
        batch_size, num_tf, _ = stacked.shape
        idx = torch.arange(num_tf, device=stacked.device)
        identity = self.embedding(idx).unsqueeze(0).expand(batch_size, -1, -1)
        return stacked + identity


class CrossTimeframeAttention(nn.Module):
    """Self-attention across the stack of timeframe embeddings.

    Input: `(batch, num_tf, embed_dim)`
    Output: `(fused (batch, embed_dim), attn_weights (batch, heads, num_tf, num_tf))`
    """

    def __init__(
        self,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_heads: int = DEFAULT_ATTENTION_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        fused = self.norm(x + attn_out)  # residual + norm, per-timeframe
        pooled = fused.mean(dim=1)  # (batch, embed_dim) — simple mean over TF slots
        return pooled, attn_weights


class TemporalAttentionEncoder(nn.Module):
    """Full S9 hierarchical-timeframe reasoning module.

    Input: a tensor `(batch, num_tf, seq_len, TIMEFRAME_FEATURE_DIM)`,
    where the `num_tf` axis is ordered per `ALL_TIMEFRAMES`
    (coarsest -> finest, i.e. D1..M1) — matching
    `MultiTimeframeSnapshot.as_matrix()`'s row order once bar histories are
    windowed into sequences.
    """

    def __init__(
        self,
        embed_dim: int = DEFAULT_EMBED_DIM,
        tf_hidden_dim: int = DEFAULT_TF_HIDDEN_DIM,
        num_heads: int = DEFAULT_ATTENTION_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.tf_encoders = nn.ModuleList(
            [PerTimeframeEncoder(embed_dim, tf_hidden_dim) for _ in ALL_TIMEFRAMES]
        )
        self.identity_embedding = TimeframeIdentityEmbedding(embed_dim, NUM_TIMEFRAMES)
        self.cross_attention = CrossTimeframeAttention(embed_dim, num_heads, dropout)

        self.alignment_head = nn.Linear(embed_dim, 1)
        self.countertrend_head = nn.Linear(embed_dim, 1)
        self.continuation_head = nn.Linear(embed_dim, 1)
        self.reversal_vs_continuation_head = nn.Linear(embed_dim, 2)

    def forward(self, timeframe_sequences: torch.Tensor) -> TemporalAttentionOutput:
        """Args:
            timeframe_sequences: `(batch, num_tf, seq_len, feature_dim)`

        Returns:
            `TemporalAttentionOutput`
        """
        batch_size, num_tf, _seq_len, _feat_dim = timeframe_sequences.shape
        assert (
            num_tf == NUM_TIMEFRAMES
        ), f"expected {NUM_TIMEFRAMES} timeframes (D1..M1), got {num_tf}"

        encoded = [
            self.tf_encoders[i](timeframe_sequences[:, i, :, :]) for i in range(num_tf)
        ]  # each (batch, embed_dim)
        stacked = torch.stack(encoded, dim=1)  # (batch, num_tf, embed_dim)
        stacked = self.identity_embedding(stacked)

        fused, attn_weights = self.cross_attention(
            stacked
        )  # (batch, embed_dim), (batch, heads, tf, tf)

        alignment_score = torch.sigmoid(self.alignment_head(fused)).squeeze(-1)
        countertrend_opportunity_score = torch.sigmoid(self.countertrend_head(fused)).squeeze(-1)
        continuation_opportunity_score = torch.sigmoid(self.continuation_head(fused)).squeeze(-1)

        reversal_logits = self.reversal_vs_continuation_head(fused)  # (batch, 2)
        reversal_probs = torch.softmax(reversal_logits, dim=-1)
        reversal_probability = reversal_probs[:, 0]
        continuation_probability = reversal_probs[:, 1]

        return TemporalAttentionOutput(
            fused_latent=fused,
            attention_weights=attn_weights,
            alignment_score=alignment_score,
            countertrend_opportunity_score=countertrend_opportunity_score,
            continuation_opportunity_score=continuation_opportunity_score,
            reversal_probability=reversal_probability,
            continuation_probability=continuation_probability,
        )

    def attention_weights_by_timeframe(self, attn_weights: torch.Tensor) -> dict[str, float]:
        """Average attention received by each timeframe slot, across heads
        and query positions — a single interpretable weight per timeframe
        for telemetry (`TemporalAssessment.attention_weights`). Purely
        descriptive; not used to alter the head outputs above.
        """
        received = attn_weights.mean(dim=(0, 1, 2))  # (num_tf,) — avg over batch, heads, queries
        total = received.sum().clamp_min(1e-9)
        normalized = (received / total).tolist()
        return {tf.value: float(w) for tf, w in zip(ALL_TIMEFRAMES, normalized)}
