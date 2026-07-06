"""Adaptive Fusion Layer — Phase 14.9A (Stream 7).

Fuses all enabled stream output vectors into a single shared latent
representation of dimension ``latent_dim`` (must match the backbone
context_dim so all existing prediction heads work unchanged).

Three fusion strategies are supported:

1. **CONCAT** — Concatenate all stream vectors, project to ``latent_dim``.
2. **WEIGHTED** — Learnable per-stream softmax weights, weighted sum,
   then project.
3. **ATTENTION** — Treat stream vectors as tokens, apply multi-head
   self-attention over them, mean-pool, then project.

Stream contribution tracking is available for training monitoring:
``fusion.last_contributions`` contains per-stream weight statistics.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from trainer.models.quant_v1.hybrid.config import AdaptiveFusionConfig, FusionStrategy


class AdaptiveFusionLayer(nn.Module):
    """Fuses N stream output vectors into the shared latent representation.

    Args:
        stream_dims: List of (name, dim) pairs for each enabled stream, in order.
        config: ``AdaptiveFusionConfig`` instance.
    """

    def __init__(
        self,
        stream_dims: list[tuple[str, int]],
        config: AdaptiveFusionConfig,
    ) -> None:
        super().__init__()
        self._config = config
        self._stream_names = [name for name, _ in stream_dims]
        self._stream_dims = [dim for _, dim in stream_dims]
        self._strategy = config.strategy
        self._latent_dim = config.latent_dim
        self._track = config.track_contributions
        self.last_contributions: dict[str, float] = {}

        n = len(stream_dims)
        total_dim = sum(self._stream_dims)

        if config.strategy == FusionStrategy.CONCAT:
            self.proj = nn.Sequential(
                nn.LayerNorm(total_dim),
                nn.Linear(total_dim, config.latent_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )

        elif config.strategy == FusionStrategy.WEIGHTED:
            # Learnable per-stream weights (before softmax)
            self.stream_logits = nn.Parameter(torch.zeros(n))
            # All streams must have same dim for weighted sum; project each first
            self.stream_projs = nn.ModuleList(
                [nn.Linear(dim, config.latent_dim) for dim in self._stream_dims]
            )
            self.out_proj = nn.Sequential(
                nn.LayerNorm(config.latent_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )

        elif config.strategy == FusionStrategy.ATTENTION:
            # Project each stream to a common embed_dim for attention
            embed_dim = config.latent_dim
            self.stream_projs = nn.ModuleList(
                [nn.Linear(dim, embed_dim) for dim in self._stream_dims]
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.out_proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, config.latent_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )

        self.norm_out = nn.LayerNorm(config.latent_dim)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def num_streams(self) -> int:
        return len(self._stream_names)

    def forward(self, stream_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse stream outputs into the shared latent representation.

        Args:
            stream_outputs: List of ``(batch, stream_dim)`` tensors, one per
                enabled stream, in the same order as ``stream_dims`` provided
                at construction.

        Returns:
            ``(batch, latent_dim)`` — the shared latent representation.
        """
        if len(stream_outputs) == 1:
            # Only one stream — fast path
            s = stream_outputs[0]
            if self._strategy == FusionStrategy.CONCAT:
                return self.norm_out(self.proj(s))
            elif self._strategy == FusionStrategy.WEIGHTED:
                out = self.stream_projs[0](s)
                return self.norm_out(self.out_proj(out))
            else:
                out = self.stream_projs[0](s).unsqueeze(1)
                attn_out, _ = self.attention(out, out, out)
                return self.norm_out(self.out_proj(attn_out.squeeze(1)))

        if self._strategy == FusionStrategy.CONCAT:
            return self._forward_concat(stream_outputs)
        elif self._strategy == FusionStrategy.WEIGHTED:
            return self._forward_weighted(stream_outputs)
        else:
            return self._forward_attention(stream_outputs)

    def _forward_concat(self, stream_outputs: list[torch.Tensor]) -> torch.Tensor:
        cat = torch.cat(stream_outputs, dim=-1)  # (batch, total_dim)
        out = self.norm_out(self.proj(cat))
        if self._track:
            self._record_concat_contributions(stream_outputs)
        return out

    def _forward_weighted(self, stream_outputs: list[torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.stream_logits, dim=0)  # (n,)
        projected = [self.stream_projs[i](stream_outputs[i]) for i in range(len(stream_outputs))]
        stacked = torch.stack(projected, dim=1)  # (batch, n, latent_dim)
        w = weights.unsqueeze(0).unsqueeze(-1)  # (1, n, 1)
        fused = (w * stacked).sum(dim=1)  # (batch, latent_dim)
        out = self.norm_out(self.out_proj(fused))
        if self._track:
            self.last_contributions = {
                name: weights[i].item() for i, name in enumerate(self._stream_names)
            }
        return out

    def _forward_attention(self, stream_outputs: list[torch.Tensor]) -> torch.Tensor:
        tokens = [self.stream_projs[i](stream_outputs[i]) for i in range(len(stream_outputs))]
        tokens_3d = torch.stack(tokens, dim=1)  # (batch, n, embed_dim)
        attn_out, attn_weights = self.attention(tokens_3d, tokens_3d, tokens_3d)
        # Mean-pool over token dimension
        pooled = attn_out.mean(dim=1)  # (batch, embed_dim)
        out = self.norm_out(self.out_proj(pooled))
        if self._track and attn_weights is not None:
            mean_weights = attn_weights.mean(dim=(0, 1))  # (n,)
            self.last_contributions = {
                name: mean_weights[i].item() for i, name in enumerate(self._stream_names)
            }
        return out

    def _record_concat_contributions(self, stream_outputs: list[torch.Tensor]) -> None:
        total_norm = sum(s.norm(dim=-1).mean().item() for s in stream_outputs)
        for i, name in enumerate(self._stream_names):
            norm_i = stream_outputs[i].norm(dim=-1).mean().item()
            self.last_contributions[name] = norm_i / (total_norm + 1e-8)

    def contribution_summary(self) -> dict[str, float]:
        """Return the most recent per-stream contribution statistics."""
        return dict(self.last_contributions)
