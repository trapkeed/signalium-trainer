"""Context Embedding Stream — Phase 14.9A (Stream 4).

Learns independent embeddings for:
  instrument, timeframe, session, hour, day, week, month, regime

Each categorical gets an ``nn.Embedding`` table.  All enabled embeddings are
concatenated and projected to ``out_dim``.

Input:  dict of ``{category_name: (batch,) long tensor}``
Output: ``(batch, out_dim)``
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from trainer.models.quant_v1.hybrid.config import (
    ContextEmbeddingStreamConfig,
    EmbeddingSpec,
)


class ContextEmbeddingStream(nn.Module):
    """Learned categorical context embeddings.

    Supports any subset of the eight canonical context categories:
    ``instrument``, ``timeframe``, ``session``, ``hour``, ``day``,
    ``week``, ``month``, ``regime``.

    Only categories whose ``EmbeddingSpec.enabled=True`` contribute.
    Categories not present in the input dict are silently skipped.

    Args:
        config: ``ContextEmbeddingStreamConfig`` instance.
    """

    def __init__(self, config: ContextEmbeddingStreamConfig) -> None:
        super().__init__()
        self._config = config
        self._enabled = config.enabled

        if not self._enabled:
            return

        self.embeddings = nn.ModuleDict()
        for name, spec in config.embeddings.items():
            if spec.enabled:
                self.embeddings[name] = nn.Embedding(
                    num_embeddings=spec.vocab_size,
                    embedding_dim=spec.embed_dim,
                )

        concat_dim = config.raw_concat_dim
        if concat_dim == 0:
            concat_dim = 8  # fallback for fully-disabled state

        self.proj = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, config.out_dim),
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
        context_ids: dict[str, torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Args:
            context_ids: dict of {category_name: (batch,) long tensor}
            batch_size: fallback for empty dict

        Returns:
            ``(batch, out_dim)``
        """
        parts: list[torch.Tensor] = []

        for name, emb in self.embeddings.items():
            if name in context_ids:
                parts.append(emb(context_ids[name]))  # (batch, embed_dim)

        if not parts:
            bs = batch_size or 1
            dev = next(self.parameters()).device
            return torch.zeros(bs, self._config.out_dim, device=dev)

        cat = torch.cat(parts, dim=-1)  # (batch, total_embed_dim)

        # If the concatenated dim differs from the projection input dim
        # (e.g. some categories missing), pad or adapt via a lazy projection.
        # We use the pre-built proj only when dim matches; otherwise fall back
        # to a zeroed projection to maintain shape.
        expected_dim = self._config.raw_concat_dim
        if cat.shape[-1] != expected_dim:
            # Partial embedding — project what we have via linear interpolation
            # using a narrow slice of the weight matrix.
            proj_weight = self.proj[1].weight  # (out_dim, expected_dim)
            out = torch.nn.functional.linear(
                cat, proj_weight[:, : cat.shape[-1]], self.proj[1].bias
            )
            out = torch.nn.functional.gelu(out)
            return out

        return self.proj(cat)
