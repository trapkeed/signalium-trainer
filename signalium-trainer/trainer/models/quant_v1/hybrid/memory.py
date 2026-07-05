"""Market Memory Stream — Phase 14.9A (Stream 5).

Encodes the Market Memory Engine state vector through a small MLP into a
fixed-dimension latent vector that the AdaptiveFusionLayer can consume.

The memory vector is treated as an external state signal — already computed
by the Market Memory Engine — and passed directly without sequence processing.

Input:  ``(batch, memory_dim)``
Output: ``(batch, out_dim)``
"""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.models.quant_v1.hybrid.config import MarketMemoryStreamConfig


class MarketMemoryStream(nn.Module):
    """Encode a Market Memory Engine state into a latent vector.

    Args:
        config: ``MarketMemoryStreamConfig`` with memory_dim, hidden_dim,
                out_dim, and dropout.
    """

    def __init__(self, config: MarketMemoryStreamConfig) -> None:
        super().__init__()
        self._config = config
        self._enabled = config.enabled

        if not self._enabled:
            return

        self.encoder = nn.Sequential(
            nn.LayerNorm(config.memory_dim),
            nn.Linear(config.memory_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
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

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Args:
            memory: ``(batch, memory_dim)`` — Market Memory Engine state.
        Returns:
            ``(batch, out_dim)``
        """
        return self.encoder(memory)
