"""Dynamic Feature Gate — Phase 14.9A (Stream 6 modifier).

The gate learns to amplify or suppress individual engineered features based on
a context signal derived from other streams.  Gate activations are constrained
to [0.1, 2.0]:

  gate = 0.1 + sigmoid(gate_net(context)) * 1.9

This guarantees:
  - Amplification (gate > 1.0) is possible.
  - Suppression (gate < 1.0) is possible.
  - No feature is ever permanently disabled (gate ≥ 0.1 always).

The gate is applied multiplicatively to the FEATURE VECTOR before the
EngineeredFeatureStream processes it.

Example::

    gate = DynamicFeatureGate(config)
    gated_features = gate(engineered_features, context_from_raw_stream)
    # → pass gated_features to EngineeredFeatureStream
"""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.models.quant_v1.hybrid.config import DynamicFeatureGateConfig


class DynamicFeatureGate(nn.Module):
    """Context-aware multiplicative gate for engineered features.

    Produces a per-feature gate weight in [0.1, 2.0] conditioned on an
    external context vector (typically the raw market stream output).

    Args:
        feature_dim: Number of features to gate (matches engineered_features.input_dim).
        config: ``DynamicFeatureGateConfig`` instance.
    """

    GATE_MIN: float = 0.1
    GATE_RANGE: float = 1.9  # sigmoid → [0,1] → scaled to [0.1, 2.0]

    def __init__(self, feature_dim: int, config: DynamicFeatureGateConfig) -> None:
        super().__init__()
        self._config = config
        self._enabled = config.enabled
        self._feature_dim = feature_dim

        if not self._enabled:
            return

        self.gate_net = nn.Sequential(
            nn.LayerNorm(config.context_dim),
            nn.Linear(config.context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, feature_dim),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply context-conditioned multiplicative gate to features.

        Args:
            features: ``(batch, feature_dim)`` — raw engineered features.
            context: ``(batch, context_dim)`` — gate conditioning signal.

        Returns:
            Tuple of:
                - ``gated_features``: ``(batch, feature_dim)`` — gated output.
                - ``gate_weights``: ``(batch, feature_dim)`` — gate values in
                  [0.1, 2.0] (useful for visualisation / training monitoring).
        """
        if not self._enabled:
            return features, torch.ones_like(features)

        raw_gate = self.gate_net(context)  # (batch, feature_dim)
        gate = self.GATE_MIN + torch.sigmoid(raw_gate) * self.GATE_RANGE
        gated = features * gate
        return gated, gate

    def gate_statistics(self, gate_weights: torch.Tensor) -> dict[str, float]:
        """Compute gate activation statistics for training monitoring.

        Args:
            gate_weights: ``(batch, feature_dim)`` gate values.

        Returns:
            Dict with mean, min, max, std across all features in the batch.
        """
        with torch.no_grad():
            return {
                "gate_mean": gate_weights.mean().item(),
                "gate_min": gate_weights.min().item(),
                "gate_max": gate_weights.max().item(),
                "gate_std": gate_weights.std().item(),
                "gate_amplified_frac": (gate_weights > 1.0).float().mean().item(),
                "gate_suppressed_frac": (gate_weights < 1.0).float().mean().item(),
            }
