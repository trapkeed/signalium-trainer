"""Input Encoder and Feature Normalizer for QuantModelV1."""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.models.quant_v1.config import InputEncoderConfig


class FeatureNormalizer(nn.Module):
    """Per-feature Layer Normalization applied to the input tensor.

    Normalizes the last dimension (feature dimension) of each timestep
    independently, keeping temporal structure intact.

    Args:
        input_dim: Number of input features per timestep.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input features.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Normalized tensor of the same shape.
        """
        return self.norm(x)


class InputEncoder(nn.Module):
    """Projects raw engineered features into the model's hidden space.

    Architecture:
        FeatureNormalizer → Linear(input_dim → hidden_dim) → GELU → Dropout

    The projection is optional — when disabled, the raw input_dim must equal
    the backbone's expected input size.

    Args:
        config: InputEncoderConfig.
    """

    def __init__(self, config: InputEncoderConfig) -> None:
        super().__init__()
        self._config = config
        self.output_dim = config.hidden_dim if config.use_projection else config.input_dim

        self.normalizer: nn.Module = (
            FeatureNormalizer(config.input_dim)
            if config.use_layer_norm
            else nn.Identity()
        )

        if config.use_projection:
            self.projection: nn.Module = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(p=config.projection_dropout),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw input features.

        Args:
            x: Raw feature tensor of shape (batch, seq_len, input_dim).

        Returns:
            Encoded tensor of shape (batch, seq_len, output_dim).
        """
        x = self.normalizer(x)
        x = self.projection(x)
        return x
