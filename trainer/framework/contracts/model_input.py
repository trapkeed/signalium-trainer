"""ModelInput — standardised input contract for all Signalium models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelInput:
    """Standardised input container for all Signalium model forward passes.

    All models in the framework accept a ``ModelInput`` in their public
    ``predict()`` interface. The ``features`` tensor is required; all other
    fields are optional and reserved for future capabilities.

    Attributes:
        features: The primary feature tensor of shape
                  (batch, seq_len, feature_dim). Required.
        mask: Optional boolean attention mask of shape (batch, seq_len).
              True = valid timestep, False = padding. Reserved for future
              Transformer / variable-length sequence support.
        metadata: Optional dict of per-sample metadata (e.g. symbol,
                  timestamp, dataset_version). Not used in forward pass.
        auxiliary: Optional dict for any extra tensors a specific model
                   may require (e.g. market regime embedding).
    """

    features: torch.Tensor
    mask: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None
    auxiliary: dict[str, torch.Tensor] | None = None

    def validate(self) -> None:
        """Validate tensor shapes and dtypes.

        Raises:
            ValueError: If ``features`` does not have exactly 3 dimensions.
            TypeError: If ``features`` is not a float tensor.
        """
        if self.features.ndim != 3:
            raise ValueError(
                f"ModelInput.features must have 3 dimensions "
                f"(batch, seq_len, feature_dim), got shape {tuple(self.features.shape)}"
            )
        if not self.features.is_floating_point():
            raise TypeError(
                f"ModelInput.features must be a float tensor, got {self.features.dtype}"
            )
        if self.mask is not None:
            if self.mask.shape != self.features.shape[:2]:
                raise ValueError(
                    f"ModelInput.mask shape {tuple(self.mask.shape)} "
                    f"must match features[:2] {tuple(self.features.shape[:2])}"
                )

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return self.features.shape[0]

    @property
    def seq_len(self) -> int:
        """Sequence length (number of timesteps)."""
        return self.features.shape[1]

    @property
    def feature_dim(self) -> int:
        """Feature dimensionality per timestep."""
        return self.features.shape[2]

    def to(self, device: torch.device) -> ModelInput:
        """Move all tensors to the specified device.

        Args:
            device: Target device.

        Returns:
            New ModelInput with all tensors on the target device.
        """
        return ModelInput(
            features=self.features.to(device),
            mask=self.mask.to(device) if self.mask is not None else None,
            metadata=self.metadata,
            auxiliary=(
                {k: v.to(device) for k, v in self.auxiliary.items()}
                if self.auxiliary is not None
                else None
            ),
        )
