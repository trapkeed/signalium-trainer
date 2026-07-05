"""BaseExternalEncoder — root abstract contract for all future encoder plugins.

Any encoder added in a future phase (cross-asset, news, macro, sentiment, …)
MUST inherit from ``BaseExternalEncoder`` and be registered via the
``PluginRegistry``.  This contract ensures the core model never needs to
change its ``_encode()`` method to accommodate new data streams.

Design principles:
  - Pure interface — no implementation, no instance state.
  - Output shape contract: ``(batch_size, output_dim)`` — same as existing
    hybrid encoder streams, so fusion layer accepts it unchanged.
  - ``get_metadata()`` ensures every plugin is self-describing.
  - ``validate_output()`` provides a safety check before the tensor enters
    the fusion layer.

Do NOT instantiate this class directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch


class BaseExternalEncoder(ABC):
    """Abstract base class for all future external encoder plugins.

    Implementors MUST provide:
        - :attr:`name`         — unique registry key (lowercase, underscored)
        - :attr:`output_dim`   — dimensionality of the encoder output tensor
        - :meth:`forward`      — ``(**kwargs) → (batch_size, output_dim)``
        - :meth:`get_metadata` — version, status, and configuration info

    The encoder output will be passed to the HybridEncoder's
    AdaptiveFusionLayer as an additional stream — without any changes to
    existing streams or prediction heads.

    Example (future implementation skeleton)::

        from trainer.interfaces.base import BaseExternalEncoder
        from trainer.plugins.registry import PluginRegistry

        @PluginRegistry.register("cross_asset_v1", version="1.0.0")
        class CrossAssetEncoder(BaseExternalEncoder):

            @property
            def name(self) -> str:
                return "cross_asset_v1"

            @property
            def output_dim(self) -> int:
                return 128

            def forward(self, cross_asset_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
                ...  # future implementation

            def get_metadata(self) -> dict:
                return {"version": "1.0.0", "status": "experimental"}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique registry identifier — lowercase alphanumeric with underscores."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output dimensionality.

        Must equal the ``latent_dim`` configured in the fusion layer so that
        the encoder output can be concatenated with other stream outputs.
        """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode external data into the shared latent space.

        Args:
            *args, **kwargs: Encoder-specific inputs (defined by the
                concrete implementation).

        Returns:
            ``(batch_size, output_dim)`` encoded representation tensor.
        """

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return version and configuration metadata.

        Must include at minimum::

            {
                "version": str,    # semantic version e.g. "1.0.0"
                "status": str,     # "experimental" | "production" | "future"
            }

        Additional fields (input_schema, output_schema, dependencies, …)
        are encouraged but not required.
        """

    # ── Provided concrete helpers ─────────────────────────────────────────

    def validate_output(self, output: torch.Tensor, batch_size: int) -> bool:
        """Validate that the encoder output has the expected shape.

        Args:
            output: Encoder output tensor to validate.
            batch_size: Expected batch size.

        Returns:
            True if shape matches ``(batch_size, self.output_dim)``.
        """
        return (
            output.dim() == 2
            and int(output.shape[0]) == batch_size
            and int(output.shape[1]) == self.output_dim
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, output_dim={self.output_dim})"
