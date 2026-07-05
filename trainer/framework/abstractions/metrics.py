"""BaseMetrics — framework-level abstract metrics interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMetrics(ABC):
    """Abstract base for all metrics engines in the Signalium Model Framework.

    A metrics engine accumulates predictions and targets across batches
    and computes aggregate statistics at the end of an epoch.

    Concrete implementations must implement :meth:`update`, :meth:`compute`,
    and :meth:`reset`.
    """

    @abstractmethod
    def update(self, logits: Any, targets: Any) -> None:
        """Accumulate predictions and targets for one batch.

        Args:
            logits: Raw model logits for the batch.
            targets: Ground truth class indices for the batch.
        """

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute aggregate metrics over all accumulated batches.

        Returns:
            Dict mapping metric name → scalar float value.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all accumulated state (call at the start of each epoch)."""

    def compute_and_reset(self) -> dict[str, float]:
        """Compute metrics and immediately reset state.

        Convenience method combining :meth:`compute` and :meth:`reset`.

        Returns:
            Dict mapping metric name → scalar float value.
        """
        result = self.compute()
        self.reset()
        return result
