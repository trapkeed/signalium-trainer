"""Early stopping callback."""

from __future__ import annotations

from loguru import logger


class EarlyStopping:
    """Monitor a metric and signal when training should stop.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: "min" (lower is better) or "max" (higher is better).
        verbose: Whether to log status on each call.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self._counter = 0
        self._best: float | None = None
        self._should_stop = False

    @property
    def should_stop(self) -> bool:
        """True when early stopping criterion has been met."""
        return self._should_stop

    @property
    def best_score(self) -> float | None:
        """Best metric value seen so far."""
        return self._best

    def step(self, metric: float) -> bool:
        """Evaluate a new metric value.

        Args:
            metric: Current epoch metric.

        Returns:
            True if training should stop.
        """
        improved = self._is_improvement(metric)

        if improved:
            self._best = metric
            self._counter = 0
            if self.verbose:
                logger.debug("EarlyStopping: improved — best={:.6f}", metric)
        else:
            self._counter += 1
            if self.verbose:
                logger.debug(
                    "EarlyStopping: no improvement {}/{} — best={:.6f}, current={:.6f}",
                    self._counter,
                    self.patience,
                    self._best,
                    metric,
                )

        if self._counter >= self.patience:
            self._should_stop = True
            logger.info(
                "EarlyStopping triggered after {} epochs without improvement", self.patience
            )

        return self._should_stop

    def reset(self) -> None:
        """Reset state (call before a new training run)."""
        self._counter = 0
        self._best = None
        self._should_stop = False

    def _is_improvement(self, metric: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return metric < self._best - self.min_delta
        return metric > self._best + self.min_delta
