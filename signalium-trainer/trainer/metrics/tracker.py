"""Running metrics tracker — accumulates epoch metrics over training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class MetricsTracker:
    """Accumulates metrics across epochs and provides aggregation helpers.

    Example::

        tracker = MetricsTracker()
        for epoch in range(N):
            metrics = train_one_epoch(...)
            tracker.update(metrics)

        history = tracker.history()       # list[dict]
        best_loss = tracker.best("val_loss", mode="min")
    """

    def __init__(self) -> None:
        self._history: list[dict[str, float]] = []

    def update(self, metrics: dict[str, float]) -> None:
        """Record a new epoch's metrics.

        Args:
            metrics: Dict of metric name → scalar value.
        """
        self._history.append(dict(metrics))

    def history(self) -> list[dict[str, float]]:
        """Return the full metrics history (one dict per epoch)."""
        return list(self._history)

    def best(self, metric: str, mode: str = "min") -> float | None:
        """Find the best value for a named metric.

        Args:
            metric: Metric name to query.
            mode: "min" or "max".

        Returns:
            Best value, or None if the metric was never recorded.
        """
        values = [ep[metric] for ep in self._history if metric in ep]
        if not values:
            return None
        return min(values) if mode == "min" else max(values)

    def last(self, metric: str) -> float | None:
        """Return the most recent value of a named metric."""
        for ep in reversed(self._history):
            if metric in ep:
                return ep[metric]
        return None

    def to_dict(self) -> dict[str, list[float]]:
        """Pivot history into {metric_name: [values...]} format."""
        out: dict[str, list[float]] = defaultdict(list)
        for ep in self._history:
            for k, v in ep.items():
                out[k].append(v)
        return dict(out)
