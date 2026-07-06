"""BenchmarkConditions — guarantees identical experimental conditions across all runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BenchmarkConditions:
    """Defines the fixed experimental conditions for a benchmark run.

    All models in a :class:`BenchmarkRunner` run under exactly the same
    ``BenchmarkConditions`` to guarantee a fair, reproducible comparison.

    Attributes:
        random_seed: Global seed applied to Python, NumPy, and PyTorch.
        dataset_version: Dataset version string (used to fetch data from the API).
        feature_version: Feature pipeline version string.
        label_version: Label scheme version string.
        train_ratio: Fraction of data used for training (0 < ratio < 1).
        val_ratio: Fraction of data used for validation.
        max_epochs: Maximum training epochs per model.
        batch_size: Mini-batch size.
        window_length: Input sequence length (number of timesteps).
        eval_metrics: Ordered list of metric names to compute after each run.
        device: PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        num_workers: DataLoader worker count.
        extra: Arbitrary additional condition metadata.
    """

    random_seed: int = 42
    dataset_version: str = "1.0.0"
    feature_version: str = "1.0.0"
    label_version: str = "1.0.0"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    max_epochs: int = 50
    batch_size: int = 256
    window_length: int = 60
    eval_metrics: list[str] = field(
        default_factory=lambda: [
            "loss",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ]
    )
    device: str = "cpu"
    num_workers: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate constraint satisfaction.

        Raises:
            ValueError: If train+val ratio ≥ 1.0 or any ratio ≤ 0.
        """
        if self.train_ratio <= 0 or self.val_ratio <= 0:
            raise ValueError("train_ratio and val_ratio must be positive.")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) "
                "must be < 1.0 to leave room for a test split."
            )
        if not 0 < self.max_epochs:
            raise ValueError("max_epochs must be > 0.")
        if not 0 < self.batch_size:
            raise ValueError("batch_size must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "random_seed": self.random_seed,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "window_length": self.window_length,
            "eval_metrics": self.eval_metrics,
            "device": self.device,
            "num_workers": self.num_workers,
            **self.extra,
        }
