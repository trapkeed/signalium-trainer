"""HyperparameterRecord — structured snapshot of all training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HyperparameterRecord:
    """Captures all hyperparameters for a single training run.

    Structured into logical groups so records are self-documenting and
    can be compared across experiments in the leaderboard.

    Attributes:
        learning_rate: Initial learning rate.
        hidden_size: Backbone hidden dimension.
        num_layers: Number of recurrent / attention layers.
        dropout: Primary dropout rate.
        optimizer: Optimizer name (e.g. ``"adamw"``).
        scheduler: LR scheduler name (e.g. ``"cosine_annealing"``).
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        window_length: Input sequence length (number of timesteps).
        gradient_clip: Gradient norm clipping value (0 = disabled).
        weight_decay: L2 regularisation coefficient.
        warmup_steps: Number of LR warmup steps.
        label_smoothing: Label smoothing coefficient.
        loss_type: Loss function name (e.g. ``"focal"``, ``"weighted_ce"``).
        focal_gamma: Gamma for focal loss (irrelevant if loss is not focal).
        class_balance_strategy: Class-balancing strategy name.
        early_stopping_patience: Epochs without improvement before stopping.
        extra: Arbitrary additional hyperparameters.
    """

    learning_rate: float = 1e-3
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    optimizer: str = "adamw"
    scheduler: str = "cosine_annealing"
    epochs: int = 100
    batch_size: int = 256
    window_length: int = 60
    gradient_clip: float = 1.0
    weight_decay: float = 1e-4
    warmup_steps: int = 0
    label_smoothing: float = 0.0
    loss_type: str = "cross_entropy"
    focal_gamma: float = 2.0
    class_balance_strategy: str = "none"
    early_stopping_patience: int = 10
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return all hyperparameters as a flat dict."""
        base = {
            "learning_rate": self.learning_rate,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "window_length": self.window_length,
            "gradient_clip": self.gradient_clip,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "label_smoothing": self.label_smoothing,
            "loss_type": self.loss_type,
            "focal_gamma": self.focal_gamma,
            "class_balance_strategy": self.class_balance_strategy,
            "early_stopping_patience": self.early_stopping_patience,
        }
        base.update(self.extra)
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HyperparameterRecord:
        """Reconstruct from a dict.

        Known keys are mapped to structured fields; unknown keys go to ``extra``.

        Args:
            d: Dict (e.g. from JSON storage or Hydra config).

        Returns:
            HyperparameterRecord.
        """
        known = {
            "learning_rate",
            "hidden_size",
            "num_layers",
            "dropout",
            "optimizer",
            "scheduler",
            "epochs",
            "batch_size",
            "window_length",
            "gradient_clip",
            "weight_decay",
            "warmup_steps",
            "label_smoothing",
            "loss_type",
            "focal_gamma",
            "class_balance_strategy",
            "early_stopping_patience",
        }
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in d.items():
            if k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        if extra:
            kwargs["extra"] = extra
        return cls(**kwargs)

    def diff(self, other: HyperparameterRecord) -> dict[str, tuple[Any, Any]]:
        """Return hyperparameters that differ between self and other.

        Args:
            other: Another HyperparameterRecord to compare with.

        Returns:
            Dict mapping param name → (self_value, other_value).
        """
        a = self.to_dict()
        b = other.to_dict()
        all_keys = set(a) | set(b)
        return {k: (a.get(k), b.get(k)) for k in all_keys if a.get(k) != b.get(k)}
