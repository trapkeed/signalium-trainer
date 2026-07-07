"""Abstract base classes for models, trainers, evaluators, and exporters.

No concrete neural networks are implemented here.
Future models inherit from BaseQuantModel.
Future trainers inherit from BaseTrainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trainer.config.schema import TrainerConfig


class BaseQuantModel(nn.Module, ABC):
    """Abstract base for all Signalium quantitative models.

    Subclasses must implement :meth:`forward` and expose
    :attr:`feature_dim` and :attr:`num_classes` so the training
    engine can build compatible DataLoaders automatically.

    Args:
        config: Full TrainerConfig for the current run.
    """

    def __init__(self, config: TrainerConfig) -> None:
        super().__init__()
        self._config = config

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Expected dimensionality of input feature vectors."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes (1 for regression tasks)."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, feature_dim).

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """

    def get_param_count(self) -> dict[str, int]:
        """Return trainable and total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


class BaseTrainer(ABC):
    """Abstract base for model trainers.

    Concrete trainers implement the training loop, validation loop,
    and hook into the checkpoint and metrics systems.

    Args:
        model: The model to train.
        config: Full TrainerConfig for the current run.
    """

    def __init__(self, model: BaseQuantModel, config: TrainerConfig) -> None:
        self._model = model
        self._config = config

    @property
    def model(self) -> BaseQuantModel:
        """The model being trained."""
        return self._model

    @abstractmethod
    def train_epoch(
        self,
        epoch: int,
        dataloader: Any,
    ) -> dict[str, float]:
        """Run a single training epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            dataloader: Training DataLoader.

        Returns:
            Dict of metric names to float values (e.g. {"loss": 0.45}).
        """

    @abstractmethod
    def validate_epoch(
        self,
        epoch: int,
        dataloader: Any,
    ) -> dict[str, float]:
        """Run validation after a training epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            dataloader: Validation DataLoader.

        Returns:
            Dict of metric names to float values.
        """

    @abstractmethod
    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        resume_checkpoint: Path | None = None,
    ) -> dict[str, Any]:
        """Execute the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            resume_checkpoint: Optional path to resume from.

        Returns:
            Training result dict with history, best metrics, and timing.
        """


class BaseEvaluator(ABC):
    """Abstract base for model evaluators.

    Args:
        model: Trained model to evaluate.
        config: Full TrainerConfig.
    """

    def __init__(self, model: BaseQuantModel, config: TrainerConfig) -> None:
        self._model = model
        self._config = config

    @abstractmethod
    def evaluate(self, dataloader: Any) -> dict[str, Any]:
        """Run evaluation on a dataset split.

        Args:
            dataloader: DataLoader for the evaluation split.

        Returns:
            Dict of metric names to values (scalars, arrays, or objects).
        """

    @abstractmethod
    def generate_report(self, results: dict[str, Any], output_dir: Path) -> Path:
        """Generate a human-readable evaluation report.

        Args:
            results: Output from :meth:`evaluate`.
            output_dir: Directory to write the report to.

        Returns:
            Path to the written report file.
        """


class BaseExporter(ABC):
    """Abstract base for model exporters.

    Args:
        model: Trained model to export.
        config: Full TrainerConfig.
    """

    def __init__(self, model: BaseQuantModel, config: TrainerConfig) -> None:
        self._model = model
        self._config = config

    @abstractmethod
    def export(self, output_dir: Path, **kwargs: Any) -> list[Path]:
        """Export the model to one or more artifact files.

        Args:
            output_dir: Directory to write exported files to.
            **kwargs: Exporter-specific options.

        Returns:
            List of paths to the exported artifact files.
        """
