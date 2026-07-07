"""Standard evaluation implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader

from trainer.config.schema import TrainerConfig
from trainer.metrics.evaluation import EvaluationMetrics
from trainer.models.base import BaseEvaluator, BaseQuantModel
from trainer.utils.device import resolve_device


class StandardEvaluator(BaseEvaluator):
    """Evaluates a trained model over a DataLoader and computes full metrics.

    Args:
        model: Trained model.
        config: Full TrainerConfig.
        num_classes: Override detected number of classes.
    """

    def __init__(
        self,
        model: BaseQuantModel,
        config: TrainerConfig,
        num_classes: int | None = None,
    ) -> None:
        super().__init__(model, config)
        self._device = resolve_device(config.gpu.device)
        self._num_classes = num_classes or model.num_classes
        task = "binary" if self._num_classes == 2 else "multiclass"
        self._metrics = EvaluationMetrics(
            num_classes=self._num_classes, task=task, device=str(self._device)
        )

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        """Run inference and compute classification metrics.

        Args:
            dataloader: DataLoader for the test or validation split.

        Returns:
            Dict with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, etc.
        """
        self._model.eval()
        self._model.to(self._device)

        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_probs: list[torch.Tensor] = []

        for x, y in dataloader:
            x = x.to(self._device, non_blocking=True)
            with torch.autocast(
                device_type=self._device.type, enabled=self._config.training.mixed_precision
            ):
                logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_probs.append(probs.cpu())

        preds_tensor = torch.cat(all_preds)
        targets_tensor = torch.cat(all_targets)
        probs_tensor = torch.cat(all_probs)

        logger.info("Computing evaluation metrics over {} samples", len(preds_tensor))
        return self._metrics.compute(preds_tensor, targets_tensor, probs_tensor)

    def generate_report(self, results: dict[str, Any], output_dir: Path) -> Path:
        """Delegate to EvaluationReporter."""
        from trainer.evaluation.reporter import EvaluationReporter

        reporter = EvaluationReporter(output_dir)
        return reporter.write(results)
