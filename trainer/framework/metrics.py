"""MetricsEngine — reusable, model-agnostic metrics accumulator.

This is the framework-level metrics implementation. It is not tied to any
specific model architecture — any model producing class logits and integer
targets can use it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from trainer.framework.abstractions.metrics import BaseMetrics


class MetricsEngine(BaseMetrics):
    """Accumulates logits and targets across batches; computes epoch metrics.

    Tracks:
        - Loss (cross-entropy of accumulated logits vs targets)
        - Accuracy
        - Per-class accuracy
        - Macro Precision, Recall, F1
        - Weighted F1

    This engine is intentionally simple and dependency-light so it works
    without TorchMetrics or scikit-learn in environments where those are
    unavailable. For the full metric suite (ROC-AUC, ECE, confusion matrix),
    use ``trainer.metrics.EvaluationMetrics``.

    Args:
        num_classes: Number of output classes.
        class_labels: Optional human-readable class names.
    """

    def __init__(
        self,
        num_classes: int,
        class_labels: list[str] | None = None,
    ) -> None:
        self._num_classes = num_classes
        self._class_labels = class_labels or [str(i) for i in range(num_classes)]
        self._logits: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions and targets for one batch.

        Args:
            logits: Raw logit tensor of shape (batch, num_classes).
            targets: Integer class index tensor of shape (batch,).
        """
        self._logits.append(logits.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> dict[str, float]:
        """Compute all metrics over all accumulated batches.

        Returns:
            Dict with keys: loss, accuracy, precision_macro, recall_macro,
            f1_macro, f1_weighted, plus per-class accuracy_{label} entries.
        """
        if not self._logits:
            return {}

        all_logits = torch.cat(self._logits, dim=0)  # (N, C)
        all_targets = torch.cat(self._targets, dim=0)  # (N,)
        n = all_targets.size(0)

        loss = F.cross_entropy(all_logits, all_targets).item()
        preds = all_logits.argmax(dim=-1)
        accuracy = (preds == all_targets).float().mean().item()

        # Per-class stats
        precision_list, recall_list, f1_list, support_list = [], [], [], []
        per_class: dict[str, float] = {}

        for c in range(self._num_classes):
            tp = ((preds == c) & (all_targets == c)).sum().item()
            fp = ((preds == c) & (all_targets != c)).sum().item()
            fn = ((preds != c) & (all_targets == c)).sum().item()
            support = (all_targets == c).sum().item()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            cls_acc = tp / support if support > 0 else 0.0

            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            support_list.append(support)

            label = self._class_labels[c]
            per_class[f"accuracy_{label}"] = cls_acc

        total_support = sum(support_list)
        precision_macro = sum(precision_list) / self._num_classes
        recall_macro = sum(recall_list) / self._num_classes
        f1_macro = sum(f1_list) / self._num_classes
        f1_weighted = (
            sum(f * s for f, s in zip(f1_list, support_list)) / total_support
            if total_support > 0
            else 0.0
        )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            **per_class,
        }

    def reset(self) -> None:
        """Clear all accumulated logits and targets."""
        self._logits.clear()
        self._targets.clear()
