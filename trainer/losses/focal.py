"""Focal Loss — down-weights well-classified examples to focus on hard cases.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from trainer.framework.abstractions.loss import BaseLoss


class FocalLoss(BaseLoss):
    """Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0 and alpha=None, reduces to standard CrossEntropyLoss.

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
               Typical values: 0.5, 1.0, 2.0, 5.0.
        alpha: Per-class weight tensor of shape (num_classes,).
               If None, no per-class weighting is applied.
        reduction: "mean" | "sum" | "none".
        label_smoothing: Label smoothing factor in [0, 1).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw class logits of shape (batch, num_classes).
            targets: Ground truth class indices of shape (batch,).

        Returns:
            Scalar loss (or per-sample if reduction='none').
        """
        num_classes = logits.size(-1)

        # Standard cross-entropy with optional label smoothing
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # p_t: probability of the true class
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # Focal modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
