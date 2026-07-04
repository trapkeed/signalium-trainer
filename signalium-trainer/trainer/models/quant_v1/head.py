"""Classification and Confidence heads for QuantModelV1.

Both heads implement their respective framework ABCs so they are
interchangeable with future head architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.framework.abstractions.head import BaseClassificationHead, BaseConfidenceHead
from trainer.models.quant_v1.config import ClassificationHeadConfig, ConfidenceHeadConfig


class ClassificationHead(BaseClassificationHead):
    """Maps the sequence context vector to class logits.

    Architecture:
        Linear(context_dim → hidden_dim) → LayerNorm → GELU → Dropout
        → Linear(hidden_dim → num_classes)

    The softmax is applied in OutputLayer, not here, to keep logits
    available for loss computation.

    Args:
        context_dim: Dimensionality of the backbone output vector.
        config: ClassificationHeadConfig.
    """

    def __init__(self, context_dim: int, config: ClassificationHeadConfig) -> None:
        super().__init__()
        self._config = config
        self.net = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            context: Context tensor of shape (batch, context_dim).

        Returns:
            Logit tensor of shape (batch, num_classes).
        """
        return self.net(context)


class ConfidenceHead(BaseConfidenceHead):
    """Produces a scalar confidence score for each prediction.

    The confidence head is trained jointly with the classification head.
    It captures the model's internal certainty regardless of predicted class.

    Architecture:
        Linear(context_dim → hidden_dim) → GELU → Dropout
        → Linear(hidden_dim → 1) → Sigmoid

    Args:
        context_dim: Dimensionality of the backbone output vector.
        config: ConfidenceHeadConfig.
    """

    def __init__(self, context_dim: int, config: ConfidenceHeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute confidence scores.

        Args:
            context: Context tensor of shape (batch, context_dim).

        Returns:
            Confidence tensor of shape (batch, 1), values in [0, 1].
        """
        return self.net(context)
