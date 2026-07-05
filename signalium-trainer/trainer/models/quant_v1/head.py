"""Prediction heads for QuantModelV1 — Phase 14.5 multi-task extension.

All original heads are preserved intact.  Five new heads are added:
  - ExpectedMoveHead   (regression)
  - StopLossHead       (regression)
  - TakeProfitHead     (regression)
  - HoldingTimeHead    (regression)
  - TPProbabilityHead  (binary classification → probability)

Every head shares the same context vector from the backbone encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.framework.abstractions.head import (
    BaseClassificationHead,
    BaseConfidenceHead,
    BaseRegressionHead,
)
from trainer.models.quant_v1.config import (
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    RegressionHeadConfig,
    TPProbabilityHeadConfig,
)


# ---------------------------------------------------------------------------
# Original heads (unchanged — backward compatible)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 14.5 — Regression heads
# ---------------------------------------------------------------------------


class _RegressionHead(BaseRegressionHead):
    """Internal base class for scalar regression heads.

    Architecture:
        Linear(context_dim → hidden_dim) → GELU → Dropout
        → Linear(hidden_dim → 1)

    No output activation — the loss function should handle the scale.
    Use SoftPlus or ReLU activation in the subclass if non-negativity
    is required (e.g. stop-loss distance cannot be negative).

    Args:
        context_dim: Dimensionality of the backbone output vector.
        config: RegressionHeadConfig.
        nonnegative: If True, apply Softplus to guarantee output > 0.
    """

    def __init__(
        self,
        context_dim: int,
        config: RegressionHeadConfig,
        nonnegative: bool = False,
    ) -> None:
        super().__init__()
        self._config = config
        layers: list[nn.Module] = [
            nn.Linear(context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, 1),
        ]
        if nonnegative:
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute scalar regression output.

        Args:
            context: Context tensor of shape (batch, context_dim).

        Returns:
            Scalar output of shape (batch, 1).
        """
        return self.net(context)


class ExpectedMoveHead(_RegressionHead):
    """Predicts the expected price movement size.

    Unit is determined by ``config.unit`` (pips / points / atr).
    Output is unrestricted (positive = bullish move, negative = bearish).

    Args:
        context_dim: Backbone output dimensionality.
        config: RegressionHeadConfig.
    """

    def __init__(self, context_dim: int, config: RegressionHeadConfig) -> None:
        super().__init__(context_dim, config, nonnegative=False)


class StopLossHead(_RegressionHead):
    """Predicts the recommended Stop Loss distance.

    Distance is always non-negative (Softplus activation).
    Unit is determined by ``config.unit`` (pips / atr / points).

    Args:
        context_dim: Backbone output dimensionality.
        config: RegressionHeadConfig.
    """

    def __init__(self, context_dim: int, config: RegressionHeadConfig) -> None:
        super().__init__(context_dim, config, nonnegative=True)


class TakeProfitHead(_RegressionHead):
    """Predicts the recommended Take Profit distance.

    Distance is always non-negative (Softplus activation).
    Unit is determined by ``config.unit`` (pips / atr / points).

    Args:
        context_dim: Backbone output dimensionality.
        config: RegressionHeadConfig.
    """

    def __init__(self, context_dim: int, config: RegressionHeadConfig) -> None:
        super().__init__(context_dim, config, nonnegative=True)


class HoldingTimeHead(_RegressionHead):
    """Predicts the expected trade holding duration.

    Duration is always non-negative (Softplus activation).
    Unit is determined by ``config.unit`` (minutes / bars / candles).

    Args:
        context_dim: Backbone output dimensionality.
        config: RegressionHeadConfig.
    """

    def __init__(self, context_dim: int, config: RegressionHeadConfig) -> None:
        super().__init__(context_dim, config, nonnegative=True)


# ---------------------------------------------------------------------------
# Phase 14.5 — Binary classification head (TP probability)
# ---------------------------------------------------------------------------


class TPProbabilityHead(nn.Module):
    """Predicts the probability that Take Profit is reached before Stop Loss.

    Binary classification head with Sigmoid activation — outputs a
    calibrated probability in [0, 1].

    Architecture:
        Linear(context_dim → hidden_dim) → GELU → Dropout
        → Linear(hidden_dim → 1) → Sigmoid

    Args:
        context_dim: Backbone output dimensionality.
        config: TPProbabilityHeadConfig.
    """

    def __init__(self, context_dim: int, config: TPProbabilityHeadConfig) -> None:
        super().__init__()
        self._config = config
        self.net = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute TP-before-SL probability.

        Args:
            context: Context tensor of shape (batch, context_dim).

        Returns:
            Probability tensor of shape (batch, 1), values in [0, 1].
        """
        return self.net(context)
