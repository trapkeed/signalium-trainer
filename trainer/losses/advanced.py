"""Advanced loss functions for Phase 14.6 Institutional Analytics training.

Provides:
  - QuantileLoss           — asymmetric quantile (pinball) loss
  - GaussianNLLLoss        — negative log-likelihood for Gaussian outputs
  - DistributionLoss       — composite loss for distribution parameter heads
  - DynamicLossWeighting   — gradient-based automatic loss weight adaptation
  - MultiStepBCELoss       — BCE across all steps of a multi-output binary head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Quantile (Pinball) Loss
# ---------------------------------------------------------------------------


class QuantileLoss(nn.Module):
    """Asymmetric quantile (pinball) loss for conditional quantile estimation.

    Minimising this loss at quantile q trains a model to predict the q-th
    conditional quantile of the target distribution.

    L(y, ŷ) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)

    Args:
        quantile: Target quantile ∈ (0, 1). Default 0.5 = median.
    """

    def __init__(self, quantile: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self._q = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss.

        Args:
            pred: Predicted quantile values, any shape.
            target: Ground-truth values, same shape as pred.

        Returns:
            Scalar mean quantile loss.
        """
        err = target - pred
        loss = torch.max(self._q * err, (self._q - 1.0) * err)
        return loss.mean()


# ---------------------------------------------------------------------------
# Gaussian Negative Log-Likelihood Loss
# ---------------------------------------------------------------------------


class GaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood loss.

    Trains the model to predict the mean and log-variance of a Gaussian
    distribution over the target:

        NLL = 0.5 * (log_var + (y - μ)² / exp(log_var))

    This is equivalent to ``torch.nn.GaussianNLLLoss`` but with an explicit
    log-variance input (more numerically stable than variance).

    Args:
        eps: Minimum variance floor to prevent log(0). Default: 1e-6.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self._eps = eps

    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian NLL.

        Args:
            mu:      Predicted mean, shape (batch, 1) or (batch,).
            log_var: Predicted log-variance, same shape as mu.
            target:  Ground-truth, same shape as mu.

        Returns:
            Scalar mean NLL loss.
        """
        var = log_var.exp().clamp(min=self._eps)
        nll = 0.5 * (log_var + (target - mu).pow(2) / var)
        return nll.mean()


# ---------------------------------------------------------------------------
# Distribution Head Composite Loss
# ---------------------------------------------------------------------------


class DistributionLoss(nn.Module):
    """Composite loss for distribution parameter prediction heads.

    Combines:
        - Huber loss on mean
        - Quantile (median) loss at q=0.5
        - MSE on variance (target: empirical variance of the window)
        - Huber loss on skewness
        - Huber loss on ci_lower and ci_upper (if profit distribution)

    Supports both ProfitDistributionHead (6 outputs) and LossDistributionHead
    (4 outputs) via the ``n_outputs`` parameter.

    Args:
        n_outputs: 6 for profit distribution, 4 for loss distribution.
        huber_delta: Delta for Huber loss components.
    """

    PROFIT_OUTPUTS = 6
    LOSS_OUTPUTS = 4

    def __init__(self, n_outputs: int, huber_delta: float = 1.0) -> None:
        super().__init__()
        if n_outputs not in (self.PROFIT_OUTPUTS, self.LOSS_OUTPUTS):
            raise ValueError(f"n_outputs must be 6 or 4, got {n_outputs}")
        self._n = n_outputs
        self._delta = huber_delta
        self._q_loss = QuantileLoss(quantile=0.5)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute composite distribution loss.

        Args:
            pred:   Predicted distribution parameters, shape (batch, n_outputs).
            target: Ground-truth parameters, same shape.

        Returns:
            Scalar mean combined loss.
        """
        # Mean — Huber
        loss = F.huber_loss(pred[:, 0], target[:, 0], delta=self._delta)
        # Median — Quantile loss
        loss = loss + self._q_loss(pred[:, 1], target[:, 1])
        # Variance — MSE (always index 2)
        loss = loss + F.mse_loss(pred[:, 2], target[:, 2])
        if self._n == self.PROFIT_OUTPUTS:
            # Skewness — Huber (index 3)
            loss = loss + F.huber_loss(pred[:, 3], target[:, 3], delta=self._delta)
            # CI lower/upper — Huber (indices 4, 5)
            loss = loss + F.huber_loss(pred[:, 4], target[:, 4], delta=self._delta)
            loss = loss + F.huber_loss(pred[:, 5], target[:, 5], delta=self._delta)
        else:
            # Tail risk — Huber (index 3)
            loss = loss + F.huber_loss(pred[:, 3], target[:, 3], delta=self._delta)
        return loss / self._n


# ---------------------------------------------------------------------------
# Multi-Step BCE Loss (for TradeSurvivalHead, TradeDecayHead)
# ---------------------------------------------------------------------------


class MultiStepBCELoss(nn.Module):
    """Binary cross-entropy averaged across all steps of a multi-output head.

    Used for heads that output a vector of probabilities, each supervised
    with an independent binary label (e.g. TradeSurvivalHead outputs
    P(survive at bar 5), P(survive at bar 10), P(survive at bar 20)).

    Args:
        n_steps: Number of probability outputs in the prediction vector.
    """

    def __init__(self, n_steps: int) -> None:
        super().__init__()
        self._n = n_steps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean BCE across all step outputs.

        Args:
            pred:   Predicted probabilities, shape (batch, n_steps), ∈ [0,1].
            target: Binary ground-truth labels, same shape.

        Returns:
            Scalar mean BCE loss.
        """
        return F.binary_cross_entropy(pred, target.float())


# ---------------------------------------------------------------------------
# Dynamic Loss Weighting (Gradient Normalisation / Uncertainty)
# ---------------------------------------------------------------------------


class DynamicLossWeighting(nn.Module):
    """Learnable per-head loss weights via homoscedastic uncertainty weighting.

    Implements the multi-task uncertainty weighting from Kendall et al. (2018):
        L_i_weighted = (1 / (2 * σ_i²)) * L_i + log(σ_i)

    where σ_i (the task noise) is a trainable parameter.  This allows the
    network to automatically balance task losses during training.

    Args:
        n_tasks: Number of tasks to weight.
        init_log_var: Initial log(σ²) for all tasks. Default 0.0 (σ=1).
    """

    def __init__(self, n_tasks: int, init_log_var: float = 0.0) -> None:
        super().__init__()
        self._n = n_tasks
        self.log_vars = nn.Parameter(torch.full((n_tasks,), init_log_var, dtype=torch.float32))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute the total weighted loss.

        Args:
            losses: List of per-task scalar loss tensors (length == n_tasks).

        Returns:
            Scalar weighted total loss.

        Raises:
            ValueError: If len(losses) != n_tasks.
        """
        if len(losses) != self._n:
            raise ValueError(f"Expected {self._n} task losses, got {len(losses)}")
        total = torch.zeros(1, device=self.log_vars.device)
        for i, loss_i in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss_i + self.log_vars[i]
        return total.squeeze()

    @property
    def task_weights(self) -> torch.Tensor:
        """Current effective task weights (precision values)."""
        return torch.exp(-self.log_vars).detach()


# ---------------------------------------------------------------------------
# Calibration Loss (Temperature Scaling auxiliary loss)
# ---------------------------------------------------------------------------


class CalibrationLoss(nn.Module):
    """NLL-based calibration loss for temperature scaling.

    After training, this loss is minimised (on a held-out set) to find
    the optimal temperature T that calibrates the softmax probabilities.

    L_cal = -sum_i [y_i * log(softmax(logits / T)_i)]

    Args:
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temperature-scaled NLL calibration loss.

        Args:
            logits:      Raw logits (batch, num_classes).
            targets:     Integer class labels (batch,).
            temperature: Scalar temperature parameter (must be > 0).

        Returns:
            Scalar NLL loss under the scaled distribution.
        """
        scaled = logits / temperature.clamp(min=1e-3)
        return self._ce(scaled, targets)
