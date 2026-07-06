"""Temperature Scaling — post-hoc probability calibration.

A single learnable parameter T (temperature) is used to rescale all
logits before the softmax:

    p_calibrated = softmax(logits / T)

T is optimised on a held-out calibration set by minimising NLL.
T > 1 softens the distribution (reduces overconfidence).
T < 1 sharpens the distribution (corrects underconfidence).

Reference:
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
    https://arxiv.org/abs/1706.04599
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from trainer.losses.advanced import CalibrationLoss


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling calibrator.

    Usage::

        calibrator = TemperatureScaling()
        calibrator.fit(val_logits, val_labels, max_iter=500, lr=0.01)
        calibrated_probs = calibrator(model_logits)

    Args:
        init_temp: Initial temperature value. Default: 1.0 (no scaling).
    """

    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        if init_temp <= 0:
            raise ValueError(f"init_temp must be > 0, got {init_temp}")
        self.temperature = nn.Parameter(torch.tensor([init_temp], dtype=torch.float32))

    @property
    def T(self) -> float:
        """Current temperature value as a Python float."""
        return self.temperature.item()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling and return calibrated probabilities.

        Args:
            logits: Raw logits of shape (batch, num_classes).

        Returns:
            Calibrated probabilities of shape (batch, num_classes).
        """
        scaled = logits / self.temperature.clamp(min=1e-3)
        return torch.softmax(scaled, dim=-1)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits (pre-softmax).

        Args:
            logits: Raw logits of shape (batch, num_classes).

        Returns:
            Scaled logits of shape (batch, num_classes).
        """
        return logits / self.temperature.clamp(min=1e-3)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 500,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> "TemperatureScaling":
        """Optimise temperature on a held-out calibration set.

        Args:
            logits: Validation set logits of shape (N, num_classes).
            labels: Integer class labels of shape (N,).
            max_iter: Maximum L-BFGS iterations. Default: 500.
            lr: L-BFGS learning rate. Default: 0.01.
            verbose: Print temperature and loss every 50 steps.

        Returns:
            Self (fitted calibrator).
        """
        self.train()
        cal_loss = CalibrationLoss(reduction="mean")
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = cal_loss(logits, labels, self.temperature)
            loss.backward()
            return loss

        optimizer.step(_closure)
        self.eval()

        if verbose:
            with torch.no_grad():
                final_loss = cal_loss(logits, labels, self.temperature)
            print(f"TemperatureScaling: T={self.T:.4f}  NLL={final_loss.item():.4f}")

        return self
