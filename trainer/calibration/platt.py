"""Platt Scaling — post-hoc logistic regression calibration.

Fits a logistic regression model (one per class vs. rest) on top of the
raw model logits to produce calibrated probabilities.

Platt scaling is more flexible than temperature scaling — it learns an
affine transformation of the logits per class, rather than a single
global scale factor.

Reference:
    Platt (1999) "Probabilistic Outputs for SVMs and Comparisons to
    Regularised Likelihood Methods"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class PlattScaling(nn.Module):
    """Post-hoc Platt scaling calibrator.

    Learns a per-class affine transformation W, b such that:

        p_calibrated = softmax(logits @ W + b)

    where W is diagonal (scale per class) and b is a per-class bias.

    Usage::

        calibrator = PlattScaling(num_classes=3)
        calibrator.fit(val_logits, val_labels)
        calibrated_probs = calibrator(model_logits)

    Args:
        num_classes: Number of output classes.
        init_scale: Initial scale values (default 1.0 = identity).
        init_bias: Initial bias values (default 0.0).
    """

    def __init__(
        self,
        num_classes: int,
        init_scale: float = 1.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self.scale = nn.Parameter(torch.full((num_classes,), init_scale, dtype=torch.float32))
        self.bias = nn.Parameter(torch.full((num_classes,), init_bias, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling and return calibrated probabilities.

        Args:
            logits: Raw logits of shape (batch, num_classes).

        Returns:
            Calibrated probabilities of shape (batch, num_classes).
        """
        scaled = logits * self.scale.unsqueeze(0) + self.bias.unsqueeze(0)
        return torch.softmax(scaled, dim=-1)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 1000,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> PlattScaling:
        """Optimise scale and bias on a held-out calibration set.

        Args:
            logits: Validation set logits of shape (N, num_classes).
            labels: Integer class labels of shape (N,).
            max_iter: Adam iterations. Default: 1000.
            lr: Adam learning rate. Default: 0.01.
            verbose: Print loss at the end.

        Returns:
            Self (fitted calibrator).
        """
        self.train()
        ce = nn.CrossEntropyLoss()
        optimizer = optim.Adam([self.scale, self.bias], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled = logits * self.scale.unsqueeze(0) + self.bias.unsqueeze(0)
            loss = ce(scaled, labels)
            loss.backward()
            optimizer.step()

        self.eval()

        if verbose:
            with torch.no_grad():
                scaled = logits * self.scale.unsqueeze(0) + self.bias.unsqueeze(0)
                final_loss = ce(scaled, labels)
            print(f"PlattScaling: loss={final_loss.item():.4f}")

        return self
