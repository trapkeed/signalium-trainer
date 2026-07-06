"""Gradient-based feature attribution methods for QuantModelV1.

Implements:
  IntegratedGradients  — Sundararajan et al. (2017) path-integral attribution.
  GradientAttribution  — Simple gradient × input (saliency maps).
  SmoothGrad           — Smilkov et al. (2017) noise-averaged gradients.

Design invariants:
  - NEVER modifies model weights or prediction outputs.
  - Accepts the model in any mode (train / eval) and restores it afterward.
  - Returns AttributionResult with per-feature, per-timestep importance scores.
  - All gradient computation happens inside _grad_mode context manager.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AttributionResult:
    """Per-feature attribution scores from a gradient-based method.

    Attributes:
        scores: Tensor of shape ``(batch, seq_len, n_features)`` — importance
            per feature per timestep.  Positive = promotes target class,
            Negative = suppresses it.
        method: Name of the attribution method used.
        target_class: Class index for which attribution was computed.
        feature_names: Optional list of feature column names.
        aggregated: Per-feature scores reduced over the sequence dimension,
            shape ``(batch, n_features)``.  Populated by :meth:`aggregate`.
    """

    scores: torch.Tensor
    method: str
    target_class: int
    feature_names: list[str] = field(default_factory=list)
    aggregated: torch.Tensor | None = None

    def aggregate(self, reduction: str = "mean") -> AttributionResult:
        """Reduce the sequence (time) dimension.

        Args:
            reduction: ``"mean"`` | ``"sum"`` | ``"abs_mean"``.

        Returns:
            Self — sets :attr:`aggregated` in place and returns self for
            chaining.
        """
        if reduction == "mean":
            self.aggregated = self.scores.mean(dim=1)
        elif reduction == "sum":
            self.aggregated = self.scores.sum(dim=1)
        elif reduction == "abs_mean":
            self.aggregated = self.scores.abs().mean(dim=1)
        else:
            raise ValueError(f"Unknown reduction: {reduction!r}")
        return self

    def top_features(
        self,
        k: int = 10,
        sample_idx: int = 0,
        absolute: bool = True,
    ) -> list[tuple[str, float]]:
        """Return top-k features by attribution magnitude.

        Args:
            k: Number of top features to return.
            sample_idx: Batch index to use (default 0).
            absolute: Sort by absolute value when True (default).

        Returns:
            List of ``(feature_name, score)`` sorted descending by magnitude.
        """
        if self.aggregated is None:
            self.aggregate()
        assert self.aggregated is not None
        scores_1d = self.aggregated[sample_idx].detach().cpu()
        rank_key = scores_1d.abs() if absolute else scores_1d
        n = min(k, len(scores_1d))
        top_idx = rank_key.topk(n).indices.tolist()
        names = (
            self.feature_names if self.feature_names else [f"f{i}" for i in range(len(scores_1d))]
        )
        return [(names[i], float(scores_1d[i])) for i in top_idx]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (shapes, not raw tensors)."""
        top = self.top_features(k=20) if self.aggregated is not None else []
        return {
            "method": self.method,
            "target_class": self.target_class,
            "scores_shape": list(self.scores.shape),
            "top_features": top,
        }


# ---------------------------------------------------------------------------
# Gradient context helper
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _grad_mode(model: nn.Module):
    """Context manager: eval mode + enable_grad; restores training state after."""
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            yield
    finally:
        if was_training:
            model.train()


def _target_logit(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    """Extract a scalar target logit from a ``(batch, n_classes)`` tensor."""
    return logits[:, target_class].sum()


# ---------------------------------------------------------------------------
# Attribution methods
# ---------------------------------------------------------------------------


class IntegratedGradients:
    """Integrated Gradients (Sundararajan et al., 2017).

    Computes the path integral of gradients from a baseline input to the
    actual input, yielding attributions that satisfy the *Completeness* axiom:

        sum(attribution) ≈ model(input)[target] - model(baseline)[target]

    Reference: https://arxiv.org/abs/1703.01365

    Usage::

        ig = IntegratedGradients()
        result = ig.attribute(model, x, target_class=0, steps=50)
        top = result.top_features(k=10)
    """

    def attribute(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int = 0,
        steps: int = 50,
        baseline: torch.Tensor | None = None,
        feature_names: list[str] | None = None,
    ) -> AttributionResult:
        """Compute Integrated Gradients attributions.

        Args:
            model: Model implementing ``forward(tensor) → logits (batch, C)``.
            input_tensor: ``(batch, seq_len, n_features)`` input.
            target_class: Class index to explain.
            steps: Number of Riemann approximation steps (50 is a good
                default; higher = more accurate but slower).
            baseline: ``(batch, seq_len, n_features)`` reference baseline.
                Defaults to all-zeros.
            feature_names: Optional column names for readability.

        Returns:
            ``AttributionResult`` with shape ``(batch, seq_len, n_features)``.
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        delta = input_tensor - baseline
        accumulated = torch.zeros_like(input_tensor)

        with _grad_mode(model):
            for step in range(steps):
                alpha = (step + 1) / steps
                interp = (baseline + alpha * delta).detach().requires_grad_(True)
                logits = model(interp)
                score = _target_logit(logits, target_class)
                grads = torch.autograd.grad(score, interp)[0]
                accumulated += grads.detach()

        ig_scores = (accumulated / steps) * delta.detach()
        return AttributionResult(
            scores=ig_scores,
            method="integrated_gradients",
            target_class=target_class,
            feature_names=feature_names or [],
        )


class GradientAttribution:
    """Gradient × Input attribution (saliency maps).

    Computes the element-wise product of the gradient of the target class
    score with respect to the input and the input itself.  Fast but may
    under-capture non-linearities compared to Integrated Gradients.

    Usage::

        ga = GradientAttribution()
        result = ga.attribute(model, x, target_class=0)
    """

    def attribute(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int = 0,
        feature_names: list[str] | None = None,
    ) -> AttributionResult:
        """Compute Gradient × Input attributions.

        Args:
            model: Model implementing ``forward(tensor) → logits``.
            input_tensor: ``(batch, seq_len, n_features)`` input.
            target_class: Class index to explain.
            feature_names: Optional column names.

        Returns:
            ``AttributionResult`` with shape ``(batch, seq_len, n_features)``.
        """
        with _grad_mode(model):
            inp = input_tensor.detach().requires_grad_(True)
            logits = model(inp)
            score = _target_logit(logits, target_class)
            grads = torch.autograd.grad(score, inp)[0]

        scores = (grads * inp).detach()
        return AttributionResult(
            scores=scores,
            method="gradient_attribution",
            target_class=target_class,
            feature_names=feature_names or [],
        )


class SmoothGrad:
    """SmoothGrad (Smilkov et al., 2017) noise-averaged gradient attributions.

    Reduces noise in gradient-based saliency maps by averaging over
    ``n_samples`` copies of the input with added Gaussian noise.

    Reference: https://arxiv.org/abs/1706.03825

    Args:
        n_samples: Number of noisy samples to average (30–50 is typical).
        noise_std: Standard deviation of additive noise as a fraction of the
            input range (e.g. 0.15 = 15 % of max - min).

    Usage::

        sg = SmoothGrad(n_samples=30, noise_std=0.10)
        result = sg.attribute(model, x, target_class=0)
    """

    def __init__(self, n_samples: int = 50, noise_std: float = 0.15) -> None:
        self._n_samples = n_samples
        self._noise_std = noise_std

    def attribute(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int = 0,
        feature_names: list[str] | None = None,
    ) -> AttributionResult:
        """Compute SmoothGrad attributions.

        Args:
            model: Model implementing ``forward(tensor) → logits``.
            input_tensor: ``(batch, seq_len, n_features)`` input.
            target_class: Class index to explain.
            feature_names: Optional column names.

        Returns:
            ``AttributionResult`` with averaged, noise-reduced gradients.
        """
        accumulated = torch.zeros_like(input_tensor)
        inp_range = float((input_tensor.max() - input_tensor.min()).item())
        std = self._noise_std * inp_range

        with _grad_mode(model):
            for _ in range(self._n_samples):
                noise = torch.randn_like(input_tensor) * std
                noisy = (input_tensor + noise).detach().requires_grad_(True)
                logits = model(noisy)
                score = _target_logit(logits, target_class)
                grads = torch.autograd.grad(score, noisy)[0]
                accumulated += grads.detach()

        scores = accumulated / self._n_samples
        return AttributionResult(
            scores=scores,
            method="smooth_grad",
            target_class=target_class,
            feature_names=feature_names or [],
        )
