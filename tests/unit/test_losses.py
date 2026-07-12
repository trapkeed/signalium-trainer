"""Unit tests for FocalLoss and WeightedCrossEntropyLoss."""

from __future__ import annotations

import pytest
pytestmark = pytest.mark.requires_torch
import torch
import torch.nn as nn

from trainer.losses.factory import build_loss
from trainer.losses.focal import FocalLoss
from trainer.losses.weighted_ce import WeightedCrossEntropyLoss
from trainer.models.quant_v1.config import LossConfig, LossType

BATCH = 8
NUM_CLASSES = 3


@pytest.fixture()
def logits() -> torch.Tensor:
    return torch.randn(BATCH, NUM_CLASSES)


@pytest.fixture()
def targets() -> torch.Tensor:
    return torch.randint(0, NUM_CLASSES, (BATCH,))


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------


class TestFocalLoss:
    def test_output_is_scalar(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        loss = FocalLoss(gamma=2.0)
        out = loss(logits, targets)
        assert out.shape == ()

    def test_output_is_finite(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        loss = FocalLoss(gamma=2.0)
        assert torch.isfinite(loss(logits, targets))

    def test_output_non_negative(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        loss = FocalLoss(gamma=2.0)
        assert loss(logits, targets) >= 0

    def test_gamma_zero_matches_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """When gamma=0, FocalLoss should approximate CrossEntropyLoss."""
        fl = FocalLoss(gamma=0.0)
        ce = nn.CrossEntropyLoss()
        assert torch.allclose(fl(logits, targets), ce(logits, targets), atol=1e-5)

    def test_with_alpha_weights(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        alpha = torch.tensor([0.25, 0.5, 0.25])
        loss = FocalLoss(gamma=2.0, alpha=alpha)
        out = loss(logits, targets)
        assert torch.isfinite(out)

    def test_reduction_none_shape(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        loss = FocalLoss(gamma=2.0, reduction="none")
        out = loss(logits, targets)
        assert out.shape == (BATCH,)

    def test_invalid_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            FocalLoss(gamma=-1.0)

    def test_invalid_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="reduction"):
            FocalLoss(reduction="invalid")

    def test_gradients_flow(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        logits_leaf = logits.detach().requires_grad_(True)
        loss = FocalLoss(gamma=2.0)(logits_leaf, targets)
        loss.backward()
        assert logits_leaf.grad is not None


# ---------------------------------------------------------------------------
# WeightedCrossEntropyLoss
# ---------------------------------------------------------------------------


class TestWeightedCrossEntropyLoss:
    def test_output_scalar(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        loss = WeightedCrossEntropyLoss()
        assert loss(logits, targets).shape == ()

    def test_output_finite(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        loss = WeightedCrossEntropyLoss()
        assert torch.isfinite(loss(logits, targets))

    def test_with_class_weights(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        weights = torch.tensor([1.0, 2.0, 0.5])
        loss = WeightedCrossEntropyLoss(class_weights=weights)
        out = loss(logits, targets)
        assert torch.isfinite(out)

    def test_matches_nn_cross_entropy_no_weights(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        custom = WeightedCrossEntropyLoss()
        standard = nn.CrossEntropyLoss()
        assert torch.allclose(custom(logits, targets), standard(logits, targets), atol=1e-5)

    def test_reduction_none(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        loss = WeightedCrossEntropyLoss(reduction="none")
        out = loss(logits, targets)
        assert out.shape == (BATCH,)

    def test_update_weights(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        loss = WeightedCrossEntropyLoss(class_weights=torch.ones(NUM_CLASSES))
        loss.update_weights(torch.tensor([1.0, 2.0, 0.5]))
        assert torch.isfinite(loss(logits, targets))


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------


class TestLossFactory:
    def test_builds_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        cfg = LossConfig(type=LossType.CROSS_ENTROPY)
        criterion = build_loss(cfg)
        assert torch.isfinite(criterion(logits, targets))

    def test_builds_focal(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        cfg = LossConfig(type=LossType.FOCAL, focal_gamma=2.0)
        criterion = build_loss(cfg)
        assert isinstance(criterion, FocalLoss)
        assert torch.isfinite(criterion(logits, targets))

    def test_builds_weighted_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        cfg = LossConfig(type=LossType.WEIGHTED_CROSS_ENTROPY, class_weights=[1.0, 1.5, 0.5])
        criterion = build_loss(cfg)
        assert isinstance(criterion, WeightedCrossEntropyLoss)
        assert torch.isfinite(criterion(logits, targets))

    def test_focal_with_alpha(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        cfg = LossConfig(type=LossType.FOCAL, focal_gamma=2.0, focal_alpha=[0.25, 0.5, 0.25])
        criterion = build_loss(cfg)
        assert torch.isfinite(criterion(logits, targets))
