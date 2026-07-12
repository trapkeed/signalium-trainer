"""Unit tests for MultiTaskLoss — shape validation, loss validation, per-head breakdown."""

from __future__ import annotations

import pytest
pytestmark = pytest.mark.requires_torch
import torch

from trainer.losses.multi_task import MultiTaskLoss, MultiTaskLossOutput
from trainer.models.quant_v1.config import MultiTaskLossConfig, RegressionLossType

BATCH = 8
NUM_CLASSES = 3


@pytest.fixture()
def config() -> MultiTaskLossConfig:
    return MultiTaskLossConfig(
        direction_weight=1.0,
        move_weight=0.5,
        sl_weight=0.5,
        tp_weight=0.5,
        holding_time_weight=0.3,
        confidence_weight=0.2,
        tp_probability_weight=0.5,
    )


@pytest.fixture()
def criterion(config: MultiTaskLossConfig) -> MultiTaskLoss:
    return MultiTaskLoss(config=config)


def _preds(enabled_heads: list[str] | None = None) -> dict[str, torch.Tensor]:
    all_heads = {
        "direction_logits": torch.randn(BATCH, NUM_CLASSES),
        "confidence": torch.sigmoid(torch.randn(BATCH, 1)),
        "expected_move": torch.randn(BATCH, 1),
        "stop_loss": torch.relu(torch.randn(BATCH, 1)),
        "take_profit": torch.relu(torch.randn(BATCH, 1)),
        "holding_time": torch.relu(torch.randn(BATCH, 1)),
        "tp_probability": torch.sigmoid(torch.randn(BATCH, 1)),
    }
    if enabled_heads is None:
        return all_heads
    return {k: v for k, v in all_heads.items() if k in enabled_heads}


def _labels(enabled_heads: list[str] | None = None) -> dict[str, torch.Tensor]:
    all_labels = {
        "y_direction": torch.randint(0, NUM_CLASSES, (BATCH,)),
        "y_confidence": torch.rand(BATCH),
        "y_move": torch.randn(BATCH),
        "y_sl": torch.rand(BATCH),
        "y_tp": torch.rand(BATCH),
        "y_holding_time": torch.rand(BATCH) * 60,
        "y_tp_prob": torch.randint(0, 2, (BATCH,)).float(),
    }
    if enabled_heads is None:
        return all_labels
    return {k: v for k, v in all_labels.items() if k in enabled_heads}


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


class TestLossOutputType:
    def test_returns_multi_task_loss_output(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert isinstance(out, MultiTaskLossOutput)

    def test_total_is_tensor(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert isinstance(out.total, torch.Tensor)

    def test_total_is_scalar(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert out.total.ndim == 0

    def test_total_is_finite(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert torch.isfinite(out.total)


# ---------------------------------------------------------------------------
# Per-head loss components
# ---------------------------------------------------------------------------


class TestPerHeadLosses:
    def test_direction_loss_always_computed(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert out.direction is not None
        assert torch.isfinite(out.direction)

    def test_move_loss_computed_when_head_present(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert out.expected_move is not None
        assert torch.isfinite(out.expected_move)

    def test_sl_loss_computed_when_head_present(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert out.stop_loss is not None

    def test_tp_loss_computed_when_head_present(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert out.take_profit is not None

    def test_holding_time_loss_computed_when_head_present(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert out.holding_time is not None

    def test_confidence_loss_computed_when_targets_present(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert out.confidence is not None

    def test_tp_probability_loss_computed(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        assert out.tp_probability is not None


# ---------------------------------------------------------------------------
# Partial head sets (some heads missing)
# ---------------------------------------------------------------------------


class TestPartialHeads:
    def test_direction_only_works(self, criterion: MultiTaskLoss) -> None:
        preds = {"direction_logits": torch.randn(BATCH, NUM_CLASSES)}
        labels = {"y_direction": torch.randint(0, NUM_CLASSES, (BATCH,))}
        out = criterion(preds, labels)
        assert torch.isfinite(out.total)
        assert out.expected_move is None
        assert out.stop_loss is None

    def test_missing_labels_skips_head(self, criterion: MultiTaskLoss) -> None:
        preds = _preds(["direction_logits", "expected_move"])
        labels = {"y_direction": torch.randint(0, NUM_CLASSES, (BATCH,))}
        out = criterion(preds, labels)
        assert out.expected_move is None
        assert torch.isfinite(out.total)

    def test_total_is_weighted_sum_direction_only(self, config: MultiTaskLossConfig) -> None:
        criterion = MultiTaskLoss(config=config)
        preds = {"direction_logits": torch.randn(BATCH, NUM_CLASSES)}
        labels = {"y_direction": torch.randint(0, NUM_CLASSES, (BATCH,))}
        out = criterion(preds, labels)
        expected = config.direction_weight * out.direction
        assert torch.allclose(out.total, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Loss types
# ---------------------------------------------------------------------------


class TestRegressionLossTypes:
    @pytest.mark.parametrize(
        "loss_type",
        [
            RegressionLossType.MSE,
            RegressionLossType.MAE,
            RegressionLossType.HUBER,
        ],
    )
    def test_regression_loss_types(
        self, config: MultiTaskLossConfig, loss_type: RegressionLossType
    ) -> None:
        criterion = MultiTaskLoss(
            config=config,
            regression_loss_type=loss_type,
        )
        preds = {
            "direction_logits": torch.randn(BATCH, NUM_CLASSES),
            "expected_move": torch.randn(BATCH, 1),
        }
        labels = {
            "y_direction": torch.randint(0, NUM_CLASSES, (BATCH,)),
            "y_move": torch.randn(BATCH),
        }
        out = criterion(preds, labels)
        assert torch.isfinite(out.total)
        assert out.expected_move is not None


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------


class TestLossToDict:
    def test_to_dict_contains_total(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        d = out.to_dict()
        assert "loss_total" in d

    def test_to_dict_values_are_floats(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        d = out.to_dict()
        for k, v in d.items():
            assert isinstance(v, float), f"{k}: expected float, got {type(v)}"

    def test_to_dict_contains_per_head_keys(self, criterion: MultiTaskLoss) -> None:
        out = criterion(_preds(), _labels())
        d = out.to_dict()
        expected = {
            "loss_total",
            "loss_direction",
            "loss_expected_move",
            "loss_stop_loss",
            "loss_take_profit",
            "loss_holding_time",
            "loss_tp_probability",
            "loss_confidence",
        }
        assert expected.issubset(d.keys())


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestLossGradientFlow:
    def test_total_loss_has_grad(self, criterion: MultiTaskLoss) -> None:
        preds = {k: v.requires_grad_(True) for k, v in _preds().items()}
        labels = _labels()
        out = criterion(preds, labels)
        out.total.backward()
        assert preds["direction_logits"].grad is not None

    def test_backward_does_not_error(self, criterion: MultiTaskLoss) -> None:
        preds = {k: v.requires_grad_(True) for k, v in _preds().items()}
        out = criterion(preds, _labels())
        out.total.backward()
