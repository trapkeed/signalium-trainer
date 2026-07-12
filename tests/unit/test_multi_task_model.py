"""Unit tests for QuantModelV1 Phase 14.5 — multi-task heads.

Covers:
  - Output shapes for all enabled heads
  - forward() backward compatibility (returns logits only)
  - predict() returns MultiTaskQuantModelOutput
  - forward_multi_task() returns correct dict keys
  - Softplus non-negativity for SL / TP / HoldingTime heads
  - Sigmoid bounds for confidence and tp_probability heads
  - Gradient flow through all heads simultaneously
  - Architecture summary includes enabled_heads
  - Model with all heads disabled behaves like original V1
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_torch
import torch
import torch.nn as nn

from trainer.config.schema import TrainerConfig
from trainer.models.quant_v1.config import (
    BackboneConfig,
    BackboneType,
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    InputEncoderConfig,
    MultiTaskLossConfig,
    QuantV1Config,
    RegressionHeadConfig,
    RegressionUnit,
    TPProbabilityHeadConfig,
)
from trainer.models.quant_v1.model import QuantModelV1
from trainer.models.quant_v1.output import MultiTaskQuantModelOutput, QuantModelOutput

BATCH = 4
SEQ_LEN = 30
INPUT_DIM = 16
HIDDEN = 32
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_config() -> QuantV1Config:
    return QuantV1Config(
        encoder=InputEncoderConfig(input_dim=INPUT_DIM, hidden_dim=HIDDEN, projection_dropout=0.0),
        backbone=BackboneConfig(
            type=BackboneType.LSTM,
            hidden_size=HIDDEN,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            window_length=SEQ_LEN,
        ),
        classification_head=ClassificationHeadConfig(
            num_classes=NUM_CLASSES, hidden_dim=16, dropout=0.0
        ),
        confidence_head=ConfidenceHeadConfig(hidden_dim=16, dropout=0.0),
    )


def _all_heads_config() -> QuantV1Config:
    cfg = _base_config()
    head_cfg = RegressionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.expected_move_head = RegressionHeadConfig(
        enabled=True, hidden_dim=16, dropout=0.0, unit=RegressionUnit.PIPS
    )
    cfg.stop_loss_head = RegressionHeadConfig(
        enabled=True, hidden_dim=16, dropout=0.0, unit=RegressionUnit.PIPS
    )
    cfg.take_profit_head = RegressionHeadConfig(
        enabled=True, hidden_dim=16, dropout=0.0, unit=RegressionUnit.PIPS
    )
    cfg.holding_time_head = RegressionHeadConfig(
        enabled=True, hidden_dim=16, dropout=0.0, unit=RegressionUnit.MINUTES
    )
    cfg.tp_probability_head = TPProbabilityHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    return cfg


@pytest.fixture()
def x() -> torch.Tensor:
    return torch.randn(BATCH, SEQ_LEN, INPUT_DIM)


@pytest.fixture()
def base_model() -> QuantModelV1:
    """Original V1 config — all new heads disabled."""
    return QuantModelV1(TrainerConfig(), _base_config())


@pytest.fixture()
def multi_task_model() -> QuantModelV1:
    """All Phase 14.5 heads enabled."""
    return QuantModelV1(TrainerConfig(), _all_heads_config())


# ---------------------------------------------------------------------------
# Backward compatibility: forward() still returns logits
# ---------------------------------------------------------------------------


class TestForwardBackwardCompat:
    def test_forward_returns_tensor(self, base_model: QuantModelV1, x: torch.Tensor) -> None:
        out = base_model(x)
        assert isinstance(out, torch.Tensor)

    def test_forward_logits_shape(self, base_model: QuantModelV1, x: torch.Tensor) -> None:
        out = base_model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_forward_multi_task_logits_shape(
        self, multi_task_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        out = multi_task_model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_ce_loss_works_on_forward_output(
        self, multi_task_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        logits = multi_task_model(x)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = nn.CrossEntropyLoss()(logits, targets)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# predict() returns MultiTaskQuantModelOutput
# ---------------------------------------------------------------------------


class TestPredictOutput:
    def test_predict_returns_multi_task_output(
        self, multi_task_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        out = multi_task_model.predict(x)
        assert isinstance(out, MultiTaskQuantModelOutput)

    def test_probabilities_shape(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.predict(x)
        assert out.probabilities.shape == (BATCH, NUM_CLASSES)

    def test_probabilities_sum_to_one(
        self, multi_task_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        out = multi_task_model.predict(x)
        sums = out.probabilities.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_confidence_shape_and_range(
        self, multi_task_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        out = multi_task_model.predict(x)
        assert out.confidence.shape == (BATCH, 1)
        assert (out.confidence >= 0).all()
        assert (out.confidence <= 1).all()

    def test_expected_move_shape(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.predict(x)
        assert out.expected_move is not None
        assert out.expected_move.shape == (BATCH, 1)

    def test_stop_loss_nonnegative(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.predict(x)
        assert out.stop_loss is not None
        assert (out.stop_loss >= 0).all(), "SL must be non-negative"

    def test_take_profit_nonnegative(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.predict(x)
        assert out.take_profit is not None
        assert (out.take_profit >= 0).all(), "TP must be non-negative"

    def test_holding_time_nonnegative(
        self, multi_task_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        out = multi_task_model.predict(x)
        assert out.holding_time is not None
        assert (out.holding_time >= 0).all(), "Holding time must be non-negative"

    def test_tp_probability_range(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.predict(x)
        assert out.tp_probability is not None
        assert (out.tp_probability >= 0).all()
        assert (out.tp_probability <= 1).all()

    def test_to_base_output(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.predict(x)
        base = out.to_base_output()
        assert isinstance(base, QuantModelOutput)
        assert base.probabilities.shape == (BATCH, NUM_CLASSES)

    def test_prediction_timestamp_present(
        self, multi_task_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        out = multi_task_model.predict(x)
        assert out.prediction_timestamp is not None
        assert "T" in out.prediction_timestamp

    def test_model_version_present(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.predict(x)
        assert out.model_version is not None


# ---------------------------------------------------------------------------
# forward_multi_task() dict output
# ---------------------------------------------------------------------------


class TestForwardMultiTask:
    def test_always_present_keys(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.forward_multi_task(x)
        assert "direction_logits" in out
        assert "confidence" in out

    def test_optional_head_keys_present_when_enabled(
        self, multi_task_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        out = multi_task_model.forward_multi_task(x)
        for key in ("expected_move", "stop_loss", "take_profit", "holding_time", "tp_probability"):
            assert key in out, f"Expected key '{key}' missing from forward_multi_task() output"

    def test_optional_head_keys_absent_when_disabled(
        self, base_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        out = base_model.forward_multi_task(x)
        for key in ("expected_move", "stop_loss", "take_profit", "holding_time", "tp_probability"):
            assert key not in out, f"Unexpected key '{key}' in base model output"

    def test_direction_logits_shape(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.forward_multi_task(x)
        assert out["direction_logits"].shape == (BATCH, NUM_CLASSES)

    def test_all_regression_shapes(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.forward_multi_task(x)
        for key in ("expected_move", "stop_loss", "take_profit", "holding_time"):
            assert out[key].shape == (BATCH, 1), f"{key}: wrong shape"

    def test_tp_probability_shape(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        out = multi_task_model.forward_multi_task(x)
        assert out["tp_probability"].shape == (BATCH, 1)


# ---------------------------------------------------------------------------
# Gradient flow through all heads
# ---------------------------------------------------------------------------


class TestMultiTaskGradientFlow:
    def test_gradients_flow_to_all_parameters(
        self, multi_task_model: QuantModelV1, x: torch.Tensor
    ) -> None:
        from trainer.losses.multi_task import MultiTaskLoss

        multi_task_model.train()
        criterion = MultiTaskLoss(config=MultiTaskLossConfig())

        labels = {
            "y_direction": torch.randint(0, NUM_CLASSES, (BATCH,)),
            "y_move": torch.randn(BATCH),
            "y_sl": torch.rand(BATCH),
            "y_tp": torch.rand(BATCH),
            "y_holding_time": torch.rand(BATCH) * 60,
            "y_tp_prob": torch.rand(BATCH),
            "y_confidence": torch.rand(BATCH),
        }

        preds = multi_task_model.forward_multi_task(x)
        loss_out = criterion(preds, labels)
        loss_out.total.backward()

        no_grad = [
            name
            for name, p in multi_task_model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not no_grad, f"Parameters missing gradients: {no_grad}"

    def test_total_loss_is_finite(self, multi_task_model: QuantModelV1, x: torch.Tensor) -> None:
        from trainer.losses.multi_task import MultiTaskLoss

        multi_task_model.train()
        criterion = MultiTaskLoss(config=MultiTaskLossConfig())
        labels = {
            "y_direction": torch.randint(0, NUM_CLASSES, (BATCH,)),
            "y_move": torch.randn(BATCH),
            "y_sl": torch.rand(BATCH),
            "y_tp": torch.rand(BATCH),
            "y_holding_time": torch.rand(BATCH) * 60,
            "y_tp_prob": torch.rand(BATCH),
            "y_confidence": torch.rand(BATCH),
        }
        preds = multi_task_model.forward_multi_task(x)
        loss_out = criterion(preds, labels)
        assert torch.isfinite(loss_out.total)


# ---------------------------------------------------------------------------
# Architecture metadata
# ---------------------------------------------------------------------------


class TestMultiTaskArchitectureMetadata:
    def test_enabled_heads_all_present(self, multi_task_model: QuantModelV1) -> None:
        heads = multi_task_model.enabled_heads
        for expected in (
            "direction",
            "confidence",
            "expected_move",
            "stop_loss",
            "take_profit",
            "holding_time",
            "tp_probability",
        ):
            assert expected in heads

    def test_base_model_enabled_heads(self, base_model: QuantModelV1) -> None:
        heads = base_model.enabled_heads
        assert "direction" in heads
        assert "confidence" in heads
        assert "expected_move" not in heads

    def test_architecture_summary_includes_enabled_heads(
        self, multi_task_model: QuantModelV1
    ) -> None:
        summary = multi_task_model.get_architecture_summary()
        assert "enabled_heads" in summary

    def test_export_metadata_structure(self, multi_task_model: QuantModelV1) -> None:
        meta = multi_task_model.get_export_metadata()
        assert "prediction_schema" in meta
        assert "supported_outputs" in meta
        assert "output_version" in meta
        assert "loss_config" in meta
