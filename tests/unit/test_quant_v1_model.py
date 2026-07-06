"""Unit tests for QuantModelV1 — shape, forward pass, gradient flow."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trainer.config.schema import TrainerConfig
from trainer.models.quant_v1.config import (
    BackboneConfig,
    BackboneType,
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    InputEncoderConfig,
    QuantV1Config,
)
from trainer.models.quant_v1.model import QuantModelV1

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 4
SEQ_LEN = 60
INPUT_DIM = 32
HIDDEN = 64
NUM_CLASSES = 3


@pytest.fixture()
def small_config() -> QuantV1Config:
    """Minimal QuantV1Config for fast unit tests."""
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
            num_classes=NUM_CLASSES, hidden_dim=32, dropout=0.0
        ),
        confidence_head=ConfidenceHeadConfig(hidden_dim=16, dropout=0.0),
    )


@pytest.fixture()
def model(small_config: QuantV1Config) -> QuantModelV1:
    return QuantModelV1(trainer_config=TrainerConfig(), model_config=small_config)


@pytest.fixture()
def x() -> torch.Tensor:
    """Random input batch of shape (batch, seq_len, input_dim)."""
    return torch.randn(BATCH, SEQ_LEN, INPUT_DIM)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestOutputShapes:
    def test_logits_shape(self, model: QuantModelV1, x: torch.Tensor) -> None:
        logits = model(x)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_predict_probabilities_shape(self, model: QuantModelV1, x: torch.Tensor) -> None:
        out = model.predict(x)
        assert out.probabilities.shape == (BATCH, NUM_CLASSES)

    def test_predict_confidence_shape(self, model: QuantModelV1, x: torch.Tensor) -> None:
        out = model.predict(x)
        assert out.confidence.shape == (BATCH, 1)

    def test_predict_predicted_class_shape(self, model: QuantModelV1, x: torch.Tensor) -> None:
        out = model.predict(x)
        assert out.predicted_class.shape == (BATCH,)

    def test_buy_sell_notrade_shapes(self, model: QuantModelV1, x: torch.Tensor) -> None:
        out = model.predict(x)
        for attr in ("buy_prob", "sell_prob", "no_trade_prob"):
            assert getattr(out, attr).shape == (BATCH,), f"{attr} wrong shape"

    def test_probabilities_sum_to_one(self, model: QuantModelV1, x: torch.Tensor) -> None:
        out = model.predict(x)
        sums = out.probabilities.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_confidence_in_zero_one(self, model: QuantModelV1, x: torch.Tensor) -> None:
        out = model.predict(x)
        assert (out.confidence >= 0).all()
        assert (out.confidence <= 1).all()

    def test_predicted_class_valid_range(self, model: QuantModelV1, x: torch.Tensor) -> None:
        out = model.predict(x)
        assert (out.predicted_class >= 0).all()
        assert (out.predicted_class < NUM_CLASSES).all()


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradients_flow_to_all_parameters(self, model: QuantModelV1, x: torch.Tensor) -> None:
        model.train()
        logits = model(x)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()

        params_without_grad = [
            name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        assert not params_without_grad, f"No gradient for: {params_without_grad}"

    def test_gradients_non_zero(self, model: QuantModelV1, x: torch.Tensor) -> None:
        model.train()
        logits = model(x)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()

        zero_grads = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() == 0
        ]
        # Some zero grads are acceptable (e.g. bias on unused paths), not all
        assert len(zero_grads) < len(list(model.parameters())) // 2

    def test_loss_is_finite(self, model: QuantModelV1, x: torch.Tensor) -> None:
        model.train()
        logits = model(x)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = nn.CrossEntropyLoss()(logits, targets)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Architecture metadata
# ---------------------------------------------------------------------------


class TestArchitectureMetadata:
    def test_feature_dim(self, model: QuantModelV1) -> None:
        assert model.feature_dim == INPUT_DIM

    def test_num_classes(self, model: QuantModelV1) -> None:
        assert model.num_classes == NUM_CLASSES

    def test_label_names(self, model: QuantModelV1) -> None:
        assert model.label_names == ["BUY", "SELL", "NO_TRADE"]

    def test_architecture_name_contains_backbone(self, model: QuantModelV1) -> None:
        assert "LSTM" in model.architecture_name

    def test_architecture_summary_keys(self, model: QuantModelV1) -> None:
        summary = model.get_architecture_summary()
        required = {
            "architecture",
            "version",
            "input_dim",
            "num_classes",
            "backbone_type",
            "total",
            "trainable",
        }
        assert required.issubset(summary.keys())

    def test_param_count_positive(self, model: QuantModelV1) -> None:
        counts = model.get_param_count()
        assert counts["trainable"] > 0
        assert counts["total"] >= counts["trainable"]

    def test_registered_in_registry(self) -> None:
        from trainer.registry.registry import ModelRegistry

        assert "quant_v1" in ModelRegistry.list_models()


# ---------------------------------------------------------------------------
# GRU backbone variant
# ---------------------------------------------------------------------------


class TestGRUBackbone:
    def test_gru_forward_shape(self, small_config: QuantV1Config, x: torch.Tensor) -> None:
        cfg = QuantV1Config(
            encoder=small_config.encoder,
            backbone=BackboneConfig(
                type=BackboneType.GRU,
                hidden_size=HIDDEN,
                num_layers=1,
                dropout=0.0,
                bidirectional=False,
                window_length=SEQ_LEN,
            ),
            classification_head=small_config.classification_head,
            confidence_head=small_config.confidence_head,
        )
        model = QuantModelV1(TrainerConfig(), cfg)
        logits = model(x)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_bidirectional_lstm_shape(self, small_config: QuantV1Config, x: torch.Tensor) -> None:
        cfg = QuantV1Config(
            encoder=small_config.encoder,
            backbone=BackboneConfig(
                type=BackboneType.LSTM,
                hidden_size=HIDDEN,
                num_layers=1,
                dropout=0.0,
                bidirectional=True,
                window_length=SEQ_LEN,
            ),
            classification_head=small_config.classification_head,
            confidence_head=small_config.confidence_head,
        )
        model = QuantModelV1(TrainerConfig(), cfg)
        logits = model(x)
        assert logits.shape == (BATCH, NUM_CLASSES)
