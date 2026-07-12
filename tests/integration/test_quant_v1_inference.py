"""Integration tests — full inference pipeline for QuantModelV1."""

from __future__ import annotations

import pytest
pytestmark = pytest.mark.requires_torch
import torch
import torch.nn as nn

from trainer.config.schema import TrainerConfig
from trainer.losses.factory import build_loss
from trainer.models.quant_v1.config import (
    BackboneConfig,
    BackboneType,
    BalancingConfig,
    BalancingStrategy,
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    InputEncoderConfig,
    LossConfig,
    LossType,
    QuantV1Config,
)
from trainer.models.quant_v1.model import QuantModelV1
from trainer.sampling.strategy import BalancingStrategyBuilder, compute_class_weights

BATCH = 16
SEQ_LEN = 60
INPUT_DIM = 64
NUM_CLASSES = 3


@pytest.fixture(scope="module")
def model() -> QuantModelV1:
    cfg = QuantV1Config(
        encoder=InputEncoderConfig(input_dim=INPUT_DIM, hidden_dim=128, projection_dropout=0.0),
        backbone=BackboneConfig(
            type=BackboneType.LSTM,
            hidden_size=128,
            num_layers=2,
            dropout=0.0,
            window_length=SEQ_LEN,
        ),
        classification_head=ClassificationHeadConfig(num_classes=NUM_CLASSES, dropout=0.0),
        confidence_head=ConfidenceHeadConfig(dropout=0.0),
    )
    m = QuantModelV1(TrainerConfig(), cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch_input() -> torch.Tensor:
    return torch.randn(BATCH, SEQ_LEN, INPUT_DIM)


class TestInferencePipeline:
    @torch.no_grad()
    def test_predict_returns_correct_type(
        self, model: QuantModelV1, batch_input: torch.Tensor
    ) -> None:
        from trainer.models.quant_v1.output import MultiTaskQuantModelOutput, QuantModelOutput

        out = model.predict(batch_input)
        assert isinstance(out, (QuantModelOutput, MultiTaskQuantModelOutput))

    @torch.no_grad()
    def test_probabilities_consistent_with_buy_sell_notrade(
        self, model: QuantModelV1, batch_input: torch.Tensor
    ) -> None:
        out = model.predict(batch_input)
        assert torch.allclose(out.buy_prob, out.probabilities[:, 0], atol=1e-6)
        assert torch.allclose(out.sell_prob, out.probabilities[:, 1], atol=1e-6)
        assert torch.allclose(out.no_trade_prob, out.probabilities[:, 2], atol=1e-6)

    @torch.no_grad()
    def test_predicted_class_consistent_with_probabilities(
        self, model: QuantModelV1, batch_input: torch.Tensor
    ) -> None:
        out = model.predict(batch_input)
        expected = out.probabilities.argmax(dim=-1)
        assert torch.equal(out.predicted_class, expected)

    @torch.no_grad()
    def test_to_dict_has_all_keys(self, model: QuantModelV1, batch_input: torch.Tensor) -> None:
        out = model.predict(batch_input)
        d = out.to_dict()
        required = {
            "logits",
            "probabilities",
            "confidence",
            "predicted_class",
            "buy_prob",
            "sell_prob",
            "no_trade_prob",
        }
        assert required.issubset(d.keys())

    def test_training_step(self, model: QuantModelV1, batch_input: torch.Tensor) -> None:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))

        optimizer.zero_grad()
        logits = model(batch_input)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)
        model.eval()


class TestLossIntegration:
    @torch.no_grad()
    def test_focal_loss_with_model_output(
        self, model: QuantModelV1, batch_input: torch.Tensor
    ) -> None:
        cfg = LossConfig(type=LossType.FOCAL, focal_gamma=2.0)
        criterion = build_loss(cfg)
        logits = model(batch_input)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = criterion(logits, targets)
        assert torch.isfinite(loss)

    @torch.no_grad()
    def test_weighted_ce_with_class_weights(
        self, model: QuantModelV1, batch_input: torch.Tensor
    ) -> None:
        labels = [0] * 70 + [1] * 20 + [2] * 10
        weights = compute_class_weights(labels, NUM_CLASSES)
        cfg = LossConfig(
            type=LossType.WEIGHTED_CROSS_ENTROPY,
            class_weights=weights.tolist(),
        )
        criterion = build_loss(cfg)
        logits = model(batch_input)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = criterion(logits, targets)
        assert torch.isfinite(loss)


class TestBalancingIntegration:
    def test_sampler_with_model_num_classes(self, model: QuantModelV1) -> None:
        labels = [0] * 70 + [1] * 20 + [2] * 10
        cfg = BalancingConfig(strategy=BalancingStrategy.WEIGHTED_SAMPLER)
        builder = BalancingStrategyBuilder(cfg, model.num_classes)
        sampler = builder.get_sampler(labels)
        assert sampler is not None
        assert len(sampler) == len(labels)
