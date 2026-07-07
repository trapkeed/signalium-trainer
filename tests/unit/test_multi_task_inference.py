"""Unit tests for multi-task inference contract.

Validates that PyTorchInferenceAdapter returns the full Phase 14.5
inference contract (buy_prob, sell_prob, no_trade_prob, confidence,
expected_move, stop_loss, take_profit, holding_time, tp_probability,
prediction_timestamp, model_version).
"""

from __future__ import annotations

import pytest
import torch

from trainer.config.schema import TrainerConfig
from trainer.framework.inference import InferenceAdapterFactory, PyTorchInferenceAdapter
from trainer.models.quant_v1.config import (
    BackboneConfig,
    BackboneType,
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    InputEncoderConfig,
    QuantV1Config,
    RegressionHeadConfig,
    RegressionUnit,
    TPProbabilityHeadConfig,
)
from trainer.models.quant_v1.model import QuantModelV1

BATCH = 4
SEQ_LEN = 20
INPUT_DIM = 16
HIDDEN = 32
NUM_CLASSES = 3


def _multi_task_config() -> QuantV1Config:
    cfg = QuantV1Config(
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
def model() -> QuantModelV1:
    return QuantModelV1(TrainerConfig(), _multi_task_config())


@pytest.fixture()
def adapter(model: QuantModelV1) -> PyTorchInferenceAdapter:
    return InferenceAdapterFactory.from_model(model)


@pytest.fixture()
def x() -> torch.Tensor:
    return torch.randn(BATCH, SEQ_LEN, INPUT_DIM)


# ---------------------------------------------------------------------------
# Full inference contract
# ---------------------------------------------------------------------------


class TestInferenceContract:
    def test_predict_returns_dict(self, adapter: PyTorchInferenceAdapter, x: torch.Tensor) -> None:
        result = adapter.predict(x)
        assert isinstance(result, dict)

    def test_direction_probabilities_present(
        self, adapter: PyTorchInferenceAdapter, x: torch.Tensor
    ) -> None:
        result = adapter.predict(x)
        for key in ("buy_prob", "sell_prob", "no_trade_prob"):
            assert key in result, f"Missing key: {key}"

    def test_confidence_present(self, adapter: PyTorchInferenceAdapter, x: torch.Tensor) -> None:
        result = adapter.predict(x)
        assert "confidence" in result

    def test_multi_task_keys_present(
        self, adapter: PyTorchInferenceAdapter, x: torch.Tensor
    ) -> None:
        result = adapter.predict(x)
        for key in ("expected_move", "stop_loss", "take_profit", "holding_time", "tp_probability"):
            assert key in result, f"Missing multi-task key: {key}"

    def test_prediction_timestamp_present(
        self, adapter: PyTorchInferenceAdapter, x: torch.Tensor
    ) -> None:
        result = adapter.predict(x)
        assert "prediction_timestamp" in result

    def test_model_version_present(self, adapter: PyTorchInferenceAdapter, x: torch.Tensor) -> None:
        result = adapter.predict(x)
        assert "model_version" in result

    def test_predicted_class_present(
        self, adapter: PyTorchInferenceAdapter, x: torch.Tensor
    ) -> None:
        result = adapter.predict(x)
        assert "predicted_class" in result

    def test_probabilities_shape(self, adapter: PyTorchInferenceAdapter, x: torch.Tensor) -> None:
        result = adapter.predict(x)
        assert result["probabilities"].shape == (BATCH, NUM_CLASSES)

    def test_probabilities_sum_to_one(
        self, adapter: PyTorchInferenceAdapter, x: torch.Tensor
    ) -> None:
        result = adapter.predict(x)
        sums = result["probabilities"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_stop_loss_nonnegative(self, adapter: PyTorchInferenceAdapter, x: torch.Tensor) -> None:
        result = adapter.predict(x)
        assert (result["stop_loss"] >= 0).all()

    def test_take_profit_nonnegative(
        self, adapter: PyTorchInferenceAdapter, x: torch.Tensor
    ) -> None:
        result = adapter.predict(x)
        assert (result["take_profit"] >= 0).all()

    def test_tp_probability_range(self, adapter: PyTorchInferenceAdapter, x: torch.Tensor) -> None:
        result = adapter.predict(x)
        p = result["tp_probability"]
        assert (p >= 0).all() and (p <= 1).all()

    def test_confidence_range(self, adapter: PyTorchInferenceAdapter, x: torch.Tensor) -> None:
        result = adapter.predict(x)
        conf = result["confidence"]
        assert (conf >= 0).all() and (conf <= 1).all()


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------


class TestConfidenceGate:
    def test_confidence_threshold_applied(self, model: QuantModelV1, x: torch.Tensor) -> None:
        adapter = InferenceAdapterFactory.from_model(model, confidence_threshold=1.0)
        result = adapter.predict(x)
        pred = result["predicted_class"]
        assert (pred == 2).all(), "All predictions should be NO_TRADE at threshold=1.0"

    def test_no_threshold_returns_all_classes(
        self, adapter: PyTorchInferenceAdapter, x: torch.Tensor
    ) -> None:
        result = adapter.predict(x)
        assert result["predicted_class"].shape == (BATCH,)


# ---------------------------------------------------------------------------
# Backward compat: base model (no multi-task heads) adapter
# ---------------------------------------------------------------------------


class TestBaseModelAdapter:
    def test_base_model_adapter_returns_direction_keys(self, x: torch.Tensor) -> None:
        base_cfg = QuantV1Config(
            encoder=InputEncoderConfig(
                input_dim=INPUT_DIM, hidden_dim=HIDDEN, projection_dropout=0.0
            ),
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
        model = QuantModelV1(TrainerConfig(), base_cfg)
        adapter = InferenceAdapterFactory.from_model(model)
        result = adapter.predict(x)
        for key in ("buy_prob", "sell_prob", "no_trade_prob", "confidence", "predicted_class"):
            assert key in result
