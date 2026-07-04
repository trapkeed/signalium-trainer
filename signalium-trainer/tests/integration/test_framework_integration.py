"""Integration tests — framework + existing LSTM model working end-to-end.

Verifies that:
- QuantModelV1 still functions after migration to framework ABCs
- ModelFactory creates a working model
- InferenceAdapterFactory wraps the model correctly
- MetricsEngine accumulates correctly through a mini training loop
- ModelSpec can be registered and retrieved
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trainer.config.schema import TrainerConfig
from trainer.framework import (
    InferenceAdapterFactory,
    MetricsEngine,
    ModelFactory,
    ModelInput,
)
from trainer.framework.abstractions import (
    BaseClassificationHead,
    BaseConfidenceHead,
    BaseFeatureEncoder,
    BaseLoss,
    BaseSequenceEncoder,
)
from trainer.models.quant_v1.config import (
    BackboneConfig,
    BackboneType,
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    InputEncoderConfig,
    QuantV1Config,
)
from trainer.models.quant_v1.model import QuantModelV1
from trainer.models.quant_v1.encoder import InputEncoder
from trainer.models.quant_v1.backbone import LSTMBackbone, GRUBackbone, SequenceEncoder
from trainer.models.quant_v1.head import ClassificationHead, ConfidenceHead
from trainer.losses.focal import FocalLoss
from trainer.losses.weighted_ce import WeightedCrossEntropyLoss

BATCH = 8
SEQ_LEN = 20
INPUT_DIM = 32
NUM_CLASSES = 3


@pytest.fixture(scope="module")
def small_cfg() -> QuantV1Config:
    return QuantV1Config(
        encoder=InputEncoderConfig(input_dim=INPUT_DIM, hidden_dim=64, projection_dropout=0.0),
        backbone=BackboneConfig(
            type=BackboneType.LSTM,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            window_length=SEQ_LEN,
        ),
        classification_head=ClassificationHeadConfig(num_classes=NUM_CLASSES, dropout=0.0),
        confidence_head=ConfidenceHeadConfig(dropout=0.0),
    )


@pytest.fixture(scope="module")
def model(small_cfg: QuantV1Config) -> QuantModelV1:
    return QuantModelV1(TrainerConfig(), small_cfg)


@pytest.fixture(scope="module")
def x() -> torch.Tensor:
    return torch.randn(BATCH, SEQ_LEN, INPUT_DIM)


# ---------------------------------------------------------------------------
# Migration verification — quant_v1 components implement framework ABCs
# ---------------------------------------------------------------------------

class TestFrameworkMigration:
    def test_input_encoder_is_base_feature_encoder(self, small_cfg: QuantV1Config) -> None:
        enc = InputEncoder(small_cfg.encoder)
        assert isinstance(enc, BaseFeatureEncoder)

    def test_sequence_encoder_is_base_sequence_encoder(self) -> None:
        assert issubclass(SequenceEncoder, BaseSequenceEncoder)

    def test_lstm_backbone_is_base_sequence_encoder(self, small_cfg: QuantV1Config) -> None:
        bb = LSTMBackbone(64, small_cfg.backbone)
        assert isinstance(bb, BaseSequenceEncoder)

    def test_gru_backbone_is_base_sequence_encoder(self, small_cfg: QuantV1Config) -> None:
        gru_cfg = BackboneConfig(type=BackboneType.GRU, hidden_size=32, num_layers=1, dropout=0.0)
        bb = GRUBackbone(INPUT_DIM, gru_cfg)
        assert isinstance(bb, BaseSequenceEncoder)

    def test_classification_head_is_base_classification_head(self, small_cfg: QuantV1Config) -> None:
        head = ClassificationHead(64, small_cfg.classification_head)
        assert isinstance(head, BaseClassificationHead)

    def test_confidence_head_is_base_confidence_head(self, small_cfg: QuantV1Config) -> None:
        head = ConfidenceHead(64, small_cfg.confidence_head)
        assert isinstance(head, BaseConfidenceHead)

    def test_focal_loss_is_base_loss(self) -> None:
        loss = FocalLoss(gamma=2.0)
        assert isinstance(loss, BaseLoss)

    def test_weighted_ce_is_base_loss(self) -> None:
        loss = WeightedCrossEntropyLoss()
        assert isinstance(loss, BaseLoss)


# ---------------------------------------------------------------------------
# QuantModelV1 still functions after migration
# ---------------------------------------------------------------------------

class TestLSTMStillFunctions:
    @torch.no_grad()
    def test_forward_shape(self, model: QuantModelV1, x: torch.Tensor) -> None:
        logits = model(x)
        assert logits.shape == (BATCH, NUM_CLASSES)

    @torch.no_grad()
    def test_predict_probabilities_sum_to_one(self, model: QuantModelV1, x: torch.Tensor) -> None:
        out = model.predict(x)
        sums = out.probabilities.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_training_step(self, model: QuantModelV1, x: torch.Tensor) -> None:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        optimizer.zero_grad()
        loss = criterion(model(x), targets)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        model.eval()


# ---------------------------------------------------------------------------
# ModelFactory integration
# ---------------------------------------------------------------------------

class TestModelFactoryIntegration:
    def test_factory_creates_working_model(self) -> None:
        model = ModelFactory.create("quant_v1", TrainerConfig())
        x = torch.randn(BATCH, 60, 128)  # Default dims
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (BATCH, 3)

    def test_factory_from_dict_with_gru(self) -> None:
        model = ModelFactory.from_dict(
            {
                "name": "quant_v1",
                "encoder": {"input_dim": INPUT_DIM, "hidden_dim": 64},
                "backbone": {"type": "gru", "hidden_size": 64, "num_layers": 1},
            },
            TrainerConfig(),
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (BATCH, NUM_CLASSES)


# ---------------------------------------------------------------------------
# InferenceAdapter integration
# ---------------------------------------------------------------------------

class TestInferenceAdapterIntegration:
    def test_pytorch_adapter_predict(self, model: QuantModelV1, x: torch.Tensor) -> None:
        adapter = InferenceAdapterFactory.from_model(model)
        assert adapter.is_loaded
        result = adapter.predict(x)
        assert "probabilities" in result
        assert "predicted_class" in result
        assert result["probabilities"].shape == (BATCH, NUM_CLASSES)

    def test_confidence_gate_zeroes_low_confidence(self, model: QuantModelV1, x: torch.Tensor) -> None:
        from trainer.framework.contracts.model_output import CLASS_NO_TRADE
        adapter = InferenceAdapterFactory.from_model(model, confidence_threshold=2.0)
        result = adapter.predict(x)
        # Threshold > 1.0 means nothing passes → all NO_TRADE
        assert (result["predicted_class"] == CLASS_NO_TRADE).all()


# ---------------------------------------------------------------------------
# MetricsEngine integration through mini training loop
# ---------------------------------------------------------------------------

class TestMetricsEngineIntegration:
    def test_metrics_across_mini_training_loop(self, model: QuantModelV1, x: torch.Tensor) -> None:
        engine = MetricsEngine(num_classes=NUM_CLASSES, class_labels=["BUY", "SELL", "NO_TRADE"])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for _ in range(3):
            targets = torch.randint(0, NUM_CLASSES, (BATCH,))
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            engine.update(logits.detach(), targets)

        metrics = engine.compute_and_reset()
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        model.eval()


# ---------------------------------------------------------------------------
# ModelSpec registration integration
# ---------------------------------------------------------------------------

class TestModelSpecRegistration:
    def test_register_and_retrieve_spec(self) -> None:
        from trainer.models.quant_v1.registration import register_quant_v1_spec
        from trainer.registry.registry import ModelRegistry

        register_quant_v1_spec()
        spec = ModelRegistry.get_spec("quant_v1")
        assert spec is not None
        assert spec.name == "quant_v1"
        assert spec.version == "1.0.0"
        assert spec.input_schema is not None
        assert spec.input_schema.feature_dim == 128
        assert spec.output_schema is not None
        assert spec.output_schema.num_classes == 3

    def test_spec_to_dict_serialisable(self) -> None:
        from trainer.models.quant_v1.registration import register_quant_v1_spec
        from trainer.registry.registry import ModelRegistry
        import json

        register_quant_v1_spec()
        spec = ModelRegistry.get_spec("quant_v1")
        d = spec.to_dict()
        # Must be JSON-serialisable (for manifest output)
        json.dumps(d)

    def test_registry_summary_shows_quant_v1(self) -> None:
        from trainer.registry.registry import ModelRegistry
        rows = ModelRegistry.summary()
        quant_row = next((r for r in rows if r["name"] == "quant_v1"), None)
        assert quant_row is not None
        assert quant_row["class"] == "QuantModelV1"
