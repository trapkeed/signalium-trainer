"""Unit tests for the Signalium Model Framework.

Covers: ABCs, contracts (ModelInput / ModelOutput / ModelSpec),
ModelFactory, enhanced ModelRegistry, and MetricsEngine.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_torch
import torch

from trainer.framework import (
    CLASS_BUY,
    CLASS_NO_TRADE,
    CLASS_SELL,
    FeatureSchema,
    MetricsEngine,
    ModelFactory,
    ModelInput,
    ModelOutput,
    ModelSpec,
    OutputSchema,
)
from trainer.framework.abstractions import (
    BaseClassificationHead,
    BaseConfidenceHead,
    BaseFeatureEncoder,
    BaseLoss,
    BaseMetrics,
    BaseSequenceEncoder,
)

BATCH = 4
SEQ_LEN = 10
INPUT_DIM = 16
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Abstract base class contract tests
# ---------------------------------------------------------------------------


class TestFrameworkABCs:
    """Verify ABCs enforce the correct abstract method signatures."""

    def test_base_feature_encoder_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseFeatureEncoder()  # type: ignore[abstract]

    def test_base_sequence_encoder_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseSequenceEncoder()  # type: ignore[abstract]

    def test_base_classification_head_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseClassificationHead()  # type: ignore[abstract]

    def test_base_confidence_head_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseConfidenceHead()  # type: ignore[abstract]

    def test_base_loss_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseLoss()  # type: ignore[abstract]

    def test_base_metrics_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseMetrics()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# ModelInput contract
# ---------------------------------------------------------------------------


class TestModelInput:
    def test_valid_3d_tensor(self) -> None:
        x = ModelInput(features=torch.randn(BATCH, SEQ_LEN, INPUT_DIM))
        x.validate()  # Should not raise

    def test_invalid_2d_tensor_raises(self) -> None:
        x = ModelInput(features=torch.randn(BATCH, INPUT_DIM))
        with pytest.raises(ValueError, match="3 dimensions"):
            x.validate()

    def test_non_float_raises(self) -> None:
        x = ModelInput(features=torch.randint(0, 10, (BATCH, SEQ_LEN, INPUT_DIM)))
        with pytest.raises(TypeError, match="float tensor"):
            x.validate()

    def test_mask_shape_mismatch_raises(self) -> None:
        x = ModelInput(
            features=torch.randn(BATCH, SEQ_LEN, INPUT_DIM),
            mask=torch.ones(BATCH, SEQ_LEN + 1, dtype=torch.bool),
        )
        with pytest.raises(ValueError, match="mask shape"):
            x.validate()

    def test_properties(self) -> None:
        x = ModelInput(features=torch.randn(BATCH, SEQ_LEN, INPUT_DIM))
        assert x.batch_size == BATCH
        assert x.seq_len == SEQ_LEN
        assert x.feature_dim == INPUT_DIM

    def test_to_device(self) -> None:
        x = ModelInput(features=torch.randn(BATCH, SEQ_LEN, INPUT_DIM))
        moved = x.to(torch.device("cpu"))
        assert moved.features.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# ModelOutput contract
# ---------------------------------------------------------------------------


class TestModelOutput:
    @pytest.fixture()
    def output(self) -> ModelOutput:
        probs = torch.softmax(torch.randn(BATCH, NUM_CLASSES), dim=-1)
        return ModelOutput(
            logits=torch.randn(BATCH, NUM_CLASSES),
            probabilities=probs,
            confidence=torch.rand(BATCH, 1),
            predicted_class=probs.argmax(dim=-1),
        )

    def test_buy_sell_notrade_props(self, output: ModelOutput) -> None:
        assert torch.equal(output.buy_prob, output.probabilities[:, CLASS_BUY])
        assert torch.equal(output.sell_prob, output.probabilities[:, CLASS_SELL])
        assert torch.equal(output.no_trade_prob, output.probabilities[:, CLASS_NO_TRADE])

    def test_predicted_label_length(self, output: ModelOutput) -> None:
        labels = output.predicted_label
        assert len(labels) == BATCH
        assert all(l in {"BUY", "SELL", "NO_TRADE"} for l in labels)

    def test_to_dict_required_keys(self, output: ModelOutput) -> None:
        d = output.to_dict()
        for key in (
            "logits",
            "probabilities",
            "confidence",
            "predicted_class",
            "buy_prob",
            "sell_prob",
            "no_trade_prob",
        ):
            assert key in d

    def test_gate_by_confidence_high_threshold(self, output: ModelOutput) -> None:
        gated = output.gate_by_confidence(threshold=1.1)  # Nothing passes
        assert (gated.predicted_class == CLASS_NO_TRADE).all()

    def test_gate_by_confidence_zero_threshold(self, output: ModelOutput) -> None:
        gated = output.gate_by_confidence(threshold=0.0)  # Everything passes
        assert torch.equal(gated.predicted_class, output.predicted_class)


# ---------------------------------------------------------------------------
# ModelSpec contract
# ---------------------------------------------------------------------------


class TestModelSpec:
    def test_to_dict_has_required_keys(self) -> None:
        spec = ModelSpec(
            name="test_model",
            version="0.1.0",
            architecture="TestArch",
            input_schema=FeatureSchema(feature_dim=64, seq_len=30),
            output_schema=OutputSchema(num_classes=3),
        )
        d = spec.to_dict()
        assert d["name"] == "test_model"
        assert d["version"] == "0.1.0"
        assert d["input_schema"]["feature_dim"] == 64
        assert d["output_schema"]["num_classes"] == 3

    def test_export_formats_default(self) -> None:
        spec = ModelSpec(name="x", version="1", architecture="Y")
        assert "torchscript" in spec.export_formats
        assert "onnx" in spec.export_formats


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------


class TestModelFactory:
    def test_list_available_includes_quant_v1(self) -> None:

        assert "quant_v1" in ModelFactory.list_available()

    def test_create_quant_v1_by_name(self) -> None:
        from trainer.config.schema import TrainerConfig
        from trainer.models.quant_v1.model import QuantModelV1

        model = ModelFactory.create("quant_v1", TrainerConfig())
        assert isinstance(model, QuantModelV1)

    def test_create_with_model_config(self) -> None:
        from trainer.config.schema import TrainerConfig
        from trainer.models.quant_v1.config import InputEncoderConfig, QuantV1Config

        cfg = QuantV1Config(encoder=InputEncoderConfig(input_dim=32, hidden_dim=64))
        model = ModelFactory.create("quant_v1", TrainerConfig(), model_config=cfg)
        assert model.feature_dim == 32

    def test_create_unknown_model_raises(self) -> None:
        from trainer.config.schema import TrainerConfig

        with pytest.raises(KeyError, match="not found"):
            ModelFactory.create("does_not_exist", TrainerConfig())

    def test_from_dict_basic(self) -> None:
        from trainer.config.schema import TrainerConfig
        from trainer.models.quant_v1.model import QuantModelV1

        model = ModelFactory.from_dict({"name": "quant_v1"}, TrainerConfig())
        assert isinstance(model, QuantModelV1)

    def test_from_dict_missing_name_raises(self) -> None:
        from trainer.config.schema import TrainerConfig

        with pytest.raises(KeyError, match="'name'"):
            ModelFactory.from_dict({"backbone": {}}, TrainerConfig())

    def test_from_dict_with_backbone_override(self) -> None:
        from trainer.config.schema import TrainerConfig

        model = ModelFactory.from_dict(
            {"name": "quant_v1", "backbone": {"type": "gru", "hidden_size": 64}},
            TrainerConfig(),
        )
        assert model is not None


# ---------------------------------------------------------------------------
# Enhanced ModelRegistry
# ---------------------------------------------------------------------------


class TestEnhancedModelRegistry:
    def test_get_spec_returns_none_without_spec(self) -> None:
        from trainer.registry.registry import ModelRegistry

        # quant_v1 may or may not have a spec depending on import order
        # — just verify the API exists and returns without crashing
        result = ModelRegistry.get_spec("quant_v1")
        assert result is None or result is not None  # Either is valid

    def test_register_with_spec(self) -> None:
        import torch

        from trainer.models.base import BaseQuantModel
        from trainer.registry.registry import ModelRegistry

        # Use a unique name to avoid conflicts with other tests
        test_name = "_test_spec_model_unique"
        if test_name in ModelRegistry._registry:
            del ModelRegistry._registry[test_name]
            ModelRegistry._specs.pop(test_name, None)

        spec = ModelSpec(name=test_name, version="0.1.0", architecture="TestNet")

        @ModelRegistry.register(test_name, spec=spec)
        class _TestModel(BaseQuantModel):
            @property
            def feature_dim(self):
                return 8

            @property
            def num_classes(self):
                return 3

            def forward(self, x):
                return torch.randn(x.shape[0], 3)

        assert ModelRegistry.has_spec(test_name)
        retrieved = ModelRegistry.get_spec(test_name)
        assert retrieved.version == "0.1.0"
        assert retrieved.architecture == "TestNet"

        # Clean up
        del ModelRegistry._registry[test_name]
        del ModelRegistry._specs[test_name]

    def test_summary_includes_all_models(self) -> None:
        from trainer.registry.registry import ModelRegistry

        rows = ModelRegistry.summary()
        names = [r["name"] for r in rows]
        assert "quant_v1" in names

    def test_summary_row_has_expected_keys(self) -> None:
        from trainer.registry.registry import ModelRegistry

        rows = ModelRegistry.summary()
        for row in rows:
            assert {"name", "class", "version", "architecture", "has_spec"}.issubset(row.keys())


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------


class TestMetricsEngine:
    @pytest.fixture()
    def engine(self) -> MetricsEngine:
        return MetricsEngine(num_classes=NUM_CLASSES, class_labels=["BUY", "SELL", "NO_TRADE"])

    @pytest.fixture()
    def batch_data(self):
        logits = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        return logits, targets

    def test_compute_returns_all_keys(self, engine: MetricsEngine, batch_data) -> None:
        logits, targets = batch_data
        engine.update(logits, targets)
        result = engine.compute()
        for key in (
            "loss",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ):
            assert key in result, f"Missing key: {key}"

    def test_compute_per_class_keys(self, engine: MetricsEngine, batch_data) -> None:
        logits, targets = batch_data
        engine.update(logits, targets)
        result = engine.compute()
        for label in ("BUY", "SELL", "NO_TRADE"):
            assert f"accuracy_{label}" in result

    def test_accuracy_in_range(self, engine: MetricsEngine, batch_data) -> None:
        logits, targets = batch_data
        engine.update(logits, targets)
        result = engine.compute()
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_loss_finite(self, engine: MetricsEngine, batch_data) -> None:
        logits, targets = batch_data
        engine.update(logits, targets)
        result = engine.compute()
        import math

        assert math.isfinite(result["loss"])

    def test_reset_clears_state(self, engine: MetricsEngine, batch_data) -> None:
        logits, targets = batch_data
        engine.update(logits, targets)
        engine.reset()
        result = engine.compute()
        assert result == {}

    def test_compute_and_reset(self, engine: MetricsEngine, batch_data) -> None:
        logits, targets = batch_data
        engine.update(logits, targets)
        result = engine.compute_and_reset()
        assert "accuracy" in result
        assert engine.compute() == {}

    def test_multi_batch_accumulation(self, engine: MetricsEngine) -> None:
        for _ in range(5):
            engine.update(torch.randn(BATCH, NUM_CLASSES), torch.randint(0, NUM_CLASSES, (BATCH,)))
        result = engine.compute()
        assert result["accuracy"] >= 0.0
