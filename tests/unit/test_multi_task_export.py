"""Unit tests for multi-task model export metadata and manifest.

Validates:
  - get_export_metadata() structure and completeness
  - ManifestBuilder correctly embeds multi-task metadata
  - model_version and architecture_name in export
  - prediction_schema contains all enabled head descriptions
  - loss_config correctly reflects training weights
  - output_version is present
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _all_heads_config() -> QuantV1Config:
    cfg = QuantV1Config(
        encoder=InputEncoderConfig(input_dim=16, hidden_dim=32, projection_dropout=0.0),
        backbone=BackboneConfig(
            type=BackboneType.LSTM, hidden_size=32, num_layers=1,
            dropout=0.0, bidirectional=False, window_length=20,
        ),
        classification_head=ClassificationHeadConfig(num_classes=3, hidden_dim=16, dropout=0.0),
        confidence_head=ConfidenceHeadConfig(hidden_dim=16, dropout=0.0),
    )
    cfg.expected_move_head = RegressionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0, unit=RegressionUnit.PIPS)
    cfg.stop_loss_head = RegressionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0, unit=RegressionUnit.ATR)
    cfg.take_profit_head = RegressionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0, unit=RegressionUnit.ATR)
    cfg.holding_time_head = RegressionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0, unit=RegressionUnit.MINUTES)
    cfg.tp_probability_head = TPProbabilityHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.multi_task_loss = MultiTaskLossConfig(
        direction_weight=1.0, move_weight=0.4, sl_weight=0.6,
        tp_weight=0.6, holding_time_weight=0.3,
        confidence_weight=0.15, tp_probability_weight=0.5,
    )
    return cfg


@pytest.fixture()
def model() -> QuantModelV1:
    return QuantModelV1(TrainerConfig(), _all_heads_config())


@pytest.fixture()
def base_model() -> QuantModelV1:
    return QuantModelV1(TrainerConfig())


# ---------------------------------------------------------------------------
# Export metadata structure
# ---------------------------------------------------------------------------

class TestExportMetadataStructure:
    def test_export_metadata_returns_dict(self, model: QuantModelV1) -> None:
        meta = model.get_export_metadata()
        assert isinstance(meta, dict)

    def test_required_top_level_keys(self, model: QuantModelV1) -> None:
        meta = model.get_export_metadata()
        required = {"model_id", "model_version", "architecture", "phase",
                    "supported_outputs", "prediction_schema", "output_version",
                    "loss_config"}
        assert required.issubset(meta.keys())

    def test_output_version_is_string(self, model: QuantModelV1) -> None:
        meta = model.get_export_metadata()
        assert isinstance(meta["output_version"], str)
        assert meta["output_version"] == "2.0.0"

    def test_phase_is_14_5(self, model: QuantModelV1) -> None:
        meta = model.get_export_metadata()
        assert meta["phase"] == "14.5"

    def test_supported_outputs_is_list(self, model: QuantModelV1) -> None:
        meta = model.get_export_metadata()
        assert isinstance(meta["supported_outputs"], list)

    def test_supported_outputs_contains_all_heads(self, model: QuantModelV1) -> None:
        meta = model.get_export_metadata()
        for head in ("direction", "confidence", "expected_move", "stop_loss",
                     "take_profit", "holding_time", "tp_probability"):
            assert head in meta["supported_outputs"]


# ---------------------------------------------------------------------------
# Prediction schema
# ---------------------------------------------------------------------------

class TestPredictionSchema:
    def test_schema_contains_direction_fields(self, model: QuantModelV1) -> None:
        schema = model.get_export_metadata()["prediction_schema"]
        assert "buy_prob" in schema
        assert "sell_prob" in schema
        assert "no_trade_prob" in schema
        assert "confidence" in schema

    def test_schema_contains_enabled_head_fields(self, model: QuantModelV1) -> None:
        schema = model.get_export_metadata()["prediction_schema"]
        for field in ("expected_move", "stop_loss", "take_profit",
                      "holding_time", "tp_probability"):
            assert field in schema
            assert schema[field] is not None, f"{field} schema should not be None when enabled"

    def test_schema_disabled_heads_are_none(self, base_model: QuantModelV1) -> None:
        schema = base_model.get_export_metadata()["prediction_schema"]
        for field in ("expected_move", "stop_loss", "take_profit",
                      "holding_time", "tp_probability"):
            assert schema.get(field) is None, f"{field} should be None when disabled"

    def test_schema_includes_unit_information(self, model: QuantModelV1) -> None:
        schema = model.get_export_metadata()["prediction_schema"]
        assert "atr" in schema["stop_loss"].lower() or "atr" in schema["take_profit"].lower()
        assert "pips" in schema["expected_move"].lower()
        assert "minutes" in schema["holding_time"].lower()


# ---------------------------------------------------------------------------
# Loss config
# ---------------------------------------------------------------------------

class TestLossConfigExport:
    def test_loss_config_has_all_weights(self, model: QuantModelV1) -> None:
        loss_cfg = model.get_export_metadata()["loss_config"]
        required = {"direction_weight", "move_weight", "sl_weight", "tp_weight",
                    "holding_time_weight", "confidence_weight", "tp_probability_weight"}
        assert required.issubset(loss_cfg.keys())

    def test_loss_weights_match_config(self, model: QuantModelV1) -> None:
        loss_cfg = model.get_export_metadata()["loss_config"]
        assert loss_cfg["direction_weight"] == 1.0
        assert loss_cfg["sl_weight"] == 0.6
        assert loss_cfg["tp_weight"] == 0.6
        assert loss_cfg["move_weight"] == 0.4


# ---------------------------------------------------------------------------
# ManifestBuilder integration
# ---------------------------------------------------------------------------

class TestManifestBuilderIntegration:
    def test_manifest_can_embed_export_metadata(self, model: QuantModelV1) -> None:
        from trainer.exporters.manifest import ManifestBuilder

        builder = ManifestBuilder(
            model_id="signalium_quant_v1",
            version=model.model_version,
            framework="pytorch",
            architecture=model.architecture_name,
        )
        builder.set_metadata(model.get_export_metadata())
        manifest = builder.build()
        assert "metadata" in manifest
        assert manifest["metadata"]["output_version"] == "2.0.0"

    def test_manifest_is_json_serialisable(self, model: QuantModelV1) -> None:
        from trainer.exporters.manifest import ManifestBuilder

        builder = ManifestBuilder(
            model_id="signalium_quant_v1",
            version=model.model_version,
            framework="pytorch",
            architecture=model.architecture_name,
        )
        builder.set_metadata(model.get_export_metadata())
        manifest = builder.build()
        json_str = json.dumps(manifest)
        assert len(json_str) > 0

    def test_manifest_written_to_disk(self, model: QuantModelV1) -> None:
        from trainer.exporters.manifest import ManifestBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ManifestBuilder(
                model_id="signalium_quant_v1",
                version=model.model_version,
                framework="pytorch",
                architecture=model.architecture_name,
            )
            builder.set_metadata(model.get_export_metadata())
            manifest_path = builder.write(Path(tmpdir))
            assert manifest_path.exists()
            with manifest_path.open() as f:
                loaded = json.load(f)
            assert loaded["metadata"]["phase"] == "14.5"
