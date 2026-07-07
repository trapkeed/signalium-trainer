"""Integration tests — TorchScript and ONNX export for QuantModelV1."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trainer.config.schema import TrainerConfig
from trainer.exporters.manifest import ManifestBuilder
from trainer.exporters.onnx import ONNXExporter
from trainer.exporters.torchscript import TorchScriptExporter
from trainer.models.quant_v1.config import (
    BackboneConfig,
    BackboneType,
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    InputEncoderConfig,
    QuantV1Config,
)
from trainer.models.quant_v1.model import QuantModelV1

INPUT_DIM = 32
SEQ_LEN = 10
BATCH = 2
NUM_CLASSES = 3


@pytest.fixture(scope="module")
def small_model() -> QuantModelV1:
    cfg = QuantV1Config(
        encoder=InputEncoderConfig(input_dim=INPUT_DIM, hidden_dim=32, projection_dropout=0.0),
        backbone=BackboneConfig(
            type=BackboneType.LSTM,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            window_length=SEQ_LEN,
        ),
        classification_head=ClassificationHeadConfig(num_classes=NUM_CLASSES, dropout=0.0),
        confidence_head=ConfidenceHeadConfig(dropout=0.0),
    )
    return QuantModelV1(TrainerConfig(), cfg)


class TestTorchScriptExport:
    def test_export_produces_file(self, small_model: QuantModelV1, tmp_path: Path) -> None:
        cfg = TrainerConfig()
        cfg.export.export_torchscript = True
        exporter = TorchScriptExporter(
            model=small_model,
            config=cfg,
            example_input_shape=(SEQ_LEN, INPUT_DIM),
            use_trace=True,
        )
        paths = exporter.export(tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".pt"

    def test_torchscript_inference_matches_original(
        self, small_model: QuantModelV1, tmp_path: Path
    ) -> None:
        cfg = TrainerConfig()
        exporter = TorchScriptExporter(
            model=small_model,
            config=cfg,
            example_input_shape=(SEQ_LEN, INPUT_DIM),
            use_trace=True,
        )
        path = exporter.export(tmp_path)[0]
        scripted = torch.jit.load(str(path))
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)

        small_model.eval()
        scripted.eval()
        with torch.no_grad():
            orig_out = small_model(x)
            script_out = scripted(x)

        assert torch.allclose(orig_out, script_out, atol=1e-5)


class TestONNXExport:
    def test_export_produces_onnx_file(self, small_model: QuantModelV1, tmp_path: Path) -> None:
        cfg = TrainerConfig()
        exporter = ONNXExporter(
            model=small_model,
            config=cfg,
            example_input_shape=(SEQ_LEN, INPUT_DIM),
        )
        paths = exporter.export(tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".onnx"

    def test_onnx_model_validates(self, small_model: QuantModelV1, tmp_path: Path) -> None:
        import onnx

        cfg = TrainerConfig()
        exporter = ONNXExporter(
            model=small_model,
            config=cfg,
            example_input_shape=(SEQ_LEN, INPUT_DIM),
        )
        path = exporter.export(tmp_path)[0]
        model = onnx.load(str(path))
        onnx.checker.check_model(model)  # raises if invalid

    def test_onnx_inference_matches_original(
        self, small_model: QuantModelV1, tmp_path: Path
    ) -> None:
        import numpy as np
        import onnxruntime as ort

        cfg = TrainerConfig()
        exporter = ONNXExporter(
            model=small_model,
            config=cfg,
            example_input_shape=(SEQ_LEN, INPUT_DIM),
        )
        path = exporter.export(tmp_path)[0]

        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        small_model.eval()
        with torch.no_grad():
            torch_out = small_model(x).numpy()

        sess = ort.InferenceSession(str(path))
        ort_out = sess.run(None, {"input": x.numpy()})[0]

        assert np.allclose(torch_out, ort_out, atol=1e-4)


class TestManifestWithExport:
    def test_manifest_includes_exported_files(
        self, small_model: QuantModelV1, tmp_path: Path
    ) -> None:
        cfg = TrainerConfig()
        ts_exporter = TorchScriptExporter(
            model=small_model,
            config=cfg,
            example_input_shape=(SEQ_LEN, INPUT_DIM),
        )
        ts_path = ts_exporter.export(tmp_path)[0]

        builder = ManifestBuilder(
            model_id="quant_v1",
            version="1.0.0",
            framework="torchscript",
            architecture="QuantModelV1",
        )
        builder.add_file(ts_path, content_type="application/octet-stream")
        builder.set_metrics({"accuracy": 0.92, "f1_macro": 0.89})
        builder.set_training_summary(small_model.get_architecture_summary())

        manifest_path = builder.write(tmp_path)
        assert manifest_path.exists()

        import json

        data = json.loads(manifest_path.read_text())
        assert data["model_id"] == "quant_v1"
        assert len(data["files"]) == 1
        assert data["metrics"]["accuracy"] == pytest.approx(0.92)
