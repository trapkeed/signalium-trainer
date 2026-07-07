"""ONNX model exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from trainer.config.schema import TrainerConfig
from trainer.models.base import BaseExporter, BaseQuantModel


class ONNXExporter(BaseExporter):
    """Export a trained model to ONNX format.

    Args:
        model: Trained model to export.
        config: Full TrainerConfig.
        example_input_shape: Shape of a single input sample (without batch dim).
        input_names: ONNX graph input names.
        output_names: ONNX graph output names.
        dynamic_axes: Dynamic axis specification for variable-length inputs/outputs.
    """

    def __init__(
        self,
        model: BaseQuantModel,
        config: TrainerConfig,
        example_input_shape: tuple[int, ...],
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
    ) -> None:
        super().__init__(model, config)
        self._example_input_shape = example_input_shape
        self._input_names = input_names or ["input"]
        self._output_names = output_names or ["output"]
        self._dynamic_axes = dynamic_axes or {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    def export(self, output_dir: Path, **kwargs: Any) -> list[Path]:
        """Export to ONNX.

        Args:
            output_dir: Directory to write the .onnx file to.
            **kwargs: Forwarded to torch.onnx.export.

        Returns:
            List containing the path to the exported ONNX file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "model.onnx"

        self._model.eval()
        self._model.cpu()

        opset = self._config.export.onnx_opset
        example = torch.zeros(1, *self._example_input_shape)

        with torch.no_grad():
            torch.onnx.export(
                self._model,
                example,
                str(out_path),
                opset_version=opset,
                input_names=self._input_names,
                output_names=self._output_names,
                dynamic_axes=self._dynamic_axes,
                do_constant_folding=True,
                export_params=True,
                **kwargs,
            )

        self._validate_onnx(out_path)
        logger.info("ONNX export saved: {}", out_path)
        return [out_path]

    @staticmethod
    def _validate_onnx(path: Path) -> None:
        """Run onnx.checker.check_model on the exported file."""
        import onnx

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        logger.info("ONNX model validation passed")
