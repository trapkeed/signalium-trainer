"""TorchScript model exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from trainer.config.schema import TrainerConfig
from trainer.models.base import BaseExporter, BaseQuantModel


class TorchScriptExporter(BaseExporter):
    """Export a trained model to TorchScript (torch.jit.script or trace).

    Args:
        model: Trained model to export.
        config: Full TrainerConfig.
        example_input_shape: Shape of a single input sample (without batch dim).
        use_trace: If True, use torch.jit.trace (faster but less general).
                   If False, use torch.jit.script (more general).
    """

    def __init__(
        self,
        model: BaseQuantModel,
        config: TrainerConfig,
        example_input_shape: tuple[int, ...],
        use_trace: bool = True,
    ) -> None:
        super().__init__(model, config)
        self._example_input_shape = example_input_shape
        self._use_trace = use_trace

    def export(self, output_dir: Path, **kwargs: Any) -> list[Path]:
        """Export to TorchScript.

        Args:
            output_dir: Directory to write the .pt file to.
            **kwargs: Unused.

        Returns:
            List containing the path to the exported TorchScript file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "model.torchscript.pt"

        self._model.eval()
        self._model.cpu()

        if self._use_trace:
            example = torch.zeros(1, *self._example_input_shape)
            with torch.no_grad():
                scripted = torch.jit.trace(self._model, example)
        else:
            scripted = torch.jit.script(self._model)

        scripted.save(str(out_path))
        logger.info("TorchScript export saved: {}", out_path)
        return [out_path]
