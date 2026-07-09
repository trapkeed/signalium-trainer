"""Model export package — TorchScript, ONNX, manifest.

``QDEExportPipeline`` (in ``trainer.exporters.pipeline``) is the S21A unified
entrypoint that chains weights -> TorchScript/ONNX -> config/calibration/
metadata -> ``qde_manifest.json`` -> ``ArtifactRegistry`` registration in one
call. Prefer it over invoking ``TorchScriptExporter``/``ONNXExporter``/
``ManifestBuilder`` directly for any new call site that exports a full
trained model version.
"""

from trainer.exporters.manifest import ManifestBuilder
from trainer.exporters.metadata_manifest import MetadataManifestBuilder
from trainer.exporters.onnx import ONNXExporter
from trainer.exporters.pipeline import QDEExportPipeline
from trainer.exporters.torchscript import TorchScriptExporter

__all__ = [
    "ManifestBuilder",
    "MetadataManifestBuilder",
    "ONNXExporter",
    "QDEExportPipeline",
    "TorchScriptExporter",
]
