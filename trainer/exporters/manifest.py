"""Build and write export manifests with checksums."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class ManifestBuilder:
    """Assembles a manifest JSON describing exported model artifacts.

    The manifest includes:
    - File list with sizes and SHA-256 checksums
    - Version information
    - Training summary
    - Metadata

    Args:
        model_id: Signalium model identifier.
        version: Semantic version string.
        framework: "pytorch" / "torchscript" / "onnx".
        architecture: Model class name.
    """

    def __init__(
        self,
        model_id: str,
        version: str,
        framework: str,
        architecture: str,
    ) -> None:
        self._model_id = model_id
        self._version = version
        self._framework = framework
        self._architecture = architecture
        self._files: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}
        self._metrics: dict[str, float] = {}
        self._training_summary: Optional[dict[str, Any]] = None

    def add_file(self, path: Path, content_type: str = "application/octet-stream") -> None:
        """Register an artifact file.

        Args:
            path: Path to the file on disk.
            content_type: MIME type of the file.
        """
        sha256 = self._sha256(path)
        self._files.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256,
                "content_type": content_type,
            }
        )

    def set_metrics(self, metrics: dict[str, float]) -> None:
        """Attach evaluation metrics to the manifest."""
        self._metrics = dict(metrics)

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Attach arbitrary metadata."""
        self._metadata = dict(metadata)

    def set_training_summary(self, summary: dict[str, Any]) -> None:
        """Attach a training run summary."""
        self._training_summary = dict(summary)

    def build(self) -> dict[str, Any]:
        """Assemble and return the manifest dict."""
        return {
            "model_id": self._model_id,
            "version": self._version,
            "framework": self._framework,
            "architecture": self._architecture,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "files": self._files,
            "metrics": self._metrics,
            "metadata": self._metadata,
            "training_summary": self._training_summary,
        }

    def write(self, output_dir: Path, filename: str = "manifest.json") -> Path:
        """Write the manifest to a JSON file.

        Args:
            output_dir: Directory to write the manifest into.
            filename: Output filename.

        Returns:
            Path to the written manifest file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / filename
        with manifest_path.open("w") as fh:
            json.dump(self.build(), fh, indent=2)
        logger.info("Manifest written: {}", manifest_path)
        return manifest_path

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
