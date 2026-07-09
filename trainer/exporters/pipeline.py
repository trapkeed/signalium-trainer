"""S21A — Unified QDE export pipeline.

Single entrypoint chaining "trained model" -> "all export formats" ->
"qde_manifest.json" -> "ArtifactRegistry record", replacing the previously
fragmented call sites that hand-assembled TorchScript/ONNX exports and
manifests independently.

Usage::

    from trainer.exporters.pipeline import QDEExportPipeline

    pipeline = QDEExportPipeline(artifact_registry=registry)
    result = pipeline.export_version(
        model_version="v3.1",
        output_dir="exports",
        model=trained_model,
        config=trainer_config,
        example_input_shape=(60, 128),
        dataset_version="ds-v3.1",
        feature_version="feat-v2.0",
        training_commit="a1b2c3d4...",
        certification_status="CERTIFIED",
        promotion_status="NOT_DEPLOYED",
        supported_symbols=["EURUSD", "BTCUSD"],
        supported_tasks=["classification"],
    )

``result["manifest"]`` is the full ``qde_manifest.json`` content;
``result["artifacts"]`` maps artifact type -> registry artifact_id (when an
``ArtifactRegistry`` was supplied).

Notes
-----
Weights/TorchScript/ONNX export require ``torch`` to be importable. All
``torch`` imports are deferred into the methods that need them so importing
this module (or running the manifest/registry-only code paths with
pre-supplied ``weights_path``) never requires ``torch`` to be installed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from trainer.research_lab.artifact_models import ArtifactRecord, ArtifactType
from trainer.research_lab.artifact_registry import DuplicateArtifactError
from trainer.research_lab.artifact_store import ArtifactImmutabilityError, ArtifactStore

#: Canonical file names written by the pipeline (relative to the per-version
#: export directory).
WEIGHTS_FILENAME = "weights.pt"
TORCHSCRIPT_FILENAME = "model.torchscript.pt"
ONNX_FILENAME = "model.onnx"
CONFIG_FILENAME = "config.json"
CALIBRATION_FILENAME = "calibration.json"
METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "qde_manifest.json"

#: Required top-level fields for a S21A-compliant qde_manifest.json.
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "model_version",
    "artifact_hash",
    "dataset_version",
    "feature_version",
    "training_commit",
    "calibration_version",
    "training_timestamp",
    "certification_status",
    "promotion_status",
    "supported_symbols",
    "supported_tasks",
    "export_formats",
    "framework_version",
)


class QDEExportPipelineError(Exception):
    """Raised when the export pipeline cannot produce a required artifact."""


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _framework_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of a config/report object to a JSON-safe dict."""
    if obj is None:
        return {}
    if is_dataclass(obj) and not isinstance(obj, type):
        return json.loads(json.dumps(asdict(obj), default=str))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return json.loads(json.dumps(obj.to_dict(), default=str))
    if isinstance(obj, dict):
        return json.loads(json.dumps(obj, default=str))
    return {"value": str(obj)}


class QDEExportPipeline:
    """Orchestrates the full export of one trained QDE model version.

    Args:
        artifact_registry: Optional ``ArtifactRegistry`` used to persist a
            registry record for every artifact produced. When omitted, the
            pipeline still writes all files and returns their checksums, but
            nothing is registered.
    """

    def __init__(self, artifact_registry: Any | None = None) -> None:
        self._registry = artifact_registry

    # ── Public entrypoint ───────────────────────────────────────────────────

    def export_version(
        self,
        *,
        model_version: str,
        output_dir: str | Path,
        model: Any | None = None,
        weights_path: str | Path | None = None,
        config: Any | None = None,
        example_input_shape: tuple[int, ...] | None = None,
        export_formats: tuple[str, ...] = ("torchscript", "onnx"),
        calibration_report: Any | None = None,
        calibration_version: str = "",
        architecture: str = "",
        metadata: dict[str, Any] | None = None,
        dataset_version: str = "",
        feature_version: str = "",
        training_commit: str = "",
        training_timestamp: float | None = None,
        certification_status: str = "PENDING",
        promotion_status: str = "NOT_DEPLOYED",
        supported_symbols: list[str] | None = None,
        supported_tasks: list[str] | None = None,
        training_run_id: str = "",
        experiment_id: str = "",
        registered_by: str = "qde_export_pipeline",
    ) -> dict[str, Any]:
        """Export every required artifact for one trained model version.

        Produces (as applicable): weights, TorchScript, ONNX, ``config.json``,
        ``calibration.json``, ``metadata.json``, and ``qde_manifest.json`` —
        each with a recorded SHA-256 checksum — and registers each with the
        ``ArtifactRegistry`` when one was supplied.

        Either ``model`` (to export weights + TorchScript/ONNX) or a
        pre-existing ``weights_path`` must be provided; weights are the one
        mandatory artifact for every export.

        Returns:
            Dict with ``model_version``, ``output_dir``, ``manifest``,
            ``manifest_path``, ``files`` (checksum list), and ``artifacts``
            (artifact_type -> registry artifact_id, when registered).

        Raises:
            QDEExportPipelineError: neither `model` nor `weights_path` was
                given, or `weights_path` does not exist.
            ArtifactImmutabilityError: `model_version` was already exported
                under `output_dir` — artifact versions are write-once; call
                with a new `model_version` instead of re-exporting this one.
        """
        if model is None and weights_path is None:
            raise QDEExportPipelineError(
                "export_version requires either `model` (to export weights) "
                "or a pre-existing `weights_path`."
            )

        # Prefer the registry's own store so path resolution/immutability
        # policy stays centralized and consistent with `registry.store` /
        # `registry.resolve_path()` — only fall back to a store rooted at
        # `output_dir` when no registry (hence no canonical store) exists.
        store = getattr(self._registry, "store", None) or ArtifactStore(root=output_dir)
        # Write-once gate: reject outright if this version was already
        # exported (raises ArtifactImmutabilityError). Every artifact for a
        # version is produced in this one call, so the check happens before
        # anything is written rather than per-file; individual writes below
        # are additionally guarded at the filesystem level (atomic
        # create-exclusive in `ArtifactStore.write_bytes`).
        version_dir = store.reserve_version(model_version)

        exported: list[tuple[Path, str, str]] = []  # (path, artifact_type, tag)

        # ── Weights ──────────────────────────────────────────────────────
        # Every file below is written via `store.write_bytes` /
        # `store.import_file` — the store's atomic O_CREAT|O_EXCL create —
        # rather than a plain `Path.write_text`/`torch.save` call, so the
        # write-once guarantee applies uniformly to every artifact kind, not
        # just the version directory as a whole.
        if model is not None:
            w_path = self._export_weights(model, store, model_version)
        else:
            src = Path(weights_path)
            if not src.exists():
                raise QDEExportPipelineError(f"weights_path does not exist: {src!r}")
            w_path = store.import_file(model_version, WEIGHTS_FILENAME, src)
        exported.append((w_path, ArtifactType.WEIGHTS.value, "weights"))

        # ── TorchScript / ONNX ───────────────────────────────────────────
        if model is not None and config is not None and example_input_shape is not None:
            if "torchscript" in export_formats:
                ts_path = self._export_torchscript(model, config, example_input_shape, store, model_version)
                exported.append((ts_path, ArtifactType.TORCHSCRIPT.value, "torchscript"))
            if "onnx" in export_formats:
                onnx_path = self._export_onnx(model, config, example_input_shape, store, model_version)
                exported.append((onnx_path, ArtifactType.ONNX.value, "onnx"))
        elif export_formats:
            logger.warning(
                "QDEExportPipeline: skipping {} export(s) — `model`, `config`, and "
                "`example_input_shape` are all required for TorchScript/ONNX export.",
                list(export_formats),
            )

        # ── config.json ──────────────────────────────────────────────────
        config_path = store.write_text(
            model_version, CONFIG_FILENAME, json.dumps(_jsonable(config), indent=2, default=str)
        )
        exported.append((config_path, ArtifactType.CONFIG.value, "config"))

        # ── calibration.json ─────────────────────────────────────────────
        calibration_data = _jsonable(calibration_report)
        calibration_path = store.write_text(
            model_version, CALIBRATION_FILENAME, json.dumps(calibration_data, indent=2, default=str)
        )
        exported.append((calibration_path, ArtifactType.CALIBRATION.value, "calibration"))

        # ── metadata.json ────────────────────────────────────────────────
        metadata_payload: dict[str, Any] = dict(metadata or {})
        if model is not None and hasattr(model, "get_architecture_summary"):
            try:
                metadata_payload.setdefault("architecture_summary", model.get_architecture_summary())
            except Exception as exc:  # noqa: BLE001
                logger.warning("QDEExportPipeline: get_architecture_summary() failed: {}", exc)
        metadata_payload.setdefault("architecture", architecture)
        metadata_payload.setdefault("model_version", model_version)
        metadata_path = store.write_text(
            model_version, METADATA_FILENAME, json.dumps(metadata_payload, indent=2, default=str)
        )
        # metadata.json is checksummed and listed like every other exported
        # file, but it is intentionally NOT registered with the
        # ArtifactRegistry: the registry's ArtifactType enum defines exactly
        # six canonical types (WEIGHTS/CONFIG/CALIBRATION/ONNX/TORCHSCRIPT/
        # DEPLOYMENT_MANIFEST) and metadata.json does not correspond to any
        # of them — forcing it under CONFIG would create false duplicate-key
        # collisions against real config artifacts. See `_register()`.
        exported.append((metadata_path, None, "metadata"))

        # ── Checksums for every exported file ───────────────────────────
        files: list[dict[str, Any]] = []
        primary_hash = ""
        for path, artifact_type, tag in exported:
            sha256 = _sha256_file(path)
            if tag == "weights":
                primary_hash = sha256
            files.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "artifact_type": artifact_type,
                    "tag": tag,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256,
                }
            )

        # ── qde_manifest.json (full required field set) ─────────────────
        manifest: dict[str, Any] = {
            "model_version": model_version,
            "artifact_hash": primary_hash,
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "training_commit": training_commit,
            "calibration_version": calibration_version,
            "training_timestamp": training_timestamp if training_timestamp is not None else time.time(),
            "certification_status": certification_status,
            "promotion_status": promotion_status,
            "supported_symbols": supported_symbols or [],
            "supported_tasks": supported_tasks or [],
            "export_formats": list(export_formats),
            "framework_version": _framework_version(),
            # Snapshot, not a live reference: `files` is appended to below
            # (adding the manifest's own entry) *after* this dict is
            # serialised to disk, so the manifest never lists itself.
            "files": list(files),
        }
        missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in manifest]
        if missing:  # pragma: no cover — defensive; fields above are exhaustive
            raise QDEExportPipelineError(f"Manifest is missing required fields: {missing}")

        # NOTE: the manifest cannot embed a checksum of its own final byte
        # content (a file cannot hash itself — the checksum would change the
        # bytes, which would change the checksum, ad infinitum). So
        # `qde_manifest.json` on disk lists checksums only for the *other*
        # exported files; its own checksum is computed once, after the single
        # write below, and is reported/registered externally (in the
        # returned `files` list and the ArtifactRegistry record) where it is
        # guaranteed to match the real on-disk bytes exactly.
        manifest_path = store.write_text(model_version, MANIFEST_FILENAME, json.dumps(manifest, indent=2, default=str))
        manifest_sha256 = _sha256_file(manifest_path)
        files.append(
            {
                "name": manifest_path.name,
                "path": str(manifest_path),
                "artifact_type": ArtifactType.DEPLOYMENT_MANIFEST.value,
                "tag": "manifest",
                "size_bytes": manifest_path.stat().st_size,
                "sha256": manifest_sha256,
            }
        )

        # ── Register with ArtifactRegistry ──────────────────────────────
        artifacts: dict[str, str] = {}
        if self._registry is not None:
            for entry in files:
                if entry["artifact_type"] is None:
                    continue  # e.g. metadata.json — not one of the 6 canonical ArtifactTypes
                artifact_id = self._register(
                    entry=entry,
                    model_version=model_version,
                    dataset_version=dataset_version,
                    feature_version=feature_version,
                    code_commit=training_commit,
                    training_run_id=training_run_id,
                    experiment_id=experiment_id,
                    registered_by=registered_by,
                )
                if artifact_id:
                    artifacts[entry["tag"]] = artifact_id

        logger.info(
            "QDEExportPipeline: exported {} file(s) for model_version={!r} -> {}",
            len(files),
            model_version,
            version_dir,
        )

        return {
            "model_version": model_version,
            "output_dir": str(version_dir),
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "files": files,
            "artifacts": artifacts,
        }

    # ── Export steps (torch is imported lazily) ─────────────────────────────

    @staticmethod
    def _export_weights(model: Any, store: ArtifactStore, model_version: str) -> Path:
        import io

        import torch

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        # Written via `store.write_bytes` (atomic O_CREAT|O_EXCL) rather than
        # `torch.save(..., path)` directly to disk, so weights get the same
        # write-once guarantee as every other artifact kind.
        weights_path = store.write_bytes(model_version, WEIGHTS_FILENAME, buffer.getvalue())
        logger.info("QDEExportPipeline: weights exported: {}", weights_path)
        return weights_path

    @staticmethod
    def _export_torchscript(
        model: Any,
        config: Any,
        example_input_shape: tuple[int, ...],
        store: ArtifactStore,
        model_version: str,
    ) -> Path:
        import tempfile

        from trainer.exporters.torchscript import TorchScriptExporter

        exporter = TorchScriptExporter(model=model, config=config, example_input_shape=example_input_shape)
        with tempfile.TemporaryDirectory() as scratch:
            scratch_path = exporter.export(Path(scratch))[0]
            # Exporters write to an arbitrary scratch path; import the result
            # into the store so it goes through the same atomic, write-once
            # `write_bytes` path as every other artifact kind.
            return store.import_file(model_version, scratch_path.name, scratch_path)

    @staticmethod
    def _export_onnx(
        model: Any,
        config: Any,
        example_input_shape: tuple[int, ...],
        store: ArtifactStore,
        model_version: str,
    ) -> Path:
        import tempfile

        from trainer.exporters.onnx import ONNXExporter

        exporter = ONNXExporter(model=model, config=config, example_input_shape=example_input_shape)
        with tempfile.TemporaryDirectory() as scratch:
            scratch_path = exporter.export(Path(scratch))[0]
            return store.import_file(model_version, scratch_path.name, scratch_path)

    # ── Registry glue ────────────────────────────────────────────────────────

    def _register(
        self,
        entry: dict[str, Any],
        model_version: str,
        dataset_version: str,
        feature_version: str,
        code_commit: str,
        training_run_id: str,
        experiment_id: str,
        registered_by: str,
    ) -> str:
        try:
            record = ArtifactRecord.new(
                artifact_type=entry["artifact_type"],
                name=entry["name"],
                path=entry["path"],
                artifact_hash=entry["sha256"],
                training_run_id=training_run_id,
                experiment_id=experiment_id,
                model_version=model_version,
                dataset_version=dataset_version,
                feature_version=feature_version,
                code_commit=code_commit,
                size_bytes=entry["size_bytes"],
                registered_by=registered_by,
                tags=[entry["tag"]],
                notes=f"S21A unified export pipeline ({entry['tag']})",
            )
            self._registry.register_artifact(record)
            return record.artifact_id
        except DuplicateArtifactError as exc:
            # Re-exporting byte-identical content under a different version
            # legitimately collides on (type, hash) — tolerate it, the
            # artifact's provenance is already registered under its first
            # version.
            logger.warning(
                "QDEExportPipeline: artifact {!r} already registered ({}); "
                "skipping duplicate registration.",
                entry.get("name"),
                exc,
            )
            return ""
        except Exception:
            # Any other registration failure (bad record, registry outage,
            # ...) is surfaced rather than swallowed: the version directory
            # has already been reserved/written and is now write-once, so a
            # silently-unregistered artifact would be both unrecoverable
            # (can't re-export this version) and undiscoverable (no
            # registry record points at it). Failing loudly lets the caller
            # decide whether to retry registration against the already
            # exported files or export a fresh version.
            logger.error(
                "QDEExportPipeline: registration failed for artifact {!r} "
                "(model_version={!r}); files are already written to {!r} but "
                "NOT registered.",
                entry.get("name"),
                model_version,
                entry.get("path"),
            )
            raise
