"""S21A — Immutable, versioned on-disk artifact store.

Defines the physical layout backing ``ArtifactRegistry`` records: one
directory per artifact version, created once and never written to again.

Layout::

    <root>/
      <version>/
        weights.pt
        model.torchscript.pt
        model.onnx
        config.json
        calibration.json
        metadata.json
        qde_manifest.json
        ... (validation/certification/research reports, if routed here)

``ArtifactStore`` only manages the local filesystem today, but every method
is expressed in terms of ``(version, filename, kind)`` rather than raw
paths, so a future S3/GCS-backed implementation can satisfy the same
interface without callers changing.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

#: Env var used to override the default store root (mirrors
#: RESEARCH_LAB_DB_PATH's convention for the database file).
ARTIFACT_STORE_ROOT_ENV = "ARTIFACT_STORE_ROOT"
DEFAULT_ARTIFACT_STORE_ROOT = "data/artifact_store"

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.\-]")


class ArtifactImmutabilityError(Exception):
    """Raised when a caller attempts to write into an already-populated
    artifact version directory."""


class ArtifactKind(str, Enum):
    """The artifact kinds S21A requires the store to persist.

    Distinct from ``ArtifactType`` in ``artifact_models.py`` (which is the
    six-value registry dedup key) — ``ArtifactKind`` is purely a storage
    layout/categorization concept, so it can describe artifacts (e.g.
    validation/certification/research reports) that never get an
    ``ArtifactRegistry`` record of their own.
    """

    WEIGHTS = "weights"
    CONFIG = "config"
    CALIBRATION = "calibration"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    DEPLOYMENT_MANIFEST = "deployment_manifest"
    METADATA = "metadata"
    VALIDATION_REPORT = "validation_report"
    CERTIFICATION_REPORT = "certification_report"
    RESEARCH_REPORT = "research_report"


def _sanitize(segment: str) -> str:
    """Collapse a version/filename segment to safe path characters.

    Prevents path traversal (``../``) or absolute-path escapes from a
    caller-supplied version string reaching outside the store root.
    """
    cleaned = _UNSAFE_CHARS.sub("_", str(segment).strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid artifact store path segment: {segment!r}")
    return cleaned


class ArtifactStore:
    """Filesystem-backed, write-once artifact store keyed by version.

    Args:
        root: Store root directory. Defaults to the ``ARTIFACT_STORE_ROOT``
            env var, falling back to ``data/artifact_store``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = os.environ.get(ARTIFACT_STORE_ROOT_ENV, DEFAULT_ARTIFACT_STORE_ROOT)
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    # ── Path resolution ──────────────────────────────────────────────────────

    def version_dir(self, version: str) -> Path:
        """Return the (not-necessarily-existing) directory for *version*."""
        return self._root / _sanitize(version)

    def path_for(self, version: str, filename: str, kind: str | ArtifactKind | None = None) -> Path:
        """Resolve the on-disk path for *filename* under *version*.

        ``kind`` is accepted so non-export-pipeline producers (validation,
        certification, research reports) can namespace their files without
        colliding with the pipeline's flat filenames; it is optional because
        the unified export pipeline's artifact kinds already have unique
        canonical filenames (``weights.pt``, ``config.json``, ...).
        """
        directory = self.version_dir(version)
        if kind is not None:
            kind_value = kind.value if isinstance(kind, ArtifactKind) else str(kind)
            directory = directory / _sanitize(kind_value)
        return directory / _sanitize(filename)

    # ── Immutability enforcement ─────────────────────────────────────────────

    #: Sentinel file used to atomically claim a version directory. Its
    #: presence — not the presence of any particular artifact file — is the
    #: single source of truth for "this version has already been exported".
    _RESERVATION_SENTINEL = ".reserved"

    def reserve_version(self, version: str) -> Path:
        """Atomically claim the directory for a brand-new *version*.

        Uses an O_CREAT|O_EXCL sentinel file inside the version directory as
        a single-writer claim: creating that file is one atomic kernel
        syscall, so two concurrent callers racing to reserve the same
        version can never both succeed — exactly one wins, the other raises.
        This is stronger than a `directory.exists()` / "has any files" check,
        which is a check-then-act race under concurrent exporters.

        Raises:
            ArtifactImmutabilityError: *version* was already reserved/exported
                — artifact versions are write-once; produce a new version
                instead of overwriting.
        """
        directory = self.version_dir(version)
        directory.mkdir(parents=True, exist_ok=True)
        # Guard against a version that already has real files but no
        # sentinel (e.g. files written directly without ever reserving).
        if any(p for p in directory.rglob("*") if p.name != self._RESERVATION_SENTINEL):
            raise ArtifactImmutabilityError(
                f"Artifact store version {version!r} already has files under "
                f"{directory} — artifact versions are immutable and write-once. "
                "Export a new version instead of re-exporting this one."
            )
        sentinel = directory / self._RESERVATION_SENTINEL
        try:
            fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
            os.close(fd)
        except FileExistsError as exc:
            raise ArtifactImmutabilityError(
                f"Artifact store version {version!r} is already reserved/exported "
                f"under {directory} — artifact versions are immutable and write-once. "
                "Export a new version instead of re-exporting this one."
            ) from exc
        return directory

    def is_populated(self, version: str) -> bool:
        """Return True if *version* has already been reserved/exported."""
        directory = self.version_dir(version)
        return (directory / self._RESERVATION_SENTINEL).exists() or (
            directory.exists() and any(directory.rglob("*"))
        )

    # ── Writes ───────────────────────────────────────────────────────────────

    def write_bytes(
        self, version: str, filename: str, data: bytes, kind: str | ArtifactKind | None = None
    ) -> Path:
        """Write *data* to ``path_for(version, filename, kind)``.

        Raises:
            ArtifactImmutabilityError: the target file already exists.
        """
        path = self.path_for(version, filename, kind=kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic create-exclusive: os.O_CREAT | os.O_EXCL fails with EEXIST if
        # *path* already exists, and the check + creation happen as a single
        # kernel syscall — unlike a `path.exists()` check followed by a
        # separate write, this cannot lose a race between two concurrent
        # writers both targeting the same (version, filename).
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise ArtifactImmutabilityError(
                f"Artifact {path} already exists — artifacts are write-once; "
                "it cannot be overwritten in place."
            ) from exc
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        except BaseException:
            # Best-effort cleanup of a partially written file so a failed
            # write doesn't permanently "poison" this (version, filename).
            path.unlink(missing_ok=True)
            raise
        _make_read_only(path)
        logger.info("ArtifactStore: wrote {} ({} bytes)", path, len(data))
        return path

    def write_text(
        self, version: str, filename: str, text: str, kind: str | ArtifactKind | None = None
    ) -> Path:
        return self.write_bytes(version, filename, text.encode("utf-8"), kind=kind)

    def import_file(
        self, version: str, filename: str, src: str | Path, kind: str | ArtifactKind | None = None
    ) -> Path:
        """Copy an already-materialized file at *src* into the store."""
        return self.write_bytes(version, filename, Path(src).read_bytes(), kind=kind)

    # ── Reads ────────────────────────────────────────────────────────────────

    def read_bytes(self, version: str, filename: str, kind: str | ArtifactKind | None = None) -> bytes:
        return self.path_for(version, filename, kind=kind).read_bytes()

    def list_version(self, version: str) -> list[Path]:
        """Return every file stored under *version*, recursively."""
        directory = self.version_dir(version)
        if not directory.exists():
            return []
        return sorted(
            p
            for p in directory.rglob("*")
            if p.is_file() and p.name != self._RESERVATION_SENTINEL
        )

    def list_versions(self) -> list[str]:
        """Return every version directory name currently in the store."""
        if not self._root.exists():
            return []
        return sorted(p.name for p in self._root.iterdir() if p.is_dir())


def _make_read_only(path: Path) -> None:
    """Best-effort: drop write permission to reinforce immutability at the
    filesystem level. Never fails the write if chmod is unsupported."""
    try:
        os.chmod(path, 0o444)
    except OSError:  # pragma: no cover — platform-dependent, non-fatal
        pass
