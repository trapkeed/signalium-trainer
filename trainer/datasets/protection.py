"""S29 — Dataset version protection: write-once immutability + integrity
enforcement, and training audit logging on every access.

Existing infra (kept as-is, not duplicated):
- ``DatasetVersion`` (``versioning.py``) — the version record itself.
- ``DatasetValidator`` (``validation.py``) — manifest + per-file SHA-256
  checksum verification.
- ``AuditSystem`` (``trainer/mlops/audit.py``) — the general-purpose
  training/validation audit log.

What was missing: nothing stopped a caller from re-writing
``manifest.json`` for an already-published dataset version (no write-once
guard, unlike ``ArtifactStore``'s sentinel-file pattern for model
artifacts), and reads weren't automatically checksum-verified or
audit-logged. This module closes both gaps for dataset versions, mirroring
``ArtifactStore``'s reservation pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from trainer.datasets.validation import DatasetValidator, ValidationResult
from trainer.datasets.versioning import DatasetVersion
from trainer.mlops.audit import AuditEvent, AuditEventType, AuditSystem


class DatasetImmutabilityError(Exception):
    """Raised when a caller attempts to modify an already-published
    dataset version's manifest."""


class DatasetIntegrityError(Exception):
    """Raised when a dataset version fails checksum/manifest validation
    on read."""


#: Sentinel marking a dataset version directory as published/write-once,
#: mirroring ``ArtifactStore._RESERVATION_SENTINEL``.
_PUBLISHED_SENTINEL = ".published"


def publish_dataset_version(
    version: DatasetVersion,
    manifest_path: Path,
    *,
    audit: AuditSystem | None = None,
    actor: str = "system",
) -> None:
    """Write *version*'s manifest exactly once, then seal the directory.

    Raises:
        DatasetImmutabilityError: *manifest_path*'s parent directory has
            already been published — dataset versions are immutable once
            created; produce a new version instead of overwriting.
    """
    directory = manifest_path.parent
    sentinel = directory / _PUBLISHED_SENTINEL
    if sentinel.exists():
        raise DatasetImmutabilityError(
            f"Dataset version {version.version!r} at {directory} has already "
            "been published — dataset versions are immutable and write-once. "
            "Create a new version instead of overwriting this one."
        )

    version.save(manifest_path)

    try:
        fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        os.close(fd)
    except FileExistsError as exc:  # pragma: no cover — race between two publishers
        raise DatasetImmutabilityError(
            f"Dataset version {version.version!r} at {directory} was published "
            "concurrently by another writer."
        ) from exc

    (audit or AuditSystem()).record(
        AuditEvent(
            event_type=AuditEventType.TRAINING,
            actor=actor,
            model_id=version.dataset_id,
            description=f"Published dataset version {version.version!r}",
            metadata={"dataset_id": version.dataset_id, "version": version.version},
        )
    )
    logger.info("dataset.protection: published {} v{} at {}", version.dataset_id, version.version, directory)


def is_published(manifest_path: Path) -> bool:
    """Return True if the dataset version at *manifest_path*'s directory
    has already been sealed by ``publish_dataset_version``."""
    return (manifest_path.parent / _PUBLISHED_SENTINEL).exists()


def load_dataset_version_securely(
    dataset_dir: Path,
    *,
    manifest_name: str = "manifest.json",
    verify_checksums: bool = True,
    audit: AuditSystem | None = None,
    actor: str = "system",
) -> tuple[DatasetVersion, ValidationResult]:
    """Read-path counterpart to ``publish_dataset_version``: validate
    integrity (checksums via ``DatasetValidator``) before returning the
    parsed ``DatasetVersion``, and record a training audit event for the
    access either way (successful or failed) so every read of a protected
    dataset version is traceable.

    Raises:
        DatasetIntegrityError: checksum/manifest validation failed.
    """
    audit = audit or AuditSystem()
    validator = DatasetValidator(dataset_dir, manifest_name=manifest_name, verify_checksums=verify_checksums, fail_on_error=False)
    result = validator.validate()

    manifest_path = Path(dataset_dir) / manifest_name
    if not result.passed:
        audit.record(
            AuditEvent(
                event_type=AuditEventType.INFERENCE_ERROR,
                actor=actor,
                model_id=str(dataset_dir),
                description="Dataset version failed integrity validation on read",
                metadata={"errors": result.errors, "warnings": result.warnings},
            )
        )
        raise DatasetIntegrityError(
            f"Dataset at {dataset_dir} failed integrity validation:\n{result.summary()}"
        )

    version = DatasetVersion.from_manifest(manifest_path)
    audit.record(
        AuditEvent(
            event_type=AuditEventType.TRAINING,
            actor=actor,
            model_id=version.dataset_id,
            description=f"Read dataset version {version.version!r} (integrity verified)",
            metadata={"dataset_id": version.dataset_id, "version": version.version},
        )
    )
    return version, result


__all__ = [
    "DatasetImmutabilityError",
    "DatasetIntegrityError",
    "load_dataset_version_securely",
    "is_published",
    "publish_dataset_version",
]
