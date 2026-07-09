"""S20 — ArtifactRegistry: register, query, and verify ML artifacts.

The registry is a **metadata-only** store: it records provenance information
about artifacts that are physically stored by the training infrastructure
(checkpoints directory, ONNX/TorchScript exports, deployment packages).  It
does not copy, move, or host files.

Usage::

    registry = ArtifactRegistry(db)

    # Register a training run first (optional but recommended for full lineage)
    run = TrainingRunRecord.new(
        experiment_id="exp-abc123",
        dataset_version="ds-v3.1",
        feature_version="feat-v2.0",
        code_commit="a1b2c3d4e5f6...",
        config_hash="deadbeef..." * 4,   # 64-char SHA-256
        artifact_hash="feedcafe..." * 4, # 64-char SHA-256
    )
    registry.register_training_run(run)

    # Register each artifact produced by the run
    art = ArtifactRecord.new(
        artifact_type=ArtifactType.WEIGHTS.value,
        name="signalium_v3_weights.pt",
        path="/exports/signalium_v3/weights.pt",
        artifact_hash="feedcafe..." * 4,
        training_run_id=run.run_id,
        ...
    )
    registry.register_artifact(art)

    # Query
    artifacts = registry.list_artifacts(artifact_type="WEIGHTS", limit=10)
    provenance = registry.get_artifact("art-abc123def456")
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.research_lab.artifact_models import (
    ARTIFACT_TYPES,
    ArtifactRecord,
    ArtifactType,
    TrainingRunRecord,
    validate_artifact_hash,
    validate_artifact_type,
)
from trainer.research_lab.artifact_store import (
    ArtifactImmutabilityError,
    ArtifactKind,
    ArtifactStore,
)
from trainer.research_lab.rl_database import ResearchLabDatabase


# ── Errors ────────────────────────────────────────────────────────────────────


class DuplicateArtifactError(Exception):
    """Raised when an artifact with the same (type, hash) is already registered."""


class ArtifactHashMismatchError(Exception):
    """Raised when a file's actual SHA-256 does not match the claimed hash."""


# ── Registry ──────────────────────────────────────────────────────────────────


class ArtifactRegistry:
    """Metadata registry for all ML artifacts produced by the training pipeline.

    Args:
        db: ``ResearchLabDatabase`` instance used for persistence.
    """

    def __init__(self, db: ResearchLabDatabase, store: ArtifactStore | None = None) -> None:
        self._db = db
        self.store = store or ArtifactStore()

    # ── Training run registration ──────────────────────────────────────────────

    def register_training_run(self, record: TrainingRunRecord) -> TrainingRunRecord:
        """Persist a ``TrainingRunRecord`` and return it unchanged."""
        self._db.upsert_training_run(record.to_dict())
        logger.info(
            f"ArtifactRegistry: registered training run {record.run_id!r} "
            f"(experiment={record.experiment_id!r} commit={record.code_commit[:12]!r})"
        )
        return record

    def get_training_run(self, run_id: str) -> TrainingRunRecord | None:
        """Return the ``TrainingRunRecord`` with *run_id*, or ``None``."""
        row = self._db.get_training_run(run_id)
        return TrainingRunRecord.from_dict(row) if row else None

    def list_training_runs(
        self,
        experiment_id: str | None = None,
        pipeline_job_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List training runs, newest first."""
        return self._db.list_training_runs(
            experiment_id=experiment_id,
            pipeline_job_id=pipeline_job_id,
            limit=limit,
        )

    # ── Artifact registration ──────────────────────────────────────────────────

    def register_artifact(
        self, record: ArtifactRecord, *, verify_file: bool = False
    ) -> ArtifactRecord:
        """Register *record* in the artifact registry.

        Parameters
        ----------
        record:
            The artifact to register.  Must have a valid 64-char SHA-256
            ``artifact_hash`` (enforced by ``ArtifactRecord.new()``).
        verify_file:
            If ``True`` **and** ``record.path`` points to a readable local
            file, re-compute the SHA-256 and raise ``ArtifactHashMismatchError``
            if it does not match ``record.artifact_hash``.  Set to ``False``
            (the default) when the path is remote or not yet materialized.

        Raises
        ------
        ArtifactHashMismatchError:
            ``verify_file=True`` and the computed hash differs from the
            declared hash.
        DuplicateArtifactError:
            An artifact with the same ``(artifact_type, artifact_hash)`` is
            already in the registry.
        """
        # Optional file-level hash verification
        if verify_file and record.path:
            path = Path(record.path)
            if path.exists():
                actual = self._hash_file(path)
                if actual != record.artifact_hash:
                    raise ArtifactHashMismatchError(
                        f"Hash mismatch for {record.path!r}: "
                        f"declared={record.artifact_hash!r}, "
                        f"computed={actual!r}"
                    )

        # Duplicate detection: same (type, hash) is rejected
        existing = self._db.find_artifact_by_type_hash(
            record.artifact_type, record.artifact_hash
        )
        if existing:
            raise DuplicateArtifactError(
                f"Artifact (type={record.artifact_type!r}, "
                f"hash={record.artifact_hash[:16]}...) already registered "
                f"as {existing['artifact_id']!r}."
            )

        self._db.upsert_artifact(record.to_dict())
        logger.info(
            f"ArtifactRegistry: registered {record.artifact_type} artifact "
            f"{record.artifact_id!r} ({record.name!r})"
        )
        return record

    def write_and_register(
        self,
        *,
        artifact_type: str,
        version: str,
        filename: str,
        data: bytes,
        kind: str | ArtifactKind | None = None,
        **record_kwargs: Any,
    ) -> ArtifactRecord:
        """Write *data* into the immutable, versioned on-disk store and
        register the resulting file in one call.

        This is the preferred way for new callers to produce an artifact:
        it guarantees ``ArtifactRecord.path`` always resolves to a real
        store-managed path (never an arbitrary caller-supplied path) and
        that the write is rejected outright if *version*'s directory is
        already populated (see ``ArtifactStore.reserve_version`` /
        ``ArtifactStore.write_bytes``).

        Raises:
            ArtifactImmutabilityError: the target file (or, on first write
                for a version, the version directory itself) already exists.
        """
        path = self.store.write_bytes(version, filename, data, kind=kind)
        record = ArtifactRecord.new(
            artifact_type=artifact_type,
            name=filename,
            path=str(path),
            artifact_hash=self.hash_bytes(data),
            model_version=version,
            size_bytes=len(data),
            **record_kwargs,
        )
        return self.register_artifact(record)

    def resolve_path(self, version: str, filename: str, kind: str | ArtifactKind | None = None) -> Path:
        """Return the store path for *filename* under *version* without
        requiring the file to already exist — the canonical way for readers
        to locate an artifact instead of trusting an arbitrary caller path."""
        return self.store.path_for(version, filename, kind=kind)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        """Return the ``ArtifactRecord`` with *artifact_id*, or ``None``."""
        row = self._db.get_artifact(artifact_id)
        return ArtifactRecord.from_dict(row) if row else None

    def list_artifacts(
        self,
        artifact_type: str | None = None,
        model_version: str | None = None,
        training_run_id: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List registered artifacts (newest first) with optional filters."""
        return self._db.list_artifacts(
            artifact_type=artifact_type,
            model_version=model_version,
            training_run_id=training_run_id,
            experiment_id=experiment_id,
            limit=limit,
        )

    def artifacts_for_run(self, training_run_id: str) -> list[dict[str, Any]]:
        """Return all artifacts associated with *training_run_id*."""
        return self._db.list_artifacts(training_run_id=training_run_id, limit=10_000)

    def provenance(self, artifact_id: str) -> dict[str, Any] | None:
        """Return a combined provenance view: artifact + its training run (if any).

        Returns ``None`` if the artifact is not found.
        """
        art_row = self._db.get_artifact(artifact_id)
        if art_row is None:
            return None
        result: dict[str, Any] = {"artifact": art_row, "training_run": None}
        run_id = art_row.get("training_run_id", "")
        if run_id:
            result["training_run"] = self._db.get_training_run(run_id)
        return result

    # ── Hash utilities ────────────────────────────────────────────────────────

    @staticmethod
    def hash_file(path: Path) -> str:
        """Compute the SHA-256 hex digest of a local file."""
        return ArtifactRegistry._hash_file(path)

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Compute the SHA-256 hex digest of in-memory *data*."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                h.update(chunk)
        return h.hexdigest()
