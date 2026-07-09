"""S20 — Dataset Versioning + Artifact Management data models.

Defines:
  - ArtifactType    — six registered artifact types
  - TrainingRunRecord — full provenance for one training run
  - ArtifactRecord  — a registered artifact with its provenance chain

No model is exposed to retail users.  All access is via admin APIs only.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

_SHA256_HEX_LEN = 64


# ── Enumerations ──────────────────────────────────────────────────────────────


class ArtifactType(str, Enum):
    """The six canonical artifact types produced by the training pipeline."""

    WEIGHTS = "WEIGHTS"
    CONFIG = "CONFIG"
    CALIBRATION = "CALIBRATION"
    ONNX = "ONNX"
    TORCHSCRIPT = "TORCHSCRIPT"
    DEPLOYMENT_MANIFEST = "DEPLOYMENT_MANIFEST"


ARTIFACT_TYPES: list[str] = [e.value for e in ArtifactType]


# ── Validation helpers ────────────────────────────────────────────────────────


def validate_artifact_hash(artifact_hash: str) -> None:
    """Raise ``ValueError`` unless *artifact_hash* is a valid SHA-256 hex string."""
    if not artifact_hash or len(artifact_hash) != _SHA256_HEX_LEN:
        raise ValueError(
            f"artifact_hash must be a 64-character SHA-256 hex string, "
            f"got {len(artifact_hash)!r} chars: {artifact_hash[:20]!r}..."
        )
    try:
        int(artifact_hash, 16)
    except ValueError:
        raise ValueError(f"artifact_hash is not a valid hex string: {artifact_hash[:20]!r}...")


def validate_artifact_type(artifact_type: str) -> None:
    """Raise ``ValueError`` if *artifact_type* is not a recognised ``ArtifactType``."""
    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(
            f"Unknown artifact_type {artifact_type!r}. " f"Valid types: {ARTIFACT_TYPES}"
        )


def compute_bytes_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


# ── TrainingRunRecord ─────────────────────────────────────────────────────────


@dataclass
class TrainingRunRecord:
    """Full provenance record for one training run.

    A ``TrainingRunRecord`` ties together every ingredient that went into
    producing a set of model artifacts, making each run fully reproducible:

    - ``dataset_version``  — version tag of the dataset used for training
    - ``feature_version``  — version tag of the feature engineering pipeline
    - ``code_commit``      — git SHA of the training codebase
    - ``config_hash``      — SHA-256 of the serialised training configuration
    - ``artifact_hash``    — SHA-256 of the primary weights artifact
    - ``experiment_id``    — Research Lab experiment that triggered this run
    - ``pipeline_job_id``  — Training pipeline job that executed this run
    - ``mlflow_run_id``    — MLflow run ID (if applicable)
    """

    run_id: str
    experiment_id: str
    dataset_version: str
    feature_version: str
    code_commit: str
    config_hash: str
    artifact_hash: str
    created_at: float

    requested_by: str = ""
    pipeline_job_id: str = ""
    mlflow_run_id: str = ""
    model_version: str = ""
    label_version: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        experiment_id: str,
        dataset_version: str,
        feature_version: str,
        code_commit: str,
        config_hash: str,
        artifact_hash: str,
        requested_by: str = "",
        pipeline_job_id: str = "",
        mlflow_run_id: str = "",
        model_version: str = "",
        label_version: str = "",
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TrainingRunRecord:
        validate_artifact_hash(artifact_hash)
        return cls(
            run_id=f"run-{str(uuid.uuid4())[:12]}",
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            feature_version=feature_version,
            code_commit=code_commit,
            config_hash=config_hash,
            artifact_hash=artifact_hash,
            created_at=time.time(),
            requested_by=requested_by,
            pipeline_job_id=pipeline_job_id,
            mlflow_run_id=mlflow_run_id,
            model_version=model_version,
            label_version=label_version,
            notes=notes,
            metadata=metadata or {},
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
            "artifact_hash": self.artifact_hash,
            "created_at": round(self.created_at, 3),
            "requested_by": self.requested_by,
            "pipeline_job_id": self.pipeline_job_id,
            "mlflow_run_id": self.mlflow_run_id,
            "model_version": self.model_version,
            "label_version": self.label_version,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingRunRecord:
        return cls(
            run_id=d["run_id"],
            experiment_id=d.get("experiment_id", ""),
            dataset_version=d.get("dataset_version", ""),
            feature_version=d.get("feature_version", ""),
            code_commit=d.get("code_commit", ""),
            config_hash=d.get("config_hash", ""),
            artifact_hash=d.get("artifact_hash", ""),
            created_at=d.get("created_at", 0.0),
            requested_by=d.get("requested_by", ""),
            pipeline_job_id=d.get("pipeline_job_id", ""),
            mlflow_run_id=d.get("mlflow_run_id", ""),
            model_version=d.get("model_version", ""),
            label_version=d.get("label_version", ""),
            notes=d.get("notes", ""),
            metadata=d.get("metadata", {}),
        )


# ── ArtifactRecord ────────────────────────────────────────────────────────────


@dataclass
class ArtifactRecord:
    """A registered model artifact with full provenance metadata.

    Attributes
    ----------
    artifact_id:
        Unique stable identifier (``art-<uuid12>``).
    artifact_type:
        One of the six ``ArtifactType`` values.
    name:
        Human-readable artifact name (e.g. ``"signalium_v3_weights.pt"``).
    path:
        Logical path used by the training infrastructure.  Physical storage
        (S3, GCS, local disk) is managed by the exporter; the registry tracks
        metadata only.
    artifact_hash:
        SHA-256 hex digest of the artifact file content.  Verified on
        registration when ``verify_file=True``.
    registered_at:
        Unix timestamp of registration.

    Provenance chain
    ----------------
    training_run_id, experiment_id, model_version, dataset_version,
    feature_version, code_commit, config_hash, pipeline_job_id, mlflow_run_id.
    """

    artifact_id: str
    artifact_type: str
    name: str
    path: str
    artifact_hash: str
    registered_at: float

    # Provenance
    training_run_id: str = ""
    experiment_id: str = ""
    model_version: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    code_commit: str = ""
    config_hash: str = ""
    pipeline_job_id: str = ""
    mlflow_run_id: str = ""

    # Metadata
    size_bytes: int = 0
    registered_by: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        artifact_type: str,
        name: str,
        path: str,
        artifact_hash: str,
        training_run_id: str = "",
        experiment_id: str = "",
        model_version: str = "",
        dataset_version: str = "",
        feature_version: str = "",
        code_commit: str = "",
        config_hash: str = "",
        pipeline_job_id: str = "",
        mlflow_run_id: str = "",
        size_bytes: int = 0,
        registered_by: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        extra: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        validate_artifact_hash(artifact_hash)
        validate_artifact_type(artifact_type)
        return cls(
            artifact_id=f"art-{str(uuid.uuid4())[:12]}",
            artifact_type=artifact_type,
            name=name,
            path=path,
            artifact_hash=artifact_hash,
            registered_at=time.time(),
            training_run_id=training_run_id,
            experiment_id=experiment_id,
            model_version=model_version,
            dataset_version=dataset_version,
            feature_version=feature_version,
            code_commit=code_commit,
            config_hash=config_hash,
            pipeline_job_id=pipeline_job_id,
            mlflow_run_id=mlflow_run_id,
            size_bytes=size_bytes,
            registered_by=registered_by,
            tags=tags or [],
            notes=notes,
            extra=extra or {},
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "name": self.name,
            "path": self.path,
            "artifact_hash": self.artifact_hash,
            "registered_at": round(self.registered_at, 3),
            "training_run_id": self.training_run_id,
            "experiment_id": self.experiment_id,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
            "pipeline_job_id": self.pipeline_job_id,
            "mlflow_run_id": self.mlflow_run_id,
            "size_bytes": self.size_bytes,
            "registered_by": self.registered_by,
            "tags": self.tags,
            "notes": self.notes,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArtifactRecord:
        return cls(
            artifact_id=d["artifact_id"],
            artifact_type=d.get("artifact_type", ""),
            name=d.get("name", ""),
            path=d.get("path", ""),
            artifact_hash=d.get("artifact_hash", ""),
            registered_at=d.get("registered_at", 0.0),
            training_run_id=d.get("training_run_id", ""),
            experiment_id=d.get("experiment_id", ""),
            model_version=d.get("model_version", ""),
            dataset_version=d.get("dataset_version", ""),
            feature_version=d.get("feature_version", ""),
            code_commit=d.get("code_commit", ""),
            config_hash=d.get("config_hash", ""),
            pipeline_job_id=d.get("pipeline_job_id", ""),
            mlflow_run_id=d.get("mlflow_run_id", ""),
            size_bytes=d.get("size_bytes", 0),
            registered_by=d.get("registered_by", ""),
            tags=d.get("tags", []),
            notes=d.get("notes", ""),
            extra=d.get("extra", {}),
        )
