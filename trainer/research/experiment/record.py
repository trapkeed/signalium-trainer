"""ExperimentRecord — immutable metadata captured for every training run."""

from __future__ import annotations

import platform
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _hardware_info() -> dict[str, Any]:
    """Capture CPU, OS, and GPU information."""
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": None,
        "gpu": None,
        "gpu_memory_gb": None,
        "ram_gb": None,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception:
        pass
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        pass
    return info


@dataclass
class ExperimentRecord:
    """Immutable metadata captured at the start and end of every training run.

    All fields except ``experiment_id`` and ``created_at`` are settable after
    construction so the record can be updated incrementally as training progresses.

    Attributes:
        experiment_id: Globally unique identifier (UUID4).
        name: Human-readable experiment name.
        model_name: Registry name of the model (e.g. ``"quant_v1"``).
        model_version: Semantic version of the model (e.g. ``"1.0.0"``).
        dataset_version: Version string for the dataset used.
        feature_version: Version string for the feature pipeline.
        label_version: Version string for the label scheme.
        git_commit_hash: Git HEAD at experiment start.
        configuration: Full serialised configuration dict.
        hyperparameters: Hyperparameter key→value snapshot.
        random_seed: Global random seed used for reproducibility.
        training_duration_seconds: Wall-clock training time.
        hardware_info: CPU/GPU/RAM snapshot.
        mlflow_run_id: MLflow run ID if MLflow was active.
        metrics: Final metric dict at the end of the experiment.
        start_time: UTC datetime when the experiment started.
        end_time: UTC datetime when the experiment ended (or None if still running).
        status: ``"running"`` | ``"completed"`` | ``"failed"``.
        notes: Free-text notes about the experiment.
    """

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    model_name: str = ""
    model_version: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    git_commit_hash: str = field(default_factory=_git_commit)
    configuration: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    training_duration_seconds: float | None = None
    hardware_info: dict[str, Any] = field(default_factory=_hardware_info)
    mlflow_run_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    status: str = "running"
    notes: str = ""

    def complete(self, metrics: dict[str, float]) -> None:
        """Mark the experiment as completed and record final metrics.

        Args:
            metrics: Final evaluation metric dict.
        """
        self.metrics = metrics
        self.end_time = datetime.now(UTC)
        if self.start_time:
            self.training_duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = "completed"

    def fail(self, reason: str = "") -> None:
        """Mark the experiment as failed.

        Args:
            reason: Optional failure reason written to notes.
        """
        self.end_time = datetime.now(UTC)
        self.status = "failed"
        if reason:
            self.notes = reason

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record to a JSON-compatible dict."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "git_commit_hash": self.git_commit_hash,
            "configuration": self.configuration,
            "hyperparameters": self.hyperparameters,
            "random_seed": self.random_seed,
            "training_duration_seconds": self.training_duration_seconds,
            "hardware_info": self.hardware_info,
            "mlflow_run_id": self.mlflow_run_id,
            "metrics": self.metrics,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentRecord:
        """Deserialise from a dict (e.g. loaded from JSON).

        Args:
            d: Dict produced by :meth:`to_dict`.

        Returns:
            Reconstructed ExperimentRecord.
        """
        rec = cls.__new__(cls)
        rec.experiment_id = d.get("experiment_id", str(uuid.uuid4()))
        rec.name = d.get("name", "")
        rec.model_name = d.get("model_name", "")
        rec.model_version = d.get("model_version", "")
        rec.dataset_version = d.get("dataset_version", "")
        rec.feature_version = d.get("feature_version", "")
        rec.label_version = d.get("label_version", "")
        rec.git_commit_hash = d.get("git_commit_hash", "unknown")
        rec.configuration = d.get("configuration", {})
        rec.hyperparameters = d.get("hyperparameters", {})
        rec.random_seed = d.get("random_seed", 42)
        rec.training_duration_seconds = d.get("training_duration_seconds")
        rec.hardware_info = d.get("hardware_info", {})
        rec.mlflow_run_id = d.get("mlflow_run_id")
        rec.metrics = d.get("metrics", {})
        rec.status = d.get("status", "unknown")
        rec.notes = d.get("notes", "")
        raw_start = d.get("start_time")
        rec.start_time = datetime.fromisoformat(raw_start) if raw_start else datetime.now(UTC)
        raw_end = d.get("end_time")
        rec.end_time = datetime.fromisoformat(raw_end) if raw_end else None
        return rec
