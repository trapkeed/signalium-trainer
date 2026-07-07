"""Pydantic models for Signalium API request/response payloads."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DatasetStatus(str, Enum):
    """Dataset lifecycle status."""

    PENDING = "pending"
    READY = "ready"
    EXPIRED = "expired"
    ERROR = "error"


class ModelStatus(str, Enum):
    """Model upload status."""

    UPLOADING = "uploading"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"


class FileEntry(BaseModel):
    """A single file in a manifest."""

    name: str
    path: str
    size_bytes: int
    sha256: str
    content_type: str = "application/octet-stream"


class DatasetManifest(BaseModel):
    """Manifest returned by the dataset download endpoint."""

    dataset_id: str
    version: str
    status: DatasetStatus
    files: list[FileEntry]
    feature_version: str | None = None
    label_version: str | None = None
    created_at: datetime
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    download_url_base: str | None = None


class ModelManifest(BaseModel):
    """Manifest describing a trained model artifact."""

    model_id: str
    version: str
    framework: str
    architecture: str
    files: list[FileEntry]
    training_run_id: str | None = None
    dataset_version: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UploadModelRequest(BaseModel):
    """Request body for uploading a model to Signalium."""

    model_id: str
    version: str
    framework: str
    architecture: str
    run_id: str
    dataset_version: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UploadMetricsRequest(BaseModel):
    """Request body for uploading training metrics."""

    run_id: str
    epoch: int
    metrics: dict[str, float]
    step: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UploadLogsRequest(BaseModel):
    """Request body for uploading log chunks."""

    run_id: str
    log_content: str
    log_level: str = "INFO"
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    version: str
    timestamp: datetime
