"""Signalium REST API client package."""

from trainer.api.client import SignaliumClient
from trainer.api.exceptions import (
    APIAuthenticationError,
    APIError,
    APINotFoundError,
    APIRateLimitError,
    APITimeoutError,
)
from trainer.api.models import (
    DatasetManifest,
    ModelManifest,
    UploadMetricsRequest,
    UploadModelRequest,
)

__all__ = [
    "SignaliumClient",
    "APIError",
    "APIAuthenticationError",
    "APINotFoundError",
    "APIRateLimitError",
    "APITimeoutError",
    "DatasetManifest",
    "ModelManifest",
    "UploadMetricsRequest",
    "UploadModelRequest",
]
