"""API-specific exception hierarchy."""

from __future__ import annotations


class APIError(Exception):
    """Base exception for all Signalium API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIAuthenticationError(APIError):
    """Raised when authentication fails (HTTP 401/403)."""


class APINotFoundError(APIError):
    """Raised when a resource is not found (HTTP 404)."""


class APIRateLimitError(APIError):
    """Raised when rate limit is exceeded (HTTP 429)."""


class APITimeoutError(APIError):
    """Raised when the request times out."""


class APIServerError(APIError):
    """Raised on 5xx server errors."""


class ChecksumMismatchError(Exception):
    """Raised when a downloaded file fails its checksum verification."""

    def __init__(self, path: str, expected: str, actual: str) -> None:
        super().__init__(f"Checksum mismatch for {path}: expected {expected}, got {actual}")
        self.path = path
        self.expected = expected
        self.actual = actual
