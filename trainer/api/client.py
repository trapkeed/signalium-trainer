"""Signalium API client with retry logic and structured error handling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import requests
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trainer.api.auth import resolve_auth
from trainer.api.exceptions import (
    APIAuthenticationError,
    APIError,
    APINotFoundError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    ChecksumMismatchError,
)
from trainer.api.models import (
    DatasetManifest,
    HealthResponse,
    ModelManifest,
    UploadLogsRequest,
    UploadMetricsRequest,
    UploadModelRequest,
)
from trainer.config.schema import APIConfig

_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def _raise_for_status(response: requests.Response) -> None:
    """Map HTTP error codes to typed exceptions."""
    code = response.status_code
    if code == 200 or code < 400:
        return
    try:
        message = response.json().get("detail", response.text)
    except Exception:
        message = response.text

    if code in (401, 403):
        raise APIAuthenticationError(message, status_code=code)
    if code == 404:
        raise APINotFoundError(message, status_code=code)
    if code == 429:
        raise APIRateLimitError(message, status_code=code)
    if code >= 500:
        raise APIServerError(message, status_code=code)
    raise APIError(message, status_code=code)


class SignaliumClient:
    """HTTP client for the Signalium REST API.

    Never connects to PostgreSQL — all data flows through this client.

    Args:
        config: APIConfig instance with connection settings.
        api_key: Optional explicit API key (overrides env).
        token: Optional explicit Bearer token (overrides env).
    """

    def __init__(
        self,
        config: APIConfig,
        api_key: str | None = None,
        token: str | None = None,
    ) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._session = requests.Session()
        self._session.auth = resolve_auth(
            api_key=api_key or config.api_key,
            token=token or config.token,
        )
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "signalium-trainer/0.1.0",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """Build an absolute URL from a relative path."""
        return f"{self._base_url}/{path.lstrip('/')}"

    def _get(self, path: str, **kwargs: Any) -> requests.Response:
        """Perform a GET request with error handling."""
        try:
            response = self._session.get(self._url(path), timeout=self._config.timeout, **kwargs)
        except requests.Timeout as exc:
            raise APITimeoutError(f"GET {path} timed out") from exc
        _raise_for_status(response)
        return response

    def _post(self, path: str, **kwargs: Any) -> requests.Response:
        """Perform a POST request with error handling."""
        try:
            response = self._session.post(self._url(path), timeout=self._config.timeout, **kwargs)
        except requests.Timeout as exc:
            raise APITimeoutError(f"POST {path} timed out") from exc
        _raise_for_status(response)
        return response

    def _put(self, path: str, **kwargs: Any) -> requests.Response:
        """Perform a PUT request with error handling."""
        try:
            response = self._session.put(self._url(path), timeout=self._config.timeout, **kwargs)
        except requests.Timeout as exc:
            raise APITimeoutError(f"PUT {path} timed out") from exc
        _raise_for_status(response)
        return response

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> HealthResponse:
        """Check API health."""
        data = self._get("/health").json()
        return HealthResponse(**data)

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def get_dataset_manifest(self, dataset_id: str) -> DatasetManifest:
        """Fetch the manifest for a dataset export.

        Args:
            dataset_id: Identifier of the dataset to fetch.

        Returns:
            DatasetManifest with file list and checksums.
        """
        data = self._get(f"/datasets/{dataset_id}/manifest").json()
        return DatasetManifest(**data)

    @retry(
        retry=retry_if_exception_type((APIServerError, APIRateLimitError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def download_dataset_file(
        self,
        download_url: str,
        destination: Path,
        expected_sha256: str | None = None,
        show_progress: bool = True,
    ) -> Path:
        """Download a single dataset file with optional checksum verification.

        Args:
            download_url: Presigned or direct URL for the file.
            destination: Local path to write the file to.
            expected_sha256: If provided, the file is verified after download.
            show_progress: Whether to log download progress.

        Returns:
            Path to the downloaded file.

        Raises:
            ChecksumMismatchError: When the checksum does not match.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading {} -> {}", download_url, destination)

        sha256 = hashlib.sha256()
        downloaded = 0

        with self._session.get(download_url, stream=True, timeout=self._config.timeout) as resp:
            _raise_for_status(resp)
            total = int(resp.headers.get("content-length", 0))
            with destination.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    fh.write(chunk)
                    sha256.update(chunk)
                    downloaded += len(chunk)
                    if show_progress and total:
                        pct = downloaded / total * 100
                        logger.debug("Download progress: {:.1f}%", pct)

        if expected_sha256:
            actual = sha256.hexdigest()
            if actual != expected_sha256:
                destination.unlink(missing_ok=True)
                raise ChecksumMismatchError(str(destination), expected_sha256, actual)
            logger.info("Checksum verified: {}", destination.name)

        return destination

    # ------------------------------------------------------------------
    # Model upload operations
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((APIServerError, APIRateLimitError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def upload_model_file(
        self,
        upload_url: str,
        file_path: Path,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload a model artifact file.

        Args:
            upload_url: Presigned PUT URL for the file.
            file_path: Local path of the file to upload.
            content_type: MIME type of the file.
        """
        logger.info("Uploading {} ({} bytes)", file_path.name, file_path.stat().st_size)
        with file_path.open("rb") as fh:
            response = self._session.put(
                upload_url,
                data=fh,
                headers={"Content-Type": content_type},
                timeout=self._config.timeout * 5,
            )
        _raise_for_status(response)
        logger.info("Upload complete: {}", file_path.name)

    def register_model(self, request: UploadModelRequest) -> ModelManifest:
        """Register a trained model with Signalium.

        Args:
            request: Model registration payload.

        Returns:
            ModelManifest with server-assigned IDs.
        """
        data = self._post("/models", json=request.model_dump()).json()
        return ModelManifest(**data)

    # ------------------------------------------------------------------
    # Metrics and logs
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((APIServerError, APIRateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def upload_metrics(self, request: UploadMetricsRequest) -> None:
        """Push training metrics for a run epoch.

        Args:
            request: Metrics payload.
        """
        self._post("/runs/metrics", json=request.model_dump())

    @retry(
        retry=retry_if_exception_type((APIServerError, APIRateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def upload_logs(self, request: UploadLogsRequest) -> None:
        """Push a log chunk for a training run.

        Args:
            request: Log payload.
        """
        self._post("/runs/logs", json=request.model_dump())

    def upload_manifest(self, manifest: ModelManifest) -> None:
        """Upload the model manifest JSON.

        Args:
            manifest: Serialisable ModelManifest.
        """
        self._post(
            f"/models/{manifest.model_id}/manifest",
            json=manifest.model_dump(mode="json"),
        )
