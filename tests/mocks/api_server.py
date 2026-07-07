"""Mock Signalium API server helpers using respx (httpx-based)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

from trainer.api.client import SignaliumClient
from trainer.config.schema import APIConfig


def _make_manifest_payload(dataset_id: str = "ds_test_001") -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "version": "1.0.0",
        "status": "ready",
        "files": [
            {
                "name": "train.parquet",
                "path": "train.parquet",
                "size_bytes": 1024,
                "sha256": "a" * 64,
                "content_type": "application/octet-stream",
            }
        ],
        "feature_version": "v1",
        "label_version": "v1",
        "created_at": datetime.utcnow().isoformat(),
        "download_url_base": "https://mock-cdn.signalium.io/datasets/ds_test_001",
        "metadata": {},
    }


def make_mock_client(api_key: str = "test-key") -> SignaliumClient:
    """Create a SignaliumClient with a mocked auth (no network calls).

    Suitable for unit tests that patch individual methods.

    Args:
        api_key: Fake API key to use.

    Returns:
        SignaliumClient instance with test base URL.
    """
    cfg = APIConfig(base_url="https://api.signalium.test", api_key=api_key)
    with patch("trainer.api.auth.resolve_auth") as mock_auth:
        mock_auth.return_value = MagicMock()
        client = SignaliumClient(config=cfg, api_key=api_key)
    return client
