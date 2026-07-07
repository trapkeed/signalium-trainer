"""Integration tests for SignaliumClient using mocked HTTP responses."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import responses as responses_lib

from trainer.api.client import SignaliumClient
from trainer.api.exceptions import APINotFoundError
from trainer.config.schema import APIConfig

BASE_URL = "https://api.signalium.test"


def _make_client() -> SignaliumClient:
    cfg = APIConfig(base_url=BASE_URL, api_key="test-key")
    with patch("trainer.api.auth.resolve_auth") as mock_auth:
        mock_auth.return_value = MagicMock()
        return SignaliumClient(config=cfg, api_key="test-key")


@responses_lib.activate
def test_get_dataset_manifest_success() -> None:
    payload = {
        "dataset_id": "ds_001",
        "version": "1.0.0",
        "status": "ready",
        "files": [],
        "created_at": datetime.utcnow().isoformat(),
        "metadata": {},
    }
    responses_lib.add(
        responses_lib.GET,
        f"{BASE_URL}/datasets/ds_001/manifest",
        json=payload,
        status=200,
    )
    client = _make_client()
    manifest = client.get_dataset_manifest("ds_001")
    assert manifest.dataset_id == "ds_001"
    assert manifest.version == "1.0.0"


@responses_lib.activate
def test_get_dataset_manifest_404_raises() -> None:
    responses_lib.add(
        responses_lib.GET,
        f"{BASE_URL}/datasets/missing/manifest",
        json={"detail": "not found"},
        status=404,
    )
    client = _make_client()
    with pytest.raises(APINotFoundError):
        client.get_dataset_manifest("missing")


@responses_lib.activate
def test_upload_metrics_success() -> None:
    responses_lib.add(
        responses_lib.POST,
        f"{BASE_URL}/runs/metrics",
        json={},
        status=200,
    )
    from trainer.api.models import UploadMetricsRequest

    client = _make_client()
    req = UploadMetricsRequest(run_id="run_1", epoch=1, metrics={"loss": 0.5})
    client.upload_metrics(req)  # Should not raise
