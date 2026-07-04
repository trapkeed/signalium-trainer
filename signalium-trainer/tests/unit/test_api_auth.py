"""Unit tests for API authentication resolution."""

from __future__ import annotations

import os

import pytest

from trainer.api.auth import APIKeyAuth, BearerAuth, resolve_auth
from trainer.api.exceptions import APIAuthenticationError


def test_bearer_auth_sets_header() -> None:
    auth = BearerAuth("my-token")
    import requests
    req = requests.Request("GET", "https://example.com").prepare()
    result = auth(req)
    assert result.headers["Authorization"] == "Bearer my-token"


def test_api_key_auth_sets_header() -> None:
    auth = APIKeyAuth("my-key")
    import requests
    req = requests.Request("GET", "https://example.com").prepare()
    result = auth(req)
    assert result.headers["X-API-Key"] == "my-key"


def test_resolve_prefers_token_over_api_key() -> None:
    auth = resolve_auth(api_key="key", token="token")
    assert isinstance(auth, BearerAuth)


def test_resolve_api_key_fallback() -> None:
    auth = resolve_auth(api_key="key")
    assert isinstance(auth, APIKeyAuth)


def test_resolve_from_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALIUM_TOKEN", "env-token")
    monkeypatch.delenv("SIGNALIUM_API_KEY", raising=False)
    auth = resolve_auth()
    assert isinstance(auth, BearerAuth)


def test_resolve_from_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIGNALIUM_TOKEN", raising=False)
    monkeypatch.setenv("SIGNALIUM_API_KEY", "env-key")
    auth = resolve_auth()
    assert isinstance(auth, APIKeyAuth)


def test_resolve_raises_when_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIGNALIUM_TOKEN", raising=False)
    monkeypatch.delenv("SIGNALIUM_API_KEY", raising=False)
    with pytest.raises(APIAuthenticationError):
        resolve_auth()
