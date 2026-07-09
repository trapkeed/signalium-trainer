"""Tests for nonce-based replay protection added to trainer.research_lab.sso."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

import trainer.research_lab.sso as sso

SECRET = "test-nonce-secret"


def _make_token(payload: dict, secret: str = SECRET) -> str:
    header = {"alg": "HS256", "typ": "JWT"}

    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    h, p = b64(header), b64(payload)
    sig = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    s = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{h}.{p}.{s}"


@pytest.fixture
def secret_env(monkeypatch):
    monkeypatch.setenv("SIGNALIUM_JWT_SECRET", SECRET)
    yield SECRET


def test_nonce_store_rejects_replay():
    store = sso.NonceStore(ttl_s=60)
    assert store.consume("abc") is True
    assert store.consume("abc") is False


def test_nonce_store_allows_distinct_nonces():
    store = sso.NonceStore(ttl_s=60)
    assert store.consume("abc") is True
    assert store.consume("def") is True


def test_nonce_store_expires_after_ttl():
    store = sso.NonceStore(ttl_s=0.05)
    store.consume("abc")
    time.sleep(0.1)
    assert store.consume("abc") is True  # expired, treated as fresh


def test_validate_jwt_with_valid_nonce_succeeds(secret_env):
    now = time.time()
    payload = {"sub": "admin1", "role": "Admin", "iat": now, "exp": now + 3600, "nonce": "n-1"}
    token = _make_token(payload)
    identity = sso.validate_signalium_jwt(token)
    assert identity.nonce == "n-1"


def test_validate_jwt_rejects_replayed_nonce(secret_env):
    now = time.time()
    payload = {"sub": "admin1", "role": "Admin", "iat": now, "exp": now + 3600, "nonce": "n-replay"}
    token = _make_token(payload)
    sso.validate_signalium_jwt(token)
    with pytest.raises(sso.ReplayedNonce):
        sso.validate_signalium_jwt(token)


def test_validate_jwt_require_nonce_missing_raises(secret_env):
    now = time.time()
    payload = {"sub": "admin1", "role": "Admin", "iat": now, "exp": now + 3600}
    token = _make_token(payload)
    with pytest.raises(Exception):
        sso.validate_signalium_jwt(token, require_nonce=True)


def test_validate_jwt_shared_store_across_calls(secret_env):
    store = sso.NonceStore(ttl_s=60)
    now = time.time()
    payload = {"sub": "u1", "role": "Admin", "iat": now, "exp": now + 3600, "nonce": "shared-1"}
    token = _make_token(payload)
    sso.validate_signalium_jwt(token, nonce_store=store)
    with pytest.raises(sso.ReplayedNonce):
        sso.validate_signalium_jwt(token, nonce_store=store)
