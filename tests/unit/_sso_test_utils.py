"""Shared helper for tests exercising the real research_lab_server app.

Builds a valid HS256 Signalium JWT without depending on a third-party JWT
library, mirroring the verification algorithm in
``trainer.research_lab.sso``. Existing server-endpoint tests predate S24D
SSO and only ever set the legacy static bearer token; they now also need a
valid ``X-Signalium-Token`` to pass identity/capability checks. Defaulting
to SuperAdmin keeps those tests focused on their original behavior (they are
not testing SSO/RBAC themselves — that is covered by test_sso.py).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

DEFAULT_TEST_JWT_SECRET = "test-signalium-jwt-secret"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_signalium_jwt(
    *,
    secret: str = DEFAULT_TEST_JWT_SECRET,
    role: str = "SuperAdmin",
    sub: str = "admin",
    service: str = "signalium-test",
    now: float | None = None,
    exp_offset_s: float = 3600.0,
    jti: str | None = None,
) -> str:
    now = now if now is not None else time.time()
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": sub,
        "role": role,
        "service": service,
        "iat": now,
        "exp": now + exp_offset_s,
    }
    if jti:
        payload["jti"] = jti
    header_b64 = _b64url(json.dumps(header).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url(signature)}"
