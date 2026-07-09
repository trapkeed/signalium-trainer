"""S31 — Local encryption + transport signature verification helpers.

Two independent cryptographic concerns are handled here:

1. **Signature verification** of sync payloads sent by Signalium, using the
   same HMAC-SHA256 shared-secret pattern already used for RunPod execution
   credentials (``trainer/infra/execution_credentials.py``) — kept
   consistent rather than inventing a second scheme.
2. **At-rest encryption** of the locally persisted secret cache, using
   Fernet (AES-128-CBC + HMAC) with a key derived from
   ``TRAINER_CONFIG_MASTER_KEY``. This key never leaves the trainer host and
   is never sent to or received from Signalium — it only protects the local
   on-disk cache.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

from cryptography.fernet import Fernet, InvalidToken

CONFIG_SYNC_SHARED_SECRET_ENV = "CONFIG_SYNC_SHARED_SECRET"
CONFIG_MASTER_KEY_ENV = "TRAINER_CONFIG_MASTER_KEY"


class ConfigCryptoError(Exception):
    """Base class for config-sync crypto failures."""


class MissingSharedSecret(ConfigCryptoError):
    pass


class InvalidSyncSignature(ConfigCryptoError):
    pass


class DecryptionFailed(ConfigCryptoError):
    pass


def canonical_payload(key: str, value: str, version: int, expires_at: float | None) -> bytes:
    """Deterministic byte representation signed by Signalium for one entry."""
    return f"{key}|{value}|{version}|{expires_at if expires_at is not None else ''}".encode("utf-8")


def verify_sync_signature(
    key: str,
    value: str,
    version: int,
    expires_at: float | None,
    signature_hex: str,
    shared_secret: str | None = None,
) -> bool:
    """HMAC-SHA256 verification of a single synced secret entry.

    Mirrors ``RunPodCredentials.verify_signature`` — Signalium signs with
    the same shared secret configured here via ``CONFIG_SYNC_SHARED_SECRET``.
    """
    secret = (
        shared_secret
        if shared_secret is not None
        else os.environ.get(CONFIG_SYNC_SHARED_SECRET_ENV, "")
    )
    if not secret:
        raise MissingSharedSecret(
            f"{CONFIG_SYNC_SHARED_SECRET_ENV} is not configured — cannot verify sync signatures."
        )
    if not signature_hex:
        return False
    expected = hmac.new(
        secret.encode("utf-8"), canonical_payload(key, value, version, expires_at), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_hex)


def sign_sync_payload(
    key: str, value: str, version: int, expires_at: float | None, shared_secret: str
) -> str:
    """Helper for the issuing side (Signalium simulators / tests)."""
    return hmac.new(
        shared_secret.encode("utf-8"),
        canonical_payload(key, value, version, expires_at),
        hashlib.sha256,
    ).hexdigest()


def _derive_fernet_key(master_key: str) -> bytes:
    """Derive a valid 32-byte urlsafe-base64 Fernet key from an arbitrary
    ``TRAINER_CONFIG_MASTER_KEY`` string via SHA-256, so operators can supply
    any sufficiently random secret rather than a pre-formatted Fernet key."""
    digest = hashlib.sha256(master_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


class LocalCache:
    """Encrypts/decrypts the local secret cache at rest.

    If ``TRAINER_CONFIG_MASTER_KEY`` is unset, a process-local ephemeral key
    is generated so the cache still functions (e.g. in tests/dev) — but a
    warning is logged since this means the cache cannot survive a restart.
    """

    def __init__(self, master_key: str | None = None) -> None:
        resolved = (
            master_key if master_key is not None else os.environ.get(CONFIG_MASTER_KEY_ENV, "")
        )
        self._ephemeral = not bool(resolved)
        if self._ephemeral:
            resolved = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8")
        self._fernet = Fernet(_derive_fernet_key(resolved))

    @property
    def is_ephemeral(self) -> bool:
        """True if no persistent master key was configured — the cache will
        not be decryptable across a process restart."""
        return self._ephemeral

    def encrypt(self, plaintext: str) -> bytes:
        return self._fernet.encrypt(plaintext.encode("utf-8"))

    def decrypt(self, ciphertext: bytes) -> str:
        try:
            return self._fernet.decrypt(ciphertext).decode("utf-8")
        except InvalidToken as exc:
            raise DecryptionFailed(
                "Failed to decrypt local config cache entry — wrong master key "
                "or corrupted cache."
            ) from exc
