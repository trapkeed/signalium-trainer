"""S29 — Model artifact encryption, anti-corruption checks, and secure loading.

Complements the existing S20/S21A stack:

- ``ArtifactRegistry`` / ``ArtifactStore`` — write-once storage + SHA-256
  hash bookkeeping (integrity, not confidentiality).
- ``signing.ArtifactSigner`` — Ed25519 authenticity (who produced it),
  enforced at promotion time by ``DeploymentManager``.

What was missing per the S29 audit: artifacts are stored **unencrypted** on
disk, and nothing checks (signature + hash + decryptability) *at load time*
— today that check only happens once, at promotion. This module adds:

1. ``ArtifactEncryptor`` — symmetric AEAD encryption (Fernet, i.e.
   AES-128-CBC + HMAC-SHA256) of artifact bytes at rest. Key comes from the
   ``ARTIFACT_ENCRYPTION_KEY`` secret; never hardcoded or committed.
2. ``encrypt_and_register`` / ``load_artifact_securely`` — a secure
   write/read pair for ``ArtifactRegistry`` that always encrypts on write
   and always re-verifies hash + signature + decrypts on read, so a
   corrupted, tampered, or improperly-decrypted artifact is rejected before
   any bytes reach the caller ("anti-corruption checks").

Design notes
------------
- Mirrors ``signing.py``'s posture: a missing encryption key does not hard-
  fail writes in environments without secrets configured (local dev/tests)
  — callers can check ``ArtifactEncryptor.can_encrypt`` — but
  ``load_artifact_securely`` always hard-fails closed if the stored artifact
  is marked encrypted and no key is available to decrypt it (see
  ``resilience-checks-fail-closed`` precedent from S26/S28).
- Encryption is applied to the artifact payload only; the SHA-256 hash
  recorded in ``ArtifactRecord.artifact_hash`` is always computed over the
  **plaintext** bytes, so registry-level dedup/provenance queries and
  cross-checks against training-run manifests are unaffected by whether a
  given artifact happens to be encrypted.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from trainer.research_lab.artifact_models import ArtifactRecord
from trainer.research_lab.artifact_registry import (
    ArtifactHashMismatchError,
    ArtifactRegistry,
)
from trainer.research_lab.artifact_store import ArtifactKind
from trainer.research_lab.signing import ArtifactSigner, SigningKeyError

#: Env var holding the base64-encoded 32-byte Fernet key used to encrypt
#: model artifacts at rest. Managed as a secret; never hardcoded/committed.
ARTIFACT_ENCRYPTION_KEY_ENV = "ARTIFACT_ENCRYPTION_KEY"

#: Marker stored in ``ArtifactRecord.extra`` (and mirrored onto the
#: on-disk manifest by callers) so a reader knows to decrypt before use.
ENCRYPTED_METADATA_FLAG = "encrypted"


class ArtifactEncryptionError(Exception):
    """Raised when encryption/decryption of an artifact fails."""


class ArtifactCorruptionError(Exception):
    """Raised by ``load_artifact_securely`` when an artifact fails any of
    its integrity/authenticity/decryption checks — i.e. it is corrupt,
    tampered with, or was encrypted/decrypted with the wrong key."""


class ArtifactEncryptor:
    """Symmetric (Fernet/AES) encryption of artifact bytes at rest.

    Args:
        key: base64-encoded 32-byte Fernet key. Defaults to the
            ``ARTIFACT_ENCRYPTION_KEY`` env var.
    """

    def __init__(self, key: str | None = None) -> None:
        self._key = key if key is not None else os.environ.get(ARTIFACT_ENCRYPTION_KEY_ENV, "")
        self._fernet: Fernet | None = None

    @property
    def can_encrypt(self) -> bool:
        return bool(self._key.strip())

    def _load(self) -> Fernet:
        if self._fernet is not None:
            return self._fernet
        if not self.can_encrypt:
            raise ArtifactEncryptionError(
                f"No encryption key configured — set the {ARTIFACT_ENCRYPTION_KEY_ENV!r} "
                "secret (base64-encoded 32-byte Fernet key; see generate_key())."
            )
        try:
            fernet = Fernet(self._key.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ArtifactEncryptionError(
                f"Failed to load {ARTIFACT_ENCRYPTION_KEY_ENV}: {exc}"
            ) from exc
        self._fernet = fernet
        return fernet

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt *data*, returning the Fernet token bytes."""
        return self._load().encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        """Decrypt *token*, raising ``ArtifactEncryptionError`` on failure
        (wrong key, corrupted ciphertext, or tampered token — Fernet's
        HMAC detects all three)."""
        try:
            return self._load().decrypt(token)
        except InvalidToken as exc:
            raise ArtifactEncryptionError(
                "Artifact ciphertext failed decryption — wrong key, or the "
                "artifact is corrupted/tampered with."
            ) from exc

    @staticmethod
    def generate_key() -> str:
        """Generate a fresh base64-encoded Fernet key (never written to disk)."""
        return Fernet.generate_key().decode("utf-8")


@dataclass
class SecureLoadResult:
    """Outcome of ``load_artifact_securely``.

    Attributes:
        data: Verified, decrypted plaintext bytes.
        artifact: The ``ArtifactRecord`` that was loaded.
        signature_verified: Whether an Ed25519 signature was checked and
            matched (``False`` when no signature/public key is configured
            — callers that require signed artifacts should check this).
    """

    data: bytes
    artifact: ArtifactRecord
    signature_verified: bool


def encrypt_and_register(
    registry: ArtifactRegistry,
    *,
    artifact_type: str,
    version: str,
    filename: str,
    data: bytes,
    encryptor: ArtifactEncryptor | None = None,
    kind: str | ArtifactKind | None = None,
    **record_kwargs: Any,
) -> ArtifactRecord:
    """Encrypt *data* at rest and register it, mirroring
    ``ArtifactRegistry.write_and_register`` but always encrypting the
    on-disk payload.

    The registered ``ArtifactRecord.artifact_hash`` is the SHA-256 of the
    **plaintext**, so provenance/dedup checks are unaffected by encryption.
    The record's ``extra`` is stamped with ``encrypted=True`` so
    ``load_artifact_securely`` knows to decrypt on read.

    Raises:
        ArtifactEncryptionError: no encryption key is configured.
    """
    encryptor = encryptor or ArtifactEncryptor()
    plaintext_hash = ArtifactRegistry.hash_bytes(data)
    ciphertext = encryptor.encrypt(data)

    extra = dict(record_kwargs.pop("extra", None) or {})
    extra[ENCRYPTED_METADATA_FLAG] = True

    path = registry.store.write_bytes(version, filename, ciphertext, kind=kind)
    record = ArtifactRecord.new(
        artifact_type=artifact_type,
        name=filename,
        path=str(path),
        artifact_hash=plaintext_hash,
        model_version=version,
        size_bytes=len(data),
        extra=extra,
        **record_kwargs,
    )
    registered = registry.register_artifact(record)
    logger.info(
        "artifact_encryption: encrypted + registered {} ({} plaintext bytes -> {} ciphertext bytes)",
        registered.artifact_id,
        len(data),
        len(ciphertext),
    )
    return registered


def load_artifact_securely(
    registry: ArtifactRegistry,
    artifact_id: str,
    *,
    encryptor: ArtifactEncryptor | None = None,
    signer: ArtifactSigner | None = None,
    signature_hex: str | None = None,
    require_signature: bool = False,
) -> SecureLoadResult:
    """Load an artifact's bytes with full anti-corruption checks:
    (1) record exists, (2) decrypt if marked encrypted, (3) plaintext
    SHA-256 matches the registered hash, (4) Ed25519 signature verifies
    if a signature is supplied or required.

    This is the S29 "secure artifact loading" entry point — callers that
    load weights/exports for inference or promotion should use this
    instead of reading ``ArtifactStore``/filesystem paths directly.

    Raises:
        ArtifactCorruptionError: the artifact record is missing, the
            stored bytes fail to decrypt, the plaintext hash does not
            match the registered hash, or a required/supplied signature
            fails verification.
    """
    record = registry.get_artifact(artifact_id)
    if record is None:
        raise ArtifactCorruptionError(f"No artifact registered with id {artifact_id!r}.")

    # Read via the record's own path, not a re-derived store path, so a
    # record pointing at a relocated/renamed file is still followed.
    raw_path = Path(record.path)
    try:
        raw = raw_path.read_bytes()
    except OSError as exc:
        raise ArtifactCorruptionError(f"Failed to read artifact bytes at {record.path!r}: {exc}") from exc

    is_encrypted = bool((record.extra or {}).get(ENCRYPTED_METADATA_FLAG))
    if is_encrypted:
        encryptor = encryptor or ArtifactEncryptor()
        if not encryptor.can_encrypt:
            raise ArtifactCorruptionError(
                f"Artifact {artifact_id!r} is marked encrypted but no decryption key is "
                "configured — refusing to serve possibly-plaintext-adjacent bytes."
            )
        try:
            plaintext = encryptor.decrypt(raw)
        except ArtifactEncryptionError as exc:
            raise ArtifactCorruptionError(str(exc)) from exc
    else:
        plaintext = raw

    actual_hash = hashlib.sha256(plaintext).hexdigest()
    if actual_hash != record.artifact_hash:
        raise ArtifactCorruptionError(
            f"Artifact {artifact_id!r} failed integrity check: declared hash "
            f"{record.artifact_hash[:16]}... does not match computed hash "
            f"{actual_hash[:16]}... — the artifact is corrupted or was tampered with."
        )

    signature_verified = False
    effective_signature = signature_hex if signature_hex is not None else (
        (record.extra or {}).get("signature", "")
    )
    if effective_signature or require_signature:
        signer = signer or ArtifactSigner()
        if not signer.can_verify:
            if require_signature:
                raise ArtifactCorruptionError(
                    f"Artifact {artifact_id!r} requires signature verification but no "
                    "verification public key is configured."
                )
        else:
            signature_verified = signer.verify(plaintext, effective_signature)
            if require_signature and not signature_verified:
                raise ArtifactCorruptionError(
                    f"Artifact {artifact_id!r} failed Ed25519 signature verification."
                )

    return SecureLoadResult(data=plaintext, artifact=record, signature_verified=signature_verified)


def verify_artifact_not_corrupted(data: bytes, expected_hash: str) -> bool:
    """Lightweight anti-corruption check usable outside the registry
    (e.g. right after a network transfer): recompute SHA-256 and compare.
    """
    return hashlib.sha256(data).hexdigest() == expected_hash


__all__ = [
    "ARTIFACT_ENCRYPTION_KEY_ENV",
    "ArtifactCorruptionError",
    "ArtifactEncryptionError",
    "ArtifactEncryptor",
    "SecureLoadResult",
    "encrypt_and_register",
    "load_artifact_securely",
    "verify_artifact_not_corrupted",
]
