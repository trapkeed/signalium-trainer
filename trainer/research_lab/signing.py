"""S21A — Artifact & manifest signing.

Provides asymmetric (Ed25519) signing/verification for every artifact and
manifest produced by the unified export pipeline (``trainer/exporters/pipeline.py``),
and enforces verification before a model version is promoted into
SHADOW/CANDIDATE/ACTIVE (``DeploymentManager``).

Key management
--------------
The signing private key is never hardcoded or committed. It is read from the
``ARTIFACT_SIGNING_PRIVATE_KEY`` environment variable (a secret), holding a
PEM-encoded Ed25519 private key. The corresponding public key is read from
``ARTIFACT_SIGNING_PUBLIC_KEY`` (PEM-encoded) — the public key is not
sensitive, but keeping it in an env var means verifiers never need the
private key to check a signature.

Use ``ArtifactSigner.generate_keypair()`` to create a new PEM keypair (e.g.
for local development or to seed the secrets) — this never touches disk on
its own.

Design notes
------------
- Signing is over the raw file bytes, so any single-byte tamper (of the
  artifact or of a hand-edited checksum) is detected independently of the
  SHA-256 stored in the manifest.
- A missing private key at export time does not hard-fail the export (some
  environments, e.g. local dev/tests, may not have signing configured) —
  instead artifacts are exported *unsigned* and ``DeploymentManager``
  correctly refuses to promote them, since "missing signature" is itself
  a verification failure. This keeps the export pipeline usable without
  secrets configured while still enforcing signing at the point that
  matters: promotion.
- A missing public key at verify time is always a hard failure — there is
  no way to verify anything without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

#: Env vars holding the PEM-encoded signing keypair. The private key MUST be
#: managed as a Replit secret — never hardcoded or committed.
SIGNING_PRIVATE_KEY_ENV = "ARTIFACT_SIGNING_PRIVATE_KEY"
SIGNING_PUBLIC_KEY_ENV = "ARTIFACT_SIGNING_PUBLIC_KEY"

SIGNATURE_ALGORITHM = "ed25519"


class SigningKeyError(Exception):
    """Raised when a required signing/verification key is missing or malformed."""


class SignatureVerificationError(Exception):
    """Raised when an artifact or manifest fails signature verification."""


@dataclass
class SignatureResult:
    """Outcome of verifying one signed file.

    Attributes:
        name: File name (as recorded in the manifest).
        verified: True iff a signature was present and matched the content.
        reason: Empty when verified; a human-readable failure reason otherwise
            (``"missing signature"``, ``"invalid signature"``, ``"hash mismatch"``, ...).
    """

    name: str
    verified: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "verified": self.verified, "reason": self.reason}


class ArtifactSigner:
    """Signs and verifies artifact/manifest bytes using Ed25519.

    Args:
        private_key_pem: PEM-encoded Ed25519 private key. Defaults to the
            ``ARTIFACT_SIGNING_PRIVATE_KEY`` env var. Only needed for signing.
        public_key_pem: PEM-encoded Ed25519 public key. Defaults to the
            ``ARTIFACT_SIGNING_PUBLIC_KEY`` env var. Only needed for
            verification.
    """

    def __init__(
        self,
        private_key_pem: str | None = None,
        public_key_pem: str | None = None,
    ) -> None:
        self._private_key_pem = private_key_pem if private_key_pem is not None else os.environ.get(
            SIGNING_PRIVATE_KEY_ENV, ""
        )
        self._public_key_pem = public_key_pem if public_key_pem is not None else os.environ.get(
            SIGNING_PUBLIC_KEY_ENV, ""
        )
        self._private_key: Ed25519PrivateKey | None = None
        self._public_key: Ed25519PublicKey | None = None

    # ── Key availability ─────────────────────────────────────────────────────

    @property
    def can_sign(self) -> bool:
        return bool(self._private_key_pem.strip())

    @property
    def can_verify(self) -> bool:
        return bool(self._public_key_pem.strip())

    def _load_private_key(self) -> Ed25519PrivateKey:
        if self._private_key is not None:
            return self._private_key
        if not self.can_sign:
            raise SigningKeyError(
                f"No signing private key configured — set the {SIGNING_PRIVATE_KEY_ENV!r} "
                "secret (PEM-encoded Ed25519 private key)."
            )
        try:
            key = serialization.load_pem_private_key(
                self._private_key_pem.encode("utf-8"), password=None
            )
        except Exception as exc:  # noqa: BLE001
            raise SigningKeyError(f"Failed to parse {SIGNING_PRIVATE_KEY_ENV}: {exc}") from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise SigningKeyError(
                f"{SIGNING_PRIVATE_KEY_ENV} is not an Ed25519 private key."
            )
        self._private_key = key
        return key

    def _load_public_key(self) -> Ed25519PublicKey:
        if self._public_key is not None:
            return self._public_key
        if not self.can_verify:
            raise SigningKeyError(
                f"No verification public key configured — set the {SIGNING_PUBLIC_KEY_ENV!r} "
                "env var (PEM-encoded Ed25519 public key)."
            )
        try:
            key = serialization.load_pem_public_key(self._public_key_pem.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SigningKeyError(f"Failed to parse {SIGNING_PUBLIC_KEY_ENV}: {exc}") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise SigningKeyError(
                f"{SIGNING_PUBLIC_KEY_ENV} is not an Ed25519 public key."
            )
        self._public_key = key
        return key

    # ── Sign / verify ────────────────────────────────────────────────────────

    def sign(self, data: bytes) -> str:
        """Return the hex-encoded Ed25519 signature of *data*.

        Raises:
            SigningKeyError: no private key is configured.
        """
        key = self._load_private_key()
        return key.sign(data).hex()

    def verify(self, data: bytes, signature_hex: str) -> bool:
        """Return ``True`` iff *signature_hex* is a valid Ed25519 signature of
        *data* under the configured public key.

        Raises:
            SigningKeyError: no public key is configured.
        """
        key = self._load_public_key()
        if not signature_hex:
            return False
        try:
            signature = bytes.fromhex(signature_hex)
        except ValueError:
            return False
        try:
            key.verify(signature, data)
            return True
        except InvalidSignature:
            return False

    # ── Key generation (setup / local dev / tests) ──────────────────────────

    @staticmethod
    def generate_keypair() -> tuple[str, str]:
        """Generate a fresh Ed25519 keypair.

        Returns:
            ``(private_key_pem, public_key_pem)`` — never written to disk.
        """
        private_key = Ed25519PrivateKey.generate()
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")
        return private_pem, public_pem
