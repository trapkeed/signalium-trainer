"""S30-spec — RunPod Execution Credentials.

Trainer never stores RunPod API keys. Instead, Signalium (or an operator)
hands Trainer a short-lived, scoped credential bundle for one execution
job: a pod/endpoint reference, a worker token, an expiration timestamp,
storage credentials, and an artifact destination. This module defines that
bundle and the two checks every consumer must run before using it:
expiration and signature validity.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any


class CredentialError(Exception):
    """Base class for RunPod execution-credential failures."""


class CredentialExpired(CredentialError):
    pass


class InvalidCredentialSignature(CredentialError):
    pass


class MissingCredentialField(CredentialError):
    pass


@dataclass(frozen=True)
class RunPodCredentials:
    """A single-job execution credential bundle handed to Trainer.

    Trainer receives, never mints, these — the fields below are exactly the
    ones named in the S30 spec: pod_id, endpoint, worker_token, expiration
    timestamp, storage_credentials, artifact_destination.
    """

    pod_id: str
    endpoint: str
    worker_token: str
    expiration_timestamp: float
    storage_credentials: dict[str, Any] = field(default_factory=dict)
    artifact_destination: str = ""
    signature: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunPodCredentials":
        required = ("pod_id", "endpoint", "worker_token", "expiration_timestamp")
        for key in required:
            if key not in data:
                raise MissingCredentialField(f"RunPod credential payload missing {key!r}")
        return cls(
            pod_id=str(data["pod_id"]),
            endpoint=str(data["endpoint"]),
            worker_token=str(data["worker_token"]),
            expiration_timestamp=float(data["expiration_timestamp"]),
            storage_credentials=dict(data.get("storage_credentials") or {}),
            artifact_destination=str(data.get("artifact_destination", "")),
            signature=str(data.get("signature", "")),
        )

    def is_expired(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return now >= self.expiration_timestamp

    def _signing_payload(self) -> bytes:
        return (
            f"{self.pod_id}|{self.endpoint}|{self.worker_token}|"
            f"{self.expiration_timestamp}|{self.artifact_destination}"
        ).encode("utf-8")

    def verify_signature(self, shared_secret: str) -> bool:
        """HMAC-SHA256 signature check over the credential's identity
        fields. Signalium is expected to sign with the same shared secret
        configured on the Trainer side (``RUNPOD_CREDENTIAL_SECRET``).
        """
        if not self.signature:
            return False
        expected = hmac.new(shared_secret.encode("utf-8"), self._signing_payload(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature)

    def assert_usable(self, shared_secret: str, now: float | None = None) -> None:
        """Raise if this credential must be rejected before use.

        Both expiration and signature checks are mandatory and fail closed
        — an unsigned or expired credential is never usable, regardless of
        which check would have failed first.
        """
        if self.is_expired(now):
            raise CredentialExpired(
                f"RunPod credential for pod {self.pod_id!r} expired at {self.expiration_timestamp!r}"
            )
        if not self.verify_signature(shared_secret):
            raise InvalidCredentialSignature(
                f"RunPod credential for pod {self.pod_id!r} failed signature verification"
            )

    @staticmethod
    def sign(data: dict[str, Any], shared_secret: str) -> str:
        """Helper for the issuing side (tests / Signalium simulators) to
        produce a valid signature for a not-yet-signed credential payload."""
        creds = RunPodCredentials.from_dict({**data, "signature": ""})
        return hmac.new(shared_secret.encode("utf-8"), creds._signing_payload(), hashlib.sha256).hexdigest()
