"""S24D — Single Sign-On Service Provider.

Signalium Trainer trusts Signalium as the sole identity provider for the
entire ecosystem and operates purely as a Service Provider (SP):

* Signalium Trainer never stores administrator passwords.
* Signalium Trainer never offers public registration.
* Signalium Trainer never performs independent authentication.
* Signalium Trainer never operates as an identity provider.

This module validates JWTs *issued by Signalium* and translates them into a
:class:`SignaliumIdentity` (administrator identity + role) that the rest of
the Research Lab server uses for session establishment, permission
enforcement, and audit logging. It never issues its own tokens or accepts
credentials directly from an end user.

JWT verification is implemented without a third-party JWT library (PyJWT is
unavailable in this environment — see ``.agents/memory`` — and pulling in a
new dependency was blocked by a pre-existing Python 3.11/3.12 interpreter
mismatch in this environment). Verification uses only the stdlib plus the
already-installed ``cryptography`` package, and supports the two algorithm
families Signalium may issue with:

* ``HS256`` — shared-secret HMAC, configured via ``SIGNALIUM_JWT_SECRET``.
* ``RS256`` — RSA public-key signature, configured via
  ``SIGNALIUM_JWT_PUBLIC_KEY`` (PEM).

Environment variables:
    SIGNALIUM_JWT_ALGORITHM     "HS256" (default) or "RS256".
    SIGNALIUM_JWT_SECRET        Shared secret, required for HS256.
    SIGNALIUM_JWT_PUBLIC_KEY    PEM-encoded RSA public key, required for RS256.
    SIGNALIUM_JWT_MAX_SKEW_S    Max allowed clock skew, in seconds, between
                                 the token's ``iat`` (issued-at / request
                                 timestamp) and the verifier's current time.
                                 Default: 300 (5 minutes). This is what
                                 satisfies "validate request timestamps" —
                                 it rejects both stale replayed tokens and
                                 tokens claiming to be issued in the future.
    SIGNALIUM_ISSUER             Expected ``iss`` claim (default: "signalium").
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]


# ── Roles & Capabilities ───────────────────────────────────────────────────


class Role(str, Enum):
    """Administrative roles as issued by Signalium in the JWT ``role`` claim.

    These are the only roles this Service Provider recognizes. Any other
    value in the ``role`` claim is rejected as an unknown role.
    """

    SUPER_ADMIN = "SuperAdmin"
    ADMIN = "Admin"
    FOUNDER = "Founder"
    QUANT_RESEARCHER = "QuantResearcher"
    ML_RESEARCHER = "MLResearcher"
    DEPLOYMENT_APPROVER = "DeploymentApprover"
    AUDITOR = "Auditor"


class Capability(str, Enum):
    """Fine-grained permissions gating individual mutating actions.

    Read access to research records is implicit for any authenticated
    Signalium identity (enforced by the base session requirement, not a
    capability) — the Auditor role's "read-only, no mutation permissions"
    guarantee falls out naturally from it holding none of these
    capabilities, rather than needing a special-cased read capability.
    """

    # QuantResearcher
    EXPERIMENT_CREATION = "experiment_creation"
    DATASET_MANAGEMENT = "dataset_management"
    FEATURE_ENGINEERING = "feature_engineering"
    SHADOW_TESTING = "shadow_testing"
    RESEARCH_WORKFLOWS = "research_workflows"

    # MLResearcher
    TRAINING_RUNS = "training_runs"
    VALIDATION = "validation"
    ARCHITECTURE_RESEARCH = "architecture_research"
    RETRAINING_WORKFLOWS = "retraining_workflows"

    # DeploymentApprover
    APPROVE_PROMOTIONS = "approve_promotions"
    APPROVE_DEPLOYMENTS = "approve_deployments"
    APPROVE_ROLLBACKS = "approve_rollbacks"

    # SuperAdmin / Admin sentinel — never appears in ROLE_CAPABILITIES for
    # any other role; presence of this capability short-circuits every
    # has_capability() check to True (see has_capability()).
    UNRESTRICTED = "unrestricted"

    # S26 — QDE Scalability and Model Resilience. Gates mutating resilience
    # ops (worker restart, backup creation, disaster-recovery snapshotting,
    # auto-rollback triggering) — granted to the roles that actually operate
    # training/deployment infrastructure day-to-day, not to QuantResearcher.
    RESILIENCE_OPERATIONS = "resilience_operations"


# Visibility-only markers for Founder — these are informational (used by
# read-oriented endpoints/UI to decide what to surface) and, like all read
# access, are not required to be presented to view research records; they
# exist so Founder's grant can be expressed explicitly rather than left
# implicit.
class Visibility(str, Enum):
    QDE_LIFECYCLE = "qde_lifecycle_visibility"
    DEPLOYMENT = "deployment_visibility"
    MODEL_GOVERNANCE = "model_governance_visibility"
    # S25 — QDE Observability and Model Intelligence Health Dashboard.
    # Internal-only surface (never exposed to retail/public callers). Granted
    # to SuperAdmin/Admin (via frozenset(Visibility)), Founder, and the three
    # research/deployment roles that need it to do their jobs day-to-day.
    # Auditor is deliberately excluded — its role is read-only compliance
    # review of research *records* (experiments/approvals/audit log), not
    # live model-health telemetry.
    OBSERVABILITY_DASHBOARD = "observability_dashboard_visibility"
    # S28 — QDE Audit and Decision Traceability. Gates decision-level
    # traceability records, model lineage chains, reproducibility/replay
    # tooling, and decision-audit investigations. Administrative-only:
    # granted to all seven Signalium administrator roles (Admin, SuperAdmin,
    # Founder, QuantResearcher, MLResearcher, DeploymentApprover, Auditor)
    # and never exposed to customer- or public-facing callers.
    DECISION_TRACEABILITY = "decision_traceability_visibility"


ROLE_CAPABILITIES: dict[Role, frozenset[Capability]] = {
    Role.SUPER_ADMIN: frozenset({Capability.UNRESTRICTED}),
    Role.ADMIN: frozenset({Capability.UNRESTRICTED}),
    Role.FOUNDER: frozenset(),  # visibility-only, no mutation capabilities
    Role.QUANT_RESEARCHER: frozenset({
        Capability.EXPERIMENT_CREATION,
        Capability.DATASET_MANAGEMENT,
        Capability.FEATURE_ENGINEERING,
        Capability.SHADOW_TESTING,
        Capability.RESEARCH_WORKFLOWS,
    }),
    Role.ML_RESEARCHER: frozenset({
        Capability.TRAINING_RUNS,
        Capability.VALIDATION,
        Capability.ARCHITECTURE_RESEARCH,
        Capability.RETRAINING_WORKFLOWS,
        Capability.RESILIENCE_OPERATIONS,
    }),
    Role.DEPLOYMENT_APPROVER: frozenset({
        Capability.APPROVE_PROMOTIONS,
        Capability.APPROVE_DEPLOYMENTS,
        Capability.APPROVE_ROLLBACKS,
        Capability.RESILIENCE_OPERATIONS,
    }),
    Role.AUDITOR: frozenset(),  # read-only, no mutation permissions
}

ROLE_VISIBILITY: dict[Role, frozenset[Visibility]] = {
    Role.SUPER_ADMIN: frozenset(Visibility),
    Role.ADMIN: frozenset(Visibility),
    Role.FOUNDER: frozenset({
        Visibility.QDE_LIFECYCLE, Visibility.DEPLOYMENT, Visibility.MODEL_GOVERNANCE,
        Visibility.OBSERVABILITY_DASHBOARD, Visibility.DECISION_TRACEABILITY,
    }),
    Role.QUANT_RESEARCHER: frozenset({
        Visibility.OBSERVABILITY_DASHBOARD, Visibility.DECISION_TRACEABILITY,
    }),
    Role.ML_RESEARCHER: frozenset({
        Visibility.OBSERVABILITY_DASHBOARD, Visibility.DECISION_TRACEABILITY,
    }),
    Role.DEPLOYMENT_APPROVER: frozenset({
        Visibility.OBSERVABILITY_DASHBOARD, Visibility.DECISION_TRACEABILITY,
    }),
    # Role.AUDITOR has no other entry (excluded from OBSERVABILITY_DASHBOARD
    # per the S25 spec), but S28 explicitly requires Auditor to see decision
    # traceability/audit records — that is the core of its compliance-review
    # job — so it gets DECISION_TRACEABILITY alone.
    Role.AUDITOR: frozenset({Visibility.DECISION_TRACEABILITY}),
}


def has_capability(role: Role, capability: Capability) -> bool:
    caps = ROLE_CAPABILITIES.get(role, frozenset())
    return Capability.UNRESTRICTED in caps or capability in caps


def has_visibility(role: Role, visibility: Visibility) -> bool:
    return visibility in ROLE_VISIBILITY.get(role, frozenset())


# ── Identity ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignaliumIdentity:
    """A verified Signalium-issued administrator identity for one request."""

    admin_id: str  # `sub` claim — the Signalium administrator identifier
    role: Role
    originating_service: str  # `service` / `iss` claim
    issued_at: float  # `iat` claim, unix seconds
    expires_at: float  # `exp` claim, unix seconds
    jti: str | None = None
    nonce: str | None = None
    raw_claims: dict[str, Any] = field(default_factory=dict, repr=False)

    def has_capability(self, capability: Capability) -> bool:
        return has_capability(self.role, capability)

    def has_visibility(self, visibility: Visibility) -> bool:
        return has_visibility(self.role, visibility)


# ── Errors ──────────────────────────────────────────────────────────────────


class SSOError(Exception):
    """Base class for every JWT/identity validation failure.

    Callers (the FastAPI dependency layer) are expected to catch this and
    translate it into an HTTP 401/403, and to log a REJECTED audit entry
    with ``reason`` before doing so.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class MissingToken(SSOError):
    pass


class MalformedToken(SSOError):
    pass


class InvalidSignature(SSOError):
    pass


class TokenExpired(SSOError):
    pass


class StaleRequestTimestamp(SSOError):
    pass


class UnknownRole(SSOError):
    pass


class MissingClaim(SSOError):
    pass


class ServerMisconfigured(SSOError):
    pass


class ReplayedNonce(SSOError):
    pass


# ── Nonce replay protection ──────────────────────────────────────────────────


class NonceStore:
    """In-memory, TTL-bounded record of consumed ``nonce`` claims.

    Distinct from the existing ``jti``-based session dedup in
    ``session_manager.py`` (which deduplicates at the *session-establishment*
    layer): this store is consulted directly inside ``validate_signalium_jwt``
    so any caller of SSO validation — not just the session-establishment
    path — gets replay protection on the underlying token, independent of
    whether a session is ever created from it.

    Entries older than ``ttl_s`` are evicted lazily on each ``consume`` call
    so the store cannot grow unboundedly across a long-lived process.
    """

    def __init__(self, ttl_s: float = 600.0) -> None:
        self._ttl_s = ttl_s
        self._seen: dict[str, float] = {}

    def consume(self, nonce: str, now: float | None = None) -> bool:
        """Mark ``nonce`` as used. Returns True if this is the first time it
        has been seen within the TTL window; False if it is a replay."""
        now = now if now is not None else time.time()
        self._evict_expired(now)
        if nonce in self._seen:
            return False
        self._seen[nonce] = now
        return True

    def _evict_expired(self, now: float) -> None:
        expired = [n for n, seen_at in self._seen.items() if now - seen_at > self._ttl_s]
        for n in expired:
            del self._seen[n]

    def size(self) -> int:
        return len(self._seen)


#: Process-wide default store used by ``validate_signalium_jwt`` when the
#: caller does not inject its own (e.g. for testing isolation).
_default_nonce_store = NonceStore()


# ── JWT verification (no third-party JWT library) ───────────────────────────


def _b64url_decode(segment: str) -> bytes:
    padding_needed = -len(segment) % 4
    return base64.urlsafe_b64decode(segment + ("=" * padding_needed))


def _verify_hs256(signing_input: bytes, signature: bytes, secret: str) -> bool:
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return hmac.compare_digest(expected, signature)


def _verify_rs256(signing_input: bytes, signature: bytes, public_key_pem: str) -> bool:
    from cryptography.exceptions import InvalidSignature as _CryptoInvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    try:
        public_key.verify(  # type: ignore[union-attr]
            signature, signing_input, padding.PKCS1v15(), hashes.SHA256()
        )
        return True
    except _CryptoInvalidSignature:
        return False


def _decode_and_verify_signature(token: str) -> dict[str, Any]:
    """Split, base64-decode, and cryptographically verify a compact JWT.

    Returns the decoded payload dict on success. Raises SSOError subclasses
    on any failure — malformed structure, unsupported/misconfigured
    algorithm, or signature mismatch.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise MalformedToken("Token is not a well-formed JWT (expected header.payload.signature)")
    header_b64, payload_b64, sig_b64 = parts

    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
        signature = _b64url_decode(sig_b64)
    except Exception as exc:
        raise MalformedToken(f"Token could not be decoded: {exc!r}") from exc

    algorithm = os.environ.get("SIGNALIUM_JWT_ALGORITHM", "HS256").upper()
    header_alg = header.get("alg")
    if header_alg != algorithm:
        raise InvalidSignature(
            f"Token alg {header_alg!r} does not match configured "
            f"SIGNALIUM_JWT_ALGORITHM={algorithm!r}"
        )

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    if algorithm == "HS256":
        secret = os.environ.get("SIGNALIUM_JWT_SECRET", "")
        if not secret:
            raise ServerMisconfigured(
                "Research Lab not configured for Signalium SSO "
                "(SIGNALIUM_JWT_SECRET not set for HS256)"
            )
        ok = _verify_hs256(signing_input, signature, secret)
    elif algorithm == "RS256":
        public_key_pem = os.environ.get("SIGNALIUM_JWT_PUBLIC_KEY", "")
        if not public_key_pem:
            raise ServerMisconfigured(
                "Research Lab not configured for Signalium SSO "
                "(SIGNALIUM_JWT_PUBLIC_KEY not set for RS256)"
            )
        ok = _verify_rs256(signing_input, signature, public_key_pem)
    else:
        raise ServerMisconfigured(f"Unsupported SIGNALIUM_JWT_ALGORITHM={algorithm!r}")

    if not ok:
        raise InvalidSignature("Token signature verification failed")

    return payload


def validate_signalium_jwt(
    token: str,
    now: float | None = None,
    *,
    nonce_store: "NonceStore | None" = None,
    require_nonce: bool = False,
) -> SignaliumIdentity:
    """Validate a Signalium-issued JWT end to end and return the identity it
    asserts.

    Performs, in order:
      1. Signature verification (HS256 or RS256, per SIGNALIUM_JWT_ALGORITHM).
      2. Required-claim presence (``sub``, ``role``, ``iat``, ``exp``).
      3. Expiration check (``exp`` must be in the future).
      4. Request-timestamp freshness check (``iat`` must be within
         SIGNALIUM_JWT_MAX_SKEW_S seconds of "now", in either direction).
      5. Role validation (``role`` must be one of the seven supported roles).
      6. Optional issuer check (``iss``, if SIGNALIUM_ISSUER is configured).
      7. Nonce replay check — if the token carries a ``nonce`` claim, it is
         consumed against ``nonce_store`` (the process-wide default store if
         none is injected) and rejected if already seen. If
         ``require_nonce=True``, a token with no ``nonce`` claim at all is
         also rejected (some callers may want to mandate nonce usage even
         though it's optional at the claim-schema level).

    Raises an ``SSOError`` subclass describing the first failure encountered.
    """
    if not token:
        raise MissingToken("No Signalium token presented")

    now = now if now is not None else time.time()

    payload = _decode_and_verify_signature(token)

    for claim in ("sub", "role", "iat", "exp"):
        if claim not in payload:
            raise MissingClaim(f"Token missing required claim {claim!r}")

    exp = payload["exp"]
    if not isinstance(exp, (int, float)) or now >= exp:
        raise TokenExpired(f"Token expired at {exp!r} (now={now!r})")

    iat = payload["iat"]
    if not isinstance(iat, (int, float)):
        raise MissingClaim("Token 'iat' claim must be numeric")
    max_skew = float(os.environ.get("SIGNALIUM_JWT_MAX_SKEW_S", "300"))
    if abs(now - iat) > max_skew:
        raise StaleRequestTimestamp(
            f"Token request timestamp (iat={iat!r}) is outside the allowed "
            f"{max_skew}s clock-skew window of now={now!r}"
        )

    raw_role = payload["role"]
    try:
        role = Role(raw_role)
    except ValueError:
        raise UnknownRole(f"Unknown/unsupported administrator role {raw_role!r}") from None

    expected_issuer = os.environ.get("SIGNALIUM_ISSUER", "")
    if expected_issuer:
        actual_issuer = payload.get("iss")
        if actual_issuer != expected_issuer:
            raise InvalidSignature(
                f"Token issuer {actual_issuer!r} does not match expected {expected_issuer!r}"
            )

    originating_service = str(payload.get("service") or payload.get("iss") or "signalium")

    nonce = payload.get("nonce")
    if require_nonce and not nonce:
        raise MissingClaim("Token missing required claim 'nonce'")
    if nonce:
        store = nonce_store if nonce_store is not None else _default_nonce_store
        if not store.consume(str(nonce), now=now):
            raise ReplayedNonce(f"Nonce {nonce!r} has already been used")

    return SignaliumIdentity(
        admin_id=str(payload["sub"]),
        role=role,
        originating_service=originating_service,
        issued_at=float(iat),
        expires_at=float(exp),
        jti=payload.get("jti"),
        nonce=str(nonce) if nonce else None,
        raw_claims=payload,
    )
