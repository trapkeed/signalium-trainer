"""S29 — Inference request security: signing, replay protection, rate
limiting, and abuse detection.

Per the S29 audit, authentication of Research Lab/admin traffic already
exists (``sso.py`` JWT verification, ``apps/trainer-api`` bearer/API-key
middleware), but nothing protects the *inference* boundary — the one path
Signalium is allowed to call — against a compromised/leaked token being
replayed, a caller signing requests with a shared secret, high-volume
abuse, or scripted probing. This module supplies the missing primitives;
wiring them into a specific HTTP framework is left to the caller (e.g. an
Express/FastAPI middleware) since this trainer has no live inference HTTP
route defined yet.

Components
----------
- ``InferenceRequestSigner`` — HMAC-SHA256 request signing/verification
  over a canonical ``(caller_id, timestamp, nonce, body_hash)`` string,
  using a per-caller shared secret (never a global secret — a leaked
  signature for caller A must not let anyone forge requests as caller B).
- ``ReplayGuard`` — in-memory nonce+timestamp cache. A request is replayed
  if its ``(caller_id, nonce)`` pair was already seen, or if its timestamp
  is outside the allowed clock-skew window.
- ``RateLimiter`` — per-caller token-bucket limiter.
- ``AbuseDetector`` — sliding-window failure-rate/volume monitor that flags
  a caller as suspicious (and can be told to auto-block) without needing an
  external monitoring stack.
- ``InferenceSecurityGuard`` — composes all four into a single
  ``authorize_request()`` call an inference endpoint can use end-to-end.

None of this depends on ``sso.py``'s JWT flow — it is intentionally
decoupled so it can also protect machine-to-machine (Signalium -> Trainer)
calls that authenticate via a shared HMAC secret rather than a human SSO
session.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


class InferenceSecurityError(Exception):
    """Base class for all inference-security rejections."""


class InvalidRequestSignatureError(InferenceSecurityError):
    """Raised when an inference request's HMAC signature does not verify."""


class ReplayDetectedError(InferenceSecurityError):
    """Raised when a request's (caller, nonce) has already been consumed,
    or its timestamp falls outside the allowed clock-skew window."""


class RateLimitExceededError(InferenceSecurityError):
    """Raised when a caller has exhausted its inference rate-limit budget."""


class AbuseDetectedError(InferenceSecurityError):
    """Raised when a caller has been flagged/blocked by the abuse detector."""


# ── Request signing ──────────────────────────────────────────────────────────


class InferenceRequestSigner:
    """HMAC-SHA256 signing/verification of inference requests.

    Each caller (e.g. a specific Signalium service identity) has its own
    shared secret, supplied by the caller of this class (typically loaded
    from a per-caller secret store) — this class never manages secret
    storage itself, mirroring ``signing.py``'s stance that key management
    stays out of the crypto primitive.
    """

    @staticmethod
    def canonical_string(caller_id: str, timestamp: str, nonce: str, body: bytes) -> bytes:
        body_hash = hashlib.sha256(body).hexdigest()
        return f"{caller_id}\n{timestamp}\n{nonce}\n{body_hash}".encode("utf-8")

    @classmethod
    def sign(cls, secret: str, caller_id: str, timestamp: str, nonce: str, body: bytes) -> str:
        """Return the hex-encoded HMAC-SHA256 signature for a request."""
        message = cls.canonical_string(caller_id, timestamp, nonce, body)
        return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()

    @classmethod
    def verify(
        cls,
        secret: str,
        caller_id: str,
        timestamp: str,
        nonce: str,
        body: bytes,
        signature_hex: str,
    ) -> bool:
        """Constant-time verification of a request signature."""
        expected = cls.sign(secret, caller_id, timestamp, nonce, body)
        return hmac.compare_digest(expected, signature_hex)


# ── Replay protection ────────────────────────────────────────────────────────


@dataclass
class ReplayGuard:
    """In-memory nonce + timestamp replay guard.

    Args:
        window_seconds: Maximum allowed clock skew between the request's
            claimed timestamp and wall-clock time. Requests outside this
            window are rejected regardless of nonce novelty.
        nonce_ttl_seconds: How long a consumed nonce is remembered. Must be
            >= ``window_seconds`` so a nonce cannot be replayed after its
            record expires but while its timestamp would still be valid.
    """

    window_seconds: float = 300.0
    nonce_ttl_seconds: float = 600.0
    _seen: dict[tuple[str, str], float] = field(default_factory=dict)

    def check_and_record(
        self, caller_id: str, nonce: str, timestamp: float, *, now: float | None = None
    ) -> None:
        """Validate freshness and novelty, then record the nonce as consumed.

        Raises:
            ReplayDetectedError: the timestamp is stale/in the future beyond
                the allowed skew, or this exact (caller, nonce) was already
                consumed.
        """
        now = now if now is not None else time.time()
        self._evict_expired(now)

        if abs(now - timestamp) > self.window_seconds:
            raise ReplayDetectedError(
                f"Request timestamp for {caller_id!r} is outside the allowed "
                f"{self.window_seconds:.0f}s clock-skew window."
            )

        key = (caller_id, nonce)
        if key in self._seen:
            raise ReplayDetectedError(
                f"Nonce {nonce!r} for caller {caller_id!r} has already been used — "
                "possible replay attack."
            )
        self._seen[key] = now

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, ts in self._seen.items() if now - ts > self.nonce_ttl_seconds]
        for k in expired:
            del self._seen[k]

    def size(self) -> int:
        return len(self._seen)


# ── Rate limiting ────────────────────────────────────────────────────────────


@dataclass
class _TokenBucket:
    capacity: float
    tokens: float
    last_refill: float


class RateLimiter:
    """Per-caller token-bucket rate limiter.

    Args:
        capacity: Maximum burst size (tokens in a full bucket).
        refill_per_second: Tokens restored per second (steady-state rate).
    """

    def __init__(self, capacity: float = 60.0, refill_per_second: float = 1.0) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._buckets: dict[str, _TokenBucket] = {}

    def allow(self, caller_id: str, *, cost: float = 1.0, now: float | None = None) -> bool:
        """Return True and consume ``cost`` tokens if the caller has budget,
        else return False without consuming anything."""
        now = now if now is not None else time.time()
        bucket = self._buckets.get(caller_id)
        if bucket is None:
            bucket = _TokenBucket(capacity=self.capacity, tokens=self.capacity, last_refill=now)
            self._buckets[caller_id] = bucket

        elapsed = max(0.0, now - bucket.last_refill)
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
        bucket.last_refill = now

        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True
        return False

    def enforce(self, caller_id: str, *, cost: float = 1.0, now: float | None = None) -> None:
        """Raise ``RateLimitExceededError`` if the caller is over budget."""
        if not self.allow(caller_id, cost=cost, now=now):
            raise RateLimitExceededError(
                f"Caller {caller_id!r} exceeded its inference rate limit "
                f"({self.capacity:.0f} burst / {self.refill_per_second:.2f} req/s)."
            )

    def remaining(self, caller_id: str) -> float:
        bucket = self._buckets.get(caller_id)
        return bucket.tokens if bucket else self.capacity


# ── Abuse detection ──────────────────────────────────────────────────────────


@dataclass
class AbuseDetector:
    """Sliding-window abuse/anomaly detector for inference callers.

    Tracks recent outcomes per caller (success/failure/rejected) and flags
    a caller as suspicious once its failure rate or raw volume within
    ``window_seconds`` crosses the configured thresholds. Distinct from
    ``RateLimiter``: rate limiting caps *volume* regardless of outcome;
    abuse detection reacts to *behavior* (e.g. a flood of signature or
    auth failures indicates probing, not legitimate high-volume use).

    Args:
        window_seconds: Sliding window length.
        max_events: Max total events (of any outcome) in the window before
            flagging as abusive volume.
        max_failure_rate: Failure fraction (0-1) in the window above which
            a caller is flagged, once ``min_events_for_rate`` is met.
        min_events_for_rate: Minimum events observed before failure-rate
            flagging kicks in (avoids flagging on a single early failure).
        block_after_flags: Number of times a caller must be flagged before
            it is auto-blocked (``is_blocked`` becomes True permanently
            until ``reset``).
    """

    window_seconds: float = 60.0
    max_events: int = 120
    max_failure_rate: float = 0.5
    min_events_for_rate: int = 5
    block_after_flags: int = 3

    _events: dict[str, Deque[tuple[float, bool]]] = field(default_factory=dict)
    _flag_counts: dict[str, int] = field(default_factory=dict)
    _blocked: set[str] = field(default_factory=set)

    def record(self, caller_id: str, *, success: bool, now: float | None = None) -> None:
        """Record one inference outcome for *caller_id*."""
        now = now if now is not None else time.time()
        events = self._events.setdefault(caller_id, deque())
        events.append((now, success))
        self._evict_expired(events, now)

    def _evict_expired(self, events: Deque[tuple[float, bool]], now: float) -> None:
        while events and now - events[0][0] > self.window_seconds:
            events.popleft()

    def is_suspicious(self, caller_id: str, *, now: float | None = None) -> bool:
        """Return True if *caller_id*'s recent behavior crosses either
        the volume or failure-rate threshold."""
        now = now if now is not None else time.time()
        events = self._events.get(caller_id)
        if not events:
            return False
        self._evict_expired(events, now)
        if not events:
            return False

        if len(events) > self.max_events:
            return True

        if len(events) >= self.min_events_for_rate:
            failures = sum(1 for _, success in events if not success)
            if failures / len(events) > self.max_failure_rate:
                return True
        return False

    def flag(self, caller_id: str) -> bool:
        """Increment the flag count for *caller_id* and auto-block it once
        ``block_after_flags`` is reached. Returns True if now blocked."""
        self._flag_counts[caller_id] = self._flag_counts.get(caller_id, 0) + 1
        if self._flag_counts[caller_id] >= self.block_after_flags:
            self._blocked.add(caller_id)
            logger.warning(
                "AbuseDetector: caller {} auto-blocked after {} flags",
                caller_id,
                self._flag_counts[caller_id],
            )
        return caller_id in self._blocked

    def is_blocked(self, caller_id: str) -> bool:
        return caller_id in self._blocked

    def reset(self, caller_id: str) -> None:
        """Clear all state for *caller_id* (e.g. after manual admin review)."""
        self._events.pop(caller_id, None)
        self._flag_counts.pop(caller_id, None)
        self._blocked.discard(caller_id)


# ── Composed guard ───────────────────────────────────────────────────────────


class InferenceSecurityGuard:
    """Composes signing, replay protection, rate limiting, and abuse
    detection into a single authorization call for an inference endpoint.

    Callers are expected to look up each caller's HMAC secret themselves
    (e.g. from a secrets store keyed by ``caller_id``) and pass it in — this
    class holds no secrets.
    """

    def __init__(
        self,
        *,
        replay_guard: ReplayGuard | None = None,
        rate_limiter: RateLimiter | None = None,
        abuse_detector: AbuseDetector | None = None,
    ) -> None:
        self.replay_guard = replay_guard or ReplayGuard()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.abuse_detector = abuse_detector or AbuseDetector()

    def authorize_request(
        self,
        *,
        caller_id: str,
        caller_secret: str,
        timestamp: float,
        nonce: str,
        body: bytes,
        signature_hex: str,
        now: float | None = None,
    ) -> None:
        """Run all four checks in order: abuse block -> signature ->
        replay -> rate limit. Raises the first applicable
        ``InferenceSecurityError`` subclass; records the outcome with the
        abuse detector either way.

        Order matters: an already-blocked caller is rejected before doing
        any crypto work (cheap short-circuit); signature verification runs
        before replay/rate checks so a forged request can't be used to
        exhaust a legitimate caller's rate budget.
        """
        now = now if now is not None else time.time()

        if self.abuse_detector.is_blocked(caller_id):
            raise AbuseDetectedError(f"Caller {caller_id!r} is blocked due to prior abuse flags.")

        timestamp_str = f"{timestamp:.6f}"
        try:
            if not InferenceRequestSigner.verify(
                caller_secret, caller_id, timestamp_str, nonce, body, signature_hex
            ):
                raise InvalidRequestSignatureError(
                    f"Signature verification failed for caller {caller_id!r}."
                )

            self.replay_guard.check_and_record(caller_id, nonce, timestamp, now=now)
            self.rate_limiter.enforce(caller_id, now=now)
        except InferenceSecurityError:
            self.abuse_detector.record(caller_id, success=False, now=now)
            if self.abuse_detector.is_suspicious(caller_id, now=now):
                self.abuse_detector.flag(caller_id)
            raise

        self.abuse_detector.record(caller_id, success=True, now=now)


__all__ = [
    "AbuseDetectedError",
    "AbuseDetector",
    "InferenceRequestSigner",
    "InferenceSecurityError",
    "InferenceSecurityGuard",
    "InvalidRequestSignatureError",
    "RateLimitExceededError",
    "RateLimiter",
    "ReplayDetectedError",
    "ReplayGuard",
]
