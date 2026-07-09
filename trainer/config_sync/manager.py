"""S31 — Trainer Configuration Manager.

The single entry point for everything the "TRAINER ENVIRONMENT OWNERSHIP
REFACTOR" spec asks for:

* receiving synchronized secrets from Signalium (signature-verified)
* decrypting/persisting them locally (versioned, encrypted at rest)
* rollback / rotation
* status + validation (stale config, missing creds, invalid signatures,
  expired secrets, unsupported provider configuration)
* offline fallback — Trainer keeps functioning off the last-known-good
  cached copy if Signalium is unreachable, and reports that state honestly
  via ``status()`` rather than hiding it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.config_sync.crypto import (
    MissingSharedSecret,
    verify_sync_signature,
)
from trainer.config_sync.ownership import (
    SUPPORTED_MARKET_DATA_PROVIDERS,
    SUPPORTED_RUNPOD_PROVIDERS,
    ConfigCategory,
    OwnershipMode,
    classify,
    is_syncable,
)
from trainer.config_sync.store import ConfigEntry, TrainerConfigStore

#: A synced entry is considered stale if it hasn't been refreshed in this
#: many seconds, even if it isn't yet expired. Purely advisory (surfaced in
#: /config/status), never blocks reads.
DEFAULT_STALE_AFTER_S = 24 * 3600.0

_PROVIDER_KEY_PREFIX = {
    "oanda": ConfigCategory.MARKET_DATA,
    "tiingo": ConfigCategory.MARKET_DATA,
    "ea_feed": ConfigCategory.MARKET_DATA,
    "runpod": ConfigCategory.RUNPOD,
}


@dataclass
class SyncResult:
    key: str
    accepted: bool
    version: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "accepted": self.accepted,
            "version": self.version,
            "reason": self.reason,
        }


@dataclass
class ValidationIssue:
    key: str
    kind: str  # stale | missing | invalid_signature | expired | unsupported_provider
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "kind": self.kind, "detail": self.detail}


@dataclass
class SyncPayloadEntry:
    key: str
    value: str
    signature: str
    expires_at: float | None = None
    provider: str | None = None


class ConfigSyncError(Exception):
    """Raised for hard failures in sync/rotate flows (bad request shape)."""


class TrainerConfigManager:
    """Facade over :class:`TrainerConfigStore` implementing the S31 spec.

    Args:
        store: Injectable for tests. Defaults to a
            :class:`TrainerConfigStore` at the default path.
        shared_secret: Injectable override for
            ``CONFIG_SYNC_SHARED_SECRET`` (tests only).
        required_keys: Keys expected to be present for the deployment to be
            considered fully configured (used by ``validate()`` for
            "missing credentials"). Defaults to all known
            Signalium-synced/exclusive keys.
        stale_after_s: Threshold for flagging a synced entry as stale.
    """

    def __init__(
        self,
        store: TrainerConfigStore | None = None,
        shared_secret: str | None = None,
        required_keys: list[str] | None = None,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
    ) -> None:
        self._store = store if store is not None else TrainerConfigStore()
        self._shared_secret = shared_secret
        self._stale_after_s = stale_after_s
        self._required_keys = (
            required_keys
            if required_keys is not None
            else ["OANDA_API_KEY", "OANDA_ACCOUNT_ID", "TIINGO_API_KEY", "RUNPOD_API_KEY"]
        )
        self._last_sync_attempt: dict[str, float] = {}
        self._last_sync_error: dict[str, str] = {}

    # ── Sync ─────────────────────────────────────────────────────────────

    def sync(self, entries: list[SyncPayloadEntry], now: float | None = None) -> list[SyncResult]:
        """Accept a batch of synced secrets from Signalium.

        Every entry is independently validated — one bad/rejected entry in
        a batch never blocks the others. Ownership rules are enforced
        (trainer-owned keys are never accepted here) and every signature is
        verified before anything is persisted.
        """
        now = now if now is not None else time.time()
        results: list[SyncResult] = []
        for entry in entries:
            self._last_sync_attempt[entry.key] = now
            result = self._sync_one(entry, now)
            if not result.accepted:
                self._last_sync_error[entry.key] = result.reason
            else:
                self._last_sync_error.pop(entry.key, None)
            results.append(result)
        return results

    def _sync_one(self, entry: SyncPayloadEntry, now: float) -> SyncResult:
        classification = classify(entry.key)
        if classification is None:
            return SyncResult(entry.key, False, reason="unknown configuration key")
        ownership, category = classification
        if not is_syncable(entry.key):
            return SyncResult(
                entry.key, False, reason="trainer-owned key cannot be synced from Signalium"
            )
        if entry.provider is not None:
            supported = (
                SUPPORTED_MARKET_DATA_PROVIDERS
                if category is ConfigCategory.MARKET_DATA
                else (
                    SUPPORTED_RUNPOD_PROVIDERS if category is ConfigCategory.RUNPOD else frozenset()
                )
            )
            if entry.provider not in supported:
                return SyncResult(
                    entry.key,
                    False,
                    reason=f"unsupported provider configuration: {entry.provider!r}",
                )
        try:
            next_version = self._store.list_versions(entry.key)
            candidate_version = (next_version[0].version + 1) if next_version else 1
            valid = verify_sync_signature(
                entry.key,
                entry.value,
                candidate_version,
                entry.expires_at,
                entry.signature,
                shared_secret=self._shared_secret,
            )
        except MissingSharedSecret as exc:
            return SyncResult(entry.key, False, reason=str(exc))
        if not valid:
            return SyncResult(entry.key, False, reason="invalid signature")
        if entry.expires_at is not None and entry.expires_at <= now:
            return SyncResult(entry.key, False, reason="secret already expired at sync time")
        version = self._store.save(
            key=entry.key,
            value=entry.value,
            category=category,
            ownership=ownership,
            expires_at=entry.expires_at,
            signature_valid=True,
            now=now,
        )
        logger.info(
            "config_sync: accepted key={} category={} ownership={} version={}",
            entry.key,
            category.value,
            ownership.value,
            version,
        )
        return SyncResult(entry.key, True, version=version)

    # ── Rollback / rotation ──────────────────────────────────────────────

    def rollback(self, key: str, to_version: int) -> bool:
        ok = self._store.rollback(key, to_version)
        if ok:
            logger.warning("config_sync: rolled back key={} to version={}", key, to_version)
        return ok

    def rotate(self, key: str) -> dict[str, Any]:
        """Trigger rotation for *key*.

        For Trainer-owned keys, rotation must happen locally — this method
        just reports that fact (rotation of Trainer-owned material is out
        of scope for this manager; see ``ArtifactSigner``/key-management
        tooling for that).

        For Signalium-owned/synced keys, Trainer cannot mint a new secret
        itself: it marks the current cached version as expired-immediately
        so every consumer is forced onto the offline-fallback/missing path
        until the next successful sync delivers a fresh value, and records
        the rotation request for audit/status visibility.
        """
        classification = classify(key)
        if classification is None:
            raise ConfigSyncError(f"unknown configuration key: {key!r}")
        ownership, _category = classification
        if ownership is OwnershipMode.TRAINER_OWNED:
            return {
                "key": key,
                "ownership": ownership.value,
                "action": "no-op",
                "detail": (
                    "Trainer-owned keys are rotated locally (outside the sync "
                    "manager) — nothing to request from Signalium."
                ),
            }
        expired = self._store.mark_expired(key)
        logger.warning("config_sync: rotation requested for key={}", key)
        return {
            "key": key,
            "ownership": ownership.value,
            "action": "invalidated_pending_resync",
            "had_cached_value": expired,
            "detail": "Cached value invalidated; awaiting next POST /config/sync from Signalium.",
        }

    # ── Reads ────────────────────────────────────────────────────────────

    def get(self, key: str) -> ConfigEntry | None:
        """Return the currently active entry for *key*, if any — this is
        the offline-fallback read path: it returns the last-known-good
        cached value even if Signalium is unreachable right now."""
        return self._store.get_active(key)

    def current_version(self, key: str | None = None) -> dict[str, Any]:
        if key is not None:
            entry = self._store.get_active(key)
            return {"key": key, "version": entry.version if entry else None}
        return {entry.key: entry.version for entry in self._store.all_active()}

    def status(self, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        entries = {entry.key: entry for entry in self._store.all_active()}
        report: dict[str, Any] = {"generated_at": now, "keys": {}}
        for key, entry in entries.items():
            age_s = now - entry.synced_at
            report["keys"][key] = {
                **entry.to_dict(),
                "age_s": age_s,
                "is_stale": (
                    entry.ownership is not OwnershipMode.TRAINER_OWNED
                    and age_s > self._stale_after_s
                ),
                "last_sync_error": self._last_sync_error.get(key, ""),
            }
        issues = [issue.to_dict() for issue in self.validate(now=now)]
        report["issues"] = issues
        report["healthy"] = not any(
            issue["kind"] in ("missing", "invalid_signature", "expired") for issue in issues
        )
        return report

    def validate(self, now: float | None = None) -> list[ValidationIssue]:
        """Run all S31 §8 validation checks and return every issue found."""
        now = now if now is not None else time.time()
        issues: list[ValidationIssue] = []
        active_by_key = {entry.key: entry for entry in self._store.all_active()}

        for key in self._required_keys:
            entry = active_by_key.get(key)
            if entry is None:
                issues.append(ValidationIssue(key, "missing", "no synced credential present"))
                continue
            if entry.is_expired(now):
                issues.append(
                    ValidationIssue(key, "expired", f"expired at {entry.expires_at} (now={now})")
                )
            age_s = now - entry.synced_at
            if age_s > self._stale_after_s:
                issues.append(
                    ValidationIssue(
                        key,
                        "stale",
                        f"last synced {age_s:.0f}s ago (> {self._stale_after_s:.0f}s threshold)",
                    )
                )

        for key, reason in self._last_sync_error.items():
            if reason == "invalid signature":
                issues.append(ValidationIssue(key, "invalid_signature", reason))
            elif reason.startswith("unsupported provider"):
                issues.append(ValidationIssue(key, "unsupported_provider", reason))

        for entry in active_by_key.values():
            if entry.key in self._required_keys:
                continue
            if entry.is_expired(now):
                issues.append(
                    ValidationIssue(
                        entry.key, "expired", f"expired at {entry.expires_at} (now={now})"
                    )
                )
        return issues


_default_manager: TrainerConfigManager | None = None


def get_default_manager() -> TrainerConfigManager:
    """Process-wide singleton, lazily constructed (mirrors the pattern used
    by other trainer subsystems that avoid import-time side effects)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TrainerConfigManager()
    return _default_manager
