"""S30-spec — Signalium <-> Trainer Synchronization & Reconciliation.

Runs on a fixed interval (default 60s) reconciling two disjoint ownership
domains:
  - Signalium-owned: identities, permissions, subscriptions, API keys, infra
  - Trainer-owned:    experiments, datasets, artifacts, memory, models, deployments

Each side's records are reconciled independently via a supplied
``fetch_remote`` + ``apply_local`` pair per domain. Failures are retried
with backoff up to a limit, then routed to a dead-letter queue (DLQ)
instead of blocking the next cycle; every apply attempt (success, retry,
or DLQ) is recorded to an audit log. Idempotency reuses the existing
DB-level pattern (INSERT OR IGNORE / ON CONFLICT) at the domain's own
storage layer — this module's contribution is idempotency at the
*reconciliation-record* level: the same remote record version is never
reapplied twice, tracked via a (domain, record_id, version) key.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class ReconciliationDomain(str, Enum):
    # Signalium-owned
    IDENTITIES = "identities"
    PERMISSIONS = "permissions"
    SUBSCRIPTIONS = "subscriptions"
    API_KEYS = "api_keys"
    INFRA = "infra"
    SYMBOLS = "symbols"
    USER_STATUS = "user_status"
    AUTONOMOUS_TRADING_PERMISSIONS = "autonomous_trading_permissions"
    # Trainer-owned
    EXPERIMENTS = "experiments"
    DATASETS = "datasets"
    ARTIFACTS = "artifacts"
    MEMORY = "memory"
    MODELS = "models"
    DEPLOYMENTS = "deployments"


SIGNALIUM_OWNED = {
    ReconciliationDomain.IDENTITIES,
    ReconciliationDomain.PERMISSIONS,
    ReconciliationDomain.SUBSCRIPTIONS,
    ReconciliationDomain.API_KEYS,
    ReconciliationDomain.INFRA,
    ReconciliationDomain.SYMBOLS,
    ReconciliationDomain.USER_STATUS,
    ReconciliationDomain.AUTONOMOUS_TRADING_PERMISSIONS,
}
TRAINER_OWNED = {
    ReconciliationDomain.EXPERIMENTS,
    ReconciliationDomain.DATASETS,
    ReconciliationDomain.ARTIFACTS,
    ReconciliationDomain.MEMORY,
    ReconciliationDomain.MODELS,
    ReconciliationDomain.DEPLOYMENTS,
}


@dataclass
class RemoteRecord:
    record_id: str
    version: str  # any string/hash that changes when the record's content changes
    payload: dict[str, Any]


@dataclass
class DomainHandler:
    domain: ReconciliationDomain
    fetch_remote: Callable[[], list[RemoteRecord]]
    apply_local: Callable[[RemoteRecord], None]
    max_retries: int = 3
    retry_backoff_s: float = 2.0


@dataclass
class AuditEntry:
    domain: str
    record_id: str
    version: str
    outcome: str  # "applied" | "skipped_idempotent" | "retrying" | "dead_lettered"
    detail: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "record_id": self.record_id,
            "version": self.version,
            "outcome": self.outcome,
            "detail": self.detail,
            "at": self.at,
        }


@dataclass
class DeadLetter:
    domain: str
    record_id: str
    version: str
    error: str
    attempts: int
    first_failed_at: float
    last_failed_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "record_id": self.record_id,
            "version": self.version,
            "error": self.error,
            "attempts": self.attempts,
            "first_failed_at": self.first_failed_at,
            "last_failed_at": self.last_failed_at,
        }


class SyncReconciler:
    """In-process reconciliation loop. ``run_once()`` performs a single
    pass over every registered domain and is exercised directly by tests;
    ``start()``/``stop()`` wrap it in a background-thread interval loop."""

    def __init__(self, interval_s: float = 60.0) -> None:
        self._interval_s = interval_s
        self._handlers: dict[ReconciliationDomain, DomainHandler] = {}
        self._applied_versions: dict[tuple[str, str], str] = (
            {}
        )  # (domain, record_id) -> last applied version
        self._audit_log: list[AuditEntry] = []
        self._dlq: dict[tuple[str, str], DeadLetter] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def register(self, handler: DomainHandler) -> None:
        self._handlers[handler.domain] = handler

    def _record_audit(self, entry: AuditEntry) -> None:
        self._audit_log.append(entry)
        logger.info(
            f"SyncReconciler[{entry.domain}] {entry.record_id} v{entry.version}: {entry.outcome} {entry.detail}"
        )

    def _apply_with_retry(self, handler: DomainHandler, record: RemoteRecord) -> None:
        key = (handler.domain.value, record.record_id)

        # Reconciliation-record idempotency: never reapply the same version twice.
        if self._applied_versions.get(key) == record.version:
            self._record_audit(
                AuditEntry(
                    domain=handler.domain.value,
                    record_id=record.record_id,
                    version=record.version,
                    outcome="skipped_idempotent",
                )
            )
            return

        attempt = 0
        last_error = ""
        first_failed_at = 0.0
        while attempt <= handler.max_retries:
            try:
                handler.apply_local(record)
                self._applied_versions[key] = record.version
                self._dlq.pop(key, None)
                self._record_audit(
                    AuditEntry(
                        domain=handler.domain.value,
                        record_id=record.record_id,
                        version=record.version,
                        outcome="applied",
                        detail=f"attempt {attempt + 1}",
                    )
                )
                return
            except Exception as exc:  # noqa: BLE001 — any apply failure is retryable-then-DLQ
                last_error = f"{exc!r}\n{traceback.format_exc(limit=3)}"
                if first_failed_at == 0.0:
                    first_failed_at = time.time()
                attempt += 1
                if attempt <= handler.max_retries:
                    self._record_audit(
                        AuditEntry(
                            domain=handler.domain.value,
                            record_id=record.record_id,
                            version=record.version,
                            outcome="retrying",
                            detail=f"attempt {attempt} failed: {exc!r}",
                        )
                    )
                    time.sleep(handler.retry_backoff_s * attempt)

        existing = self._dlq.get(key)
        attempts_total = (existing.attempts if existing else 0) + attempt
        self._dlq[key] = DeadLetter(
            domain=handler.domain.value,
            record_id=record.record_id,
            version=record.version,
            error=last_error,
            attempts=attempts_total,
            first_failed_at=existing.first_failed_at if existing else first_failed_at,
            last_failed_at=time.time(),
        )
        self._record_audit(
            AuditEntry(
                domain=handler.domain.value,
                record_id=record.record_id,
                version=record.version,
                outcome="dead_lettered",
                detail=last_error,
            )
        )

    def run_once(self, domains: list[ReconciliationDomain] | None = None) -> dict[str, Any]:
        with self._lock:
            targets = domains if domains is not None else list(self._handlers.keys())
            summary: dict[str, Any] = {"domains_run": [], "errors": []}
            for domain in targets:
                handler = self._handlers.get(domain)
                if handler is None:
                    summary["errors"].append(f"no handler registered for domain {domain.value!r}")
                    continue
                try:
                    records = handler.fetch_remote()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"SyncReconciler: fetch_remote failed for {domain.value}: {exc!r}")
                    summary["errors"].append(f"{domain.value}: fetch_remote failed: {exc!r}")
                    continue
                for record in records:
                    self._apply_with_retry(handler, record)
                summary["domains_run"].append(domain.value)
            return summary

    def dead_letters(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self._dlq.values()]

    def requeue_dead_letter(self, domain: ReconciliationDomain, record_id: str) -> bool:
        """Manual operator action: drop a DLQ entry so the next fetch_remote
        pass reattempts it (it will only actually reapply if fetch_remote
        still returns it, i.e. it's still pending upstream)."""
        key = (domain.value, record_id)
        return self._dlq.pop(key, None) is not None

    def audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._audit_log[-limit:]]

    # ── Background loop ──────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.run_once()
                except Exception as exc:  # noqa: BLE001 — the loop must survive a bad cycle
                    logger.error(f"SyncReconciler background loop error: {exc!r}")
                self._stop_event.wait(self._interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True, name="sync-reconciler")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
