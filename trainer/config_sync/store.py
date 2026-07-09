"""S31 — Encrypted, versioned local secret cache.

SQLite-backed store for synced/local configuration entries. Every write
creates a new version row rather than mutating in place, so rollback and
audit history are "free" — the active version is just the highest version
number that has not been explicitly rolled back past.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from trainer.config_sync.crypto import LocalCache
from trainer.config_sync.ownership import ConfigCategory, OwnershipMode

DEFAULT_DB_PATH = "data/trainer_config_cache.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS config_versions (
    key TEXT NOT NULL,
    version INTEGER NOT NULL,
    ciphertext BLOB NOT NULL,
    category TEXT NOT NULL,
    ownership TEXT NOT NULL,
    synced_at REAL NOT NULL,
    expires_at REAL,
    signature_valid INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (key, version)
);
CREATE INDEX IF NOT EXISTS idx_config_versions_key ON config_versions(key);
"""


@dataclass
class ConfigEntry:
    key: str
    version: int
    value: str
    category: ConfigCategory
    ownership: OwnershipMode
    synced_at: float
    expires_at: float | None
    active: bool

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "version": self.version,
            "category": self.category.value,
            "ownership": self.ownership.value,
            "synced_at": self.synced_at,
            "expires_at": self.expires_at,
            "active": self.active,
            "is_expired": self.is_expired(),
        }


class TrainerConfigStore:
    """Versioned encrypted key/value store, one row per version.

    Args:
        db_path: SQLite file path. Defaults to ``TRAINER_CONFIG_DB_PATH``
            env var or ``data/trainer_config_cache.db``.
        cache: A :class:`LocalCache` instance (injectable for tests).
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, cache: LocalCache | None = None) -> None:
        self._db_path = db_path
        self._cache = cache if cache is not None else LocalCache()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Writes ───────────────────────────────────────────────────────────

    def save(
        self,
        key: str,
        value: str,
        category: ConfigCategory,
        ownership: OwnershipMode,
        expires_at: float | None = None,
        signature_valid: bool = True,
        now: float | None = None,
    ) -> int:
        """Persist a new version of *key*, deactivating any prior active
        version. Returns the new version number."""
        now = now if now is not None else time.time()
        cur = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM config_versions WHERE key = ?", (key,)
        )
        next_version = int(cur.fetchone()[0]) + 1
        ciphertext = self._cache.encrypt(value)
        self._conn.execute(
            "UPDATE config_versions SET active = 0 WHERE key = ? AND active = 1", (key,)
        )
        self._conn.execute(
            "INSERT INTO config_versions "
            "(key, version, ciphertext, category, ownership, synced_at, expires_at, "
            "signature_valid, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (
                key,
                next_version,
                ciphertext,
                category.value,
                ownership.value,
                now,
                expires_at,
                int(signature_valid),
            ),
        )
        self._conn.commit()
        return next_version

    def mark_expired(self, key: str, now: float | None = None) -> bool:
        """Force the active version of *key* to be treated as expired
        immediately (used by rotation: invalidate-now, wait-for-next-sync)."""
        now = now if now is not None else time.time()
        cur = self._conn.execute(
            "UPDATE config_versions SET expires_at = ? WHERE key = ? AND active = 1", (now, key)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def rollback(self, key: str, to_version: int) -> bool:
        """Reactivate a prior version, deactivating the current one.

        Does not delete the version that was rolled back from — it remains
        in history and could be rolled forward to again.
        """
        cur = self._conn.execute(
            "SELECT 1 FROM config_versions WHERE key = ? AND version = ?", (key, to_version)
        )
        if cur.fetchone() is None:
            return False
        self._conn.execute(
            "UPDATE config_versions SET active = 0 WHERE key = ? AND active = 1", (key,)
        )
        self._conn.execute(
            "UPDATE config_versions SET active = 1 WHERE key = ? AND version = ?",
            (key, to_version),
        )
        self._conn.commit()
        return True

    # ── Reads ────────────────────────────────────────────────────────────

    def _row_to_entry(self, row: tuple[Any, ...]) -> ConfigEntry:
        key, version, ciphertext, category, ownership, synced_at, expires_at, _sig, active = row
        return ConfigEntry(
            key=key,
            version=version,
            value=self._cache.decrypt(ciphertext),
            category=ConfigCategory(category),
            ownership=OwnershipMode(ownership),
            synced_at=synced_at,
            expires_at=expires_at,
            active=bool(active),
        )

    def get_active(self, key: str) -> ConfigEntry | None:
        cur = self._conn.execute(
            "SELECT key, version, ciphertext, category, ownership, synced_at, expires_at, "
            "signature_valid, active FROM config_versions WHERE key = ? AND active = 1",
            (key,),
        )
        row = cur.fetchone()
        return self._row_to_entry(row) if row else None

    def get_version(self, key: str, version: int) -> ConfigEntry | None:
        cur = self._conn.execute(
            "SELECT key, version, ciphertext, category, ownership, synced_at, expires_at, "
            "signature_valid, active FROM config_versions WHERE key = ? AND version = ?",
            (key, version),
        )
        row = cur.fetchone()
        return self._row_to_entry(row) if row else None

    def list_versions(self, key: str) -> list[ConfigEntry]:
        cur = self._conn.execute(
            "SELECT key, version, ciphertext, category, ownership, synced_at, expires_at, "
            "signature_valid, active FROM config_versions WHERE key = ? ORDER BY version DESC",
            (key,),
        )
        return [self._row_to_entry(row) for row in cur.fetchall()]

    def list_keys(self) -> list[str]:
        cur = self._conn.execute("SELECT DISTINCT key FROM config_versions ORDER BY key")
        return [row[0] for row in cur.fetchall()]

    def all_active(self) -> list[ConfigEntry]:
        cur = self._conn.execute(
            "SELECT key, version, ciphertext, category, ownership, synced_at, expires_at, "
            "signature_valid, active FROM config_versions WHERE active = 1 ORDER BY key"
        )
        return [self._row_to_entry(row) for row in cur.fetchall()]
