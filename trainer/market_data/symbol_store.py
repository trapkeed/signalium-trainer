"""S33-spec — SQLite persistence for symbol normalization state.

``SymbolNormalizer`` is pure in-memory by design (fast, dependency-free,
fully unit-testable). This module is the optional persistence layer that
sits alongside it: admin-defined aliases, per-symbol enable/disable state,
and a log of every normalization event survive process restarts and are
queryable for the admin diagnostics surface.

Deliberately independent of ``trainer.research_lab`` — market data
normalization is infrastructure shared by QDE and Research Lab alike, not
owned by either (mirrors the `trainer.qde` / `trainer.research_lab`
one-way-dependency convention: neither should own this).
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/symbol_registry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbol_aliases (
    raw TEXT PRIMARY KEY,
    canonical TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'admin',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS symbol_state (
    canonical TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS normalization_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw TEXT NOT NULL,
    canonical TEXT,
    matched_via TEXT NOT NULL,
    broker TEXT,
    enabled INTEGER NOT NULL,
    at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_normalization_events_at ON normalization_events(at);
CREATE INDEX IF NOT EXISTS idx_normalization_events_canonical ON normalization_events(canonical);
"""


@dataclass
class StoredAlias:
    raw: str
    canonical: str
    source: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "canonical": self.canonical, "source": self.source, "created_at": self.created_at}


class SymbolStore:
    """Injectable SQLite-backed persistence for ``SymbolNormalizer``.

    ``db_path`` follows the same convention as ``ResearchLabDatabase`` —
    configurable via the ``SYMBOL_STORE_DB_PATH`` env var, or pass an
    explicit path (e.g. ``:memory:`` for tests) for dependency injection.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path if db_path is not None else os.environ.get("SYMBOL_STORE_DB_PATH", DEFAULT_DB_PATH)
        if self._db_path != ":memory:":
            parent = os.path.dirname(self._db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Aliases ──────────────────────────────────────────────────────────────

    def save_alias(self, raw: str, canonical: str, source: str = "admin") -> None:
        self._conn.execute(
            "INSERT INTO symbol_aliases (raw, canonical, source, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(raw) DO UPDATE SET canonical=excluded.canonical, source=excluded.source, "
            "created_at=excluded.created_at",
            (raw.upper(), canonical.upper(), source, time.time()),
        )
        self._conn.commit()

    def remove_alias(self, raw: str) -> None:
        self._conn.execute("DELETE FROM symbol_aliases WHERE raw = ?", (raw.upper(),))
        self._conn.commit()

    def list_aliases(self, canonical: str | None = None) -> list[StoredAlias]:
        if canonical is not None:
            rows = self._conn.execute(
                "SELECT * FROM symbol_aliases WHERE canonical = ? ORDER BY raw", (canonical.upper(),)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM symbol_aliases ORDER BY canonical, raw").fetchall()
        return [StoredAlias(r["raw"], r["canonical"], r["source"], r["created_at"]) for r in rows]

    # ── Symbol enable/disable state ─────────────────────────────────────────

    def set_enabled(self, canonical: str, enabled: bool) -> None:
        self._conn.execute(
            "INSERT INTO symbol_state (canonical, enabled, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(canonical) DO UPDATE SET enabled=excluded.enabled, updated_at=excluded.updated_at",
            (canonical.upper(), 1 if enabled else 0, time.time()),
        )
        self._conn.commit()

    def get_state(self) -> dict[str, bool]:
        rows = self._conn.execute("SELECT canonical, enabled FROM symbol_state").fetchall()
        return {r["canonical"]: bool(r["enabled"]) for r in rows}

    # ── Normalization event log ─────────────────────────────────────────────

    def log_event(self, raw: str, canonical: str | None, matched_via: str, enabled: bool, broker: str | None = None) -> None:
        try:
            self._conn.execute(
                "INSERT INTO normalization_events (raw, canonical, matched_via, broker, enabled, at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (raw, canonical, matched_via, broker, 1 if enabled else 0, time.time()),
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001 — event logging must never break normalization
            logger.warning(f"SymbolStore: failed to persist normalization event for {raw!r}: {exc!r}")

    def recent_events(self, limit: int = 200, unresolved_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM normalization_events"
        if unresolved_only:
            query += " WHERE canonical IS NULL"
        query += " ORDER BY at DESC LIMIT ?"
        rows = self._conn.execute(query, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def event_stats(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) AS c FROM normalization_events").fetchone()["c"]
        unresolved = self._conn.execute(
            "SELECT COUNT(*) AS c FROM normalization_events WHERE canonical IS NULL"
        ).fetchone()["c"]
        by_via = self._conn.execute(
            "SELECT matched_via, COUNT(*) AS c FROM normalization_events GROUP BY matched_via"
        ).fetchall()
        return {
            "total_events": total,
            "unresolved_events": unresolved,
            "by_matched_via": {r["matched_via"]: r["c"] for r in by_via},
        }
