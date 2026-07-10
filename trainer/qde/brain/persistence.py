"""QDE Brain — persistence layer.

Thread-safe, WAL-mode SQLite persistence for `BrainState`, following the
same pattern as `qde_observer_database.QDEObserverDatabase`: the full
serialized state is upserted per-instrument (`brain_state`) so a restarted
process resumes without losing accumulated narrative/hypothesis/personality
context (cross-session continuity), plus append-only logs for reflections
and experiences for audit/telemetry.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS brain_state (
    instrument      TEXT PRIMARY KEY,
    tick_count      INTEGER NOT NULL DEFAULT 0,
    updated_at      REAL NOT NULL,
    data_json       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS brain_reflections (
    reflection_id   TEXT PRIMARY KEY,
    instrument      TEXT NOT NULL,
    created_at      REAL NOT NULL,
    data_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_brain_reflections_instrument ON brain_reflections(instrument);

CREATE TABLE IF NOT EXISTS brain_experiences (
    experience_id   TEXT PRIMARY KEY,
    instrument      TEXT NOT NULL,
    kind            TEXT NOT NULL,
    created_at      REAL NOT NULL,
    data_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_brain_experiences_instrument ON brain_experiences(instrument);
"""


class QDEBrainDatabase:
    """SQLite-backed persistence for per-symbol Brain state + history."""

    def __init__(self, db_path: str = "data/qde_brain.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"QDEBrainDatabase: opened {self._path}")

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Brain state (current snapshot per instrument) ────────────────────

    def upsert_state(self, state: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO brain_state (instrument, tick_count, updated_at, data_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(instrument) DO UPDATE SET
                tick_count=excluded.tick_count,
                updated_at=excluded.updated_at,
                data_json=excluded.data_json
            """,
            (
                state["instrument"],
                state.get("tick_count", 0),
                state.get("last_updated_at", time.time()),
                json.dumps(state),
            ),
        )
        self._conn.commit()

    def get_state(self, instrument: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM brain_state WHERE instrument = ?", (instrument,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["data_json"])

    def list_states(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM brain_state ORDER BY updated_at DESC"
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Reflections (append-only) ────────────────────────────────────────

    def log_reflection(self, reflection: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO brain_reflections
                (reflection_id, instrument, created_at, data_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                reflection["reflection_id"],
                reflection["instrument"],
                reflection.get("created_at", time.time()),
                json.dumps(reflection),
            ),
        )
        self._conn.commit()

    def list_reflections(
        self, instrument: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        if instrument:
            rows = self._conn.execute(
                "SELECT data_json FROM brain_reflections WHERE instrument = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (instrument, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data_json FROM brain_reflections ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Experiences (append-only) ────────────────────────────────────────

    def log_experience(self, experience: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO brain_experiences
                (experience_id, instrument, kind, created_at, data_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                experience["experience_id"],
                experience["instrument"],
                experience.get("kind", ""),
                experience.get("created_at", time.time()),
                json.dumps(experience),
            ),
        )
        self._conn.commit()

    def list_experiences(
        self, instrument: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        if instrument:
            rows = self._conn.execute(
                "SELECT data_json FROM brain_experiences WHERE instrument = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (instrument, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data_json FROM brain_experiences ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]
