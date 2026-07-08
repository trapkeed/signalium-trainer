"""Phase S7 — Market Observer persistence layer.

Thread-safe, WAL-mode SQLite database for `QDEMarketObserver` state.
Follows the same pattern as `trainer/research_lab/rl_database.py` and
`trainer/microstructure/micro_database.py`: complex fields as JSON blobs,
commonly-filtered scalars promoted to their own indexed columns.

Two tables:
  - `observer_state`: one row per instrument, the *current* memory snapshot
    (upserted every tick) — lets a restarted process resume without
    replaying history.
  - `observer_transitions`: append-only log of every state change, used for
    telemetry/dashboards and auditing overtrading/revenge-trade guard trips.

Usage::

    db = QDEObserverDatabase("data/qde_observer.db")
    db.upsert_state(memory.to_dict())
    db.log_transition(result.to_dict())
    rows = db.list_transitions(instrument="EURUSD")
    db.close()
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observer_state (
    instrument      TEXT PRIMARY KEY,
    state           TEXT NOT NULL,
    previous_state  TEXT NOT NULL,
    readiness_score REAL NOT NULL DEFAULT 0,
    consecutive_losses INTEGER NOT NULL DEFAULT 0,
    cooldown_until  REAL NOT NULL DEFAULT 0,
    updated_at      REAL NOT NULL,
    data_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_observer_state_state ON observer_state(state);

CREATE TABLE IF NOT EXISTS observer_transitions (
    transition_id   TEXT PRIMARY KEY,
    instrument      TEXT NOT NULL,
    state           TEXT NOT NULL,
    previous_state  TEXT NOT NULL,
    event           TEXT NOT NULL,
    guard_tripped   TEXT NOT NULL DEFAULT '',
    readiness_score REAL NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    data_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_observer_transitions_instrument ON observer_transitions(instrument);
CREATE INDEX IF NOT EXISTS idx_observer_transitions_event      ON observer_transitions(event);
CREATE INDEX IF NOT EXISTS idx_observer_transitions_created_at ON observer_transitions(created_at);
"""


class QDEObserverDatabase:
    """SQLite-backed persistence for per-symbol observer state + history."""

    def __init__(self, db_path: str = "data/qde_observer.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"QDEObserverDatabase: opened {self._path}")

    # ── Connection management ────────────────────────────────────────────

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

    # ── Observer state (current snapshot per instrument) ─────────────────

    def upsert_state(self, state: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO observer_state
                (instrument, state, previous_state, readiness_score,
                 consecutive_losses, cooldown_until, updated_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instrument) DO UPDATE SET
                state=excluded.state,
                previous_state=excluded.previous_state,
                readiness_score=excluded.readiness_score,
                consecutive_losses=excluded.consecutive_losses,
                cooldown_until=excluded.cooldown_until,
                updated_at=excluded.updated_at,
                data_json=excluded.data_json
            """,
            (
                state["instrument"],
                state["state"],
                state["previous_state"],
                state.get("readiness_score", 0.0),
                state.get("consecutive_losses", 0),
                state.get("cooldown_until", 0.0),
                state.get("last_updated_at", time.time()),
                json.dumps(state),
            ),
        )
        self._conn.commit()

    def get_state(self, instrument: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM observer_state WHERE instrument = ?", (instrument,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_states(self, state: str | None = None) -> list[dict[str, Any]]:
        if state:
            rows = self._conn.execute(
                "SELECT data_json FROM observer_state WHERE state = ? ORDER BY instrument",
                (state,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data_json FROM observer_state ORDER BY instrument"
            ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Transition log (append-only history) ─────────────────────────────

    def log_transition(self, result: dict[str, Any]) -> str:
        transition_id = result.get("observation_id") or str(uuid.uuid4())[:12]
        self._conn.execute(
            """
            INSERT OR REPLACE INTO observer_transitions
                (transition_id, instrument, state, previous_state, event,
                 guard_tripped, readiness_score, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition_id,
                result["instrument"],
                result["state"],
                result["previous_state"],
                result["event"],
                result.get("guard_tripped", ""),
                result.get("readiness_score", 0.0),
                result.get("unix_ts", time.time()),
                json.dumps(result),
            ),
        )
        self._conn.commit()
        return transition_id

    def list_transitions(
        self,
        instrument: str | None = None,
        event: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if instrument:
            clauses.append("instrument = ?")
            params.append(instrument)
        if event:
            clauses.append("event = ?")
            params.append(event)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT data_json FROM observer_transitions
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def count_transitions_since(self, instrument: str, since_ts: float, event: str | None = None) -> int:
        if event:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM observer_transitions "
                "WHERE instrument = ? AND created_at >= ? AND event = ?",
                (instrument, since_ts, event),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM observer_transitions "
                "WHERE instrument = ? AND created_at >= ?",
                (instrument, since_ts),
            ).fetchone()
        return int(row["n"]) if row else 0
