"""S32 — SQLite persistence for the Setup Memory Engine.

Follows the same conventions as ``trainer/memory/memory_database.py`` and
``trainer/qde/qde_observer_database.py``: one row per (symbol, timeframe)
holding the *current* record, plus an append-only transitions table for
the full audit trail (aging/strengthening/weakening/invalidation history).
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from trainer.setup_memory.models import Direction, SetupRecord, SetupState, TransitionEvent

DEFAULT_DB_PATH = "data/setup_memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS setup_memory (
    symbol                     TEXT NOT NULL,
    timeframe                  TEXT NOT NULL,
    broker_symbol              TEXT NOT NULL,
    state                      TEXT NOT NULL,
    state_confidence           REAL NOT NULL DEFAULT 0.0,
    state_entered_at           REAL NOT NULL,
    last_updated_at            REAL NOT NULL,
    expected_direction         TEXT NOT NULL DEFAULT 'UNKNOWN',
    expected_move_size         REAL NOT NULL DEFAULT 0.0,
    expected_holding_period_s  REAL NOT NULL DEFAULT 0.0,
    consecutive_weakenings     INTEGER NOT NULL DEFAULT 0,
    consecutive_strengthenings INTEGER NOT NULL DEFAULT 0,
    invalidation_reason        TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (symbol, timeframe)
);

CREATE TABLE IF NOT EXISTS setup_memory_transitions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol            TEXT NOT NULL,
    timeframe         TEXT NOT NULL,
    at                REAL NOT NULL,
    from_state        TEXT NOT NULL,
    to_state          TEXT NOT NULL,
    reason            TEXT NOT NULL DEFAULT '',
    confidence_after  REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_setup_memory_transitions_key
    ON setup_memory_transitions(symbol, timeframe);
"""


class SetupMemoryStore:
    """Direct SQLite access — ``SetupMemoryEngine`` is the intended public
    API; this class only knows how to read/write rows."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Reads ────────────────────────────────────────────────────────────

    def get(self, symbol: str, timeframe: str) -> SetupRecord | None:
        cur = self._conn.execute(
            "SELECT symbol, timeframe, broker_symbol, state, state_confidence, "
            "state_entered_at, last_updated_at, expected_direction, expected_move_size, "
            "expected_holding_period_s, consecutive_weakenings, consecutive_strengthenings, "
            "invalidation_reason FROM setup_memory WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def all_records(self) -> list[SetupRecord]:
        cur = self._conn.execute(
            "SELECT symbol, timeframe, broker_symbol, state, state_confidence, "
            "state_entered_at, last_updated_at, expected_direction, expected_move_size, "
            "expected_holding_period_s, consecutive_weakenings, consecutive_strengthenings, "
            "invalidation_reason FROM setup_memory ORDER BY symbol, timeframe"
        )
        return [self._row_to_record(row) for row in cur.fetchall()]

    def history(self, symbol: str, timeframe: str, limit: int = 100) -> list[TransitionEvent]:
        cur = self._conn.execute(
            "SELECT at, from_state, to_state, reason, confidence_after "
            "FROM setup_memory_transitions WHERE symbol = ? AND timeframe = ? "
            "ORDER BY id DESC LIMIT ?",
            (symbol, timeframe, limit),
        )
        return [
            TransitionEvent(
                at=row[0],
                from_state=SetupState(row[1]),
                to_state=SetupState(row[2]),
                reason=row[3],
                confidence_after=row[4],
            )
            for row in cur.fetchall()
        ]

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> SetupRecord:
        return SetupRecord(
            symbol=row[0],
            timeframe=row[1],
            broker_symbol=row[2],
            state=SetupState(row[3]),
            state_confidence=row[4],
            state_entered_at=row[5],
            last_updated_at=row[6],
            expected_direction=Direction(row[7]),
            expected_move_size=row[8],
            expected_holding_period_s=row[9],
            consecutive_weakenings=row[10],
            consecutive_strengthenings=row[11],
            invalidation_reason=row[12],
        )

    # ── Writes ───────────────────────────────────────────────────────────

    def upsert(self, record: SetupRecord) -> None:
        self._conn.execute(
            "INSERT INTO setup_memory (symbol, timeframe, broker_symbol, state, "
            "state_confidence, state_entered_at, last_updated_at, expected_direction, "
            "expected_move_size, expected_holding_period_s, consecutive_weakenings, "
            "consecutive_strengthenings, invalidation_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol, timeframe) DO UPDATE SET "
            "broker_symbol=excluded.broker_symbol, state=excluded.state, "
            "state_confidence=excluded.state_confidence, "
            "state_entered_at=excluded.state_entered_at, "
            "last_updated_at=excluded.last_updated_at, "
            "expected_direction=excluded.expected_direction, "
            "expected_move_size=excluded.expected_move_size, "
            "expected_holding_period_s=excluded.expected_holding_period_s, "
            "consecutive_weakenings=excluded.consecutive_weakenings, "
            "consecutive_strengthenings=excluded.consecutive_strengthenings, "
            "invalidation_reason=excluded.invalidation_reason",
            (
                record.symbol,
                record.timeframe,
                record.broker_symbol,
                record.state.value,
                record.state_confidence,
                record.state_entered_at,
                record.last_updated_at,
                record.expected_direction.value,
                record.expected_move_size,
                record.expected_holding_period_s,
                record.consecutive_weakenings,
                record.consecutive_strengthenings,
                record.invalidation_reason,
            ),
        )
        self._conn.commit()

    def append_transition(
        self,
        symbol: str,
        timeframe: str,
        from_state: SetupState,
        to_state: SetupState,
        reason: str,
        confidence_after: float,
        now: float | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO setup_memory_transitions "
            "(symbol, timeframe, at, from_state, to_state, reason, confidence_after) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                symbol,
                timeframe,
                now if now is not None else time.time(),
                from_state.value,
                to_state.value,
                reason,
                confidence_after,
            ),
        )
        self._conn.commit()

    def delete(self, symbol: str, timeframe: str) -> None:
        self._conn.execute(
            "DELETE FROM setup_memory WHERE symbol = ? AND timeframe = ?", (symbol, timeframe)
        )
        self._conn.commit()
