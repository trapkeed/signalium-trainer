"""Phase S9 — Temporal feature store.

Thread-safe, WAL-mode SQLite persistence for hierarchical-timeframe data.
Follows the same pattern as `qde_observer_database.py`/
`qde_account_database.py`: JSON blob for the full record, indexed scalar
columns for common filters.

Two tables:
  - `timeframe_bars`: append-only raw OHLCV bars per
    instrument+timeframe+close_time — the durable source the synchronizer
    reads from to rebuild a `MultiTimeframeSnapshot` as-of any timestamp.
  - `temporal_assessments`: append-only log of every `TemporalAssessment`
    the model has produced, for replay/telemetry/training-data export.

Usage::

    store = TemporalFeatureStore("data/qde_timeframe.db")
    store.add_bar(bar.to_dict())
    snapshot = store.build_snapshot("EURUSD", as_of=ts, synchronizer=sync)
    store.log_assessment(assessment.to_dict())
    store.close()
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.qde_timeframe_models import MultiTimeframeSnapshot, TimeframeBar
from trainer.qde.qde_timeframe_sync import TimeframeSynchronizer
from trainer.qde.qde_timeframe_types import ALL_TIMEFRAMES, Timeframe

_SCHEMA = """
CREATE TABLE IF NOT EXISTS timeframe_bars (
    instrument   TEXT NOT NULL,
    timeframe    TEXT NOT NULL,
    close_time   REAL NOT NULL,
    open_time    REAL NOT NULL DEFAULT 0,
    data_json    TEXT NOT NULL,
    PRIMARY KEY (instrument, timeframe, close_time)
);
CREATE INDEX IF NOT EXISTS idx_timeframe_bars_lookup
    ON timeframe_bars(instrument, timeframe, close_time);

CREATE TABLE IF NOT EXISTS temporal_assessments (
    assessment_id TEXT PRIMARY KEY,
    instrument    TEXT NOT NULL,
    as_of         REAL NOT NULL,
    alignment_score REAL NOT NULL DEFAULT 0,
    reversal_probability REAL NOT NULL DEFAULT 0.5,
    created_at    REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_temporal_assessments_instrument
    ON temporal_assessments(instrument, as_of);
"""

_BARS_PER_TIMEFRAME_LIMIT = 200  # enough history for the sync pipeline's lookback windows


class TemporalFeatureStore:
    """SQLite-backed persistence for per-timeframe bars + model assessments."""

    def __init__(self, db_path: str = "data/qde_timeframe.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"TemporalFeatureStore: opened {self._path}")

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

    # ── Bars (append-only, upsert-on-close_time for idempotent re-writes) ──

    def add_bar(self, bar: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO timeframe_bars (instrument, timeframe, close_time, open_time, data_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(instrument, timeframe, close_time) DO UPDATE SET
                data_json=excluded.data_json,
                open_time=excluded.open_time
            """,
            (
                bar["instrument"],
                bar["timeframe"],
                bar["close_time"],
                bar.get("open_time", 0.0),
                json.dumps(bar),
            ),
        )
        self._conn.commit()

    def add_bars(self, bars: list[dict[str, Any]]) -> None:
        for bar in bars:
            self.add_bar(bar)

    def list_bars(
        self,
        instrument: str,
        timeframe: Timeframe,
        as_of: float,
        limit: int = _BARS_PER_TIMEFRAME_LIMIT,
    ) -> list[TimeframeBar]:
        """Bars with `close_time <= as_of`, oldest -> newest — the exact
        no-lookahead slice the synchronizer needs.
        """
        rows = self._conn.execute(
            """
            SELECT data_json FROM timeframe_bars
            WHERE instrument = ? AND timeframe = ? AND close_time <= ?
            ORDER BY close_time DESC
            LIMIT ?
            """,
            (instrument, timeframe.value, as_of, limit),
        ).fetchall()
        bars = [TimeframeBar.from_dict(json.loads(r["data_json"])) for r in rows]
        bars.reverse()  # oldest -> newest
        return bars

    def build_snapshot(
        self,
        instrument: str,
        as_of: float,
        synchronizer: TimeframeSynchronizer | None = None,
    ) -> MultiTimeframeSnapshot:
        """Rehydrate a `MultiTimeframeSnapshot` from stored bars as-of `as_of`."""
        sync = synchronizer or TimeframeSynchronizer()
        bars_by_timeframe = {tf: self.list_bars(instrument, tf, as_of) for tf in ALL_TIMEFRAMES}
        return sync.align(instrument, as_of, bars_by_timeframe)

    # ── Temporal assessments (append-only history) ───────────────────────

    def log_assessment(self, assessment: dict[str, Any]) -> str:
        assessment_id = assessment["assessment_id"]
        self._conn.execute(
            """
            INSERT OR REPLACE INTO temporal_assessments
                (assessment_id, instrument, as_of, alignment_score,
                 reversal_probability, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assessment_id,
                assessment["instrument"],
                assessment["as_of"],
                assessment.get("alignment_score", 0.0),
                assessment.get("reversal_probability", 0.5),
                assessment["as_of"],
                json.dumps(assessment),
            ),
        )
        self._conn.commit()
        return assessment_id

    def list_assessments(
        self,
        instrument: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if instrument:
            rows = self._conn.execute(
                """
                SELECT data_json FROM temporal_assessments
                WHERE instrument = ?
                ORDER BY as_of DESC
                LIMIT ?
                """,
                (instrument, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data_json FROM temporal_assessments ORDER BY as_of DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]
