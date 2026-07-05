"""Phase 21.5 — Lifecycle Database.

SQLite persistence for the Trade Lifecycle Intelligence Engine.

Tables:
  trade_lifecycle     — master record per trade
  lifecycle_events    — event log
  transition_rules    — state machine rules catalogue
  state_history       — full state history per trade
  lifecycle_analytics — aggregated analytics snapshots
  optimisation_results — optimisation recommendations

Uses a persistent in-memory connection (`:memory:`) or a file path.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_states import LifecycleState


_DDL = """
CREATE TABLE IF NOT EXISTS trade_lifecycle (
    trade_id       TEXT    PRIMARY KEY,
    instrument     TEXT    NOT NULL,
    direction      TEXT    NOT NULL,
    current_state  TEXT    NOT NULL,
    confidence     REAL    NOT NULL DEFAULT 0.0,
    is_terminal    INTEGER NOT NULL DEFAULT 0,
    is_position_open INTEGER NOT NULL DEFAULT 0,
    age_hours      REAL    DEFAULT 0.0,
    state_count    INTEGER DEFAULT 0,
    event_count    INTEGER DEFAULT 0,
    tags           TEXT    DEFAULT '[]',
    notes          TEXT    DEFAULT '',
    created_at     REAL    NOT NULL,
    updated_at     REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS lifecycle_events (
    event_id    TEXT PRIMARY KEY,
    trade_id    TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    state       TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    description TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'INFO',
    data        TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS transition_rules (
    rule_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    from_state       TEXT    NOT NULL,
    to_state         TEXT    NOT NULL,
    entry_conditions TEXT    DEFAULT '[]',
    exit_conditions  TEXT    DEFAULT '[]',
    validation_rules TEXT    DEFAULT '[]',
    timeout_hours    REAL    DEFAULT 0.0,
    base_confidence  REAL    DEFAULT 0.0,
    expected_benefit REAL    DEFAULT 0.0,
    risk_rating      TEXT    DEFAULT 'MEDIUM',
    description      TEXT    DEFAULT '',
    UNIQUE(from_state, to_state)
);

CREATE TABLE IF NOT EXISTS state_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id     TEXT    NOT NULL,
    state        TEXT    NOT NULL,
    timestamp    REAL    NOT NULL,
    confidence   REAL    NOT NULL,
    reason       TEXT    NOT NULL,
    duration_h   REAL    DEFAULT 0.0,
    market_ctx   TEXT    DEFAULT '{}',
    position_mtr TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS lifecycle_analytics (
    snapshot_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_ts      REAL    NOT NULL,
    total_trades     INTEGER DEFAULT 0,
    completed_trades INTEGER DEFAULT 0,
    failed_trades    INTEGER DEFAULT 0,
    cancelled_trades INTEGER DEFAULT 0,
    expired_trades   INTEGER DEFAULT 0,
    invalidated_trades INTEGER DEFAULT 0,
    avg_lifecycle_h  REAL    DEFAULT 0.0,
    failure_rate     REAL    DEFAULT 0.0,
    cancellation_rate REAL   DEFAULT 0.0,
    expiration_rate  REAL    DEFAULT 0.0,
    state_analytics  TEXT    DEFAULT '{}',
    transition_matrix TEXT   DEFAULT '{}',
    top_paths        TEXT    DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS optimisation_results (
    result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    analysed_at    REAL    NOT NULL,
    total_analysed INTEGER DEFAULT 0,
    findings       TEXT    DEFAULT '[]',
    summary        TEXT    DEFAULT '',
    key_insight    TEXT    DEFAULT '',
    top_priority   TEXT    DEFAULT ''
);
"""


class LifecycleDatabase:
    """SQLite persistence for the Trade Lifecycle Intelligence Engine."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        if db_path == ":memory:":
            self._persistent_conn: Optional[sqlite3.Connection] = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
        else:
            self._persistent_conn = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        if self._persistent_conn is not None:
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ── Trade Lifecycle ───────────────────────────────────────────────────────

    def upsert_lifecycle(self, summary: Dict) -> None:
        sql = """
        INSERT INTO trade_lifecycle
            (trade_id, instrument, direction, current_state, confidence,
             is_terminal, is_position_open, age_hours, state_count, event_count,
             tags, notes, created_at, updated_at)
        VALUES
            (:trade_id, :instrument, :direction, :current_state, :confidence,
             :is_terminal, :is_position_open, :age_hours, :state_count, :event_count,
             :tags, :notes, :created_at, :updated_at)
        ON CONFLICT(trade_id) DO UPDATE SET
            current_state    = excluded.current_state,
            confidence       = excluded.confidence,
            is_terminal      = excluded.is_terminal,
            is_position_open = excluded.is_position_open,
            age_hours        = excluded.age_hours,
            state_count      = excluded.state_count,
            event_count      = excluded.event_count,
            tags             = excluded.tags,
            notes            = excluded.notes,
            updated_at       = excluded.updated_at
        """
        row = {
            "trade_id":        summary.get("trade_id", ""),
            "instrument":      summary.get("instrument", ""),
            "direction":       summary.get("direction", "BUY"),
            "current_state":   summary.get("state", summary.get("current_state", "IDEA")),
            "confidence":      summary.get("confidence", 0.0),
            "is_terminal":     int(summary.get("is_terminal", False)),
            "is_position_open": int(summary.get("is_position_open", False)),
            "age_hours":       summary.get("age_hours", 0.0),
            "state_count":     summary.get("state_count", 0),
            "event_count":     summary.get("event_count", 0),
            "tags":            json.dumps(summary.get("tags", [])),
            "notes":           summary.get("notes", ""),
            "created_at":      summary.get("created_at", time.time()),
            "updated_at":      summary.get("updated_at", time.time()),
        }
        with self._conn() as conn:
            conn.execute(sql, row)
            if self._persistent_conn:
                conn.commit()

    def get_lifecycle(self, trade_id: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trade_lifecycle WHERE trade_id = ?", (trade_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_lifecycles(
        self,
        state:      Optional[str] = None,
        instrument: Optional[str] = None,
        limit:      int           = 100,
    ) -> List[Dict]:
        conditions, params = [], []
        if state:
            conditions.append("current_state = ?"); params.append(state)
        if instrument:
            conditions.append("instrument = ?"); params.append(instrument)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM trade_lifecycle {where} ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Events ────────────────────────────────────────────────────────────────

    def insert_event(self, event_dict: Dict) -> None:
        sql = """
        INSERT OR IGNORE INTO lifecycle_events
            (event_id, trade_id, event_type, state, timestamp, description, severity, data)
        VALUES
            (:event_id, :trade_id, :event_type, :state, :timestamp, :description, :severity, :data)
        """
        row = dict(event_dict)
        row["data"] = json.dumps(row.get("data", {}))
        with self._conn() as conn:
            conn.execute(sql, row)
            if self._persistent_conn:
                conn.commit()

    def get_events_for_trade(self, trade_id: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM lifecycle_events WHERE trade_id = ? ORDER BY timestamp",
                (trade_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_critical_events(self, limit: int = 50) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM lifecycle_events WHERE severity = 'CRITICAL' "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── State History ─────────────────────────────────────────────────────────

    def insert_state_record(self, trade_id: str, record: Dict) -> None:
        sql = """
        INSERT INTO state_history
            (trade_id, state, timestamp, confidence, reason, duration_h, market_ctx, position_mtr)
        VALUES
            (:trade_id, :state, :timestamp, :confidence, :reason, :duration_h, :market_ctx, :position_mtr)
        """
        with self._conn() as conn:
            conn.execute(sql, {
                "trade_id":    trade_id,
                "state":       record.get("state", ""),
                "timestamp":   record.get("timestamp", time.time()),
                "confidence":  record.get("confidence", 0.0),
                "reason":      record.get("reason", ""),
                "duration_h":  record.get("duration_h", 0.0),
                "market_ctx":  json.dumps(record.get("market_context") or {}),
                "position_mtr": json.dumps(record.get("position_metrics") or {}),
            })
            if self._persistent_conn:
                conn.commit()

    def get_state_history(self, trade_id: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM state_history WHERE trade_id = ? ORDER BY timestamp",
                (trade_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Transition Rules ──────────────────────────────────────────────────────

    def upsert_transition_rule(self, rule: Dict) -> None:
        sql = """
        INSERT INTO transition_rules
            (from_state, to_state, entry_conditions, exit_conditions, validation_rules,
             timeout_hours, base_confidence, expected_benefit, risk_rating, description)
        VALUES
            (:from_state, :to_state, :entry_conditions, :exit_conditions, :validation_rules,
             :timeout_hours, :base_confidence, :expected_benefit, :risk_rating, :description)
        ON CONFLICT(from_state, to_state) DO UPDATE SET
            entry_conditions  = excluded.entry_conditions,
            exit_conditions   = excluded.exit_conditions,
            validation_rules  = excluded.validation_rules,
            timeout_hours     = excluded.timeout_hours,
            base_confidence   = excluded.base_confidence,
            expected_benefit  = excluded.expected_benefit,
            risk_rating       = excluded.risk_rating,
            description       = excluded.description
        """
        with self._conn() as conn:
            conn.execute(sql, {
                "from_state":       rule.get("from_state", ""),
                "to_state":         rule.get("to_state", ""),
                "entry_conditions": json.dumps(rule.get("entry_conditions", [])),
                "exit_conditions":  json.dumps(rule.get("exit_conditions", [])),
                "validation_rules": json.dumps(rule.get("validation_rules", [])),
                "timeout_hours":    rule.get("timeout_hours", 0.0),
                "base_confidence":  rule.get("base_confidence", 0.0),
                "expected_benefit": rule.get("expected_benefit", 0.0),
                "risk_rating":      rule.get("risk_rating", "MEDIUM"),
                "description":      rule.get("description", ""),
            })
            if self._persistent_conn:
                conn.commit()

    def get_transition_rules(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM transition_rules").fetchall()
        return [dict(r) for r in rows]

    # ── Analytics ─────────────────────────────────────────────────────────────

    def save_analytics_snapshot(self, analytics: Dict) -> int:
        sql = """
        INSERT INTO lifecycle_analytics
            (snapshot_ts, total_trades, completed_trades, failed_trades,
             cancelled_trades, expired_trades, invalidated_trades,
             avg_lifecycle_h, failure_rate, cancellation_rate, expiration_rate,
             state_analytics, transition_matrix, top_paths)
        VALUES
            (:snapshot_ts, :total_trades, :completed_trades, :failed_trades,
             :cancelled_trades, :expired_trades, :invalidated_trades,
             :avg_lifecycle_h, :failure_rate, :cancellation_rate, :expiration_rate,
             :state_analytics, :transition_matrix, :top_paths)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, {
                "snapshot_ts":       time.time(),
                "total_trades":      analytics.get("total_trades", 0),
                "completed_trades":  analytics.get("completed_trades", 0),
                "failed_trades":     analytics.get("failed_trades", 0),
                "cancelled_trades":  analytics.get("cancelled_trades", 0),
                "expired_trades":    analytics.get("expired_trades", 0),
                "invalidated_trades": analytics.get("invalidated_trades", 0),
                "avg_lifecycle_h":   analytics.get("avg_lifecycle_h", 0.0),
                "failure_rate":      analytics.get("failure_rate", 0.0),
                "cancellation_rate": analytics.get("cancellation_rate", 0.0),
                "expiration_rate":   analytics.get("expiration_rate", 0.0),
                "state_analytics":   json.dumps(analytics.get("state_analytics", {})),
                "transition_matrix": json.dumps(analytics.get("transition_matrix", {})),
                "top_paths":         json.dumps(analytics.get("top_paths", [])),
            })
            if self._persistent_conn:
                conn.commit()
            return cur.lastrowid or 0

    def get_latest_analytics(self) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM lifecycle_analytics ORDER BY snapshot_ts DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ── Optimisation ──────────────────────────────────────────────────────────

    def save_optimisation(self, result: Dict) -> int:
        sql = """
        INSERT INTO optimisation_results
            (analysed_at, total_analysed, findings, summary, key_insight, top_priority)
        VALUES
            (:analysed_at, :total_analysed, :findings, :summary, :key_insight, :top_priority)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, {
                "analysed_at":    time.time(),
                "total_analysed": result.get("total_analysed", 0),
                "findings":       json.dumps(result.get("findings", [])),
                "summary":        result.get("summary", ""),
                "key_insight":    result.get("key_insight", ""),
                "top_priority":   result.get("top_priority") or "",
            })
            if self._persistent_conn:
                conn.commit()
            return cur.lastrowid or 0

    def get_optimisation_results(self, limit: int = 10) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM optimisation_results ORDER BY analysed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        with self._conn() as conn:
            tl  = conn.execute("SELECT COUNT(*) FROM trade_lifecycle").fetchone()[0]
            ev  = conn.execute("SELECT COUNT(*) FROM lifecycle_events").fetchone()[0]
            sh  = conn.execute("SELECT COUNT(*) FROM state_history").fetchone()[0]
            tr  = conn.execute("SELECT COUNT(*) FROM transition_rules").fetchone()[0]
            an  = conn.execute("SELECT COUNT(*) FROM lifecycle_analytics").fetchone()[0]
            opt = conn.execute("SELECT COUNT(*) FROM optimisation_results").fetchone()[0]
        return {
            "trade_lifecycle":     tl,
            "lifecycle_events":    ev,
            "state_history":       sh,
            "transition_rules":    tr,
            "lifecycle_analytics": an,
            "optimisation_results": opt,
        }
