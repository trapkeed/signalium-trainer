"""Phase S8 — Account Awareness persistence layer.

Thread-safe, WAL-mode SQLite database for historical account state. Follows
the same pattern as `trainer/qde/qde_observer_database.py` and
`trainer/research_lab/rl_database.py`: complex fields as JSON blobs,
commonly-filtered scalars promoted to their own indexed columns.

Two tables:
  - `account_state`: one row per account_id, the *current* snapshot +
    engineered features + latest risk assessment (upserted every tick) —
    lets a restarted process resume without replaying history.
  - `account_history`: append-only log of every `AccountSnapshot` +
    `AccountRiskAssessment` pair, used by the feature pipeline for
    lookback (equity high-water-mark, trade-frequency baseline, pnl trend)
    and by telemetry/dashboards.

Usage::

    db = QDEAccountDatabase("data/qde_account.db")
    db.record(snapshot.to_dict(), features.to_dict(), assessment.to_dict())
    history = db.list_history("default", limit=500)
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
CREATE TABLE IF NOT EXISTS account_state (
    account_id          TEXT PRIMARY KEY,
    balance             REAL NOT NULL DEFAULT 0,
    equity              REAL NOT NULL DEFAULT 0,
    equity_high_water_mark REAL NOT NULL DEFAULT 0,
    current_drawdown_pct REAL NOT NULL DEFAULT 0,
    risk_signal         REAL NOT NULL DEFAULT 0.5,
    action              TEXT NOT NULL DEFAULT 'NORMAL',
    forces_no_trade     INTEGER NOT NULL DEFAULT 0,
    updated_at          REAL NOT NULL,
    snapshot_json       TEXT NOT NULL DEFAULT '{}',
    features_json       TEXT NOT NULL DEFAULT '{}',
    assessment_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_account_state_action ON account_state(action);

CREATE TABLE IF NOT EXISTS account_history (
    record_id           TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL,
    balance             REAL NOT NULL DEFAULT 0,
    equity              REAL NOT NULL DEFAULT 0,
    current_drawdown_pct REAL NOT NULL DEFAULT 0,
    winning_streak      INTEGER NOT NULL DEFAULT 0,
    losing_streak       INTEGER NOT NULL DEFAULT 0,
    trades_today        INTEGER NOT NULL DEFAULT 0,
    risk_signal         REAL NOT NULL DEFAULT 0.5,
    action              TEXT NOT NULL DEFAULT 'NORMAL',
    forces_no_trade     INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    snapshot_json       TEXT NOT NULL DEFAULT '{}',
    features_json       TEXT NOT NULL DEFAULT '{}',
    assessment_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_account_history_account_id ON account_history(account_id);
CREATE INDEX IF NOT EXISTS idx_account_history_created_at ON account_history(created_at);
CREATE INDEX IF NOT EXISTS idx_account_history_action ON account_history(action);
"""


class QDEAccountDatabase:
    """SQLite-backed persistence for account state + history."""

    def __init__(self, db_path: str = "data/qde_account.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"QDEAccountDatabase: opened {self._path}")

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

    # ── Record (current state + append-only history in one call) ─────────

    def record(
        self,
        snapshot: dict[str, Any],
        features: dict[str, Any],
        assessment: dict[str, Any],
    ) -> str:
        account_id = snapshot["account_id"]
        now = assessment.get("unix_ts", time.time())

        self._conn.execute(
            """
            INSERT INTO account_state
                (account_id, balance, equity, equity_high_water_mark,
                 current_drawdown_pct, risk_signal, action, forces_no_trade,
                 updated_at, snapshot_json, features_json, assessment_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                balance=excluded.balance,
                equity=excluded.equity,
                equity_high_water_mark=excluded.equity_high_water_mark,
                current_drawdown_pct=excluded.current_drawdown_pct,
                risk_signal=excluded.risk_signal,
                action=excluded.action,
                forces_no_trade=excluded.forces_no_trade,
                updated_at=excluded.updated_at,
                snapshot_json=excluded.snapshot_json,
                features_json=excluded.features_json,
                assessment_json=excluded.assessment_json
            """,
            (
                account_id,
                snapshot.get("balance", 0.0),
                snapshot.get("equity", 0.0),
                features.get("equity_high_water_mark", 0.0),
                snapshot.get("current_drawdown_pct", 0.0),
                assessment.get("risk_signal", 0.5),
                assessment.get("action", "NORMAL"),
                1 if assessment.get("forces_no_trade") else 0,
                now,
                json.dumps(snapshot),
                json.dumps(features),
                json.dumps(assessment),
            ),
        )

        record_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            """
            INSERT INTO account_history
                (record_id, account_id, balance, equity, current_drawdown_pct,
                 winning_streak, losing_streak, trades_today, risk_signal,
                 action, forces_no_trade, created_at, snapshot_json,
                 features_json, assessment_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                account_id,
                snapshot.get("balance", 0.0),
                snapshot.get("equity", 0.0),
                snapshot.get("current_drawdown_pct", 0.0),
                snapshot.get("winning_streak", 0),
                snapshot.get("losing_streak", 0),
                snapshot.get("trades_today", 0),
                assessment.get("risk_signal", 0.5),
                assessment.get("action", "NORMAL"),
                1 if assessment.get("forces_no_trade") else 0,
                now,
                json.dumps(snapshot),
                json.dumps(features),
                json.dumps(assessment),
            ),
        )
        self._conn.commit()
        return record_id

    # ── Current state ──────────────────────────────────────────────────────

    def get_state(self, account_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT snapshot_json, features_json, assessment_json "
            "FROM account_state WHERE account_id = ?",
            (account_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "snapshot": json.loads(row["snapshot_json"]),
            "features": json.loads(row["features_json"]),
            "assessment": json.loads(row["assessment_json"]),
        }

    def list_accounts(self, action: str | None = None) -> list[dict[str, Any]]:
        if action:
            rows = self._conn.execute(
                "SELECT account_id, action, risk_signal, updated_at "
                "FROM account_state WHERE action = ? ORDER BY account_id",
                (action,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT account_id, action, risk_signal, updated_at "
                "FROM account_state ORDER BY account_id"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── History (append-only, used for feature-pipeline lookback) ─────────

    def list_history(
        self,
        account_id: str,
        limit: int = 500,
        since_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Returns snapshots ordered oldest -> newest (natural for trend math)."""
        if since_ts is not None:
            rows = self._conn.execute(
                """
                SELECT snapshot_json, features_json, assessment_json, created_at
                FROM account_history
                WHERE account_id = ? AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (account_id, since_ts, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT snapshot_json, features_json, assessment_json, created_at
                FROM account_history
                WHERE account_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (account_id, limit),
            ).fetchall()
        records = [
            {
                "snapshot": json.loads(r["snapshot_json"]),
                "features": json.loads(r["features_json"]),
                "assessment": json.loads(r["assessment_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        records.reverse()
        return records

    def count_trades_since(self, account_id: str, since_ts: float) -> int:
        """Approximates trades-in-window via trades_today deltas is caller's
        job; this returns the count of history rows (ticks) recorded in the
        window, useful as an activity-rate denominator."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM account_history "
            "WHERE account_id = ? AND created_at >= ?",
            (account_id, since_ts),
        ).fetchone()
        return int(row["n"]) if row else 0
