"""Phase 21 — Execution Database.

SQLite-backed persistence for execution plans, trade scores,
sizing decisions, management plans, and optimization history.

Uses the same persistent-connection pattern as HRSADatabase / PortfolioDatabase
for :memory: testing.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.execution.execution_planner import ExecutionPlan


_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_plans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument  TEXT    NOT NULL,
    direction   TEXT    NOT NULL,
    grade       TEXT    NOT NULL,
    score       REAL    NOT NULL,
    executable  INTEGER NOT NULL,
    plan_json   TEXT    NOT NULL,
    created_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument      TEXT    NOT NULL,
    overall_score   REAL    NOT NULL,
    grade           TEXT    NOT NULL,
    forecast_quality REAL   NOT NULL,
    entry_quality   REAL    NOT NULL,
    risk_score      REAL    NOT NULL,
    score_json      TEXT    NOT NULL,
    created_at      REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS sizing_decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument  TEXT    NOT NULL,
    direction   TEXT    NOT NULL,
    method      TEXT    NOT NULL,
    lots        REAL    NOT NULL,
    risk_pct    REAL    NOT NULL,
    sizing_json TEXT    NOT NULL,
    created_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS management_plans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument      TEXT    NOT NULL,
    primary_action  TEXT    NOT NULL,
    plan_json       TEXT    NOT NULL,
    created_at      REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS optimization_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument  TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    detail      TEXT    NOT NULL,
    created_at  REAL    NOT NULL
);
"""

_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ep_instrument  ON execution_plans  (instrument);
CREATE INDEX IF NOT EXISTS idx_ep_grade       ON execution_plans  (grade);
CREATE INDEX IF NOT EXISTS idx_ts_instrument  ON trade_scores     (instrument);
CREATE INDEX IF NOT EXISTS idx_sd_instrument  ON sizing_decisions (instrument);
CREATE INDEX IF NOT EXISTS idx_mp_instrument  ON management_plans (instrument);
CREATE INDEX IF NOT EXISTS idx_oh_instrument  ON optimization_history (instrument);
"""


class ExecutionDatabase:
    """Persist execution intelligence to SQLite.

    Parameters
    ----------
    db_path:
        File path or ``":memory:"`` for in-memory testing.
        For ``:memory:`` a single persistent connection is reused so the
        schema survives across calls.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path       = db_path
        self._persistent_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.executescript(_INDEX_SQL)
            conn.commit()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        if self._db_path == ":memory:":
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._persistent_conn.row_factory = sqlite3.Row
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ── Write operations ──────────────────────────────────────────────────────

    def store_plan(self, plan: ExecutionPlan) -> int:
        now = time.time()
        plan_json = json.dumps(plan.to_dict())
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO execution_plans
                   (instrument, direction, grade, score, executable, plan_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    plan.instrument,
                    plan.direction,
                    plan.score.grade.value,
                    plan.score.overall_score,
                    int(plan.is_executable()),
                    plan_json,
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO trade_scores
                   (instrument, overall_score, grade, forecast_quality,
                    entry_quality, risk_score, score_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    plan.instrument,
                    plan.score.overall_score,
                    plan.score.grade.value,
                    plan.score.forecast_quality,
                    plan.score.entry_quality,
                    plan.score.risk_score,
                    json.dumps(plan.score.to_dict()),
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO sizing_decisions
                   (instrument, direction, method, lots, risk_pct, sizing_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    plan.instrument,
                    plan.direction,
                    plan.sizing.method,
                    plan.sizing.lots,
                    plan.sizing.risk_pct,
                    json.dumps(plan.sizing.to_dict()),
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO management_plans
                   (instrument, primary_action, plan_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    plan.instrument,
                    plan.management.primary_action.value,
                    json.dumps(plan.management.to_dict()),
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def log_optimization(self, instrument: str, event_type: str, detail: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO optimization_history
                   (instrument, event_type, detail, created_at)
                   VALUES (?, ?, ?, ?)""",
                (instrument, event_type, detail, time.time()),
            )
            conn.commit()

    # ── Read operations ───────────────────────────────────────────────────────

    def latest_plans(self, n: int = 10) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT plan_json FROM execution_plans ORDER BY created_at DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [json.loads(r["plan_json"]) for r in rows]

    def plans_by_instrument(self, instrument: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT plan_json FROM execution_plans WHERE instrument = ? ORDER BY created_at DESC",
                (instrument,),
            ).fetchall()
        return [json.loads(r["plan_json"]) for r in rows]

    def top_scored_plans(self, n: int = 5) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT plan_json FROM execution_plans ORDER BY score DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [json.loads(r["plan_json"]) for r in rows]

    def executable_plans(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT plan_json FROM execution_plans WHERE executable = 1 ORDER BY score DESC",
            ).fetchall()
        return [json.loads(r["plan_json"]) for r in rows]

    def sizing_history(self, instrument: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT sizing_json FROM sizing_decisions WHERE instrument = ? ORDER BY created_at DESC",
                (instrument,),
            ).fetchall()
        return [json.loads(r["sizing_json"]) for r in rows]

    def optimization_log(self, instrument: Optional[str] = None, limit: int = 50) -> List[Dict]:
        with self._conn() as conn:
            if instrument:
                rows = conn.execute(
                    "SELECT * FROM optimization_history WHERE instrument = ? ORDER BY created_at DESC LIMIT ?",
                    (instrument, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM optimization_history ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict:
        with self._conn() as conn:
            n_plans  = conn.execute("SELECT COUNT(*) FROM execution_plans").fetchone()[0]
            n_exec   = conn.execute("SELECT COUNT(*) FROM execution_plans WHERE executable=1").fetchone()[0]
            n_scores = conn.execute("SELECT COUNT(*) FROM trade_scores").fetchone()[0]
            n_sizing = conn.execute("SELECT COUNT(*) FROM sizing_decisions").fetchone()[0]
            n_mgmt   = conn.execute("SELECT COUNT(*) FROM management_plans").fetchone()[0]
            n_opt    = conn.execute("SELECT COUNT(*) FROM optimization_history").fetchone()[0]
            avg_score_row = conn.execute(
                "SELECT AVG(score) FROM execution_plans"
            ).fetchone()
            avg_score = round(avg_score_row[0] or 0.0, 4)
        return {
            "execution_plans":    n_plans,
            "executable_plans":   n_exec,
            "trade_scores":       n_scores,
            "sizing_decisions":   n_sizing,
            "management_plans":   n_mgmt,
            "optimization_events": n_opt,
            "average_score":      avg_score,
        }
