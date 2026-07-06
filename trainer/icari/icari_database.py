"""Phase 24 — ICARI Database.

SQLite persistence for ICARI.
8 tables: capital_allocation, risk_budgets, allocation_history,
portfolio_risk, scenario_results, capital_health, optimization_results,
drawdown_events.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.icari.icari_models import (
    AllocationResult,
    CapitalHealth,
    DrawdownEvent,
    OptimizationResult,
    PortfolioRisk,
    RiskBudget,
    ScenarioResult,
)

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS capital_allocation (
    request_id      TEXT PRIMARY KEY,
    instrument      TEXT NOT NULL,
    status          TEXT NOT NULL,
    allocation_frac REAL DEFAULT 0,
    allocation_usd  REAL DEFAULT 0,
    confidence      REAL DEFAULT 0,
    rationale       TEXT DEFAULT '',
    timestamp       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_budgets (
    budget_type TEXT PRIMARY KEY,
    limit_frac  REAL NOT NULL,
    used_frac   REAL DEFAULT 0,
    label       TEXT DEFAULT '',
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS allocation_history (
    history_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument   TEXT,
    status       TEXT,
    allocation_frac REAL,
    confidence   REAL,
    timestamp    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_risk (
    risk_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    var_95       REAL DEFAULT 0,
    var_99       REAL DEFAULT 0,
    cvar_95      REAL DEFAULT 0,
    max_dd       REAL DEFAULT 0,
    port_vol     REAL DEFAULT 0,
    div_ratio    REAL DEFAULT 1,
    concentration REAL DEFAULT 0,
    timestamp    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS scenario_results (
    result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario       TEXT NOT NULL,
    portfolio_return REAL,
    portfolio_var  REAL,
    max_drawdown   REAL,
    resilience     REAL,
    timestamp      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS capital_health (
    health_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    efficiency      REAL DEFAULT 0,
    risk_score      REAL DEFAULT 0,
    allocation_q    REAL DEFAULT 0,
    diversification REAL DEFAULT 0,
    stability       REAL DEFAULT 0,
    utilisation     REAL DEFAULT 0,
    overall         REAL DEFAULT 0,
    grade           TEXT DEFAULT 'FAIR',
    timestamp       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS optimization_results (
    result_id   TEXT PRIMARY KEY,
    objective   TEXT NOT NULL,
    weights     TEXT DEFAULT '{}',
    metrics     TEXT DEFAULT '{}',
    converged   INTEGER DEFAULT 0,
    iterations  INTEGER DEFAULT 0,
    timestamp   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS drawdown_events (
    event_id   TEXT PRIMARY KEY,
    peak       REAL,
    trough     REAL,
    drawdown   REAL,
    mode       TEXT,
    recovered  INTEGER DEFAULT 0,
    start_ts   REAL NOT NULL,
    end_ts     REAL
);

CREATE INDEX IF NOT EXISTS idx_alloc_instrument ON capital_allocation(instrument);
CREATE INDEX IF NOT EXISTS idx_alloc_status     ON capital_allocation(status);
CREATE INDEX IF NOT EXISTS idx_history_ts       ON allocation_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_risk_ts          ON portfolio_risk(timestamp);
CREATE INDEX IF NOT EXISTS idx_health_ts        ON capital_health(timestamp);
CREATE INDEX IF NOT EXISTS idx_scenario_type    ON scenario_results(scenario);
"""


class IcariDatabase:
    """SQLite persistence for ICARI."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._persistent_conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_TABLES)

    @contextlib.contextmanager
    def _conn(self):
        if self._db_path == ":memory:":
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:")
                self._persistent_conn.row_factory = sqlite3.Row
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _commit(self, conn) -> None:
        if self._db_path == ":memory:":
            conn.commit()

    # ── Allocation ────────────────────────────────────────────────────────────

    def save_allocation(self, result: AllocationResult) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO capital_allocation
                   (request_id, instrument, status, allocation_frac, allocation_usd, confidence, rationale, timestamp)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    result.request_id,
                    result.instrument,
                    result.status.value,
                    result.allocation_frac,
                    result.allocation_usd,
                    result.confidence,
                    result.rationale,
                    result.timestamp,
                ),
            )
            conn.execute(
                """INSERT INTO allocation_history
                   (instrument, status, allocation_frac, confidence, timestamp)
                   VALUES (?,?,?,?,?)""",
                (
                    result.instrument,
                    result.status.value,
                    result.allocation_frac,
                    result.confidence,
                    result.timestamp,
                ),
            )
            self._commit(conn)

    def count_allocations(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM capital_allocation").fetchone()[0]

    def recent_allocations(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM capital_allocation ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def allocation_stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status, COUNT(*) as cnt, AVG(confidence) as avg_conf FROM capital_allocation GROUP BY status"
            ).fetchall()
        return {
            r["status"]: {"count": r["cnt"], "avg_confidence": round(r["avg_conf"] or 0.0, 4)}
            for r in row
        }

    # ── Risk budgets ──────────────────────────────────────────────────────────

    def save_budget(self, budget: RiskBudget) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO risk_budgets
                   (budget_type, limit_frac, used_frac, label, updated_at)
                   VALUES (?,?,?,?,?)""",
                (budget.budget_type.value, budget.limit, budget.used, budget.label, time.time()),
            )
            self._commit(conn)

    def count_budgets(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM risk_budgets").fetchone()[0]

    # ── Portfolio risk ────────────────────────────────────────────────────────

    def save_portfolio_risk(self, pr: PortfolioRisk) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO portfolio_risk
                   (var_95, var_99, cvar_95, max_dd, port_vol, div_ratio, concentration, timestamp)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    pr.var_95,
                    pr.var_99,
                    pr.cvar_95,
                    pr.max_expected_drawdown,
                    pr.portfolio_volatility,
                    pr.diversification_ratio,
                    pr.risk_concentration,
                    time.time(),
                ),
            )
            self._commit(conn)

    def count_risk_snapshots(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM portfolio_risk").fetchone()[0]

    def latest_risk(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_risk ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ── Scenario results ──────────────────────────────────────────────────────

    def save_scenario(self, result: ScenarioResult) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO scenario_results
                   (scenario, portfolio_return, portfolio_var, max_drawdown, resilience, timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (
                    result.scenario.value,
                    result.portfolio_return,
                    result.portfolio_var,
                    result.max_drawdown,
                    result.resilience_score,
                    time.time(),
                ),
            )
            self._commit(conn)

    def count_scenarios(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM scenario_results").fetchone()[0]

    def worst_scenario(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM scenario_results ORDER BY resilience ASC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ── Capital health ────────────────────────────────────────────────────────

    def save_health(self, health: CapitalHealth) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO capital_health
                   (efficiency, risk_score, allocation_q, diversification, stability, utilisation, overall, grade, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    health.efficiency_score,
                    health.risk_score,
                    health.allocation_quality,
                    health.diversification,
                    health.stability,
                    health.utilisation_score,
                    health.overall_score,
                    health.grade.value,
                    health.timestamp,
                ),
            )
            self._commit(conn)

    def count_health_snapshots(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM capital_health").fetchone()[0]

    def latest_health(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM capital_health ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ── Optimization ──────────────────────────────────────────────────────────

    def save_optimization(self, result: OptimizationResult) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO optimization_results
                   (result_id, objective, weights, metrics, converged, iterations, timestamp)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    result.result_id,
                    result.objective.value,
                    json.dumps(result.weights),
                    json.dumps(result.metrics),
                    int(result.converged),
                    result.iterations,
                    result.timestamp,
                ),
            )
            self._commit(conn)

    def count_optimizations(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM optimization_results").fetchone()[0]

    # ── Drawdown events ───────────────────────────────────────────────────────

    def save_drawdown_event(self, event: DrawdownEvent) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO drawdown_events
                   (event_id, peak, trough, drawdown, mode, recovered, start_ts, end_ts)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    event.event_id,
                    event.peak,
                    event.trough,
                    event.drawdown,
                    event.mode.value,
                    int(event.recovered),
                    event.start_ts,
                    event.end_ts,
                ),
            )
            self._commit(conn)

    def count_drawdown_events(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM drawdown_events").fetchone()[0]

    def unrecovered_events(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM drawdown_events WHERE recovered=0").fetchall()
        return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "allocations": self.count_allocations(),
            "risk_snapshots": self.count_risk_snapshots(),
            "health_snapshots": self.count_health_snapshots(),
            "scenarios": self.count_scenarios(),
            "optimizations": self.count_optimizations(),
            "drawdown_events": self.count_drawdown_events(),
            "budgets": self.count_budgets(),
        }
