"""Phase 20 — Portfolio Database.

SQLite persistence for:
  - Portfolio recommendations
  - Correlation matrices
  - Exposure history
  - Optimization results
  - Scenario results
  - Allocation history
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager


class PortfolioDatabase:
    """SQLite persistence layer for Phase 20 Portfolio Intelligence."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS portfolio_recommendations (
        rec_id          TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        instrument_count INTEGER,
        expected_return REAL,
        expected_volatility REAL,
        expected_sharpe REAL,
        diversification_score REAL,
        regime          TEXT,
        method          TEXT,
        weights_json    TEXT,
        metrics_json    TEXT
    );

    CREATE TABLE IF NOT EXISTS correlation_matrices (
        matrix_id       TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        method          TEXT,
        instruments_json TEXT,
        pairs_json      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_corr_ts ON correlation_matrices(timestamp);

    CREATE TABLE IF NOT EXISTS exposure_history (
        exp_id          TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        gross_exposure  REAL,
        net_exposure    REAL,
        long_exposure   REAL,
        short_exposure  REAL,
        diversification_ratio REAL,
        alert_count     INTEGER,
        by_asset_json   TEXT,
        by_currency_json TEXT,
        by_region_json  TEXT,
        alerts_json     TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_exp_ts ON exposure_history(timestamp);

    CREATE TABLE IF NOT EXISTS optimization_results (
        opt_id          TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        objective       TEXT,
        expected_sharpe REAL,
        expected_return REAL,
        expected_volatility REAL,
        diversification_score REAL,
        iterations      INTEGER,
        converged       INTEGER,
        weights_json    TEXT,
        metrics_json    TEXT
    );

    CREATE TABLE IF NOT EXISTS scenario_results (
        scenario_id     TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        scenario_type   TEXT,
        base_sharpe     REAL,
        stressed_sharpe REAL,
        return_impact   REAL,
        volatility_impact REAL,
        resilience_score REAL,
        instrument_impacts_json TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_scenario_type ON scenario_results(scenario_type);

    CREATE TABLE IF NOT EXISTS allocation_history (
        alloc_id        TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        method          TEXT,
        regime          TEXT,
        instrument_count INTEGER,
        expected_return REAL,
        expected_sharpe REAL,
        budget_feasible INTEGER,
        weights_json    TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_alloc_ts ON allocation_history(timestamp);
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._path = db_path
        self._is_memory = db_path == ":memory:"
        self._persistent_conn: sqlite3.Connection | None = None
        if self._is_memory:
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.execute("PRAGMA journal_mode=WAL")
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        if self._is_memory and self._persistent_conn is not None:
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ── Portfolio Recommendations ──────────────────────────────────────────

    def save_recommendation(self, rec: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO portfolio_recommendations
                   (rec_id, timestamp, instrument_count, expected_return,
                    expected_volatility, expected_sharpe, diversification_score,
                    regime, method, weights_json, metrics_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec["rec_id"],
                    rec.get("timestamp", ""),
                    rec.get("instrument_count", 0),
                    rec.get("expected_return", 0.0),
                    rec.get("expected_volatility", 0.0),
                    rec.get("expected_sharpe", 0.0),
                    rec.get("diversification_score", 0.0),
                    rec.get("regime", ""),
                    rec.get("method", ""),
                    json.dumps(rec.get("weights", {})),
                    json.dumps(rec.get("metrics", {})),
                ),
            )

    def get_recommendation(self, rec_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_recommendations WHERE rec_id = ?", (rec_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["weights"] = json.loads(d.pop("weights_json", "{}"))
        d["metrics"] = json.loads(d.pop("metrics_json", "{}"))
        return d

    def recommendations_history(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio_recommendations ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["weights"] = json.loads(d.pop("weights_json", "{}"))
            d["metrics"] = json.loads(d.pop("metrics_json", "{}"))
            result.append(d)
        return result

    # ── Correlation Matrices ───────────────────────────────────────────────

    def save_correlation_matrix(
        self, matrix_id: str, timestamp: str, method: str, instruments: list[str], pairs: dict
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO correlation_matrices
                   (matrix_id, timestamp, method, instruments_json, pairs_json)
                   VALUES (?,?,?,?,?)""",
                (matrix_id, timestamp, method, json.dumps(instruments), json.dumps(pairs)),
            )

    def get_correlation_matrix(self, matrix_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM correlation_matrices WHERE matrix_id = ?", (matrix_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["instruments"] = json.loads(d.pop("instruments_json", "[]"))
        d["pairs"] = json.loads(d.pop("pairs_json", "{}"))
        return d

    def latest_correlation_matrix(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM correlation_matrices ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["instruments"] = json.loads(d.pop("instruments_json", "[]"))
        d["pairs"] = json.loads(d.pop("pairs_json", "{}"))
        return d

    # ── Exposure History ───────────────────────────────────────────────────

    def save_exposure(self, exp: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO exposure_history
                   (exp_id, timestamp, gross_exposure, net_exposure,
                    long_exposure, short_exposure, diversification_ratio,
                    alert_count, by_asset_json, by_currency_json,
                    by_region_json, alerts_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    exp["exp_id"],
                    exp.get("timestamp", ""),
                    exp.get("gross_exposure", 0.0),
                    exp.get("net_exposure", 0.0),
                    exp.get("long_exposure", 0.0),
                    exp.get("short_exposure", 0.0),
                    exp.get("diversification_ratio", 0.0),
                    exp.get("alert_count", 0),
                    json.dumps(exp.get("by_asset_class", {})),
                    json.dumps(exp.get("by_currency", {})),
                    json.dumps(exp.get("by_region", {})),
                    json.dumps(exp.get("alerts", [])),
                ),
            )

    def exposure_history(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM exposure_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["by_asset_class"] = json.loads(d.pop("by_asset_json", "{}"))
            d["by_currency"] = json.loads(d.pop("by_currency_json", "{}"))
            d["by_region"] = json.loads(d.pop("by_region_json", "{}"))
            d["alerts"] = json.loads(d.pop("alerts_json", "[]"))
            result.append(d)
        return result

    # ── Optimization Results ───────────────────────────────────────────────

    def save_optimization(self, opt: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO optimization_results
                   (opt_id, timestamp, objective, expected_sharpe,
                    expected_return, expected_volatility, diversification_score,
                    iterations, converged, weights_json, metrics_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    opt["opt_id"],
                    opt.get("timestamp", ""),
                    opt.get("objective", ""),
                    opt.get("expected_sharpe", 0.0),
                    opt.get("expected_return", 0.0),
                    opt.get("expected_volatility", 0.0),
                    opt.get("diversification_score", 0.0),
                    opt.get("iterations", 0),
                    int(opt.get("converged", False)),
                    json.dumps(opt.get("weights", {})),
                    json.dumps(opt.get("metrics", {})),
                ),
            )

    def optimization_history(self, objective: str | None = None, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            if objective:
                rows = conn.execute(
                    "SELECT * FROM optimization_results WHERE objective = ? ORDER BY timestamp DESC LIMIT ?",
                    (objective, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM optimization_results ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["weights"] = json.loads(d.pop("weights_json", "{}"))
            d["metrics"] = json.loads(d.pop("metrics_json", "{}"))
            result.append(d)
        return result

    # ── Scenario Results ───────────────────────────────────────────────────

    def save_scenario(self, sc: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO scenario_results
                   (scenario_id, timestamp, scenario_type, base_sharpe,
                    stressed_sharpe, return_impact, volatility_impact,
                    resilience_score, instrument_impacts_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    sc["scenario_id"],
                    sc.get("timestamp", ""),
                    sc.get("scenario_type", ""),
                    sc.get("base_sharpe", 0.0),
                    sc.get("stressed_sharpe", 0.0),
                    sc.get("return_impact", 0.0),
                    sc.get("volatility_impact", 0.0),
                    sc.get("resilience_score", 0.0),
                    json.dumps(sc.get("instrument_impacts", {})),
                ),
            )

    def scenario_history(self, scenario_type: str | None = None, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            if scenario_type:
                rows = conn.execute(
                    "SELECT * FROM scenario_results WHERE scenario_type = ? ORDER BY timestamp DESC LIMIT ?",
                    (scenario_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM scenario_results ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["instrument_impacts"] = json.loads(d.pop("instrument_impacts_json", "{}"))
            result.append(d)
        return result

    # ── Allocation History ─────────────────────────────────────────────────

    def save_allocation(self, alloc: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO allocation_history
                   (alloc_id, timestamp, method, regime, instrument_count,
                    expected_return, expected_sharpe, budget_feasible, weights_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    alloc["alloc_id"],
                    alloc.get("timestamp", ""),
                    alloc.get("method", ""),
                    alloc.get("regime", ""),
                    alloc.get("instrument_count", 0),
                    alloc.get("expected_return", 0.0),
                    alloc.get("expected_sharpe", 0.0),
                    int(alloc.get("budget_feasible", True)),
                    json.dumps(alloc.get("weights", {})),
                ),
            )

    def allocation_history(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM allocation_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["weights"] = json.loads(d.pop("weights_json", "{}"))
            result.append(d)
        return result

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._conn() as conn:
            recs = conn.execute("SELECT COUNT(*) FROM portfolio_recommendations").fetchone()[0]
            corrs = conn.execute("SELECT COUNT(*) FROM correlation_matrices").fetchone()[0]
            exps = conn.execute("SELECT COUNT(*) FROM exposure_history").fetchone()[0]
            opts = conn.execute("SELECT COUNT(*) FROM optimization_results").fetchone()[0]
            scens = conn.execute("SELECT COUNT(*) FROM scenario_results").fetchone()[0]
            allocs = conn.execute("SELECT COUNT(*) FROM allocation_history").fetchone()[0]
            avg_sharpe = conn.execute(
                "SELECT AVG(expected_sharpe) FROM portfolio_recommendations"
            ).fetchone()[0]
        return {
            "recommendations": recs,
            "correlation_matrices": corrs,
            "exposures": exps,
            "optimizations": opts,
            "scenarios": scens,
            "allocations": allocs,
            "average_sharpe": round(avg_sharpe, 4) if avg_sharpe else None,
        }
