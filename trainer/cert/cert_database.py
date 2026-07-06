"""Phase 25 — Certification Database (SQLite, 6 tables)."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from collections.abc import Iterator
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

_SCHEMA = """
-- 1. Certification runs (master log)
CREATE TABLE IF NOT EXISTS certification_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,
    model_id        TEXT    NOT NULL,
    model_name      TEXT,
    model_version   TEXT,
    requester       TEXT,
    started_at      REAL    NOT NULL,
    completed_at    REAL,
    decision_status TEXT,
    overall_score   REAL,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_cr_model_id ON certification_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_cr_decision ON certification_runs(decision_status);

-- 2. Certification results (per-stage detail)
CREATE TABLE IF NOT EXISTS certification_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    stage           TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    score           REAL,
    findings_json   TEXT,
    warnings_json   TEXT,
    failures_json   TEXT,
    duration_ms     REAL,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_cres_run_id ON certification_results(run_id, stage);

-- 3. Certification criteria (checklist items)
CREATE TABLE IF NOT EXISTS certification_criteria (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    check_id        TEXT    NOT NULL,
    category        TEXT,
    description     TEXT,
    mandatory       INTEGER DEFAULT 1,
    status          TEXT    NOT NULL,
    value_json      TEXT,
    threshold       REAL,
    rationale       TEXT,
    risk_level      TEXT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_cc_run_id ON certification_criteria(run_id, status);

-- 4. Benchmark results
CREATE TABLE IF NOT EXISTS benchmark_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    metric          TEXT    NOT NULL,
    value           REAL,
    unit            TEXT,
    threshold       REAL,
    status          TEXT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_br_run_id ON benchmark_results(run_id, metric);

-- 5. Audit packages
CREATE TABLE IF NOT EXISTS audit_packages (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                TEXT    NOT NULL UNIQUE,
    certification_report  TEXT,
    validation_report     TEXT,
    benchmark_report      TEXT,
    stress_test_report    TEXT,
    security_report       TEXT,
    operational_checklist TEXT,
    executive_summary     TEXT,
    generated_at          REAL,
    created_at            REAL    NOT NULL DEFAULT (unixepoch())
);

-- 6. Production approvals (human sign-off)
CREATE TABLE IF NOT EXISTS production_approvals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    model_id        TEXT    NOT NULL,
    approved_by     TEXT    NOT NULL,
    approval_status TEXT    NOT NULL,
    conditions_json TEXT,
    approved_at     REAL    NOT NULL DEFAULT (unixepoch()),
    expires_at      REAL,
    notes           TEXT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_pa_model_id ON production_approvals(model_id);
CREATE INDEX IF NOT EXISTS idx_pa_run_id   ON production_approvals(run_id);
"""


class CertDatabase:
    """SQLite persistence layer for the Certification Platform."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._path = db_path
        if db_path == ":memory:":
            self._persistent_conn: sqlite3.Connection | None = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.executescript(_SCHEMA)
            self._persistent_conn.commit()
        else:
            self._persistent_conn = None
            with self._conn() as conn:
                conn.executescript(_SCHEMA)
        logger.debug(f"[CertDB] Initialised at {db_path!r}")

    # ── Context manager ────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._persistent_conn is not None:
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ── Write: Certification Runs ──────────────────────────────────────────

    def save_run(self, rec: dict[str, Any]) -> int:
        sql = """
        INSERT INTO certification_runs
            (run_id, model_id, model_name, model_version, requester,
             started_at, completed_at, decision_status, overall_score)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(run_id) DO UPDATE SET
            completed_at    = excluded.completed_at,
            decision_status = excluded.decision_status,
            overall_score   = excluded.overall_score
        """
        with self._conn() as conn:
            cur = conn.execute(
                sql,
                (
                    rec.get("run_id", ""),
                    rec.get("model_id", ""),
                    rec.get("model_name"),
                    rec.get("model_version"),
                    rec.get("requester"),
                    rec.get("started_at", time.time()),
                    rec.get("completed_at"),
                    rec.get("decision_status"),
                    rec.get("overall_score"),
                ),
            )
            return cur.lastrowid or 0

    # ── Write: Stage Results ───────────────────────────────────────────────

    def save_stage_result(self, rec: dict[str, Any]) -> int:
        sql = """
        INSERT INTO certification_results
            (run_id, stage, status, score, findings_json,
             warnings_json, failures_json, duration_ms)
        VALUES (?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(
                sql,
                (
                    rec.get("run_id", ""),
                    rec.get("stage", ""),
                    rec.get("status", "PASS"),
                    rec.get("score"),
                    json.dumps(rec.get("findings", [])),
                    json.dumps(rec.get("warnings", [])),
                    json.dumps(rec.get("failures", [])),
                    rec.get("duration_ms"),
                ),
            )
            return cur.lastrowid or 0

    # ── Write: Checklist Items ─────────────────────────────────────────────

    def save_checklist_item(self, run_id: str, rec: dict[str, Any]) -> int:
        sql = """
        INSERT INTO certification_criteria
            (run_id, check_id, category, description, mandatory,
             status, value_json, threshold, rationale, risk_level)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(
                sql,
                (
                    run_id,
                    rec.get("check_id", ""),
                    rec.get("category"),
                    rec.get("description"),
                    1 if rec.get("mandatory", True) else 0,
                    rec.get("status", "PASS"),
                    json.dumps(rec.get("value")),
                    rec.get("threshold"),
                    rec.get("rationale"),
                    rec.get("risk_level"),
                ),
            )
            return cur.lastrowid or 0

    # ── Write: Benchmark Results ───────────────────────────────────────────

    def save_benchmark(self, rec: dict[str, Any]) -> int:
        sql = """
        INSERT INTO benchmark_results
            (run_id, metric, value, unit, threshold, status)
        VALUES (?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(
                sql,
                (
                    rec.get("run_id", ""),
                    rec.get("metric", ""),
                    rec.get("value"),
                    rec.get("unit"),
                    rec.get("threshold"),
                    rec.get("status", "PASS"),
                ),
            )
            return cur.lastrowid or 0

    # ── Write: Audit Package ───────────────────────────────────────────────

    def save_audit_package(self, rec: dict[str, Any]) -> int:
        sql = """
        INSERT INTO audit_packages
            (run_id, certification_report, validation_report,
             benchmark_report, stress_test_report, security_report,
             operational_checklist, executive_summary, generated_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(run_id) DO UPDATE SET
            certification_report  = excluded.certification_report,
            validation_report     = excluded.validation_report,
            benchmark_report      = excluded.benchmark_report,
            stress_test_report    = excluded.stress_test_report,
            security_report       = excluded.security_report,
            operational_checklist = excluded.operational_checklist,
            executive_summary     = excluded.executive_summary,
            generated_at          = excluded.generated_at
        """
        with self._conn() as conn:
            cur = conn.execute(
                sql,
                (
                    rec.get("run_id", ""),
                    rec.get("certification_report"),
                    rec.get("validation_report"),
                    rec.get("benchmark_report"),
                    rec.get("stress_test_report"),
                    rec.get("security_report"),
                    rec.get("operational_checklist"),
                    rec.get("executive_summary"),
                    rec.get("generated_at", time.time()),
                ),
            )
            return cur.lastrowid or 0

    # ── Write: Production Approvals ────────────────────────────────────────

    def save_approval(self, rec: dict[str, Any]) -> int:
        sql = """
        INSERT INTO production_approvals
            (run_id, model_id, approved_by, approval_status,
             conditions_json, approved_at, expires_at, notes)
        VALUES (?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(
                sql,
                (
                    rec.get("run_id", ""),
                    rec.get("model_id", ""),
                    rec.get("approved_by", "SYSTEM"),
                    rec.get("approval_status", "PENDING"),
                    json.dumps(rec.get("conditions", [])),
                    rec.get("approved_at", time.time()),
                    rec.get("expires_at"),
                    rec.get("notes"),
                ),
            )
            return cur.lastrowid or 0

    # ── Read ───────────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM certification_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_runs_for_model(self, model_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM certification_runs WHERE model_id=? "
                "ORDER BY started_at DESC LIMIT ?",
                (model_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stage_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM certification_results WHERE run_id=? " "ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_checklist(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM certification_criteria WHERE run_id=? " "ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_benchmarks(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_results WHERE run_id=? ORDER BY metric",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_audit_package(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM audit_packages WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_approvals(self, model_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM production_approvals WHERE model_id=? " "ORDER BY approved_at DESC",
                (model_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_certified(self, model_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM certification_runs "
                "WHERE model_id=? AND decision_status IN ('CERTIFIED','CERTIFIED_WITH_CONDITIONS') "
                "ORDER BY completed_at DESC LIMIT 1",
                (model_id,),
            ).fetchone()
        return dict(row) if row else None

    def table_count(self, table: str) -> int:
        with self._conn() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return row[0] if row else 0
