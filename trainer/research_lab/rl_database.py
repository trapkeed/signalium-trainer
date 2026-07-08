"""Research Lab — SQLite persistence layer.

Thread-safe, WAL-mode SQLite database for the Research Lab subsystem.
Follows the same pattern as trainer/mlops/database.py.

All complex fields are serialised as JSON blobs. Scalar columns that are
commonly used for filtering are extracted to their own columns so queries
remain fast without full-table JSON scans.

Usage::

    db = ResearchLabDatabase("data/research_lab.db")
    db.upsert_experiment(record.to_dict())
    rows = db.list_experiments(status="COMPLETED")
    db.close()
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

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
-- Experiments (tracking every training run)
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id  TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    qde_version    TEXT NOT NULL,
    status         TEXT NOT NULL,
    researcher     TEXT NOT NULL DEFAULT '',
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    data_json      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_experiments_qde_version ON experiments(qde_version);
CREATE INDEX IF NOT EXISTS idx_experiments_status       ON experiments(status);

-- Datasets
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    data_json   TEXT NOT NULL DEFAULT '{}'
);

-- Dataset generation jobs
CREATE TABLE IF NOT EXISTS dataset_generation_jobs (
    job_id      TEXT PRIMARY KEY,
    dataset_id  TEXT NOT NULL DEFAULT '',
    name        TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    data_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_gen_jobs_status     ON dataset_generation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_gen_jobs_dataset_id ON dataset_generation_jobs(dataset_id);

-- Backtests
CREATE TABLE IF NOT EXISTS backtests (
    backtest_id   TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    created_at    REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_backtests_experiment ON backtests(experiment_id);

-- Deployment gate results (validation)
CREATE TABLE IF NOT EXISTS gate_results (
    gate_id       TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    passed        INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_gates_experiment ON gate_results(experiment_id);

-- Failure events
CREATE TABLE IF NOT EXISTS failures (
    failure_id  TEXT PRIMARY KEY,
    qde_version TEXT NOT NULL,
    category    TEXT NOT NULL,
    severity    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'OPEN',
    created_at  REAL NOT NULL,
    data_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_failures_qde_version ON failures(qde_version);
CREATE INDEX IF NOT EXISTS idx_failures_status      ON failures(status);
CREATE INDEX IF NOT EXISTS idx_failures_severity    ON failures(severity);

-- Version comparisons
CREATE TABLE IF NOT EXISTS comparisons (
    comparison_id TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);

-- Shadow tests (metadata only; decisions stored separately)
CREATE TABLE IF NOT EXISTS shadow_tests (
    test_id            TEXT PRIMARY KEY,
    challenger_version TEXT NOT NULL,
    champion_version   TEXT NOT NULL,
    status             TEXT NOT NULL,
    started_at         REAL NOT NULL,
    data_json          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_shadow_tests_challenger ON shadow_tests(challenger_version);
CREATE INDEX IF NOT EXISTS idx_shadow_tests_status     ON shadow_tests(status);

-- Shadow decisions (high-volume; one row per decision)
CREATE TABLE IF NOT EXISTS shadow_decisions (
    record_id   TEXT PRIMARY KEY,
    test_id     TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    data_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_shadow_decisions_test ON shadow_decisions(test_id);

-- Deployment records (promotion approvals)
CREATE TABLE IF NOT EXISTS deployments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    action        TEXT NOT NULL,
    experiment_id TEXT NOT NULL DEFAULT '',
    approver      TEXT NOT NULL DEFAULT '',
    timestamp     REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_deployments_experiment ON deployments(experiment_id);

-- Rollback requests (rollback approval workflow)
CREATE TABLE IF NOT EXISTS rollback_requests (
    request_id         TEXT PRIMARY KEY,
    current_version_id TEXT NOT NULL,
    target_version_id  TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'PENDING',
    requested_by       TEXT NOT NULL DEFAULT '',
    created_at         REAL NOT NULL,
    data_json          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_rollback_status ON rollback_requests(status);

-- Lifecycle authorizations (sole-authority gate for promote/rollback/archive/retire)
CREATE TABLE IF NOT EXISTS lifecycle_authorizations (
    authorization_id TEXT PRIMARY KEY,
    action           TEXT NOT NULL,
    version_id       TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'PENDING',
    requested_by     TEXT NOT NULL DEFAULT '',
    created_at       REAL NOT NULL,
    data_json        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_auth_version ON lifecycle_authorizations(version_id);
CREATE INDEX IF NOT EXISTS idx_lifecycle_auth_status ON lifecycle_authorizations(status);

-- Parameter searches
CREATE TABLE IF NOT EXISTS parameter_searches (
    search_id   TEXT PRIMARY KEY,
    qde_version TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    data_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_searches_qde_version ON parameter_searches(qde_version);

-- Regime analysis reports
CREATE TABLE IF NOT EXISTS regime_reports (
    report_id   TEXT PRIMARY KEY,
    qde_version TEXT NOT NULL,
    created_at  REAL NOT NULL,
    data_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_regime_reports_version ON regime_reports(qde_version);

-- Feature discovery
CREATE TABLE IF NOT EXISTS features (
    feature_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'CANDIDATE',
    created_at REAL NOT NULL,
    data_json  TEXT NOT NULL DEFAULT '{}'
);

-- Benchmarks
CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    data_json    TEXT NOT NULL DEFAULT '{}'
);

-- Research notes (standalone, addressable notes; not embedded in other records)
CREATE TABLE IF NOT EXISTS research_notes (
    note_id            TEXT PRIMARY KEY,
    author             TEXT NOT NULL DEFAULT '',
    linked_entity_type TEXT NOT NULL DEFAULT '',
    linked_entity_id   TEXT NOT NULL DEFAULT '',
    created_at         REAL NOT NULL,
    data_json          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_research_notes_author ON research_notes(author);
CREATE INDEX IF NOT EXISTS idx_research_notes_linked_entity
    ON research_notes(linked_entity_type, linked_entity_id);
"""


# ── Database ──────────────────────────────────────────────────────────────────


class ResearchLabDatabase:
    """SQLite-backed persistence for the Research Lab.

    Thread-safe. Uses WAL journal mode for concurrent reads.
    All complex data stored as JSON blobs; scalar filter columns extracted.
    """

    def __init__(self, db_path: str = "data/research_lab.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"ResearchLabDatabase: opened {self._path}")

    # ── Connection management ──────────────────────────────────────────────────

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

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _j(obj: Any) -> str:
        return json.dumps(obj, default=str)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        raw = d.pop("data_json", "{}")
        data = json.loads(raw) if raw else {}
        data.update(d)
        return data

    # ── Experiments ────────────────────────────────────────────────────────────

    def upsert_experiment(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO experiments
                (experiment_id, name, qde_version, status, researcher, created_at, updated_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                status     = excluded.status,
                updated_at = excluded.updated_at,
                data_json  = excluded.data_json
            """,
            (
                record["experiment_id"],
                record.get("name", ""),
                record.get("qde_version", ""),
                record.get("status", "PENDING"),
                record.get("researcher", ""),
                record.get("created_at", 0.0),
                record.get("updated_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_experiments(
        self,
        qde_version: str | None = None,
        status: str | None = None,
        researcher: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT data_json FROM experiments WHERE 1=1"
        params: list[Any] = []
        if qde_version:
            query += " AND qde_version = ?"
            params.append(qde_version)
        if status:
            query += " AND status = ?"
            params.append(status)
        if researcher:
            query += " AND researcher = ?"
            params.append(researcher)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Datasets ───────────────────────────────────────────────────────────────

    def upsert_dataset(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO datasets (dataset_id, name, version, created_at, data_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET data_json = excluded.data_json
            """,
            (
                record["dataset_id"],
                record.get("name", ""),
                record.get("version", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM datasets WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_datasets(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM datasets ORDER BY created_at DESC"
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Dataset Generation Jobs ────────────────────────────────────────────────

    def upsert_generation_job(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO dataset_generation_jobs
                (job_id, dataset_id, name, status, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status    = excluded.status,
                data_json = excluded.data_json
            """,
            (
                record["job_id"],
                record.get("dataset_id", ""),
                record.get("name", ""),
                record.get("status", "PENDING"),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_generation_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM dataset_generation_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_generation_jobs(
        self,
        dataset_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT data_json FROM dataset_generation_jobs WHERE 1=1"
        params: list[Any] = []
        if dataset_id:
            query += " AND dataset_id = ?"
            params.append(dataset_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Backtests ──────────────────────────────────────────────────────────────

    def upsert_backtest(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO backtests (backtest_id, experiment_id, created_at, data_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(backtest_id) DO UPDATE SET data_json = excluded.data_json
            """,
            (
                record["backtest_id"],
                record.get("experiment_id", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def list_backtests(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id:
            rows = self._conn.execute(
                "SELECT data_json FROM backtests WHERE experiment_id = ? ORDER BY created_at DESC",
                (experiment_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data_json FROM backtests ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Gate Results (Validation) ──────────────────────────────────────────────

    def upsert_gate_result(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO gate_results (gate_id, experiment_id, passed, created_at, data_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(gate_id) DO UPDATE SET data_json = excluded.data_json
            """,
            (
                record["gate_id"],
                record.get("experiment_id", ""),
                1 if record.get("passed") else 0,
                0.0,
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_gate_result(self, gate_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM gate_results WHERE gate_id = ?", (gate_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_gate_results(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id:
            rows = self._conn.execute(
                "SELECT data_json FROM gate_results WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT data_json FROM gate_results").fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Failures ───────────────────────────────────────────────────────────────

    def upsert_failure(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO failures
                (failure_id, qde_version, category, severity, status, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(failure_id) DO UPDATE SET
                status    = excluded.status,
                data_json = excluded.data_json
            """,
            (
                record["failure_id"],
                record.get("qde_version", ""),
                record.get("category", ""),
                record.get("severity", ""),
                record.get("status", "OPEN"),
                record.get("detected_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_failure(self, failure_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM failures WHERE failure_id = ?", (failure_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_failures(
        self,
        category: str | None = None,
        severity: str | None = None,
        qde_version: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT data_json FROM failures WHERE 1=1"
        params: list[Any] = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if qde_version:
            query += " AND qde_version = ?"
            params.append(qde_version)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Comparisons ────────────────────────────────────────────────────────────

    def upsert_comparison(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO comparisons (comparison_id, created_at, data_json)
            VALUES (?, ?, ?)
            ON CONFLICT(comparison_id) DO UPDATE SET data_json = excluded.data_json
            """,
            (record["comparison_id"], record.get("created_at", 0.0), self._j(record)),
        )
        self._conn.commit()

    def get_comparison(self, comparison_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM comparisons WHERE comparison_id = ?", (comparison_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_comparisons(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM comparisons ORDER BY created_at DESC"
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Shadow Tests ───────────────────────────────────────────────────────────

    def upsert_shadow_test(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO shadow_tests
                (test_id, challenger_version, champion_version, status, started_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(test_id) DO UPDATE SET
                status    = excluded.status,
                data_json = excluded.data_json
            """,
            (
                record["test_id"],
                record.get("challenger_version", ""),
                record.get("champion_version", ""),
                record.get("status", "RUNNING"),
                record.get("started_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_shadow_test(self, test_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM shadow_tests WHERE test_id = ?", (test_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_shadow_tests(
        self,
        challenger_version: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT data_json FROM shadow_tests WHERE 1=1"
        params: list[Any] = []
        if challenger_version:
            query += " AND challenger_version = ?"
            params.append(challenger_version)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY started_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def save_shadow_decision(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO shadow_decisions (record_id, test_id, recorded_at, data_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                record["record_id"],
                record.get("test_id", ""),
                record.get("recorded_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def count_shadow_decisions(self, test_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM shadow_decisions WHERE test_id = ?", (test_id,)
        ).fetchone()
        return int(row["n"]) if row else 0

    def list_shadow_decisions(self, test_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM shadow_decisions WHERE test_id = ? ORDER BY recorded_at",
            (test_id,),
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Deployments (Promotion Approvals) ──────────────────────────────────────

    def save_deployment(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO deployments (action, experiment_id, approver, timestamp, data_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.get("action", ""),
                record.get("experiment_id", ""),
                record.get("approver", record.get("rejector", "")),
                record.get("timestamp", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def list_deployments(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM deployments ORDER BY timestamp DESC"
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Rollback Requests ──────────────────────────────────────────────────────

    def upsert_rollback_request(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO rollback_requests
                (request_id, current_version_id, target_version_id, status,
                 requested_by, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                status    = excluded.status,
                data_json = excluded.data_json
            """,
            (
                record["request_id"],
                record.get("current_version_id", ""),
                record.get("target_version_id", ""),
                record.get("status", "PENDING"),
                record.get("requested_by", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_rollback_request(self, request_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM rollback_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_rollback_requests(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT data_json FROM rollback_requests WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data_json FROM rollback_requests ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Lifecycle Authorizations ─────────────────────────────────────────────────

    def upsert_lifecycle_authorization(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO lifecycle_authorizations
                (authorization_id, action, version_id, status, requested_by, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(authorization_id) DO UPDATE SET
                status    = excluded.status,
                data_json = excluded.data_json
            """,
            (
                record["authorization_id"],
                record.get("action", ""),
                record.get("version_id", ""),
                record.get("status", "PENDING"),
                record.get("requested_by", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_lifecycle_authorization(self, authorization_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM lifecycle_authorizations WHERE authorization_id = ?",
            (authorization_id,),
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_lifecycle_authorizations(
        self,
        version_id: str | None = None,
        action: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT data_json FROM lifecycle_authorizations WHERE 1=1"
        params: list[Any] = []
        if version_id:
            query += " AND version_id = ?"
            params.append(version_id)
        if action:
            query += " AND action = ?"
            params.append(action)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def find_approved_lifecycle_authorization(
        self, version_id: str, action: str
    ) -> dict[str, Any] | None:
        """Find the most recent APPROVED, unconsumed authorization for a
        given version + action. Used by the gateway to check permission
        before executing a lifecycle mutation."""
        rows = self._conn.execute(
            """
            SELECT data_json FROM lifecycle_authorizations
            WHERE version_id = ? AND action = ? AND status = 'APPROVED'
            ORDER BY created_at DESC LIMIT 1
            """,
            (version_id, action),
        ).fetchall()
        return json.loads(rows[0]["data_json"]) if rows else None

    # ── Parameter Searches ─────────────────────────────────────────────────────

    def upsert_search(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO parameter_searches (search_id, qde_version, status, created_at, data_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(search_id) DO UPDATE SET
                status    = excluded.status,
                data_json = excluded.data_json
            """,
            (
                record["search_id"],
                record.get("qde_version", ""),
                record.get("status", "RUNNING"),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_search(self, search_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM parameter_searches WHERE search_id = ?", (search_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_searches(self, qde_version: str | None = None) -> list[dict[str, Any]]:
        if qde_version:
            rows = self._conn.execute(
                "SELECT data_json FROM parameter_searches WHERE qde_version = ? ORDER BY created_at DESC",
                (qde_version,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data_json FROM parameter_searches ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Regime Reports ─────────────────────────────────────────────────────────

    def upsert_regime_report(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO regime_reports (report_id, qde_version, created_at, data_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(report_id) DO UPDATE SET data_json = excluded.data_json
            """,
            (
                record["report_id"],
                record.get("qde_version", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_regime_report(self, report_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM regime_reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_regime_reports(self, qde_version: str | None = None) -> list[dict[str, Any]]:
        if qde_version:
            rows = self._conn.execute(
                "SELECT data_json FROM regime_reports WHERE qde_version = ? ORDER BY created_at DESC",
                (qde_version,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data_json FROM regime_reports ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Features ───────────────────────────────────────────────────────────────

    def upsert_feature(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO features (feature_id, name, status, created_at, data_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(feature_id) DO UPDATE SET
                status    = excluded.status,
                data_json = excluded.data_json
            """,
            (
                record["feature_id"],
                record.get("name", ""),
                record.get("status", "CANDIDATE"),
                record.get("discovered_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_feature(self, feature_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM features WHERE feature_id = ?", (feature_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_features(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT data_json FROM features WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT data_json FROM features").fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Benchmarks ─────────────────────────────────────────────────────────────

    def upsert_benchmark(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO benchmarks (benchmark_id, name, created_at, data_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(benchmark_id) DO UPDATE SET data_json = excluded.data_json
            """,
            (
                record["benchmark_id"],
                record.get("name", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_benchmark(self, benchmark_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM benchmarks WHERE benchmark_id = ?", (benchmark_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_benchmarks(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM benchmarks ORDER BY created_at DESC"
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Research Notes ─────────────────────────────────────────────────────────

    def upsert_research_note(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO research_notes
                (note_id, author, linked_entity_type, linked_entity_id, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(note_id) DO UPDATE SET
                author             = excluded.author,
                linked_entity_type = excluded.linked_entity_type,
                linked_entity_id   = excluded.linked_entity_id,
                data_json          = excluded.data_json
            """,
            (
                record["note_id"],
                record.get("author", ""),
                record.get("linked_entity_type", ""),
                record.get("linked_entity_id", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_research_note(self, note_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM research_notes WHERE note_id = ?", (note_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_research_notes(
        self,
        author: str | None = None,
        linked_entity_type: str | None = None,
        linked_entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT data_json FROM research_notes WHERE 1=1"
        params: list[Any] = []
        if author:
            query += " AND author = ?"
            params.append(author)
        if linked_entity_type:
            query += " AND linked_entity_type = ?"
            params.append(linked_entity_type)
        if linked_entity_id:
            query += " AND linked_entity_id = ?"
            params.append(linked_entity_id)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        tables = [
            "experiments",
            "datasets",
            "dataset_generation_jobs",
            "backtests",
            "gate_results",
            "failures",
            "comparisons",
            "shadow_tests",
            "shadow_decisions",
            "deployments",
            "rollback_requests",
            "lifecycle_authorizations",
            "parameter_searches",
            "regime_reports",
            "features",
            "benchmarks",
            "research_notes",
        ]
        result: dict[str, int] = {}
        for table in tables:
            row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            result[table] = int(row["n"]) if row else 0
        return result
