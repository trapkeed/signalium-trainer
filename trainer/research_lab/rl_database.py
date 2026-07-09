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

_TELEMETRY_SCHEMA = """
-- ── Telemetry Warehouse (S20) ─────────────────────────────────────────────────
-- Eight append-only streams from Signalium. Nothing is ever deleted.
-- All share: event_id (PK), model_version, symbol, timestamp, data_json.
-- The scalar columns (model_version, symbol, timestamp) are extracted for fast
-- range and filter queries; all other fields live in data_json.

CREATE TABLE IF NOT EXISTS tel_decisions (
    event_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tel_dec_mv  ON tel_decisions(model_version);
CREATE INDEX IF NOT EXISTS idx_tel_dec_sym ON tel_decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_tel_dec_ts  ON tel_decisions(timestamp);

CREATE TABLE IF NOT EXISTS tel_trades (
    event_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tel_trd_mv  ON tel_trades(model_version);
CREATE INDEX IF NOT EXISTS idx_tel_trd_sym ON tel_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_tel_trd_ts  ON tel_trades(timestamp);

CREATE TABLE IF NOT EXISTS tel_positions (
    event_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tel_pos_mv  ON tel_positions(model_version);
CREATE INDEX IF NOT EXISTS idx_tel_pos_sym ON tel_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_tel_pos_ts  ON tel_positions(timestamp);

CREATE TABLE IF NOT EXISTS tel_failures (
    event_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tel_fail_mv  ON tel_failures(model_version);
CREATE INDEX IF NOT EXISTS idx_tel_fail_sym ON tel_failures(symbol);
CREATE INDEX IF NOT EXISTS idx_tel_fail_ts  ON tel_failures(timestamp);

CREATE TABLE IF NOT EXISTS tel_reasoning (
    event_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tel_rsn_mv  ON tel_reasoning(model_version);
CREATE INDEX IF NOT EXISTS idx_tel_rsn_sym ON tel_reasoning(symbol);
CREATE INDEX IF NOT EXISTS idx_tel_rsn_ts  ON tel_reasoning(timestamp);

CREATE TABLE IF NOT EXISTS tel_execution (
    event_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tel_exec_mv  ON tel_execution(model_version);
CREATE INDEX IF NOT EXISTS idx_tel_exec_sym ON tel_execution(symbol);
CREATE INDEX IF NOT EXISTS idx_tel_exec_ts  ON tel_execution(timestamp);

CREATE TABLE IF NOT EXISTS tel_counterfactuals (
    event_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tel_cf_mv  ON tel_counterfactuals(model_version);
CREATE INDEX IF NOT EXISTS idx_tel_cf_sym ON tel_counterfactuals(symbol);
CREATE INDEX IF NOT EXISTS idx_tel_cf_ts  ON tel_counterfactuals(timestamp);

CREATE TABLE IF NOT EXISTS tel_lifecycle (
    event_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tel_lce_mv  ON tel_lifecycle(model_version);
CREATE INDEX IF NOT EXISTS idx_tel_lce_sym ON tel_lifecycle(symbol);
CREATE INDEX IF NOT EXISTS idx_tel_lce_ts  ON tel_lifecycle(timestamp);
"""

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

-- Automatically-detected failure reports (S20 Failure Investigation Engine)
CREATE TABLE IF NOT EXISTS failure_reports (
    report_id     TEXT PRIMARY KEY,
    failure_kind  TEXT NOT NULL,
    severity_score REAL NOT NULL,
    model_version  TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_failure_reports_kind     ON failure_reports(failure_kind);
CREATE INDEX IF NOT EXISTS idx_failure_reports_created  ON failure_reports(created_at);
CREATE INDEX IF NOT EXISTS idx_failure_reports_severity ON failure_reports(severity_score DESC);
CREATE INDEX IF NOT EXISTS idx_failure_reports_mv       ON failure_reports(model_version);

-- Label datasets produced by LabelDiscoveryEngine (S20)
CREATE TABLE IF NOT EXISTS label_datasets (
    dataset_id    TEXT PRIMARY KEY,
    label_type    TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    sample_count  INTEGER NOT NULL DEFAULT 0,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_label_datasets_label_type ON label_datasets(label_type);
CREATE INDEX IF NOT EXISTS idx_label_datasets_mv        ON label_datasets(model_version);
CREATE INDEX IF NOT EXISTS idx_label_datasets_created   ON label_datasets(created_at);

-- Retraining recommendations produced by the RetrainingRecommendationEngine (S20)
CREATE TABLE IF NOT EXISTS retraining_recommendations (
    recommendation_id TEXT PRIMARY KEY,
    model_version     TEXT NOT NULL DEFAULT '',
    trigger_dimension TEXT NOT NULL,
    urgency           TEXT NOT NULL DEFAULT 'LOW',
    status            TEXT NOT NULL DEFAULT 'PROPOSED',
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    data_json         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_rr_mv        ON retraining_recommendations(model_version);
CREATE INDEX IF NOT EXISTS idx_rr_status    ON retraining_recommendations(status);
CREATE INDEX IF NOT EXISTS idx_rr_dimension ON retraining_recommendations(trigger_dimension);
CREATE INDEX IF NOT EXISTS idx_rr_created   ON retraining_recommendations(created_at DESC);

-- Training runs — provenance records (S20 Dataset Versioning)
CREATE TABLE IF NOT EXISTS training_runs (
    run_id            TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL DEFAULT '',
    pipeline_job_id   TEXT NOT NULL DEFAULT '',
    model_version     TEXT NOT NULL DEFAULT '',
    created_at        REAL NOT NULL,
    data_json         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_training_runs_experiment   ON training_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_pipeline_job ON training_runs(pipeline_job_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_created      ON training_runs(created_at DESC);

-- Registered artifacts — metadata-only (S20 Artifact Management)
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id      TEXT PRIMARY KEY,
    artifact_type    TEXT NOT NULL,
    artifact_hash    TEXT NOT NULL,
    training_run_id  TEXT NOT NULL DEFAULT '',
    model_version    TEXT NOT NULL DEFAULT '',
    experiment_id    TEXT NOT NULL DEFAULT '',
    registered_at    REAL NOT NULL,
    data_json        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_artifacts_type         ON artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_run          ON artifacts(training_run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_model        ON artifacts(model_version);
CREATE INDEX IF NOT EXISTS idx_artifacts_experiment   ON artifacts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_registered   ON artifacts(registered_at DESC);
-- Duplicate guard: (type, hash) must be unique across all registered artifacts.
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_type_hash
    ON artifacts(artifact_type, artifact_hash);

-- Decision quality reports — confidence-vs-outcome + expectancy breakdowns (S22B)
CREATE TABLE IF NOT EXISTS decision_quality_reports (
    report_id     TEXT PRIMARY KEY,
    model_version TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_dqr_mv      ON decision_quality_reports(model_version);
CREATE INDEX IF NOT EXISTS idx_dqr_created ON decision_quality_reports(created_at DESC);

-- Model health drift alerts (S20)
CREATE TABLE IF NOT EXISTS health_alerts (
    alert_id      TEXT PRIMARY KEY,
    dimension     TEXT NOT NULL,
    severity      TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_ha_dimension     ON health_alerts(dimension);
CREATE INDEX IF NOT EXISTS idx_ha_severity      ON health_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_ha_model_version ON health_alerts(model_version);
CREATE INDEX IF NOT EXISTS idx_ha_created       ON health_alerts(created_at DESC);

-- Training Pipeline jobs (S20)
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    job_id            TEXT PRIMARY KEY,
    status            TEXT NOT NULL DEFAULT 'QUEUED',
    experiment_id     TEXT NOT NULL DEFAULT '',
    recommendation_id TEXT NOT NULL DEFAULT '',
    requested_by      TEXT NOT NULL DEFAULT '',
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    data_json         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_pj_status            ON pipeline_jobs(status);
CREATE INDEX IF NOT EXISTS idx_pj_experiment_id     ON pipeline_jobs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_pj_recommendation_id ON pipeline_jobs(recommendation_id);
CREATE INDEX IF NOT EXISTS idx_pj_created           ON pipeline_jobs(created_at DESC);

-- Human Approval Gate requests (S20)
CREATE TABLE IF NOT EXISTS approval_requests (
    request_id   TEXT PRIMARY KEY,
    gate_type    TEXT NOT NULL,
    resource_id  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'PENDING',
    requested_by TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    expires_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    data_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_apr_gate_type   ON approval_requests(gate_type);
CREATE INDEX IF NOT EXISTS idx_apr_resource_id ON approval_requests(resource_id);
CREATE INDEX IF NOT EXISTS idx_apr_status      ON approval_requests(status);
CREATE INDEX IF NOT EXISTS idx_apr_created     ON approval_requests(created_at DESC);
-- Partial unique index: at most one PENDING request per (gate_type, resource_id).
-- This enforces idempotency at the DB level, preventing duplicate open requests
-- even under concurrent callers (SQLite serialises writes in WAL mode; the
-- UNIQUE constraint is the last line of defence against races).
CREATE UNIQUE INDEX IF NOT EXISTS idx_apr_gate_resource_pending
    ON approval_requests(gate_type, resource_id) WHERE status = 'PENDING';

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

-- Model deployment state — one row per model version, current deployment status (S20 Task #13)
CREATE TABLE IF NOT EXISTS model_deployments (
    model_version   TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'NOT_DEPLOYED',
    previous_status TEXT NOT NULL DEFAULT '',
    weights_url     TEXT NOT NULL DEFAULT '',
    updated_at      REAL NOT NULL,
    data_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_mdep_status ON model_deployments(status);

-- Deployment events — immutable audit log of all stage transitions and Signalium notifications (S20 Task #13)
CREATE TABLE IF NOT EXISTS deployment_events (
    event_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    from_status   TEXT NOT NULL DEFAULT '',
    to_status     TEXT NOT NULL DEFAULT '',
    actor         TEXT NOT NULL DEFAULT '',
    approval_id   TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    data_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_devt_mv      ON deployment_events(model_version);
CREATE INDEX IF NOT EXISTS idx_devt_type    ON deployment_events(event_type);
CREATE INDEX IF NOT EXISTS idx_devt_created ON deployment_events(created_at DESC);
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
        self._conn.executescript(_TELEMETRY_SCHEMA)
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

    # ── Label Datasets (S20 Label Discovery Engine) ───────────────────────────

    def upsert_label_dataset(self, record: dict[str, Any]) -> None:
        """Persist a LabelDataset produced by the LabelDiscoveryEngine."""
        self._conn.execute(
            """
            INSERT INTO label_datasets
                (dataset_id, label_type, model_version, created_at, sample_count, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                sample_count = excluded.sample_count,
                data_json    = excluded.data_json
            """,
            (
                record["dataset_id"],
                record.get("label_type", ""),
                record.get("model_version", ""),
                record.get("created_at", 0.0),
                record.get("sample_count", 0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_label_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM label_datasets WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_label_datasets(
        self,
        label_type: str | None = None,
        model_version: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM label_datasets WHERE 1=1"
        params: list[Any] = []
        if label_type:
            sql += " AND label_type = ?"
            params.append(label_type)
        if model_version:
            sql += " AND model_version = ?"
            params.append(model_version)
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Failure Reports (S20 Investigation Engine) ────────────────────────────

    def upsert_failure_report(self, record: dict[str, Any]) -> None:
        """Persist a FailureReport.  Uses upsert so re-runs can overwrite stale reports."""
        self._conn.execute(
            """
            INSERT INTO failure_reports
                (report_id, failure_kind, severity_score, model_version, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_id) DO UPDATE SET
                severity_score = excluded.severity_score,
                data_json      = excluded.data_json
            """,
            (
                record["report_id"],
                record.get("failure_kind", ""),
                record.get("severity_score", 0.0),
                record.get("model_version", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_failure_report(self, report_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM failure_reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_failure_reports(
        self,
        failure_kind: str | None = None,
        model_version: str | None = None,
        min_severity: float = 0.0,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM failure_reports WHERE severity_score >= ?"
        params: list[Any] = [min_severity]
        if failure_kind:
            sql += " AND failure_kind = ?"
            params.append(failure_kind)
        if model_version:
            sql += " AND model_version = ?"
            params.append(model_version)
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since)
        if until is not None:
            sql += " AND created_at <= ?"
            params.append(until)
        sql += " ORDER BY severity_score DESC, created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Model Health Alerts (S20) ─────────────────────────────────────────────

    def upsert_health_alert(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO health_alerts
                (alert_id, dimension, severity, model_version, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(alert_id) DO UPDATE SET
                dimension     = excluded.dimension,
                severity      = excluded.severity,
                model_version = excluded.model_version,
                created_at    = excluded.created_at,
                data_json     = excluded.data_json
            """,
            (
                record["alert_id"],
                record.get("dimension", ""),
                record.get("severity", ""),
                record.get("model_version", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_health_alert(self, alert_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM health_alerts WHERE alert_id = ?", (alert_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_health_alerts(
        self,
        dimension: str | None = None,
        severity: str | None = None,
        model_version: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM health_alerts WHERE 1=1"
        params: list[Any] = []
        if dimension:
            sql += " AND dimension = ?"
            params.append(dimension)
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        if model_version:
            sql += " AND model_version = ?"
            params.append(model_version)
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Decision Quality Reports (S22B) ───────────────────────────────────────

    def upsert_decision_quality_report(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO decision_quality_reports
                (report_id, model_version, created_at, data_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(report_id) DO UPDATE SET
                model_version = excluded.model_version,
                created_at    = excluded.created_at,
                data_json     = excluded.data_json
            """,
            (
                record["report_id"],
                record.get("model_version", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_decision_quality_report(self, report_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM decision_quality_reports WHERE report_id = ?",
            (report_id,),
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_decision_quality_reports(
        self,
        model_version: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM decision_quality_reports WHERE 1=1"
        params: list[Any] = []
        if model_version:
            sql += " AND model_version = ?"
            params.append(model_version)
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Training Runs + Artifact Registry (S20) ───────────────────────────────

    def upsert_training_run(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO training_runs
                (run_id, experiment_id, pipeline_job_id, model_version, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                data_json = excluded.data_json
            """,
            (
                record["run_id"],
                record.get("experiment_id", ""),
                record.get("pipeline_job_id", ""),
                record.get("model_version", ""),
                record.get("created_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_training_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM training_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_training_runs(
        self,
        experiment_id: str | None = None,
        pipeline_job_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM training_runs WHERE 1=1"
        params: list[Any] = []
        if experiment_id:
            sql += " AND experiment_id = ?"
            params.append(experiment_id)
        if pipeline_job_id:
            sql += " AND pipeline_job_id = ?"
            params.append(pipeline_job_id)
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def upsert_artifact(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO artifacts
                (artifact_id, artifact_type, artifact_hash, training_run_id,
                 model_version, experiment_id, registered_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                data_json = excluded.data_json
            """,
            (
                record["artifact_id"],
                record.get("artifact_type", ""),
                record.get("artifact_hash", ""),
                record.get("training_run_id", ""),
                record.get("model_version", ""),
                record.get("experiment_id", ""),
                record.get("registered_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def find_artifact_by_type_hash(
        self, artifact_type: str, artifact_hash: str
    ) -> dict[str, Any] | None:
        """Return an existing artifact with the same (type, hash), or None."""
        row = self._conn.execute(
            "SELECT data_json FROM artifacts WHERE artifact_type = ? AND artifact_hash = ?",
            (artifact_type, artifact_hash),
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_artifacts(
        self,
        artifact_type: str | None = None,
        model_version: str | None = None,
        training_run_id: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM artifacts WHERE 1=1"
        params: list[Any] = []
        if artifact_type:
            sql += " AND artifact_type = ?"
            params.append(artifact_type)
        if model_version:
            sql += " AND model_version = ?"
            params.append(model_version)
        if training_run_id:
            sql += " AND training_run_id = ?"
            params.append(training_run_id)
        if experiment_id:
            sql += " AND experiment_id = ?"
            params.append(experiment_id)
        sql += " ORDER BY registered_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Training Pipeline Jobs (S20) ──────────────────────────────────────────

    def upsert_pipeline_job(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO pipeline_jobs
                (job_id, status, experiment_id, recommendation_id, requested_by,
                 created_at, updated_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status     = excluded.status,
                updated_at = excluded.updated_at,
                data_json  = excluded.data_json
            """,
            (
                record["job_id"],
                record.get("status", "QUEUED"),
                record.get("experiment_id") or "",
                record.get("recommendation_id") or "",
                record.get("requested_by", ""),
                record.get("created_at", 0.0),
                record.get("updated_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_pipeline_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM pipeline_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_pipeline_jobs(
        self,
        status: str | None = None,
        experiment_id: str | None = None,
        recommendation_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM pipeline_jobs WHERE 1=1"
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if experiment_id:
            sql += " AND experiment_id = ?"
            params.append(experiment_id)
        if recommendation_id:
            sql += " AND recommendation_id = ?"
            params.append(recommendation_id)
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Human Approval Gates (S20) ────────────────────────────────────────────

    def upsert_approval_request(self, record: dict[str, Any]) -> None:
        """Upsert an approval request.

        For new PENDING rows uses INSERT OR IGNORE so that a concurrent caller
        racing to create the same (gate_type, resource_id) PENDING request is
        silently discarded — the partial unique index `idx_apr_gate_resource_pending`
        guarantees at most one open PENDING per (gate_type, resource_id).

        For status transitions (PENDING→APPROVED/REJECTED/EXPIRED/CONSUMED) the
        row already exists, so the INSERT hits ON CONFLICT(request_id) and updates
        the scalar status + data_json in place. The partial index does not fire for
        non-PENDING rows, so there is no conflict on the unique gate+resource index.
        """
        status = record.get("status", "PENDING")
        params = (
            record["request_id"],
            record.get("gate_type", ""),
            record.get("resource_id", ""),
            status,
            record.get("requested_by", ""),
            record.get("created_at", 0.0),
            record.get("expires_at", 0.0),
            record.get("updated_at", 0.0),
            self._j(record),
        )
        if status == "PENDING":
            # INSERT OR IGNORE: if partial unique index fires (concurrent create),
            # the row is silently dropped; the caller re-fetches the existing row.
            self._conn.execute(
                """
                INSERT OR IGNORE INTO approval_requests
                    (request_id, gate_type, resource_id, status, requested_by,
                     created_at, expires_at, updated_at, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
        else:
            # Status transition: update the existing row by request_id.
            self._conn.execute(
                """
                INSERT INTO approval_requests
                    (request_id, gate_type, resource_id, status, requested_by,
                     created_at, expires_at, updated_at, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    status     = excluded.status,
                    updated_at = excluded.updated_at,
                    data_json  = excluded.data_json
                """,
                params,
            )
        self._conn.commit()

    def get_approval_request(self, request_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM approval_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_approval_requests(
        self,
        gate_type: str | None = None,
        resource_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM approval_requests WHERE 1=1"
        params: list[Any] = []
        if gate_type:
            sql += " AND gate_type = ?"
            params.append(gate_type)
        if resource_id:
            sql += " AND resource_id = ?"
            params.append(resource_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def atomic_consume_approval(
        self,
        gate_type: str,
        resource_id: str,
        consumed_by: str,
        consumed_at: float,
    ) -> int:
        """Atomically mark a single APPROVED request as CONSUMED.

        Uses UPDATE WHERE status='APPROVED' + rowcount check to prevent
        double-consume. Returns the number of rows updated (1 = success, 0 = blocked).
        """
        import json as _json
        now = consumed_at
        # We need to also update the data_json to reflect CONSUMED status.
        # First, get the matching APPROVED row.
        row = self._conn.execute(
            """
            SELECT request_id, data_json FROM approval_requests
            WHERE gate_type = ? AND resource_id = ? AND status = 'APPROVED'
            LIMIT 1
            """,
            (gate_type, resource_id),
        ).fetchone()
        if row is None:
            return 0

        # Build the updated data_json payload before touching the DB, so that
        # if anything raises we have not committed a partial state.
        data = _json.loads(row["data_json"])
        data["status"] = "CONSUMED"
        data["consumed_at"] = round(now, 3)
        data["consumed_by"] = consumed_by
        data["updated_at"] = round(now, 3)
        data.setdefault("audit_log", []).append({
            "event": "consumed",
            "actor": consumed_by,
            "timestamp": round(now, 3),
            "reason": "Gated action executed",
            "metadata": {},
        })

        # Single explicit transaction: update scalar status + JSON payload in
        # one atomic block with a single COMMIT.  The WHERE status='APPROVED'
        # clause is the optimistic lock; if another caller already consumed the
        # row, rowcount returns 0 and we roll back.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            cursor = self._conn.execute(
                """
                UPDATE approval_requests
                SET status = 'CONSUMED', updated_at = ?, data_json = ?
                WHERE request_id = ? AND status = 'APPROVED'
                """,
                (now, self._j(data), row["request_id"]),
            )
            if cursor.rowcount == 0:
                self._conn.execute("ROLLBACK")
                return 0
            self._conn.execute("COMMIT")
        except Exception:
            try:
                self._conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        return 1

    # ── Retraining Recommendations (S20) ──────────────────────────────────────

    def upsert_retraining_recommendation(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO retraining_recommendations
                (recommendation_id, model_version, trigger_dimension, urgency,
                 status, created_at, updated_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(recommendation_id) DO UPDATE SET
                status     = excluded.status,
                urgency    = excluded.urgency,
                updated_at = excluded.updated_at,
                data_json  = excluded.data_json
            """,
            (
                record["recommendation_id"],
                record.get("model_version", ""),
                record.get("trigger_dimension", ""),
                record.get("urgency", "LOW"),
                record.get("status", "PROPOSED"),
                record.get("created_at", 0.0),
                record.get("updated_at", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def get_retraining_recommendation(self, recommendation_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM retraining_recommendations WHERE recommendation_id = ?",
            (recommendation_id,),
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_retraining_recommendations(
        self,
        status: str | None = None,
        model_version: str | None = None,
        trigger_dimension: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM retraining_recommendations WHERE 1=1"
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if model_version:
            sql += " AND model_version = ?"
            params.append(model_version)
        if trigger_dimension:
            sql += " AND trigger_dimension = ?"
            params.append(trigger_dimension)
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    # ── Telemetry Warehouse (S20) ──────────────────────────────────────────────

    _TELEMETRY_TABLES = (
        "tel_decisions",
        "tel_trades",
        "tel_positions",
        "tel_failures",
        "tel_reasoning",
        "tel_execution",
        "tel_counterfactuals",
        "tel_lifecycle",
    )

    def insert_telemetry_event(self, table: str, event_id: str, record: dict[str, Any]) -> None:
        """Append a single telemetry event to the appropriate table.

        Append-only: uses INSERT OR IGNORE so duplicate event_ids are silently
        skipped (idempotent re-delivery from Signalium).
        """
        if table not in self._TELEMETRY_TABLES:
            raise ValueError(f"Unknown telemetry table: {table!r}")
        self._conn.execute(
            f"""
            INSERT OR IGNORE INTO {table}
                (event_id, model_version, symbol, timestamp, data_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_id,
                record.get("model_version", ""),
                record.get("symbol", ""),
                record.get("timestamp", 0.0),
                self._j(record),
            ),
        )
        self._conn.commit()

    def query_telemetry(self, table: str, q: Any) -> list[dict[str, Any]]:
        """Typed read from a telemetry table with optional filters.

        Args:
            table: One of the _TELEMETRY_TABLES names.
            q: TelemetryQuery dataclass (model_version, symbol, since, until, limit).

        Returns:
            List of event dicts, ordered by timestamp DESC.
        """
        if table not in self._TELEMETRY_TABLES:
            raise ValueError(f"Unknown telemetry table: {table!r}")
        sql = f"SELECT data_json FROM {table} WHERE 1=1"
        params: list[Any] = []
        if q.model_version:
            sql += " AND model_version = ?"
            params.append(q.model_version)
        if q.symbol:
            sql += " AND symbol = ?"
            params.append(q.symbol)
        if q.since is not None:
            sql += " AND timestamp >= ?"
            params.append(q.since)
        if q.until is not None:
            sql += " AND timestamp <= ?"
            params.append(q.until)
        sql += " ORDER BY timestamp DESC"
        limit = getattr(q, "limit", 1000)
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def telemetry_stream_counts(self) -> dict[str, int]:
        """Row counts for all eight telemetry streams."""
        result: dict[str, int] = {}
        for table in self._TELEMETRY_TABLES:
            row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            result[table] = int(row["n"]) if row else 0
        return result

    # ── Model Deployments (S20 Task #13) ──────────────────────────────────────

    def upsert_model_deployment(self, record: dict[str, Any]) -> None:
        """Upsert current deployment status for a model version.

        Refreshes all scalar filter columns alongside data_json on conflict.
        """
        self._conn.execute(
            """
            INSERT INTO model_deployments
                (model_version, status, previous_status, weights_url, updated_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_version) DO UPDATE SET
                status          = excluded.status,
                previous_status = excluded.previous_status,
                weights_url     = excluded.weights_url,
                updated_at      = excluded.updated_at,
                data_json       = excluded.data_json
            """,
            (
                record["model_version"],
                record.get("status", "NOT_DEPLOYED"),
                record.get("previous_status", ""),
                record.get("weights_url", ""),
                record.get("updated_at", 0.0),
                json.dumps(record),
            ),
        )
        self._conn.commit()

    def get_model_deployment(self, model_version: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM model_deployments WHERE model_version = ?",
            (model_version,),
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_model_deployments(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM model_deployments WHERE 1=1"
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY updated_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        return [json.loads(r["data_json"]) for r in self._conn.execute(sql, params).fetchall()]

    def retire_active_deployments(self, except_model_version: str) -> int:
        """Atomically retire all ACTIVE model versions except the specified one.

        Returns the number of rows retired.
        """
        now = __import__("time").time()
        rows = self._conn.execute(
            "SELECT model_version, data_json FROM model_deployments WHERE status = 'ACTIVE' AND model_version != ?",
            (except_model_version,),
        ).fetchall()
        count = 0
        for row in rows:
            rec = json.loads(row["data_json"])
            rec["previous_status"] = rec.get("status", "ACTIVE")
            rec["status"] = "RETIRED"
            rec["updated_at"] = now
            self._conn.execute(
                """
                UPDATE model_deployments
                SET status = 'RETIRED', previous_status = ?, updated_at = ?, data_json = ?
                WHERE model_version = ? AND status = 'ACTIVE'
                """,
                (rec["previous_status"], now, json.dumps(rec), row["model_version"]),
            )
            count += self._conn.execute("SELECT changes()").fetchone()[0]
        self._conn.commit()
        return count

    # ── Deployment Events (S20 Task #13) ──────────────────────────────────────

    def insert_deployment_event(self, record: dict[str, Any]) -> None:
        """Append an immutable deployment event to the audit log."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO deployment_events
                (event_id, model_version, event_type, from_status, to_status,
                 actor, approval_id, created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["event_id"],
                record.get("model_version", ""),
                record.get("event_type", ""),
                record.get("from_status", ""),
                record.get("to_status", ""),
                record.get("actor", ""),
                record.get("approval_id", ""),
                record.get("created_at", 0.0),
                json.dumps(record),
            ),
        )
        self._conn.commit()

    def list_deployment_events(
        self,
        model_version: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM deployment_events WHERE 1=1"
        params: list[Any] = []
        if model_version:
            sql += " AND model_version = ?"
            params.append(model_version)
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        return [json.loads(r["data_json"]) for r in self._conn.execute(sql, params).fetchall()]

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
        ] + list(self._TELEMETRY_TABLES) + [
            "failure_reports", "label_datasets",
            "retraining_recommendations", "approval_requests", "pipeline_jobs",
            "training_runs", "artifacts", "health_alerts",
            "model_deployments", "deployment_events",
            "decision_quality_reports",
        ]
        result: dict[str, int] = {}
        for table in tables:
            row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            result[table] = int(row["n"]) if row else 0
        return result
