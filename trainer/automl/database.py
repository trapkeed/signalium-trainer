"""Phase 16 — AutoML Database.

SQLite-backed storage for all AutoML platform data.

Stores:
  - Search jobs
  - Experiments (trial records)
  - Trial history (intermediate scores)
  - Hyperparameters
  - Objective scores
  - Resource usage snapshots
  - Pruning decisions
  - Promotion decisions

Uses Python's built-in ``sqlite3`` — no extra dependencies.

Usage::

    db = AutoMLDatabase("experiments/automl.db")
    job_id = db.save_job(job)
    record_id = db.save_record(record)
    records = db.list_records(job_id=job_id, limit=50)
    db.close()
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from trainer.automl.experiment import ExperimentRecord, SearchJob

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS search_jobs (
    job_id          TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    n_trials        INTEGER NOT NULL DEFAULT 20,
    objective       TEXT NOT NULL,
    objectives_json TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'pending',
    budget_type     TEXT NOT NULL DEFAULT 'trials',
    budget_value    REAL NOT NULL DEFAULT 20,
    pruner_type     TEXT NOT NULL DEFAULT 'median',
    parallel_workers INTEGER NOT NULL DEFAULT 1,
    best_score      REAL,
    champion_trial_id TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    search_space_name TEXT
);

CREATE TABLE IF NOT EXISTS experiment_records (
    record_id           TEXT PRIMARY KEY,
    trial_id            TEXT NOT NULL,
    job_id              TEXT NOT NULL,
    job_name            TEXT,
    strategy            TEXT,
    backbone            TEXT,
    random_seed         INTEGER,
    dataset_version     TEXT,
    feature_version     TEXT,
    label_version       TEXT,
    git_commit          TEXT,
    model_version       TEXT,
    hardware            TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    primary_score       REAL,
    composite_research_score REAL,
    promotion_eligible  INTEGER NOT NULL DEFAULT 0,
    overall_rank        INTEGER,
    training_duration_seconds REAL,
    validation_duration_seconds REAL,
    pruned              INTEGER NOT NULL DEFAULT 0,
    pruning_step        INTEGER,
    pruning_reason      TEXT,
    error_message       TEXT,
    retry_count         INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    notes               TEXT,
    config_json         TEXT NOT NULL DEFAULT '{}',
    per_objective_scores_json TEXT NOT NULL DEFAULT '{}',
    validation_results_json   TEXT NOT NULL DEFAULT '{}',
    promotion_decision_json   TEXT NOT NULL DEFAULT '{}',
    per_instrument_ranks_json TEXT NOT NULL DEFAULT '{}',
    per_timeframe_ranks_json  TEXT NOT NULL DEFAULT '{}',
    per_regime_ranks_json     TEXT NOT NULL DEFAULT '{}',
    resource_usage_json       TEXT NOT NULL DEFAULT '{}',
    intermediate_scores_json  TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (job_id) REFERENCES search_jobs(job_id)
);

CREATE TABLE IF NOT EXISTS pruning_decisions (
    decision_id TEXT PRIMARY KEY,
    trial_id    TEXT NOT NULL,
    job_id      TEXT NOT NULL,
    step        INTEGER NOT NULL,
    score       REAL NOT NULL,
    threshold   REAL,
    prune       INTEGER NOT NULL DEFAULT 0,
    reason      TEXT,
    decided_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_decisions (
    decision_id     TEXT PRIMARY KEY,
    trial_id        TEXT NOT NULL,
    job_id          TEXT NOT NULL,
    model_version   TEXT,
    is_eligible     INTEGER NOT NULL DEFAULT 0,
    auto_approved   INTEGER NOT NULL DEFAULT 0,
    manually_approved INTEGER NOT NULL DEFAULT 0,
    champion_status INTEGER NOT NULL DEFAULT 0,
    rollback_eligible INTEGER NOT NULL DEFAULT 0,
    decided_at      TEXT NOT NULL,
    rules_json      TEXT NOT NULL DEFAULT '[]',
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_records_job ON experiment_records(job_id);
CREATE INDEX IF NOT EXISTS idx_records_status ON experiment_records(status);
CREATE INDEX IF NOT EXISTS idx_records_score ON experiment_records(primary_score DESC);
CREATE INDEX IF NOT EXISTS idx_pruning_trial ON pruning_decisions(trial_id);
CREATE INDEX IF NOT EXISTS idx_promotion_trial ON promotion_decisions(trial_id);
"""


# ---------------------------------------------------------------------------
# AutoML database
# ---------------------------------------------------------------------------


class AutoMLDatabase:
    """SQLite database for all AutoML platform data.

    Thread-safe with connection-per-thread semantics.

    Args:
        db_path: Path to the SQLite database file.
            Defaults to ``experiments/automl/automl.db``.
    """

    def __init__(self, db_path: str = "experiments/automl/automl.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"AutoMLDatabase initialized at {self.db_path}")

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Search jobs
    # -----------------------------------------------------------------------

    def save_job(self, job: SearchJob) -> str:
        """Insert or replace a search job. Returns job_id."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO search_jobs
            (job_id, name, strategy, n_trials, objective, objectives_json,
             status, budget_type, budget_value, pruner_type, parallel_workers,
             best_score, champion_trial_id, created_at, started_at,
             completed_at, metadata_json, search_space_name)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                job.job_id,
                job.name,
                job.strategy,
                job.n_trials,
                job.objective,
                json.dumps(job.objectives),
                job.status,
                job.budget_type,
                job.budget_value,
                job.pruner_type,
                job.parallel_workers,
                job.best_score,
                job.champion_trial_id,
                job.created_at,
                job.started_at or None,
                job.completed_at or None,
                json.dumps(job.metadata),
                job.search_space_name,
            ),
        )
        self._conn.commit()
        return job.job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a search job dict or None."""
        row = self._conn.execute("SELECT * FROM search_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return self._job_row_to_dict(row)

    def list_jobs(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List search jobs, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM search_jobs WHERE status = ? " "ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM search_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._job_row_to_dict(r) for r in rows]

    def update_job_status(
        self,
        job_id: str,
        status: str,
        best_score: float | None = None,
        champion_trial_id: str | None = None,
    ) -> None:
        """Update job status and optionally champion info."""
        self._conn.execute(
            """UPDATE search_jobs
               SET status = ?,
                   best_score = COALESCE(?, best_score),
                   champion_trial_id = COALESCE(?, champion_trial_id)
               WHERE job_id = ?""",
            (status, best_score, champion_trial_id, job_id),
        )
        self._conn.commit()

    def _job_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["objectives"] = json.loads(d.pop("objectives_json", "[]"))
        d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
        return d

    # -----------------------------------------------------------------------
    # Experiment records
    # -----------------------------------------------------------------------

    def save_record(self, record: ExperimentRecord) -> str:
        """Insert or replace an experiment record. Returns record_id."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO experiment_records
            (record_id, trial_id, job_id, job_name, strategy, backbone,
             random_seed, dataset_version, feature_version, label_version,
             git_commit, model_version, hardware, status, primary_score,
             composite_research_score, promotion_eligible, overall_rank,
             training_duration_seconds, validation_duration_seconds,
             pruned, pruning_step, pruning_reason, error_message,
             retry_count, created_at, notes, config_json,
             per_objective_scores_json, validation_results_json,
             promotion_decision_json, per_instrument_ranks_json,
             per_timeframe_ranks_json, per_regime_ranks_json,
             resource_usage_json, intermediate_scores_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.record_id,
                record.trial_id,
                record.job_id,
                record.job_name,
                record.strategy,
                record.backbone,
                record.random_seed,
                record.dataset_version,
                record.feature_version,
                record.label_version,
                record.git_commit,
                record.model_version,
                record.hardware,
                record.status,
                record.primary_score,
                record.composite_research_score,
                int(record.promotion_eligible),
                record.overall_rank,
                record.training_duration_seconds,
                record.validation_duration_seconds,
                int(record.pruned),
                record.pruning_step,
                record.pruning_reason or None,
                record.error_message or None,
                record.retry_count,
                record.created_at,
                record.notes or None,
                json.dumps(record.config),
                json.dumps(record.per_objective_scores),
                json.dumps(record.validation_results),
                json.dumps(record.promotion_decision),
                json.dumps(record.per_instrument_ranks),
                json.dumps(record.per_timeframe_ranks),
                json.dumps(record.per_regime_ranks),
                json.dumps(record.resource_usage),
                json.dumps(list(record.intermediate_scores)),
            ),
        )
        self._conn.commit()
        return record.record_id

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        """Return an experiment record dict or None."""
        row = self._conn.execute(
            "SELECT * FROM experiment_records WHERE record_id = ?", (record_id,)
        ).fetchone()
        if not row:
            return None
        return self._record_row_to_dict(row)

    def get_records_for_job(self, job_id: str, status: str | None = None) -> list[dict[str, Any]]:
        """Return all experiment records for a job."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM experiment_records WHERE job_id = ? AND status = ? "
                "ORDER BY primary_score DESC",
                (job_id, status),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM experiment_records WHERE job_id = ? "
                "ORDER BY primary_score DESC NULLS LAST",
                (job_id,),
            ).fetchall()
        return [self._record_row_to_dict(r) for r in rows]

    def list_records(
        self,
        job_id: str | None = None,
        status: str | None = None,
        backbone: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List experiment records with optional filters."""
        clauses = []
        params: list[Any] = []
        if job_id:
            clauses.append("job_id = ?")
            params.append(job_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if backbone:
            clauses.append("backbone = ?")
            params.append(backbone)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params += [limit, offset]
        rows = self._conn.execute(
            f"SELECT * FROM experiment_records {where} "
            f"ORDER BY primary_score DESC NULLS LAST LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [self._record_row_to_dict(r) for r in rows]

    def update_record_rank(self, record_id: str, rank: int) -> None:
        self._conn.execute(
            "UPDATE experiment_records SET overall_rank = ? WHERE record_id = ?", (rank, record_id)
        )
        self._conn.commit()

    def _record_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in (
            "config_json",
            "per_objective_scores_json",
            "validation_results_json",
            "promotion_decision_json",
            "per_instrument_ranks_json",
            "per_timeframe_ranks_json",
            "per_regime_ranks_json",
            "resource_usage_json",
        ):
            new_key = key.replace("_json", "")
            d[new_key] = json.loads(d.pop(key, "{}") or "{}")
        d["intermediate_scores"] = json.loads(d.pop("intermediate_scores_json", "[]") or "[]")
        d["promotion_eligible"] = bool(d.get("promotion_eligible", 0))
        d["pruned"] = bool(d.get("pruned", 0))
        return d

    # -----------------------------------------------------------------------
    # Pruning decisions
    # -----------------------------------------------------------------------

    def save_pruning_decision(
        self,
        trial_id: str,
        job_id: str,
        step: int,
        score: float,
        threshold: float,
        prune: bool,
        reason: str = "",
    ) -> str:
        decision_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO pruning_decisions
               (decision_id, trial_id, job_id, step, score, threshold,
                prune, reason, decided_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                decision_id,
                trial_id,
                job_id,
                step,
                score,
                threshold,
                int(prune),
                reason,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()
        return decision_id

    def get_pruning_history(self, trial_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM pruning_decisions WHERE trial_id = ? ORDER BY step", (trial_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Promotion decisions
    # -----------------------------------------------------------------------

    def save_promotion_decision(
        self,
        trial_id: str,
        job_id: str,
        model_version: str,
        is_eligible: bool,
        auto_approved: bool,
        manually_approved: bool,
        champion_status: bool,
        rollback_eligible: bool,
        rules: list[dict[str, Any]],
        notes: str = "",
    ) -> str:
        decision_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO promotion_decisions
               (decision_id, trial_id, job_id, model_version, is_eligible,
                auto_approved, manually_approved, champion_status,
                rollback_eligible, decided_at, rules_json, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                decision_id,
                trial_id,
                job_id,
                model_version,
                int(is_eligible),
                int(auto_approved),
                int(manually_approved),
                int(champion_status),
                int(rollback_eligible),
                datetime.now(UTC).isoformat(),
                json.dumps(rules),
                notes or None,
            ),
        )
        self._conn.commit()
        return decision_id

    def get_promotion_history(
        self, job_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        if job_id:
            rows = self._conn.execute(
                "SELECT * FROM promotion_decisions WHERE job_id = ? "
                "ORDER BY decided_at DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM promotion_decisions ORDER BY decided_at DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["rules"] = json.loads(d.pop("rules_json", "[]") or "[]")
            for field in (
                "is_eligible",
                "auto_approved",
                "manually_approved",
                "champion_status",
                "rollback_eligible",
            ):
                d[field] = bool(d.get(field, 0))
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Analytics queries
    # -----------------------------------------------------------------------

    def best_record(self, job_id: str | None = None) -> dict[str, Any] | None:
        """Return the record with the highest primary_score."""
        if job_id:
            row = self._conn.execute(
                "SELECT * FROM experiment_records WHERE job_id = ? "
                "ORDER BY primary_score DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM experiment_records " "ORDER BY primary_score DESC LIMIT 1"
            ).fetchone()
        return self._record_row_to_dict(row) if row else None

    def score_history(self, job_id: str) -> list[dict[str, Any]]:
        """Return (created_at, primary_score) pairs for a job."""
        rows = self._conn.execute(
            "SELECT created_at, primary_score FROM experiment_records "
            "WHERE job_id = ? AND status = 'completed' ORDER BY created_at",
            (job_id,),
        ).fetchall()
        return [{"created_at": r["created_at"], "score": r["primary_score"]} for r in rows]

    def backbone_summary(self, job_id: str | None = None) -> list[dict[str, Any]]:
        """Return per-backbone avg/max scores."""
        if job_id:
            rows = self._conn.execute(
                "SELECT backbone, COUNT(*) as n, AVG(primary_score) as avg_score, "
                "MAX(primary_score) as max_score "
                "FROM experiment_records WHERE job_id = ? AND status = 'completed' "
                "GROUP BY backbone ORDER BY avg_score DESC",
                (job_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT backbone, COUNT(*) as n, AVG(primary_score) as avg_score, "
                "MAX(primary_score) as max_score "
                "FROM experiment_records WHERE status = 'completed' "
                "GROUP BY backbone ORDER BY avg_score DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Return overall database statistics."""
        jobs = self._conn.execute("SELECT COUNT(*) as n FROM search_jobs").fetchone()["n"]
        records = self._conn.execute("SELECT COUNT(*) as n FROM experiment_records").fetchone()["n"]
        completed = self._conn.execute(
            "SELECT COUNT(*) as n FROM experiment_records WHERE status = 'completed'"
        ).fetchone()["n"]
        pruned = self._conn.execute(
            "SELECT COUNT(*) as n FROM experiment_records WHERE pruned = 1"
        ).fetchone()["n"]
        promotions = self._conn.execute(
            "SELECT COUNT(*) as n FROM promotion_decisions WHERE is_eligible = 1"
        ).fetchone()["n"]
        return {
            "total_jobs": jobs,
            "total_records": records,
            "completed_records": completed,
            "pruned_records": pruned,
            "eligible_promotions": promotions,
        }

    def close(self) -> None:
        """Close the database connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
