"""Phase 15 — Validation Database.

SQLite-backed storage for all validation run results.

Stores every validation run with:
  - Configuration (JSON)
  - Metrics (JSON)
  - Charts (JSON — chart data dicts, not binary images)
  - Calibration (JSON)
  - Drift (JSON)
  - Leaderboard results (JSON)
  - Promotion decisions (JSON)
  - Model metadata (JSON)
  - Git commit
  - Experiment ID
  - Dataset / Feature / Label versions

Uses Python's built-in ``sqlite3`` — no extra dependencies.

Usage::

    db = ValidationDatabase("experiments/validation.db")
    run_id = db.save_run(run_data)
    run = db.get_run(run_id)
    runs = db.list_runs(limit=50)
    db.close()
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Validation run record
# ---------------------------------------------------------------------------


@dataclass
class ValidationRunRecord:
    """A complete validation run record stored in the database.

    Attributes:
        run_id: UUID of this validation run.
        model_id: Model that was validated.
        model_version: Semantic version of the model.
        run_timestamp: UTC ISO-8601 timestamp.
        configuration: Full validation configuration dict.
        statistical_metrics: Statistical metrics dict.
        trading_metrics: Trading metrics dict.
        calibration: Calibration report dict.
        drift: Drift report dict.
        monte_carlo: Monte Carlo results dict.
        robustness: Robustness report dict.
        leaderboard_score: Overall leaderboard score.
        promotion_decision: Promotion decision dict.
        model_metadata: Model metadata dict.
        git_commit: Git commit hash.
        experiment_id: Experiment UUID.
        dataset_version: Dataset version string.
        feature_version: Feature version string.
        label_version: Label version string.
        visualization_data: Chart data dicts.
        notes: Free-text notes.
        status: "completed" | "failed" | "partial".
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    model_version: str = ""
    run_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    configuration: dict[str, Any] = field(default_factory=dict)
    statistical_metrics: dict[str, Any] = field(default_factory=dict)
    trading_metrics: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    monte_carlo: dict[str, Any] = field(default_factory=dict)
    robustness: dict[str, Any] = field(default_factory=dict)
    leaderboard_score: float = 0.0
    promotion_decision: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    experiment_id: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    visualization_data: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    status: str = "completed"


# ---------------------------------------------------------------------------
# Validation Database
# ---------------------------------------------------------------------------


class ValidationDatabase:
    """SQLite-backed validation run store.

    Thread-safe via a per-instance lock.

    Args:
        db_path: Path to the SQLite database file.
            Created if it doesn't exist.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS validation_runs (
        run_id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        model_version TEXT,
        run_timestamp TEXT,
        configuration TEXT,
        statistical_metrics TEXT,
        trading_metrics TEXT,
        calibration TEXT,
        drift TEXT,
        monte_carlo TEXT,
        robustness TEXT,
        leaderboard_score REAL,
        promotion_decision TEXT,
        model_metadata TEXT,
        git_commit TEXT,
        experiment_id TEXT,
        dataset_version TEXT,
        feature_version TEXT,
        label_version TEXT,
        visualization_data TEXT,
        notes TEXT,
        status TEXT DEFAULT 'completed'
    );

    CREATE INDEX IF NOT EXISTS idx_model_id ON validation_runs(model_id);
    CREATE INDEX IF NOT EXISTS idx_run_timestamp ON validation_runs(run_timestamp);
    CREATE INDEX IF NOT EXISTS idx_status ON validation_runs(status);
    CREATE INDEX IF NOT EXISTS idx_experiment_id ON validation_runs(experiment_id);
    """

    def __init__(self, db_path: str | Path = "validation.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info(f"ValidationDatabase opened: {self._path}")

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()

    # ── Write operations ──────────────────────────────────────────────────

    def save_run(self, record: ValidationRunRecord) -> str:
        """Save a complete validation run.

        Args:
            record: The run record to persist.

        Returns:
            The run_id of the saved record.
        """
        row = (
            record.run_id,
            record.model_id,
            record.model_version,
            record.run_timestamp,
            json.dumps(record.configuration, default=str),
            json.dumps(record.statistical_metrics, default=str),
            json.dumps(record.trading_metrics, default=str),
            json.dumps(record.calibration, default=str),
            json.dumps(record.drift, default=str),
            json.dumps(record.monte_carlo, default=str),
            json.dumps(record.robustness, default=str),
            record.leaderboard_score,
            json.dumps(record.promotion_decision, default=str),
            json.dumps(record.model_metadata, default=str),
            record.git_commit,
            record.experiment_id,
            record.dataset_version,
            record.feature_version,
            record.label_version,
            json.dumps(record.visualization_data, default=str),
            record.notes,
            record.status,
        )
        sql = """
            INSERT OR REPLACE INTO validation_runs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """
        with self._lock:
            self._conn.execute(sql, row)
            self._conn.commit()
        logger.debug(f"ValidationDatabase: saved run {record.run_id}")
        return record.run_id

    def update_status(self, run_id: str, status: str) -> None:
        """Update the status of an existing run."""
        with self._lock:
            self._conn.execute(
                "UPDATE validation_runs SET status=? WHERE run_id=?",
                (status, run_id),
            )
            self._conn.commit()

    def delete_run(self, run_id: str) -> bool:
        """Delete a run record."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM validation_runs WHERE run_id=?", (run_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    # ── Read operations ───────────────────────────────────────────────────

    def get_run(self, run_id: str) -> Optional[ValidationRunRecord]:
        """Retrieve a single run by ID."""
        cursor = self._conn.execute("SELECT * FROM validation_runs WHERE run_id=?", (run_id,))
        row = cursor.fetchone()
        return self._row_to_record(dict(row)) if row else None

    def list_runs(
        self,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        experiment_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "run_timestamp",
        ascending: bool = False,
    ) -> list[ValidationRunRecord]:
        """List runs with optional filtering and pagination.

        Args:
            model_id: Filter by model.
            status: Filter by status ("completed", "failed", "partial").
            experiment_id: Filter by experiment.
            limit: Max rows to return.
            offset: Row offset for pagination.
            order_by: Column to sort by.
            ascending: Sort direction.

        Returns:
            List of ``ValidationRunRecord``.
        """
        conditions = []
        params: list[Any] = []
        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        direction = "ASC" if ascending else "DESC"
        # Whitelist order_by to prevent injection
        safe_columns = {
            "run_timestamp",
            "model_id",
            "leaderboard_score",
            "status",
        }
        col = order_by if order_by in safe_columns else "run_timestamp"
        sql = (
            f"SELECT * FROM validation_runs{where} " f"ORDER BY {col} {direction} LIMIT ? OFFSET ?"
        )
        params += [limit, offset]
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def count_runs(self, model_id: Optional[str] = None) -> int:
        """Count total runs, optionally filtered by model."""
        if model_id:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM validation_runs WHERE model_id=?", (model_id,)
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM validation_runs")
        return int(cursor.fetchone()[0])

    def get_best_run(self, model_id: str) -> Optional[ValidationRunRecord]:
        """Return the run with the highest leaderboard score for a model."""
        cursor = self._conn.execute(
            "SELECT * FROM validation_runs WHERE model_id=? "
            "ORDER BY leaderboard_score DESC LIMIT 1",
            (model_id,),
        )
        row = cursor.fetchone()
        return self._row_to_record(dict(row)) if row else None

    def get_latest_run(self, model_id: str) -> Optional[ValidationRunRecord]:
        """Return the most recent run for a model."""
        cursor = self._conn.execute(
            "SELECT * FROM validation_runs WHERE model_id=? " "ORDER BY run_timestamp DESC LIMIT 1",
            (model_id,),
        )
        row = cursor.fetchone()
        return self._row_to_record(dict(row)) if row else None

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug("ValidationDatabase: connection closed.")

    # ── Private ────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> ValidationRunRecord:
        """Convert a SQLite row dict to a ValidationRunRecord."""

        def _parse(key: str, default: Any = None) -> Any:
            val = row.get(key)
            if val is None:
                return default if default is not None else {}
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return val
            return val

        return ValidationRunRecord(
            run_id=row.get("run_id", ""),
            model_id=row.get("model_id", ""),
            model_version=row.get("model_version", ""),
            run_timestamp=row.get("run_timestamp", ""),
            configuration=_parse("configuration"),
            statistical_metrics=_parse("statistical_metrics"),
            trading_metrics=_parse("trading_metrics"),
            calibration=_parse("calibration"),
            drift=_parse("drift"),
            monte_carlo=_parse("monte_carlo"),
            robustness=_parse("robustness"),
            leaderboard_score=float(row.get("leaderboard_score", 0.0)),
            promotion_decision=_parse("promotion_decision"),
            model_metadata=_parse("model_metadata"),
            git_commit=row.get("git_commit", ""),
            experiment_id=row.get("experiment_id", ""),
            dataset_version=row.get("dataset_version", ""),
            feature_version=row.get("feature_version", ""),
            label_version=row.get("label_version", ""),
            visualization_data=_parse("visualization_data"),
            notes=row.get("notes", ""),
            status=row.get("status", "completed"),
        )
