"""Phase 19.3 — HRSA Persistence Layer.

SQLite-backed storage for all hierarchical scoring data.

Tables:
  hrsa_raw_metrics       — Level 1 metric values per experiment
  hrsa_category_scores   — Level 2 category scores per experiment
  hrsa_score_history     — Level 3 composite scores (time series)
  hrsa_normalization     — Normalisation parameter snapshots
  hrsa_benchmarks        — Registered benchmark records
  hrsa_grades            — Grade history
  hrsa_explanations      — Score explanations (JSON blobs)
  hrsa_whatif            — What-if simulation results

All tables use WAL journal mode for concurrent reads.

Usage::

    db = HRSADatabase(":memory:")  # or path/to/hrsa.db
    db.save(hierarchical_score)

    retrieved = db.get_score(score_id)
    history   = db.composite_history(experiment_id)
    leaderboard = db.leaderboard(top_n=20)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# HRSADatabase
# ---------------------------------------------------------------------------


class HRSADatabase:
    """SQLite persistence for the HRSA scoring pipeline."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS hrsa_raw_metrics (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        score_id        TEXT    NOT NULL,
        experiment_id   TEXT    NOT NULL,
        metric_name     TEXT    NOT NULL,
        raw_value       REAL    NOT NULL,
        recorded_at     TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS hrsa_category_scores (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        score_id        TEXT    NOT NULL,
        experiment_id   TEXT    NOT NULL,
        category        TEXT    NOT NULL,
        display_name    TEXT    NOT NULL,
        score           REAL    NOT NULL,
        confidence      REAL    NOT NULL,
        weight          REAL    NOT NULL,
        grade           TEXT    NOT NULL,
        metric_count    INTEGER NOT NULL,
        coverage        REAL    NOT NULL,
        metric_scores   TEXT    NOT NULL,   -- JSON
        raw_values      TEXT    NOT NULL,   -- JSON
        recorded_at     TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS hrsa_score_history (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        score_id                TEXT    NOT NULL UNIQUE,
        experiment_id           TEXT    NOT NULL,
        name                    TEXT    NOT NULL,
        composite               REAL    NOT NULL,
        confidence              REAL    NOT NULL,
        institutional_grade     TEXT    NOT NULL,
        research_grade          TEXT    NOT NULL,
        promotion_recommendation TEXT   NOT NULL,
        risk_rating             TEXT    NOT NULL,
        future_potential        TEXT    NOT NULL,
        instrument              TEXT    NOT NULL DEFAULT '',
        timeframe               TEXT    NOT NULL DEFAULT '',
        backbone                TEXT    NOT NULL DEFAULT '',
        model_family            TEXT    NOT NULL DEFAULT '',
        level1_metric_count     INTEGER NOT NULL DEFAULT 0,
        total_weight_used       REAL    NOT NULL DEFAULT 0.0,
        scored_at               TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS hrsa_normalization (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        score_id        TEXT    NOT NULL,
        category        TEXT    NOT NULL,
        metric_name     TEXT    NOT NULL,
        raw_value       REAL    NOT NULL,
        normalised      REAL    NOT NULL,
        method          TEXT    NOT NULL,
        direction       TEXT    NOT NULL,
        recorded_at     TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS hrsa_benchmarks (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id   TEXT    NOT NULL,
        name            TEXT    NOT NULL,
        role            TEXT    NOT NULL,
        composite       REAL    NOT NULL,
        confidence      REAL    NOT NULL,
        institutional_grade TEXT NOT NULL,
        category_scores TEXT    NOT NULL,   -- JSON
        generation      TEXT    NOT NULL DEFAULT '',
        registered_at   TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS hrsa_grades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        score_id        TEXT    NOT NULL,
        experiment_id   TEXT    NOT NULL,
        composite       REAL    NOT NULL,
        institutional_grade     TEXT NOT NULL,
        research_grade          TEXT NOT NULL,
        promotion_recommendation TEXT NOT NULL,
        risk_rating     TEXT    NOT NULL,
        future_potential TEXT   NOT NULL,
        graded_at       TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS hrsa_explanations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        score_id        TEXT    NOT NULL,
        experiment_id   TEXT    NOT NULL,
        explanation_json TEXT   NOT NULL,
        generated_at    TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS hrsa_whatif (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        simulation_id   TEXT    NOT NULL,
        experiment_id   TEXT    NOT NULL,
        scenario_name   TEXT    NOT NULL,
        original_composite REAL NOT NULL,
        simulated_composite REAL NOT NULL,
        delta           REAL    NOT NULL,
        simulation_json TEXT    NOT NULL,
        simulated_at    TEXT    NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_hrsa_history_exp
        ON hrsa_score_history(experiment_id);
    CREATE INDEX IF NOT EXISTS idx_hrsa_cat_exp
        ON hrsa_category_scores(experiment_id);
    CREATE INDEX IF NOT EXISTS idx_hrsa_raw_exp
        ON hrsa_raw_metrics(experiment_id);
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._path = db_path
        self._is_memory = db_path == ":memory:"
        # For :memory: databases, keep a single persistent connection
        # to avoid losing the schema between calls.
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
            # Yield the persistent connection; do NOT close it
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

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, hs: Any) -> None:
        """Persist a complete HierarchicalScore (all tables)."""
        ts = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            # hrsa_score_history
            ga = hs.grade_assessment
            conn.execute(
                """
                INSERT OR REPLACE INTO hrsa_score_history
                (score_id, experiment_id, name, composite, confidence,
                 institutional_grade, research_grade, promotion_recommendation,
                 risk_rating, future_potential, instrument, timeframe,
                 backbone, model_family, level1_metric_count, total_weight_used,
                 scored_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    hs.score_id,
                    hs.experiment_id,
                    hs.name,
                    hs.level3_composite,
                    hs.confidence,
                    hs.institutional_grade,
                    hs.research_grade,
                    hs.promotion_recommendation,
                    hs.risk_rating,
                    hs.future_potential,
                    hs.instrument,
                    hs.timeframe,
                    hs.backbone,
                    hs.model_family,
                    hs.level1_metric_count,
                    hs.total_weight_used,
                    hs.scored_at,
                ),
            )

            # hrsa_category_scores
            for cat, result in hs.level2_categories.items():
                conn.execute(
                    """
                    INSERT INTO hrsa_category_scores
                    (score_id, experiment_id, category, display_name,
                     score, confidence, weight, grade, metric_count, coverage,
                     metric_scores, raw_values, recorded_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        hs.score_id,
                        hs.experiment_id,
                        cat,
                        result.display_name,
                        result.score,
                        result.confidence,
                        result.weight,
                        result.grade,
                        result.metric_count,
                        result.coverage,
                        json.dumps(result.metric_scores),
                        json.dumps(result.raw_values),
                        ts,
                    ),
                )

            # hrsa_raw_metrics
            for cat, result in hs.level2_categories.items():
                for metric, raw_val in result.raw_values.items():
                    conn.execute(
                        """
                        INSERT INTO hrsa_raw_metrics
                        (score_id, experiment_id, metric_name, raw_value, recorded_at)
                        VALUES (?,?,?,?,?)
                        """,
                        (hs.score_id, hs.experiment_id, metric, raw_val, ts),
                    )

            # hrsa_grades
            if ga:
                conn.execute(
                    """
                    INSERT INTO hrsa_grades
                    (score_id, experiment_id, composite, institutional_grade,
                     research_grade, promotion_recommendation, risk_rating,
                     future_potential, graded_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        hs.score_id,
                        hs.experiment_id,
                        hs.level3_composite,
                        ga.institutional_grade,
                        ga.research_grade,
                        ga.promotion_recommendation,
                        ga.risk_rating,
                        ga.future_potential,
                        ts,
                    ),
                )

    def save_explanation(self, explanation: Any) -> None:
        ts = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO hrsa_explanations
                (score_id, experiment_id, explanation_json, generated_at)
                VALUES (?,?,?,?)
                """,
                (
                    explanation.score_id,
                    explanation.experiment_id,
                    json.dumps(explanation.to_dict()),
                    ts,
                ),
            )

    def save_whatif(self, experiment_id: str, sim: Any) -> None:
        ts = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO hrsa_whatif
                (simulation_id, experiment_id, scenario_name,
                 original_composite, simulated_composite, delta,
                 simulation_json, simulated_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    sim.simulation_id,
                    experiment_id,
                    sim.scenario_name,
                    sim.original_composite,
                    sim.simulated_composite,
                    sim.delta,
                    json.dumps(sim.to_dict()),
                    ts,
                ),
            )

    def save_benchmark(self, record: Any) -> None:
        ts = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO hrsa_benchmarks
                (experiment_id, name, role, composite, confidence,
                 institutional_grade, category_scores, generation, registered_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.experiment_id,
                    record.name,
                    record.role,
                    record.composite,
                    record.confidence,
                    record.institutional_grade,
                    json.dumps(record.category_scores),
                    record.generation,
                    ts,
                ),
            )

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_score(self, score_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM hrsa_score_history WHERE score_id = ?", (score_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def get_latest(self, experiment_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM hrsa_score_history
                WHERE experiment_id = ?
                ORDER BY scored_at DESC LIMIT 1
                """,
                (experiment_id,),
            ).fetchone()
            return dict(row) if row else None

    def composite_history(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT score_id, composite, confidence, institutional_grade, scored_at
                FROM hrsa_score_history WHERE experiment_id = ?
                ORDER BY scored_at ASC
                """,
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def category_history(self, experiment_id: str, category: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT cs.score, cs.confidence, cs.grade, hs.scored_at
                FROM hrsa_category_scores cs
                JOIN hrsa_score_history hs ON cs.score_id = hs.score_id
                WHERE cs.experiment_id = ? AND cs.category = ?
                ORDER BY hs.scored_at ASC
                """,
                (experiment_id, category),
            ).fetchall()
            return [dict(r) for r in rows]

    def metric_history(self, experiment_id: str, metric_name: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT rm.raw_value, rm.recorded_at
                FROM hrsa_raw_metrics rm
                WHERE rm.experiment_id = ? AND rm.metric_name = ?
                ORDER BY rm.recorded_at ASC
                """,
                (experiment_id, metric_name),
            ).fetchall()
            return [dict(r) for r in rows]

    def leaderboard(
        self,
        top_n: int = 20,
        instrument: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        filters = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if instrument:
            filters.append("instrument = ?")
            params.append(instrument)
        where = " AND ".join(filters)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT score_id, experiment_id, name, composite, confidence,
                       institutional_grade, research_grade, promotion_recommendation,
                       instrument, timeframe, backbone, scored_at
                FROM hrsa_score_history
                WHERE {where}
                ORDER BY composite DESC
                LIMIT ?
                """,
                [*params, top_n],
            ).fetchall()
            return [dict(r) for r in rows]

    def category_scores_for(self, score_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT category, display_name, score, confidence,
                       grade, metric_count, metric_scores
                FROM hrsa_category_scores WHERE score_id = ?
                ORDER BY score DESC
                """,
                (score_id,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["metric_scores"] = json.loads(d["metric_scores"])
                result.append(d)
            return result

    def whatif_history(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT scenario_name, original_composite, simulated_composite,
                       delta, simulated_at
                FROM hrsa_whatif WHERE experiment_id = ?
                ORDER BY simulated_at DESC
                """,
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_explanation(self, score_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT explanation_json FROM hrsa_explanations
                WHERE score_id = ? ORDER BY generated_at DESC LIMIT 1
                """,
                (score_id,),
            ).fetchone()
            if row:
                return json.loads(row["explanation_json"])
            return None

    def benchmarks_by_role(self, role: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT experiment_id, name, role, composite, confidence,
                       institutional_grade, generation, registered_at
                FROM hrsa_benchmarks WHERE role = ?
                ORDER BY composite DESC
                """,
                (role,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as n FROM hrsa_score_history").fetchone()["n"]
            avg_comp = conn.execute(
                "SELECT AVG(composite) as a FROM hrsa_score_history"
            ).fetchone()["a"]
            max_comp = conn.execute(
                "SELECT MAX(composite) as m FROM hrsa_score_history"
            ).fetchone()["m"]
            simulations = conn.execute("SELECT COUNT(*) as n FROM hrsa_whatif").fetchone()["n"]
            explanations = conn.execute("SELECT COUNT(*) as n FROM hrsa_explanations").fetchone()[
                "n"
            ]
            benchmarks = conn.execute("SELECT COUNT(*) as n FROM hrsa_benchmarks").fetchone()["n"]
        return {
            "total_scores": total,
            "average_composite": round(avg_comp, 2) if avg_comp else None,
            "max_composite": round(max_comp, 2) if max_comp else None,
            "total_simulations": simulations,
            "total_explanations": explanations,
            "total_benchmarks": benchmarks,
        }
