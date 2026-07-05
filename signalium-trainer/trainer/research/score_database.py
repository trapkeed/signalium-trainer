"""Phase 19.2 — IRSE Score Database.

SQLite persistence for all Research Score data.

Tables:
  research_scores     — composite scores + grade + metadata
  category_scores     — per-category scores for each research_score
  score_history       — time-series snapshots for trend tracking
  weight_configurations — saved weight presets with names
  benchmark_records   — registered benchmark model records
  benchmark_comparisons — results of benchmark compare() calls

Design:
  - WAL mode for concurrent reads
  - Thread-safe with threading.Lock
  - All dates stored as ISO 8601 UTC strings
  - No external dependencies — stdlib sqlite3 only

Usage::

    db = IRSEScoreDatabase(":memory:")
    db.save_research_score(rs.to_dict())
    results = db.query_leaderboard(limit=20, order_by="composite")
    stats = db.stats()
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# IRSEScoreDatabase
# ---------------------------------------------------------------------------


class IRSEScoreDatabase:
    """Thread-safe SQLite database for IRSE scoring data."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._path = db_path
        self._lock = threading.Lock()
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._create_tables()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _configure(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS research_scores (
                    score_id              TEXT PRIMARY KEY,
                    experiment_id         TEXT NOT NULL,
                    model_id              TEXT,
                    name                  TEXT,
                    scored_at             TEXT NOT NULL,
                    composite             REAL NOT NULL,
                    grade_label           TEXT,
                    grade_description     TEXT,
                    promotion_eligible    INTEGER DEFAULT 0,
                    deployment_eligible   INTEGER DEFAULT 0,
                    confidence_score      REAL DEFAULT 0.0,
                    confidence_level      TEXT DEFAULT '',
                    instrument            TEXT DEFAULT '',
                    timeframe             TEXT DEFAULT '',
                    backbone              TEXT DEFAULT '',
                    architecture          TEXT DEFAULT '',
                    dataset_version       TEXT DEFAULT '',
                    feature_version       TEXT DEFAULT '',
                    model_family          TEXT DEFAULT '',
                    tags                  TEXT DEFAULT '[]',
                    weights_used          TEXT DEFAULT '{}',
                    created_at            TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS category_scores (
                    id                    TEXT PRIMARY KEY,
                    score_id              TEXT NOT NULL,
                    category              TEXT NOT NULL,
                    score                 REAL NOT NULL,
                    weight                REAL NOT NULL,
                    weighted_contribution REAL NOT NULL,
                    sub_scores            TEXT DEFAULT '{}',
                    FOREIGN KEY (score_id) REFERENCES research_scores(score_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS score_history (
                    snapshot_id       TEXT PRIMARY KEY,
                    experiment_id     TEXT NOT NULL,
                    model_id          TEXT DEFAULT '',
                    name              TEXT DEFAULT '',
                    composite         REAL NOT NULL,
                    grade_label       TEXT DEFAULT '',
                    recorded_at       TEXT NOT NULL,
                    category_scores   TEXT DEFAULT '{}',
                    model_family      TEXT DEFAULT '',
                    backbone          TEXT DEFAULT '',
                    is_champion       INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS weight_configurations (
                    config_id         TEXT PRIMARY KEY,
                    name              TEXT NOT NULL UNIQUE,
                    description       TEXT DEFAULT '',
                    weights           TEXT NOT NULL,
                    created_at        TEXT DEFAULT (datetime('now')),
                    is_active         INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS benchmark_records (
                    experiment_id     TEXT PRIMARY KEY,
                    model_id          TEXT DEFAULT '',
                    name              TEXT DEFAULT '',
                    role              TEXT NOT NULL,
                    composite         REAL NOT NULL,
                    grade_label       TEXT DEFAULT '',
                    promotion_eligible INTEGER DEFAULT 0,
                    category_scores   TEXT DEFAULT '{}',
                    registered_at     TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS benchmark_comparisons (
                    comparison_id         TEXT PRIMARY KEY,
                    subject_id            TEXT NOT NULL,
                    subject_name          TEXT DEFAULT '',
                    reference_id          TEXT NOT NULL,
                    reference_name        TEXT DEFAULT '',
                    reference_role        TEXT DEFAULT '',
                    subject_composite     REAL NOT NULL,
                    reference_composite   REAL NOT NULL,
                    composite_delta       REAL NOT NULL,
                    composite_improved    INTEGER DEFAULT 0,
                    category_deltas       TEXT DEFAULT '{}',
                    stronger_categories   TEXT DEFAULT '[]',
                    weaker_categories     TEXT DEFAULT '[]',
                    compared_at           TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_rs_experiment_id
                    ON research_scores(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_rs_composite
                    ON research_scores(composite);
                CREATE INDEX IF NOT EXISTS idx_rs_grade
                    ON research_scores(grade_label);
                CREATE INDEX IF NOT EXISTS idx_cat_score_id
                    ON category_scores(score_id);
                CREATE INDEX IF NOT EXISTS idx_history_experiment
                    ON score_history(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_history_recorded
                    ON score_history(recorded_at);
            """)

    # ------------------------------------------------------------------
    # Research scores
    # ------------------------------------------------------------------

    def save_research_score(self, rs_dict: Dict[str, Any]) -> None:
        """Upsert a ResearchScore (from rs.to_dict())."""
        with self._lock:
            score_id = rs_dict["score_id"]
            self._conn.execute(
                """INSERT OR REPLACE INTO research_scores (
                    score_id, experiment_id, model_id, name, scored_at,
                    composite, grade_label, grade_description,
                    promotion_eligible, deployment_eligible,
                    confidence_score, confidence_level,
                    instrument, timeframe, backbone, architecture,
                    dataset_version, feature_version, model_family,
                    tags, weights_used
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    score_id,
                    rs_dict.get("experiment_id", ""),
                    rs_dict.get("model_id", ""),
                    rs_dict.get("name", ""),
                    rs_dict.get("scored_at", datetime.now(timezone.utc).isoformat()),
                    rs_dict.get("composite", 0.0),
                    rs_dict.get("grade_label", ""),
                    rs_dict.get("grade_description", ""),
                    int(rs_dict.get("promotion_eligible", False)),
                    int(rs_dict.get("deployment_eligible", False)),
                    rs_dict.get("confidence_score", 0.0),
                    rs_dict.get("confidence_level", ""),
                    rs_dict.get("instrument", ""),
                    rs_dict.get("timeframe", ""),
                    rs_dict.get("backbone", ""),
                    rs_dict.get("architecture", ""),
                    rs_dict.get("dataset_version", ""),
                    rs_dict.get("feature_version", ""),
                    rs_dict.get("model_family", ""),
                    json.dumps(rs_dict.get("tags", [])),
                    json.dumps(rs_dict.get("weights_used", {})),
                ),
            )

            # Upsert category scores
            self._conn.execute(
                "DELETE FROM category_scores WHERE score_id=?", (score_id,)
            )
            for cs in rs_dict.get("category_scores", []):
                self._conn.execute(
                    """INSERT INTO category_scores (
                        id, score_id, category, score, weight,
                        weighted_contribution, sub_scores
                    ) VALUES (?,?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()),
                        score_id,
                        cs["category"],
                        cs["score"],
                        cs["weight"],
                        cs["weighted_contribution"],
                        json.dumps(cs.get("sub_scores", {})),
                    ),
                )

    def get_research_score(self, score_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_scores WHERE score_id=?", (score_id,)
            ).fetchone()
            if not row:
                return None
            return self._hydrate_score(dict(row))

    def get_latest_for_experiment(
        self, experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM research_scores
                   WHERE experiment_id=?
                   ORDER BY scored_at DESC LIMIT 1""",
                (experiment_id,),
            ).fetchone()
            if not row:
                return None
            return self._hydrate_score(dict(row))

    def _hydrate_score(self, row: Dict[str, Any]) -> Dict[str, Any]:
        score_id = row["score_id"]
        cat_rows = self._conn.execute(
            "SELECT * FROM category_scores WHERE score_id=? ORDER BY category",
            (score_id,),
        ).fetchall()
        row["category_scores"] = [
            {
                "category": cr["category"],
                "score": cr["score"],
                "weight": cr["weight"],
                "weighted_contribution": cr["weighted_contribution"],
                "sub_scores": json.loads(cr["sub_scores"] or "{}"),
            }
            for cr in cat_rows
        ]
        row["tags"] = json.loads(row.get("tags") or "[]")
        row["weights_used"] = json.loads(row.get("weights_used") or "{}")
        return row

    # ------------------------------------------------------------------
    # Leaderboard queries
    # ------------------------------------------------------------------

    def query_leaderboard(
        self,
        limit: int = 50,
        order_by: str = "composite",
        descending: bool = True,
        instrument: Optional[str] = None,
        timeframe: Optional[str] = None,
        backbone: Optional[str] = None,
        architecture: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_version: Optional[str] = None,
        model_family: Optional[str] = None,
        grade_label: Optional[str] = None,
        promotion_eligible: Optional[bool] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        safe_cols = {
            "composite", "confidence_score", "scored_at", "grade_label", "name"
        }
        col = order_by if order_by in safe_cols else "composite"
        direction = "DESC" if descending else "ASC"

        conditions: List[str] = []
        params: List[Any] = []
        if instrument:
            conditions.append("instrument=?")
            params.append(instrument)
        if timeframe:
            conditions.append("timeframe=?")
            params.append(timeframe)
        if backbone:
            conditions.append("backbone=?")
            params.append(backbone)
        if architecture:
            conditions.append("architecture=?")
            params.append(architecture)
        if dataset_version:
            conditions.append("dataset_version=?")
            params.append(dataset_version)
        if feature_version:
            conditions.append("feature_version=?")
            params.append(feature_version)
        if model_family:
            conditions.append("model_family=?")
            params.append(model_family)
        if grade_label:
            conditions.append("grade_label=?")
            params.append(grade_label)
        if promotion_eligible is not None:
            conditions.append("promotion_eligible=?")
            params.append(int(promotion_eligible))
        if min_score is not None:
            conditions.append("composite>=?")
            params.append(min_score)
        if max_score is not None:
            conditions.append("composite<=?")
            params.append(max_score)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM research_scores {where} "
                f"ORDER BY {col} {direction} LIMIT ?",
                params,
            ).fetchall()
            return [self._hydrate_score(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Score history
    # ------------------------------------------------------------------

    def save_score_snapshot(self, snap: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO score_history (
                    snapshot_id, experiment_id, model_id, name, composite,
                    grade_label, recorded_at, category_scores,
                    model_family, backbone, is_champion
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    snap.get("snapshot_id", str(uuid.uuid4())),
                    snap.get("experiment_id", ""),
                    snap.get("model_id", ""),
                    snap.get("name", ""),
                    snap.get("composite", 0.0),
                    snap.get("grade_label", ""),
                    snap.get(
                        "recorded_at", datetime.now(timezone.utc).isoformat()
                    ),
                    json.dumps(snap.get("category_scores", {})),
                    snap.get("model_family", ""),
                    snap.get("backbone", ""),
                    int(snap.get("is_champion", False)),
                ),
            )

    def get_score_history(
        self, experiment_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if experiment_id:
                rows = self._conn.execute(
                    """SELECT * FROM score_history WHERE experiment_id=?
                       ORDER BY recorded_at ASC LIMIT ?""",
                    (experiment_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM score_history
                       ORDER BY recorded_at ASC LIMIT ?""",
                    (limit,),
                ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["category_scores"] = json.loads(d.get("category_scores") or "{}")
                d["is_champion"] = bool(d.get("is_champion", 0))
                result.append(d)
            return result

    # ------------------------------------------------------------------
    # Weight configurations
    # ------------------------------------------------------------------

    def save_weight_configuration(
        self,
        name: str,
        weights: Dict[str, float],
        description: str = "",
        set_active: bool = False,
    ) -> str:
        config_id = str(uuid.uuid4())
        with self._lock:
            if set_active:
                self._conn.execute(
                    "UPDATE weight_configurations SET is_active=0"
                )
            self._conn.execute(
                """INSERT OR REPLACE INTO weight_configurations (
                    config_id, name, description, weights, is_active
                ) VALUES (?,?,?,?,?)""",
                (
                    config_id,
                    name,
                    description,
                    json.dumps(weights),
                    int(set_active),
                ),
            )
        return config_id

    def get_active_weights(self) -> Optional[Dict[str, float]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT weights FROM weight_configurations WHERE is_active=1 LIMIT 1"
            ).fetchone()
            if row:
                return json.loads(row["weights"])
            return None

    def list_weight_configurations(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM weight_configurations ORDER BY created_at DESC"
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["weights"] = json.loads(d["weights"])
                d["is_active"] = bool(d["is_active"])
                result.append(d)
            return result

    # ------------------------------------------------------------------
    # Benchmark records
    # ------------------------------------------------------------------

    def save_benchmark_record(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO benchmark_records (
                    experiment_id, model_id, name, role, composite,
                    grade_label, promotion_eligible, category_scores, registered_at
                ) VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    record["experiment_id"],
                    record.get("model_id", ""),
                    record.get("name", ""),
                    record.get("role", "experimental"),
                    record.get("composite", 0.0),
                    record.get("grade_label", ""),
                    int(record.get("promotion_eligible", False)),
                    json.dumps(record.get("category_scores", {})),
                    record.get(
                        "registered_at", datetime.now(timezone.utc).isoformat()
                    ),
                ),
            )

    def save_benchmark_comparison(self, comp: Dict[str, Any]) -> None:
        with self._lock:
            subj = comp.get("subject", {})
            ref = comp.get("reference", {})
            self._conn.execute(
                """INSERT OR REPLACE INTO benchmark_comparisons (
                    comparison_id, subject_id, subject_name,
                    reference_id, reference_name, reference_role,
                    subject_composite, reference_composite,
                    composite_delta, composite_improved,
                    category_deltas, stronger_categories, weaker_categories,
                    compared_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    comp.get("comparison_id", str(uuid.uuid4())),
                    subj.get("id", ""),
                    subj.get("name", ""),
                    ref.get("id", ""),
                    ref.get("name", ""),
                    ref.get("role", ""),
                    subj.get("score", 0.0),
                    ref.get("score", 0.0),
                    comp.get("composite_delta", 0.0),
                    int(comp.get("composite_improved", False)),
                    json.dumps(comp.get("category_deltas", {})),
                    json.dumps(comp.get("stronger_categories", [])),
                    json.dumps(comp.get("weaker_categories", [])),
                    comp.get(
                        "compared_at", datetime.now(timezone.utc).isoformat()
                    ),
                ),
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            def count(table: str) -> int:
                return self._conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]

            best_row = self._conn.execute(
                "SELECT name, composite, grade_label FROM research_scores "
                "ORDER BY composite DESC LIMIT 1"
            ).fetchone()

            return {
                "research_scores": count("research_scores"),
                "category_scores": count("category_scores"),
                "score_history": count("score_history"),
                "weight_configurations": count("weight_configurations"),
                "benchmark_records": count("benchmark_records"),
                "benchmark_comparisons": count("benchmark_comparisons"),
                "best_score": dict(best_row) if best_row else None,
            }

    def close(self) -> None:
        self._conn.close()
