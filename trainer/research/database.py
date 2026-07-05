"""Phase 19 — Research Intelligence Database.

SQLite-backed storage for the Research Intelligence Platform.

Tables:
  - research_findings        : All knowledge base entries
  - feature_lifecycle        : Feature observation history
  - dataset_evolution        : Dataset version registry
  - label_quality            : Label quality batch records
  - market_evolution         : Market regime snapshots and detections
  - retraining_recommendations : Advisory recommendations
  - experiment_suggestions   : Experiment advisory suggestions
  - model_health_points      : Long-term model health timeline

Uses Python's built-in sqlite3 — no extra dependencies.
Thread-safe with WAL journal mode.

Usage::

    db = ResearchDatabase("experiments/research/research.db")
    db.save_knowledge_entry(entry.to_dict())
    db.save_feature_observation(obs.to_dict())
    db.close()
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_findings (
    entry_id         TEXT PRIMARY KEY,
    entry_type       TEXT NOT NULL,
    title            TEXT NOT NULL,
    body             TEXT,
    tags_json        TEXT NOT NULL DEFAULT '[]',
    model_id         TEXT,
    feature_name     TEXT,
    dataset_version  TEXT,
    impact_level     TEXT NOT NULL DEFAULT 'medium',
    author           TEXT NOT NULL DEFAULT 'system',
    references_json  TEXT NOT NULL DEFAULT '[]',
    metadata_json    TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_lifecycle (
    obs_id                  TEXT PRIMARY KEY,
    feature_name            TEXT NOT NULL,
    importance              REAL NOT NULL DEFAULT 0.0,
    stability               REAL NOT NULL DEFAULT 1.0,
    drift                   REAL NOT NULL DEFAULT 0.0,
    missing_rate            REAL NOT NULL DEFAULT 0.0,
    correlation_change      REAL NOT NULL DEFAULT 0.0,
    predictive_contribution REAL NOT NULL DEFAULT 0.0,
    dataset_version         TEXT,
    model_version           TEXT,
    timestamp               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_evolution (
    dataset_id              TEXT PRIMARY KEY,
    version                 TEXT NOT NULL UNIQUE,
    description             TEXT,
    date_start              TEXT,
    date_end                TEXT,
    n_instruments           INTEGER NOT NULL DEFAULT 0,
    n_samples               INTEGER NOT NULL DEFAULT 0,
    n_features              INTEGER NOT NULL DEFAULT 0,
    n_labels                INTEGER NOT NULL DEFAULT 3,
    timeframes_json         TEXT NOT NULL DEFAULT '[]',
    instruments_json        TEXT NOT NULL DEFAULT '[]',
    regimes_covered_json    TEXT NOT NULL DEFAULT '[]',
    feature_schema_version  TEXT,
    label_schema_version    TEXT,
    data_quality_score      REAL NOT NULL DEFAULT 0.0,
    completeness            REAL NOT NULL DEFAULT 1.0,
    status                  TEXT NOT NULL DEFAULT 'active',
    parent_version          TEXT,
    changes_from_parent     TEXT,
    registered_at           TEXT NOT NULL,
    metadata_json           TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS label_quality (
    batch_id         TEXT PRIMARY KEY,
    label_version    TEXT NOT NULL,
    n_samples        INTEGER NOT NULL DEFAULT 0,
    noise_estimate   REAL NOT NULL DEFAULT 0.0,
    consistency      REAL NOT NULL DEFAULT 1.0,
    entropy          REAL NOT NULL DEFAULT 0.0,
    imbalance_ratio  REAL NOT NULL DEFAULT 1.0,
    distribution_json TEXT NOT NULL DEFAULT '{}',
    dataset_version  TEXT,
    notes            TEXT,
    timestamp        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_evolution (
    snapshot_id      TEXT PRIMARY KEY,
    instrument       TEXT NOT NULL,
    timeframe        TEXT NOT NULL,
    regime           TEXT,
    volatility       REAL NOT NULL DEFAULT 0.0,
    trend_strength   REAL NOT NULL DEFAULT 0.0,
    liquidity_score  REAL NOT NULL DEFAULT 0.5,
    session          TEXT,
    correlation_score REAL NOT NULL DEFAULT 0.0,
    macro_indicator  TEXT,
    detection_json   TEXT NOT NULL DEFAULT '{}',
    timestamp        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retraining_recommendations (
    recommendation_id TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL,
    trigger          TEXT NOT NULL,
    reason           TEXT NOT NULL,
    expected_benefit TEXT,
    estimated_risk   TEXT,
    priority         INTEGER NOT NULL DEFAULT 3,
    confidence       REAL NOT NULL DEFAULT 0.5,
    status           TEXT NOT NULL DEFAULT 'pending',
    context_json     TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    acted_at         TEXT,
    acted_by         TEXT
);

CREATE TABLE IF NOT EXISTS experiment_suggestions (
    suggestion_id    TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    category         TEXT NOT NULL,
    description      TEXT,
    hypothesis       TEXT,
    expected_benefit TEXT,
    estimated_risk   TEXT,
    estimated_duration TEXT,
    priority         INTEGER NOT NULL DEFAULT 3,
    confidence       REAL NOT NULL DEFAULT 0.5,
    status           TEXT NOT NULL DEFAULT 'pending',
    config_changes_json TEXT NOT NULL DEFAULT '{}',
    context_json     TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_health_points (
    point_id         TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL,
    accuracy         REAL NOT NULL DEFAULT 0.0,
    sharpe_ratio     REAL NOT NULL DEFAULT 0.0,
    confidence       REAL NOT NULL DEFAULT 0.5,
    calibration_error REAL NOT NULL DEFAULT 0.0,
    prediction_entropy REAL NOT NULL DEFAULT 0.0,
    false_positive_rate REAL NOT NULL DEFAULT 0.0,
    false_negative_rate REAL NOT NULL DEFAULT 0.0,
    signal_rate      REAL NOT NULL DEFAULT 0.0,
    latency_ms       REAL NOT NULL DEFAULT 0.0,
    n_predictions    INTEGER NOT NULL DEFAULT 0,
    regime           TEXT,
    notes            TEXT,
    timestamp        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_findings_type ON research_findings(entry_type);
CREATE INDEX IF NOT EXISTS idx_findings_model ON research_findings(model_id);
CREATE INDEX IF NOT EXISTS idx_findings_impact ON research_findings(impact_level);
CREATE INDEX IF NOT EXISTS idx_feature_name ON feature_lifecycle(feature_name);
CREATE INDEX IF NOT EXISTS idx_dataset_version ON dataset_evolution(version);
CREATE INDEX IF NOT EXISTS idx_label_version ON label_quality(label_version);
CREATE INDEX IF NOT EXISTS idx_market_instrument ON market_evolution(instrument, timeframe);
CREATE INDEX IF NOT EXISTS idx_retrain_model ON retraining_recommendations(model_id);
CREATE INDEX IF NOT EXISTS idx_retrain_status ON retraining_recommendations(status);
CREATE INDEX IF NOT EXISTS idx_experiment_status ON experiment_suggestions(status);
CREATE INDEX IF NOT EXISTS idx_health_model ON model_health_points(model_id);
"""


# ---------------------------------------------------------------------------
# Research database
# ---------------------------------------------------------------------------


class ResearchDatabase:
    """SQLite database for the Research Intelligence Platform.

    Thread-safe (WAL mode, connection-per-thread).

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "experiments/research/research.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"ResearchDatabase initialized at {self.db_path}")

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Knowledge / Research Findings
    # -----------------------------------------------------------------------

    def save_knowledge_entry(self, entry_dict: Dict[str, Any]) -> str:
        eid = entry_dict.get("entry_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO research_findings
               (entry_id, entry_type, title, body, tags_json, model_id,
                feature_name, dataset_version, impact_level, author,
                references_json, metadata_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eid,
                entry_dict.get("entry_type", "research_note"),
                entry_dict.get("title", ""),
                entry_dict.get("body", ""),
                json.dumps(entry_dict.get("tags", [])),
                entry_dict.get("model_id"),
                entry_dict.get("feature_name"),
                entry_dict.get("dataset_version"),
                entry_dict.get("impact_level", "medium"),
                entry_dict.get("author", "system"),
                json.dumps(entry_dict.get("references", [])),
                json.dumps(entry_dict.get("metadata", {})),
                entry_dict.get("created_at",
                                datetime.now(timezone.utc).isoformat()),
            ),
        )
        self._conn.commit()
        return eid

    def list_knowledge_entries(self, entry_type: Optional[str] = None,
                                model_id: Optional[str] = None,
                                impact_level: Optional[str] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        wheres, vals = [], []
        if entry_type:
            wheres.append("entry_type = ?"); vals.append(entry_type)
        if model_id:
            wheres.append("model_id = ?"); vals.append(model_id)
        if impact_level:
            wheres.append("impact_level = ?"); vals.append(impact_level)
        where_str = f"WHERE {' AND '.join(wheres)}" if wheres else ""
        vals.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM research_findings {where_str} "
            f"ORDER BY created_at DESC LIMIT ?", vals
        ).fetchall()
        return [self._unpack_finding(dict(r)) for r in rows]

    def _unpack_finding(self, d: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("tags_json", "references_json", "metadata_json"):
            val = d.pop(k, "[]")
            key = k.replace("_json", "")
            d[key] = json.loads(val or "[]") if k != "metadata_json" else json.loads(val or "{}")
        return d

    # -----------------------------------------------------------------------
    # Feature lifecycle
    # -----------------------------------------------------------------------

    def save_feature_observation(self, obs_dict: Dict[str, Any]) -> str:
        oid = obs_dict.get("obs_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO feature_lifecycle
               (obs_id, feature_name, importance, stability, drift,
                missing_rate, correlation_change, predictive_contribution,
                dataset_version, model_version, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                oid,
                obs_dict.get("name", ""),
                obs_dict.get("importance", 0.0),
                obs_dict.get("stability", 1.0),
                obs_dict.get("drift", 0.0),
                obs_dict.get("missing_rate", 0.0),
                obs_dict.get("correlation_change", 0.0),
                obs_dict.get("predictive_contribution", 0.0),
                obs_dict.get("dataset_version", ""),
                obs_dict.get("model_version", ""),
                obs_dict.get("timestamp",
                              datetime.now(timezone.utc).isoformat()),
            ),
        )
        self._conn.commit()
        return oid

    def list_feature_observations(self, feature_name: str,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM feature_lifecycle WHERE feature_name = ? "
            "ORDER BY timestamp DESC LIMIT ?", (feature_name, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Dataset evolution
    # -----------------------------------------------------------------------

    def save_dataset(self, rec_dict: Dict[str, Any]) -> str:
        did = rec_dict.get("dataset_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO dataset_evolution
               (dataset_id, version, description, date_start, date_end,
                n_instruments, n_samples, n_features, n_labels,
                timeframes_json, instruments_json, regimes_covered_json,
                feature_schema_version, label_schema_version,
                data_quality_score, completeness, status, parent_version,
                changes_from_parent, registered_at, metadata_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                did,
                rec_dict.get("version", ""),
                rec_dict.get("description", ""),
                rec_dict.get("date_start", ""),
                rec_dict.get("date_end", ""),
                rec_dict.get("n_instruments", 0),
                rec_dict.get("n_samples", 0),
                rec_dict.get("n_features", 0),
                rec_dict.get("n_labels", 3),
                json.dumps(rec_dict.get("timeframes", [])),
                json.dumps(rec_dict.get("instruments", [])),
                json.dumps(rec_dict.get("regimes_covered", [])),
                rec_dict.get("feature_schema_version", ""),
                rec_dict.get("label_schema_version", ""),
                rec_dict.get("data_quality_score", 0.0),
                rec_dict.get("completeness", 1.0),
                rec_dict.get("status", "active"),
                rec_dict.get("parent_version"),
                rec_dict.get("changes_from_parent", ""),
                rec_dict.get("registered_at",
                              datetime.now(timezone.utc).isoformat()),
                json.dumps(rec_dict.get("metadata", {})),
            ),
        )
        self._conn.commit()
        return did

    def list_datasets(self, status: Optional[str] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM dataset_evolution WHERE status = ? "
                "ORDER BY registered_at DESC LIMIT ?", (status, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM dataset_evolution ORDER BY registered_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for k in ("timeframes_json", "instruments_json",
                       "regimes_covered_json", "metadata_json"):
                val = d.pop(k, "[]")
                key = k.replace("_json", "")
                d[key] = json.loads(val or ("[]" if k != "metadata_json" else "{}"))
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Label quality
    # -----------------------------------------------------------------------

    def save_label_batch(self, batch_dict: Dict[str, Any]) -> str:
        bid = batch_dict.get("batch_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO label_quality
               (batch_id, label_version, n_samples, noise_estimate,
                consistency, entropy, imbalance_ratio, distribution_json,
                dataset_version, notes, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                bid,
                batch_dict.get("label_version", ""),
                batch_dict.get("n_samples", 0),
                batch_dict.get("noise_estimate", 0.0),
                batch_dict.get("consistency", 1.0),
                batch_dict.get("entropy", 0.0),
                batch_dict.get("imbalance_ratio", 1.0),
                json.dumps(batch_dict.get("class_distribution", {})),
                batch_dict.get("dataset_version", ""),
                batch_dict.get("notes", ""),
                batch_dict.get("timestamp",
                                datetime.now(timezone.utc).isoformat()),
            ),
        )
        self._conn.commit()
        return bid

    def list_label_batches(self, label_version: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        if label_version:
            rows = self._conn.execute(
                "SELECT * FROM label_quality WHERE label_version = ? "
                "ORDER BY timestamp DESC LIMIT ?", (label_version, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM label_quality ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["class_distribution"] = json.loads(d.pop("distribution_json", "{}") or "{}")
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Market evolution
    # -----------------------------------------------------------------------

    def save_market_snapshot(self, snap_dict: Dict[str, Any]) -> str:
        sid = snap_dict.get("snapshot_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO market_evolution
               (snapshot_id, instrument, timeframe, regime, volatility,
                trend_strength, liquidity_score, session, correlation_score,
                macro_indicator, detection_json, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sid,
                snap_dict.get("instrument", ""),
                snap_dict.get("timeframe", ""),
                snap_dict.get("regime", ""),
                snap_dict.get("volatility", 0.0),
                snap_dict.get("trend_strength", 0.0),
                snap_dict.get("liquidity_score", 0.5),
                snap_dict.get("session", ""),
                snap_dict.get("correlation_score", 0.0),
                snap_dict.get("macro_indicator", ""),
                json.dumps(snap_dict.get("detection", {})),
                snap_dict.get("timestamp",
                               datetime.now(timezone.utc).isoformat()),
            ),
        )
        self._conn.commit()
        return sid

    def list_market_snapshots(self, instrument: Optional[str] = None,
                               timeframe: Optional[str] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        wheres, vals = [], []
        if instrument:
            wheres.append("instrument = ?"); vals.append(instrument)
        if timeframe:
            wheres.append("timeframe = ?"); vals.append(timeframe)
        where_str = f"WHERE {' AND '.join(wheres)}" if wheres else ""
        vals.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM market_evolution {where_str} "
            f"ORDER BY timestamp DESC LIMIT ?", vals
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["detection"] = json.loads(d.pop("detection_json", "{}") or "{}")
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Retraining recommendations
    # -----------------------------------------------------------------------

    def save_recommendation(self, rec_dict: Dict[str, Any]) -> str:
        rid = rec_dict.get("recommendation_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO retraining_recommendations
               (recommendation_id, model_id, trigger, reason,
                expected_benefit, estimated_risk, priority, confidence,
                status, context_json, created_at, acted_at, acted_by)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rid,
                rec_dict.get("model_id", ""),
                rec_dict.get("trigger", ""),
                rec_dict.get("reason", ""),
                rec_dict.get("expected_benefit", ""),
                rec_dict.get("estimated_risk", ""),
                rec_dict.get("priority", 3),
                rec_dict.get("confidence", 0.5),
                rec_dict.get("status", "pending"),
                json.dumps(rec_dict.get("context", {})),
                rec_dict.get("created_at",
                              datetime.now(timezone.utc).isoformat()),
                rec_dict.get("acted_at"),
                rec_dict.get("acted_by"),
            ),
        )
        self._conn.commit()
        return rid

    def list_recommendations(self, model_id: Optional[str] = None,
                               status: Optional[str] = None,
                               limit: int = 50) -> List[Dict[str, Any]]:
        wheres, vals = [], []
        if model_id:
            wheres.append("model_id = ?"); vals.append(model_id)
        if status:
            wheres.append("status = ?"); vals.append(status)
        where_str = f"WHERE {' AND '.join(wheres)}" if wheres else ""
        vals.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM retraining_recommendations {where_str} "
            f"ORDER BY created_at DESC LIMIT ?", vals
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["context"] = json.loads(d.pop("context_json", "{}") or "{}")
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Experiment suggestions
    # -----------------------------------------------------------------------

    def save_experiment(self, exp_dict: Dict[str, Any]) -> str:
        sid = exp_dict.get("suggestion_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO experiment_suggestions
               (suggestion_id, title, category, description, hypothesis,
                expected_benefit, estimated_risk, estimated_duration,
                priority, confidence, status, config_changes_json,
                context_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sid,
                exp_dict.get("title", ""),
                exp_dict.get("category", ""),
                exp_dict.get("description", ""),
                exp_dict.get("hypothesis", ""),
                exp_dict.get("expected_benefit", ""),
                exp_dict.get("estimated_risk", ""),
                exp_dict.get("estimated_duration", ""),
                exp_dict.get("priority", 3),
                exp_dict.get("confidence", 0.5),
                exp_dict.get("status", "pending"),
                json.dumps(exp_dict.get("config_changes", {})),
                json.dumps(exp_dict.get("trigger_context", {})),
                exp_dict.get("created_at",
                              datetime.now(timezone.utc).isoformat()),
            ),
        )
        self._conn.commit()
        return sid

    def list_experiments(self, status: Optional[str] = None,
                          category: Optional[str] = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
        wheres, vals = [], []
        if status:
            wheres.append("status = ?"); vals.append(status)
        if category:
            wheres.append("category = ?"); vals.append(category)
        where_str = f"WHERE {' AND '.join(wheres)}" if wheres else ""
        vals.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM experiment_suggestions {where_str} "
            f"ORDER BY priority ASC, created_at DESC LIMIT ?", vals
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["config_changes"] = json.loads(d.pop("config_changes_json", "{}") or "{}")
            d["trigger_context"] = json.loads(d.pop("context_json", "{}") or "{}")
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Model health points
    # -----------------------------------------------------------------------

    def save_health_point(self, pt_dict: Dict[str, Any]) -> str:
        pid = pt_dict.get("point_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO model_health_points
               (point_id, model_id, accuracy, sharpe_ratio, confidence,
                calibration_error, prediction_entropy,
                false_positive_rate, false_negative_rate, signal_rate,
                latency_ms, n_predictions, regime, notes, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                pid,
                pt_dict.get("model_id", ""),
                pt_dict.get("accuracy", 0.0),
                pt_dict.get("sharpe_ratio", 0.0),
                pt_dict.get("confidence", 0.5),
                pt_dict.get("calibration_error", 0.0),
                pt_dict.get("prediction_entropy", 0.0),
                pt_dict.get("false_positive_rate", 0.0),
                pt_dict.get("false_negative_rate", 0.0),
                pt_dict.get("signal_rate", 0.0),
                pt_dict.get("latency_ms", 0.0),
                pt_dict.get("n_predictions", 0),
                pt_dict.get("regime", ""),
                pt_dict.get("notes", ""),
                pt_dict.get("timestamp",
                             datetime.now(timezone.utc).isoformat()),
            ),
        )
        self._conn.commit()
        return pid

    def list_health_points(self, model_id: str,
                            limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM model_health_points WHERE model_id = ? "
            "ORDER BY timestamp ASC LIMIT ?", (model_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        def _count(table: str, where: str = "") -> int:
            sql = f"SELECT COUNT(*) as n FROM {table}"
            if where:
                sql += f" WHERE {where}"
            return self._conn.execute(sql).fetchone()["n"]

        return {
            "total_knowledge_entries": _count("research_findings"),
            "total_feature_observations": _count("feature_lifecycle"),
            "total_datasets": _count("dataset_evolution"),
            "active_datasets": _count("dataset_evolution", "status = 'active'"),
            "total_label_batches": _count("label_quality"),
            "total_market_snapshots": _count("market_evolution"),
            "total_recommendations": _count("retraining_recommendations"),
            "pending_recommendations": _count("retraining_recommendations",
                                               "status = 'pending'"),
            "total_experiments": _count("experiment_suggestions"),
            "pending_experiments": _count("experiment_suggestions",
                                           "status = 'pending'"),
            "total_health_points": _count("model_health_points"),
        }

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
