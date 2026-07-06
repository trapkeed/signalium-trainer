"""Phase 18 — MLOps Database.

SQLite-backed storage for the Institutional MLOps Platform.

Tables:
  - model_registry         : All registered model records
  - lifecycle_transitions  : Every lifecycle state change
  - deployments            : All deployment events
  - health_metrics         : Rolling health/drift metrics
  - alerts                 : All generated alerts
  - audit_events           : Full immutable audit trail
  - retraining_policies    : Active policies
  - retraining_recommendations : Advisory recommendations

Uses Python's built-in sqlite3 — no extra dependencies.
Thread-safe with WAL journal mode.

Usage::

    db = MLOpsDatabase("experiments/mlops/mlops.db")
    db.save_model(record)
    db.save_alert(alert)
    stats = db.stats()
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

from trainer.mlops.registry import ModelRecord

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_registry (
    model_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    training_date   TEXT NOT NULL,
    architecture    TEXT,
    backbone        TEXT,
    parameter_count INTEGER NOT NULL DEFAULT 0,
    dataset_version TEXT,
    feature_version TEXT,
    label_version   TEXT,
    git_commit      TEXT,
    lifecycle_state TEXT NOT NULL DEFAULT 'draft',
    promotion_status TEXT NOT NULL DEFAULT 'pending',
    deployment_status TEXT NOT NULL DEFAULT 'not_deployed',
    retirement_status TEXT NOT NULL DEFAULT 'active',
    hardware_used   TEXT,
    training_duration_seconds REAL NOT NULL DEFAULT 0.0,
    parent_model_id TEXT,
    child_model_ids TEXT NOT NULL DEFAULT '[]',
    notes           TEXT,
    training_config_json TEXT NOT NULL DEFAULT '{}',
    validation_report_json TEXT NOT NULL DEFAULT '{}',
    approval_history_json TEXT NOT NULL DEFAULT '[]',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lifecycle_transitions (
    transition_id   TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    from_state      TEXT NOT NULL,
    to_state        TEXT NOT NULL,
    actor           TEXT NOT NULL,
    reason          TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    timestamp       TEXT NOT NULL,
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

CREATE TABLE IF NOT EXISTS deployments (
    deployment_id   TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    target          TEXT NOT NULL,
    deploy_type     TEXT NOT NULL,
    approver        TEXT,
    actor           TEXT,
    reason          TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    from_state      TEXT,
    to_state        TEXT,
    deployed_at     TEXT NOT NULL,
    rolled_back_at  TEXT,
    rollback_reason TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS health_metrics (
    metric_id       TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    confidence_drift REAL NOT NULL DEFAULT 0.0,
    prediction_drift REAL NOT NULL DEFAULT 0.0,
    calibration_drift REAL NOT NULL DEFAULT 0.0,
    performance_degradation REAL NOT NULL DEFAULT 0.0,
    latency_degradation REAL NOT NULL DEFAULT 0.0,
    distribution_drift REAL NOT NULL DEFAULT 0.0,
    is_healthy      INTEGER NOT NULL DEFAULT 1,
    drift_alerts_json TEXT NOT NULL DEFAULT '[]',
    n_records       INTEGER NOT NULL DEFAULT 0,
    snapshot_json   TEXT NOT NULL DEFAULT '{}',
    recorded_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        TEXT PRIMARY KEY,
    level           TEXT NOT NULL,
    model_id        TEXT,
    category        TEXT NOT NULL,
    message         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    acknowledged_by TEXT,
    acknowledged_at TEXT,
    resolved_at     TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    actor           TEXT NOT NULL,
    model_id        TEXT,
    description     TEXT,
    outcome         TEXT NOT NULL DEFAULT 'success',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    session_id      TEXT,
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retraining_recommendations (
    recommendation_id TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    trigger         TEXT NOT NULL,
    reason          TEXT NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 2,
    status          TEXT NOT NULL DEFAULT 'pending',
    policy_id       TEXT,
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    acted_at        TEXT,
    acted_by        TEXT
);

CREATE INDEX IF NOT EXISTS idx_registry_state ON model_registry(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_registry_name ON model_registry(name);
CREATE INDEX IF NOT EXISTS idx_lifecycle_model ON lifecycle_transitions(model_id);
CREATE INDEX IF NOT EXISTS idx_deployments_model ON deployments(model_id);
CREATE INDEX IF NOT EXISTS idx_health_model ON health_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level);
CREATE INDEX IF NOT EXISTS idx_audit_model ON audit_events(model_id);
CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_retrain_model ON retraining_recommendations(model_id);
"""


# ---------------------------------------------------------------------------
# MLOps database
# ---------------------------------------------------------------------------


class MLOpsDatabase:
    """SQLite database for the Institutional MLOps Platform.

    Thread-safe (WAL mode, connection-per-thread).

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "experiments/mlops/mlops.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"MLOpsDatabase initialized at {self.db_path}")

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
    # Model registry
    # -----------------------------------------------------------------------

    def save_model(self, record: ModelRecord) -> str:
        self._conn.execute(
            """INSERT OR REPLACE INTO model_registry
               (model_id, name, version, training_date, architecture, backbone,
                parameter_count, dataset_version, feature_version, label_version,
                git_commit, lifecycle_state, promotion_status, deployment_status,
                retirement_status, hardware_used, training_duration_seconds,
                parent_model_id, child_model_ids, notes, training_config_json,
                validation_report_json, approval_history_json, metadata_json,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.model_id,
                record.name,
                record.version,
                record.training_date,
                record.architecture,
                record.backbone,
                record.parameter_count,
                record.dataset_version,
                record.feature_version,
                record.label_version,
                record.git_commit,
                record.lifecycle_state,
                record.promotion_status,
                record.deployment_status,
                record.retirement_status,
                record.hardware_used,
                record.training_duration_seconds,
                record.parent_model_id,
                json.dumps(record.child_model_ids),
                record.notes,
                json.dumps(record.training_config),
                json.dumps(record.validation_report),
                json.dumps([a.to_dict() for a in record.approval_history]),
                json.dumps(record.metadata),
                record.created_at,
                record.updated_at,
            ),
        )
        self._conn.commit()
        return record.model_id

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM model_registry WHERE model_id = ?", (model_id,)
        ).fetchone()
        return self._model_row(row) if row else None

    def list_models(
        self, lifecycle_state: Optional[str] = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        if lifecycle_state:
            rows = self._conn.execute(
                "SELECT * FROM model_registry WHERE lifecycle_state = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (lifecycle_state, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM model_registry ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._model_row(r) for r in rows]

    def _model_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for k in (
            "child_model_ids",
            "training_config_json",
            "validation_report_json",
            "metadata_json",
        ):
            v = d.pop(k, None) or "{}"
            key = k.replace("_json", "").replace("training_config", "training_config")
            d[key.replace("_json", "")] = json.loads(v) if v else {}
        d["child_model_ids"] = (
            json.loads(d.pop("child_model_ids", "[]") or "[]") if "child_model_ids" in d else []
        )
        d["approval_history"] = (
            json.loads(d.pop("approval_history_json", "[]") or "[]")
            if "approval_history_json" in d
            else []
        )
        for k in list(d.keys()):
            if k.endswith("_json"):
                d[k[:-5]] = json.loads(d.pop(k) or "{}")
        return d

    def update_model_state(
        self,
        model_id: str,
        lifecycle_state: str,
        deployment_status: str = "",
        promotion_status: str = "",
    ) -> None:
        sets = ["lifecycle_state = ?", "updated_at = ?"]
        vals: List[Any] = [lifecycle_state, datetime.now(timezone.utc).isoformat()]
        if deployment_status:
            sets.append("deployment_status = ?")
            vals.append(deployment_status)
        if promotion_status:
            sets.append("promotion_status = ?")
            vals.append(promotion_status)
        vals.append(model_id)
        self._conn.execute(
            f"UPDATE model_registry SET {', '.join(sets)} WHERE model_id = ?",
            vals,
        )
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Lifecycle transitions
    # -----------------------------------------------------------------------

    def save_transition(self, transition_dict: Dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO lifecycle_transitions
               (transition_id, model_id, from_state, to_state, actor,
                reason, metadata_json, timestamp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                transition_dict.get("transition_id", str(uuid.uuid4())),
                transition_dict.get("model_id", ""),
                transition_dict.get("from_state", ""),
                transition_dict.get("to_state", ""),
                transition_dict.get("actor", "system"),
                transition_dict.get("reason", ""),
                json.dumps(transition_dict.get("metadata", {})),
                transition_dict.get("timestamp", datetime.now(timezone.utc).isoformat()),
            ),
        )
        self._conn.commit()

    def get_lifecycle_history(self, model_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM lifecycle_transitions WHERE model_id = ? ORDER BY timestamp", (model_id,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Deployments
    # -----------------------------------------------------------------------

    def save_deployment(self, deploy_dict: Dict[str, Any]) -> str:
        did = deploy_dict.get("deployment_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO deployments
               (deployment_id, model_id, target, deploy_type, approver, actor,
                reason, status, from_state, to_state, deployed_at,
                rolled_back_at, rollback_reason, metadata_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                did,
                deploy_dict.get("model_id", ""),
                deploy_dict.get("target", ""),
                deploy_dict.get("deploy_type", "manual"),
                deploy_dict.get("approver", "system"),
                deploy_dict.get("actor", "system"),
                deploy_dict.get("reason", ""),
                deploy_dict.get("status", "active"),
                deploy_dict.get("from_state", ""),
                deploy_dict.get("to_state", ""),
                deploy_dict.get("deployed_at", datetime.now(timezone.utc).isoformat()),
                deploy_dict.get("rolled_back_at"),
                deploy_dict.get("rollback_reason"),
                json.dumps(deploy_dict.get("metadata", {})),
            ),
        )
        self._conn.commit()
        return did

    def list_deployments(
        self, model_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        if model_id:
            rows = self._conn.execute(
                "SELECT * FROM deployments WHERE model_id = ? " "ORDER BY deployed_at DESC LIMIT ?",
                (model_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM deployments ORDER BY deployed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Health metrics
    # -----------------------------------------------------------------------

    def save_health_report(self, report_dict: Dict[str, Any]) -> str:
        metric_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO health_metrics
               (metric_id, model_id, confidence_drift, prediction_drift,
                calibration_drift, performance_degradation, latency_degradation,
                distribution_drift, is_healthy, drift_alerts_json,
                n_records, snapshot_json, recorded_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                metric_id,
                report_dict.get("model_id", ""),
                report_dict.get("confidence_drift", 0.0),
                report_dict.get("prediction_drift", 0.0),
                report_dict.get("calibration_drift", 0.0),
                report_dict.get("performance_degradation", 0.0),
                report_dict.get("latency_degradation", 0.0),
                report_dict.get("distribution_drift", 0.0),
                int(report_dict.get("is_healthy", True)),
                json.dumps(report_dict.get("drift_alerts", [])),
                report_dict.get("n_records", 0),
                json.dumps(report_dict),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return metric_id

    def get_latest_health(self, model_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT snapshot_json FROM health_metrics WHERE model_id = ? "
            "ORDER BY recorded_at DESC LIMIT 1",
            (model_id,),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["snapshot_json"])

    def list_health_history(self, model_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM health_metrics WHERE model_id = ? " "ORDER BY recorded_at DESC LIMIT ?",
            (model_id, limit),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["drift_alerts"] = json.loads(d.pop("drift_alerts_json", "[]") or "[]")
            d.pop("snapshot_json", None)
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Alerts
    # -----------------------------------------------------------------------

    def save_alert(self, alert_dict: Dict[str, Any]) -> str:
        aid = alert_dict.get("alert_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO alerts
               (alert_id, level, model_id, category, message, status,
                acknowledged_by, acknowledged_at, resolved_at,
                metadata_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                aid,
                alert_dict.get("level", "WARNING"),
                alert_dict.get("model_id", ""),
                alert_dict.get("category", "system"),
                alert_dict.get("message", ""),
                alert_dict.get("status", "active"),
                alert_dict.get("acknowledged_by"),
                alert_dict.get("acknowledged_at"),
                alert_dict.get("resolved_at"),
                json.dumps(alert_dict.get("metadata", {})),
                alert_dict.get("created_at", datetime.now(timezone.utc).isoformat()),
            ),
        )
        self._conn.commit()
        return aid

    def list_alerts(
        self, status: Optional[str] = None, model_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        wheres, vals = [], []
        if status:
            wheres.append("status = ?")
            vals.append(status)
        if model_id:
            wheres.append("model_id = ?")
            vals.append(model_id)
        where_str = f"WHERE {' AND '.join(wheres)}" if wheres else ""
        vals.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM alerts {where_str} ORDER BY created_at DESC LIMIT ?",
            vals,
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Audit events
    # -----------------------------------------------------------------------

    def save_audit_event(self, event_dict: Dict[str, Any]) -> str:
        eid = event_dict.get("event_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO audit_events
               (event_id, event_type, actor, model_id, description,
                outcome, metadata_json, session_id, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                eid,
                event_dict.get("event_type", "system"),
                event_dict.get("actor", "system"),
                event_dict.get("model_id", ""),
                event_dict.get("description", ""),
                event_dict.get("outcome", "success"),
                json.dumps(event_dict.get("metadata", {})),
                event_dict.get("session_id"),
                event_dict.get("timestamp", datetime.now(timezone.utc).isoformat()),
            ),
        )
        self._conn.commit()
        return eid

    def list_audit_events(
        self, event_type: Optional[str] = None, model_id: Optional[str] = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        wheres, vals = [], []
        if event_type:
            wheres.append("event_type = ?")
            vals.append(event_type)
        if model_id:
            wheres.append("model_id = ?")
            vals.append(model_id)
        where_str = f"WHERE {' AND '.join(wheres)}" if wheres else ""
        vals.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM audit_events {where_str} ORDER BY timestamp DESC LIMIT ?",
            vals,
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Retraining recommendations
    # -----------------------------------------------------------------------

    def save_recommendation(self, rec_dict: Dict[str, Any]) -> str:
        rid = rec_dict.get("recommendation_id", str(uuid.uuid4()))
        self._conn.execute(
            """INSERT OR REPLACE INTO retraining_recommendations
               (recommendation_id, model_id, trigger, reason, priority,
                status, policy_id, metrics_json, created_at, acted_at, acted_by)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rid,
                rec_dict.get("model_id", ""),
                rec_dict.get("trigger", ""),
                rec_dict.get("reason", ""),
                rec_dict.get("priority", 2),
                rec_dict.get("status", "pending"),
                rec_dict.get("policy_id", ""),
                json.dumps(rec_dict.get("metrics", {})),
                rec_dict.get("created_at", datetime.now(timezone.utc).isoformat()),
                rec_dict.get("acted_at"),
                rec_dict.get("acted_by"),
            ),
        )
        self._conn.commit()
        return rid

    def list_recommendations(
        self, model_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        wheres, vals = [], []
        if model_id:
            wheres.append("model_id = ?")
            vals.append(model_id)
        if status:
            wheres.append("status = ?")
            vals.append(status)
        where_str = f"WHERE {' AND '.join(wheres)}" if wheres else ""
        vals.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM retraining_recommendations {where_str} "
            f"ORDER BY created_at DESC LIMIT ?",
            vals,
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d.pop("metrics_json", "{}") or "{}")
            result.append(d)
        return result

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
            "total_models": _count("model_registry"),
            "total_transitions": _count("lifecycle_transitions"),
            "total_deployments": _count("deployments"),
            "active_deployments": _count("deployments", "status = 'active'"),
            "total_alerts": _count("alerts"),
            "active_alerts": _count("alerts", "status = 'active'"),
            "critical_alerts": _count("alerts", "level = 'CRITICAL' AND status = 'active'"),
            "total_audit_events": _count("audit_events"),
            "pending_recommendations": _count("retraining_recommendations", "status = 'pending'"),
        }

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
