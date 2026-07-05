"""Phase 19.1 — Lineage Database.

SQLite-backed persistence for the Phase 19.1 Experiment Lineage,
Reproducibility, Cost Tracking, and Research Timeline subsystems.

Tables:
  lineage_nodes       — one row per LineageNode
  lineage_edges       — parent-child relationships
  environment_snapshots — EnvironmentSnapshot per node
  dataset_snapshots   — DatasetSnapshot per node
  feature_snapshots   — FeatureSnapshot per node
  model_snapshots     — ModelSnapshot per node
  validation_snapshots — ValidationSnapshot per node
  cost_records        — ExperimentCost per entity
  timeline_events     — TimelineEvent records
  knowledge_graph     — serialised KnowledgeGraph (single-row JSON blob)

Zero new runtime dependencies — stdlib sqlite3 only.
Thread-safe with WAL journal mode.

Usage::

    db = LineageDatabase("experiments/research/lineage.db")
    db.save_node(node.to_dict())
    db.save_environment(node_id, env.to_dict())
    db.save_cost(cost.to_dict())
    db.close()
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


_SCHEMA = """
CREATE TABLE IF NOT EXISTS lineage_nodes (
    node_id         TEXT PRIMARY KEY,
    name            TEXT NOT NULL DEFAULT '',
    researcher      TEXT NOT NULL DEFAULT 'system',
    purpose         TEXT NOT NULL DEFAULT '',
    parent_id       TEXT,
    child_ids_json  TEXT NOT NULL DEFAULT '[]',
    branch          TEXT NOT NULL DEFAULT 'main',
    project         TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'planned',
    created_at      TEXT NOT NULL,
    completed_at    TEXT,
    notes           TEXT NOT NULL DEFAULT '',
    tags_json       TEXT NOT NULL DEFAULT '[]',
    gpu_hours       REAL NOT NULL DEFAULT 0.0,
    cpu_hours       REAL NOT NULL DEFAULT 0.0,
    training_cost_usd   REAL NOT NULL DEFAULT 0.0,
    storage_cost_usd    REAL NOT NULL DEFAULT 0.0,
    validation_cost_usd REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS environment_snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    node_id         TEXT NOT NULL,
    data_json       TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES lineage_nodes(node_id)
);

CREATE TABLE IF NOT EXISTS dataset_snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    node_id         TEXT NOT NULL,
    data_json       TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES lineage_nodes(node_id)
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    node_id         TEXT NOT NULL,
    data_json       TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES lineage_nodes(node_id)
);

CREATE TABLE IF NOT EXISTS model_snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    node_id         TEXT NOT NULL,
    data_json       TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES lineage_nodes(node_id)
);

CREATE TABLE IF NOT EXISTS validation_snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    node_id         TEXT NOT NULL,
    data_json       TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES lineage_nodes(node_id)
);

CREATE TABLE IF NOT EXISTS cost_records (
    cost_id         TEXT PRIMARY KEY,
    entity_id       TEXT NOT NULL,
    entity_type     TEXT NOT NULL DEFAULT 'experiment',
    project         TEXT NOT NULL DEFAULT '',
    branch          TEXT NOT NULL DEFAULT '',
    researcher      TEXT NOT NULL DEFAULT 'system',
    recorded_at     TEXT NOT NULL,
    data_json       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS timeline_events (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    entity_id       TEXT NOT NULL DEFAULT '',
    entity_type     TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    researcher      TEXT NOT NULL DEFAULT 'system',
    recorded_at     TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS knowledge_graph_blobs (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    data_json       TEXT NOT NULL,
    saved_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lineage_nodes_parent ON lineage_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_project ON lineage_nodes(project);
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_branch ON lineage_nodes(branch);
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_status ON lineage_nodes(status);
CREATE INDEX IF NOT EXISTS idx_timeline_events_type ON timeline_events(event_type);
CREATE INDEX IF NOT EXISTS idx_timeline_events_entity ON timeline_events(entity_id);
CREATE INDEX IF NOT EXISTS idx_cost_records_entity ON cost_records(entity_id);
CREATE INDEX IF NOT EXISTS idx_cost_records_project ON cost_records(project);
"""


class LineageDatabase:
    """Thread-safe SQLite persistence for Phase 19.1 lineage data."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.debug(f"[LineageDatabase] opened: {db_path}")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Lineage nodes
    # ------------------------------------------------------------------

    def save_node(self, d: Dict[str, Any]) -> None:
        sql = """
        INSERT OR REPLACE INTO lineage_nodes
          (node_id, name, researcher, purpose, parent_id, child_ids_json,
           branch, project, status, created_at, completed_at, notes,
           tags_json, gpu_hours, cpu_hours, training_cost_usd,
           storage_cost_usd, validation_cost_usd)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                d["node_id"], d.get("name", ""), d.get("researcher", "system"),
                d.get("purpose", ""), d.get("parent_id"),
                json.dumps(d.get("child_ids", [])),
                d.get("branch", "main"), d.get("project", ""),
                d.get("status", "planned"), d["created_at"],
                d.get("completed_at"),
                d.get("notes", ""),
                json.dumps(d.get("tags", [])),
                d.get("gpu_hours", 0.0), d.get("cpu_hours", 0.0),
                d.get("training_cost_usd", 0.0),
                d.get("storage_cost_usd", 0.0),
                d.get("validation_cost_usd", 0.0),
            ))

    def load_all_nodes(self) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM lineage_nodes ORDER BY created_at"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["child_ids"] = json.loads(d.pop("child_ids_json", "[]"))
            d["tags"] = json.loads(d.pop("tags_json", "[]"))
            result.append(d)
        return result

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM lineage_nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["child_ids"] = json.loads(d.pop("child_ids_json", "[]"))
        d["tags"] = json.loads(d.pop("tags_json", "[]"))
        return d

    # ------------------------------------------------------------------
    # Snapshots (shared pattern)
    # ------------------------------------------------------------------

    def _save_snapshot(
        self, table: str, snapshot_id: str, node_id: str, data: Dict[str, Any]
    ) -> None:
        sql = f"""
        INSERT OR REPLACE INTO {table}
          (snapshot_id, node_id, data_json, captured_at)
        VALUES (?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                snapshot_id, node_id,
                json.dumps(data),
                data.get("captured_at", datetime.now(timezone.utc).isoformat()),
            ))

    def _load_snapshot(
        self, table: str, node_id: str
    ) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            f"SELECT data_json FROM {table} WHERE node_id=? ORDER BY captured_at DESC LIMIT 1",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def save_environment(self, node_id: str, data: Dict[str, Any]) -> None:
        self._save_snapshot("environment_snapshots", data["snapshot_id"], node_id, data)

    def load_environment(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._load_snapshot("environment_snapshots", node_id)

    def save_dataset_snapshot(self, node_id: str, data: Dict[str, Any]) -> None:
        self._save_snapshot("dataset_snapshots", data["snapshot_id"], node_id, data)

    def load_dataset_snapshot(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._load_snapshot("dataset_snapshots", node_id)

    def save_feature_snapshot(self, node_id: str, data: Dict[str, Any]) -> None:
        self._save_snapshot("feature_snapshots", data["snapshot_id"], node_id, data)

    def load_feature_snapshot(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._load_snapshot("feature_snapshots", node_id)

    def save_model_snapshot(self, node_id: str, data: Dict[str, Any]) -> None:
        self._save_snapshot("model_snapshots", data["snapshot_id"], node_id, data)

    def load_model_snapshot(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._load_snapshot("model_snapshots", node_id)

    def save_validation_snapshot(self, node_id: str, data: Dict[str, Any]) -> None:
        self._save_snapshot("validation_snapshots", data["snapshot_id"], node_id, data)

    def load_validation_snapshot(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._load_snapshot("validation_snapshots", node_id)

    # ------------------------------------------------------------------
    # Cost records
    # ------------------------------------------------------------------

    def save_cost(self, data: Dict[str, Any]) -> None:
        sql = """
        INSERT OR REPLACE INTO cost_records
          (cost_id, entity_id, entity_type, project, branch, researcher,
           recorded_at, data_json)
        VALUES (?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                data["cost_id"], data["entity_id"],
                data.get("entity_type", "experiment"),
                data.get("project", ""), data.get("branch", ""),
                data.get("researcher", "system"),
                data.get("recorded_at", datetime.now(timezone.utc).isoformat()),
                json.dumps(data),
            ))

    def load_all_costs(self) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT data_json FROM cost_records ORDER BY recorded_at"
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def load_cost_by_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT data_json FROM cost_records WHERE entity_id=? ORDER BY recorded_at DESC LIMIT 1",
            (entity_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    # ------------------------------------------------------------------
    # Timeline events
    # ------------------------------------------------------------------

    def save_timeline_event(self, data: Dict[str, Any]) -> None:
        sql = """
        INSERT OR REPLACE INTO timeline_events
          (event_id, event_type, entity_id, entity_type, description,
           researcher, recorded_at, attributes_json)
        VALUES (?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                data["event_id"], data["event_type"],
                data.get("entity_id", ""), data.get("entity_type", ""),
                data.get("description", ""), data.get("researcher", "system"),
                data.get("recorded_at", datetime.now(timezone.utc).isoformat()),
                json.dumps(data.get("attributes", {})),
            ))

    def load_all_timeline_events(self) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM timeline_events ORDER BY recorded_at"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["attributes"] = json.loads(d.pop("attributes_json", "{}"))
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Knowledge graph blob
    # ------------------------------------------------------------------

    def save_knowledge_graph(self, data: Dict[str, Any]) -> None:
        sql = """
        INSERT OR REPLACE INTO knowledge_graph_blobs (id, data_json, saved_at)
        VALUES (1, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                json.dumps(data),
                datetime.now(timezone.utc).isoformat(),
            ))

    def load_knowledge_graph(self) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT data_json FROM knowledge_graph_blobs WHERE id=1"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        conn = self._conn()
        return {
            "lineage_nodes": conn.execute(
                "SELECT COUNT(*) FROM lineage_nodes"
            ).fetchone()[0],
            "cost_records": conn.execute(
                "SELECT COUNT(*) FROM cost_records"
            ).fetchone()[0],
            "timeline_events": conn.execute(
                "SELECT COUNT(*) FROM timeline_events"
            ).fetchone()[0],
            "environment_snapshots": conn.execute(
                "SELECT COUNT(*) FROM environment_snapshots"
            ).fetchone()[0],
            "model_snapshots": conn.execute(
                "SELECT COUNT(*) FROM model_snapshots"
            ).fetchone()[0],
            "dataset_snapshots": conn.execute(
                "SELECT COUNT(*) FROM dataset_snapshots"
            ).fetchone()[0],
            "validation_snapshots": conn.execute(
                "SELECT COUNT(*) FROM validation_snapshots"
            ).fetchone()[0],
        }
