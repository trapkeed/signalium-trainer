"""Phase 17 — AEIP Ensemble Database.

SQLite-backed storage for the Adaptive Ensemble Intelligence Platform.

Stores:
  - Expert models
  - Expert metrics
  - Ensemble results
  - Meta model state
  - Weight history
  - Voting history
  - Routing decisions
  - Performance history

Uses Python's built-in ``sqlite3`` — no extra dependencies.

Usage::

    db = EnsembleDatabase("experiments/ensemble/ensemble.db")
    db.save_expert(expert)
    db.save_ensemble_result(result_dict)
    results = db.list_ensemble_results(limit=50)
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

from trainer.ensemble.expert import ExpertModel

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS expert_models (
    expert_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    model_family    TEXT,
    specializations TEXT NOT NULL DEFAULT '[]',
    instruments     TEXT NOT NULL DEFAULT '[]',
    timeframes      TEXT NOT NULL DEFAULT '[]',
    version         TEXT,
    checkpoint_path TEXT,
    model_version   TEXT,
    phase           TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    weight          REAL NOT NULL DEFAULT 1.0,
    recent_accuracy REAL NOT NULL DEFAULT 0.0,
    recent_sharpe   REAL NOT NULL DEFAULT 0.0,
    calibration_error REAL NOT NULL DEFAULT 0.0,
    n_predictions   INTEGER NOT NULL DEFAULT 0,
    n_correct       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ensemble_results (
    result_id       TEXT PRIMARY KEY,
    buy_prob        REAL NOT NULL,
    sell_prob       REAL NOT NULL,
    no_trade_prob   REAL NOT NULL,
    confidence      REAL NOT NULL,
    uncertainty     REAL NOT NULL,
    predicted_class TEXT NOT NULL,
    strategy        TEXT,
    n_experts       INTEGER NOT NULL DEFAULT 0,
    agreement_score REAL NOT NULL DEFAULT 0.0,
    regime          TEXT,
    instrument      TEXT,
    timeframe       TEXT,
    true_class      TEXT,
    is_correct      INTEGER,
    weight_map_json TEXT NOT NULL DEFAULT '{}',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta_model_state (
    state_id        TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    meta_version    TEXT,
    n_training_samples INTEGER NOT NULL DEFAULT 0,
    state_json      TEXT NOT NULL DEFAULT '{}',
    saved_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS weight_history (
    entry_id        TEXT PRIMARY KEY,
    expert_id       TEXT NOT NULL,
    weight          REAL NOT NULL,
    regime          TEXT,
    instrument      TEXT,
    timeframe       TEXT,
    recorded_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS routing_decisions (
    decision_id     TEXT PRIMARY KEY,
    regime          TEXT,
    instrument      TEXT,
    timeframe       TEXT,
    selected_experts TEXT NOT NULL DEFAULT '[]',
    weights_json    TEXT NOT NULL DEFAULT '{}',
    routing_reason  TEXT,
    n_experts_selected INTEGER NOT NULL DEFAULT 0,
    decided_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS performance_history (
    record_id       TEXT PRIMARY KEY,
    expert_id       TEXT NOT NULL,
    predicted_class TEXT,
    true_class      TEXT,
    profit_pnl      REAL NOT NULL DEFAULT 0.0,
    confidence      REAL NOT NULL DEFAULT 0.5,
    uncertainty     REAL NOT NULL DEFAULT 0.5,
    regime          TEXT,
    instrument      TEXT,
    timeframe       TEXT,
    is_correct      INTEGER,
    recorded_at     TEXT NOT NULL,
    FOREIGN KEY (expert_id) REFERENCES expert_models(expert_id)
);

CREATE TABLE IF NOT EXISTS voting_history (
    vote_id         TEXT PRIMARY KEY,
    result_id       TEXT NOT NULL,
    expert_id       TEXT NOT NULL,
    voted_class     TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.5,
    weight          REAL NOT NULL DEFAULT 0.0,
    recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ensemble_created ON ensemble_results(created_at);
CREATE INDEX IF NOT EXISTS idx_ensemble_strategy ON ensemble_results(strategy);
CREATE INDEX IF NOT EXISTS idx_perf_expert ON performance_history(expert_id);
CREATE INDEX IF NOT EXISTS idx_weight_expert ON weight_history(expert_id);
CREATE INDEX IF NOT EXISTS idx_routing_regime ON routing_decisions(regime);
"""


# ---------------------------------------------------------------------------
# Ensemble database
# ---------------------------------------------------------------------------


class EnsembleDatabase:
    """SQLite database for the Adaptive Ensemble Intelligence Platform.

    Thread-safe with connection-per-thread semantics.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "experiments/ensemble/ensemble.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"EnsembleDatabase initialized at {self.db_path}")

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
    # Expert models
    # -----------------------------------------------------------------------

    def save_expert(self, expert: ExpertModel) -> str:
        self._conn.execute(
            """INSERT OR REPLACE INTO expert_models
               (expert_id, name, model_family, specializations, instruments,
                timeframes, version, checkpoint_path, model_version, phase,
                status, weight, recent_accuracy, recent_sharpe,
                calibration_error, n_predictions, n_correct,
                created_at, updated_at, metadata_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                expert.expert_id,
                expert.name,
                expert.model_family,
                json.dumps(expert.specializations),
                json.dumps(expert.instruments),
                json.dumps(expert.timeframes),
                expert.version,
                expert.checkpoint_path,
                expert.model_version,
                expert.phase,
                expert.status,
                expert.weight,
                expert.recent_accuracy,
                expert.recent_sharpe,
                expert.calibration_error,
                expert.n_predictions,
                expert.n_correct,
                expert.created_at,
                expert.updated_at,
                json.dumps(expert.metadata),
            ),
        )
        self._conn.commit()
        return expert.expert_id

    def get_expert(self, expert_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM expert_models WHERE expert_id = ?", (expert_id,)
        ).fetchone()
        return self._expert_row(row) if row else None

    def list_experts(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM expert_models WHERE status = ? ORDER BY created_at", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM expert_models ORDER BY created_at").fetchall()
        return [self._expert_row(r) for r in rows]

    def _expert_row(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("specializations", "instruments", "timeframes"):
            d[key] = json.loads(d.get(key, "[]") or "[]")
        d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
        return d

    def update_expert_status(self, expert_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE expert_models SET status = ?, updated_at = ? WHERE expert_id = ?",
            (status, datetime.now(UTC).isoformat(), expert_id),
        )
        self._conn.commit()

    def update_expert_weight(self, expert_id: str, weight: float) -> None:
        self._conn.execute(
            "UPDATE expert_models SET weight = ?, updated_at = ? WHERE expert_id = ?",
            (weight, datetime.now(UTC).isoformat(), expert_id),
        )
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Ensemble results
    # -----------------------------------------------------------------------

    def save_ensemble_result(
        self,
        result: dict[str, Any],
        regime: str = "",
        instrument: str = "",
        timeframe: str = "",
        true_class: str | None = None,
    ) -> str:
        result_id = result.get("result_id") or str(uuid.uuid4())
        is_correct = None
        if true_class is not None:
            is_correct = int(result.get("predicted_class") == true_class)
        self._conn.execute(
            """INSERT OR REPLACE INTO ensemble_results
               (result_id, buy_prob, sell_prob, no_trade_prob, confidence,
                uncertainty, predicted_class, strategy, n_experts,
                agreement_score, regime, instrument, timeframe, true_class,
                is_correct, weight_map_json, metadata_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                result_id,
                result.get("buy_prob", 0.33),
                result.get("sell_prob", 0.33),
                result.get("no_trade_prob", 0.34),
                result.get("confidence", 0.5),
                result.get("uncertainty", 0.5),
                result.get("predicted_class", "no_trade"),
                result.get("strategy", ""),
                result.get("n_experts", 0),
                result.get("agreement_score", 0.0),
                regime,
                instrument,
                timeframe,
                true_class,
                is_correct,
                json.dumps(result.get("weight_map", {})),
                json.dumps(result.get("metadata", {})),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()
        return result_id

    def list_ensemble_results(
        self, strategy: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if strategy:
            rows = self._conn.execute(
                "SELECT * FROM ensemble_results WHERE strategy = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (strategy, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM ensemble_results ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["weight_map"] = json.loads(d.pop("weight_map_json", "{}") or "{}")
            d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
            d["is_correct"] = bool(d["is_correct"]) if d["is_correct"] is not None else None
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Meta model state
    # -----------------------------------------------------------------------

    def save_meta_model(self, meta_dict: dict[str, Any]) -> str:
        state_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO meta_model_state
               (state_id, model_id, meta_version, n_training_samples, state_json, saved_at)
               VALUES (?,?,?,?,?,?)""",
            (
                state_id,
                meta_dict.get("model_id", ""),
                meta_dict.get("meta_version", "1.0"),
                meta_dict.get("n_training_samples", 0),
                json.dumps(meta_dict),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()
        return state_id

    def get_latest_meta_model(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT state_json FROM meta_model_state ORDER BY saved_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return json.loads(row["state_json"])

    # -----------------------------------------------------------------------
    # Weight history
    # -----------------------------------------------------------------------

    def save_weight_entry(
        self,
        expert_id: str,
        weight: float,
        regime: str = "",
        instrument: str = "",
        timeframe: str = "",
    ) -> None:
        self._conn.execute(
            """INSERT INTO weight_history
               (entry_id, expert_id, weight, regime, instrument, timeframe, recorded_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                expert_id,
                weight,
                regime,
                instrument,
                timeframe,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def get_weight_history(self, expert_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM weight_history WHERE expert_id = ? " "ORDER BY recorded_at DESC LIMIT ?",
            (expert_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Routing decisions
    # -----------------------------------------------------------------------

    def save_routing_decision(self, decision_dict: dict[str, Any]) -> str:
        did = decision_dict.get("decision_id") or str(uuid.uuid4())
        self._conn.execute(
            """INSERT OR REPLACE INTO routing_decisions
               (decision_id, regime, instrument, timeframe, selected_experts,
                weights_json, routing_reason, n_experts_selected, decided_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                did,
                decision_dict.get("regime", ""),
                decision_dict.get("instrument", ""),
                decision_dict.get("timeframe", ""),
                json.dumps(decision_dict.get("selected_expert_ids", [])),
                json.dumps(decision_dict.get("weights", {})),
                decision_dict.get("routing_reason", ""),
                decision_dict.get("n_experts_selected", 0),
                decision_dict.get("timestamp", datetime.now(UTC).isoformat()),
            ),
        )
        self._conn.commit()
        return did

    def list_routing_decisions(
        self, regime: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if regime:
            rows = self._conn.execute(
                "SELECT * FROM routing_decisions WHERE regime = ? "
                "ORDER BY decided_at DESC LIMIT ?",
                (regime, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM routing_decisions ORDER BY decided_at DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["selected_experts"] = json.loads(d.get("selected_experts", "[]") or "[]")
            d["weights"] = json.loads(d.pop("weights_json", "{}") or "{}")
            result.append(d)
        return result

    # -----------------------------------------------------------------------
    # Performance history
    # -----------------------------------------------------------------------

    def save_performance_record(
        self,
        expert_id: str,
        predicted_class: str,
        true_class: str | None = None,
        profit_pnl: float = 0.0,
        confidence: float = 0.5,
        uncertainty: float = 0.5,
        regime: str = "",
        instrument: str = "",
        timeframe: str = "",
    ) -> str:
        record_id = str(uuid.uuid4())
        is_correct = None
        if true_class is not None:
            is_correct = int(predicted_class == true_class)
        self._conn.execute(
            """INSERT INTO performance_history
               (record_id, expert_id, predicted_class, true_class, profit_pnl,
                confidence, uncertainty, regime, instrument, timeframe,
                is_correct, recorded_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record_id,
                expert_id,
                predicted_class,
                true_class,
                profit_pnl,
                confidence,
                uncertainty,
                regime,
                instrument,
                timeframe,
                is_correct,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()
        return record_id

    def get_performance_history(self, expert_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM performance_history WHERE expert_id = ? "
            "ORDER BY recorded_at DESC LIMIT ?",
            (expert_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Voting history
    # -----------------------------------------------------------------------

    def save_vote(
        self, result_id: str, expert_id: str, voted_class: str, confidence: float, weight: float
    ) -> None:
        self._conn.execute(
            """INSERT INTO voting_history
               (vote_id, result_id, expert_id, voted_class, confidence, weight, recorded_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                result_id,
                expert_id,
                voted_class,
                confidence,
                weight,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def get_voting_history(self, result_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM voting_history WHERE result_id = ? ORDER BY recorded_at", (result_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        n_experts = self._conn.execute("SELECT COUNT(*) as n FROM expert_models").fetchone()["n"]
        n_active = self._conn.execute(
            "SELECT COUNT(*) as n FROM expert_models WHERE status IN ('active','champion','experimental')"
        ).fetchone()["n"]
        n_results = self._conn.execute("SELECT COUNT(*) as n FROM ensemble_results").fetchone()["n"]
        n_correct = self._conn.execute(
            "SELECT COUNT(*) as n FROM ensemble_results WHERE is_correct = 1"
        ).fetchone()["n"]
        n_weights = self._conn.execute("SELECT COUNT(*) as n FROM weight_history").fetchone()["n"]
        n_routes = self._conn.execute("SELECT COUNT(*) as n FROM routing_decisions").fetchone()["n"]
        return {
            "total_experts": n_experts,
            "active_experts": n_active,
            "total_ensemble_results": n_results,
            "correct_ensemble_results": n_correct,
            "ensemble_accuracy": n_correct / n_results if n_results > 0 else 0.0,
            "total_weight_entries": n_weights,
            "total_routing_decisions": n_routes,
        }

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
