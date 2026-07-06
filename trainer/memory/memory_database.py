"""Phase 21.6 — Memory Database.

SQLite persistence for the Decision Memory system.
9 tables: decision_memory, decision_snapshots, decision_reasoning,
          decision_outcomes, counterfactual_analysis, decision_scores,
          decision_patterns, knowledge_base, decision_lineage.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_models import (
    DecisionRecord,
    DecisionSnapshot,
    DecisionReasoning,
    DecisionOutcome,
    DecisionScore,
    CounterfactualResult,
)
from trainer.memory.decision_types import (
    DecisionType,
    OutcomeType,
    DecisionGrade,
)

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS decision_memory (
    decision_id   TEXT PRIMARY KEY,
    decision_type TEXT NOT NULL,
    instrument    TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    trade_id      TEXT DEFAULT '',
    lineage_id    TEXT DEFAULT '',
    tags          TEXT DEFAULT '[]',
    notes         TEXT DEFAULT '',
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS decision_snapshots (
    decision_id        TEXT PRIMARY KEY,
    price              REAL,
    atr                REAL,
    spread             REAL,
    volatility         REAL,
    session            TEXT,
    liquidity_estimate REAL,
    market_regime      TEXT,
    volatility_regime  TEXT,
    timeframe          TEXT,
    forecast_direction TEXT,
    forecast_confidence REAL,
    expected_move      REAL,
    expected_holding_h REAL,
    tp_probability     REAL,
    sl_probability     REAL,
    model_version      TEXT,
    champion_model     TEXT,
    research_score     REAL,
    ensemble_output    REAL,
    portfolio_exposure REAL,
    correlation_risk   REAL,
    risk_metrics       TEXT DEFAULT '{}',
    feature_snapshot   TEXT DEFAULT '{}',
    expert_weights     TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS decision_reasoning (
    decision_id            TEXT PRIMARY KEY,
    primary_reasons        TEXT DEFAULT '[]',
    supporting_factors     TEXT DEFAULT '[]',
    contradicting_factors  TEXT DEFAULT '[]',
    risk_factors           TEXT DEFAULT '[]',
    uncertainty_sources    TEXT DEFAULT '[]',
    alternatives_considered TEXT DEFAULT '[]',
    final_rationale        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS decision_outcomes (
    decision_id              TEXT PRIMARY KEY,
    outcome_type             TEXT NOT NULL,
    actual_direction_correct INTEGER,
    actual_pnl               REAL DEFAULT 0.0,
    actual_pnl_pct           REAL DEFAULT 0.0,
    hit_tp                   INTEGER DEFAULT 0,
    hit_sl                   INTEGER DEFAULT 0,
    holding_time_h           REAL DEFAULT 0.0,
    market_event             TEXT DEFAULT '',
    notes                    TEXT DEFAULT '',
    recorded_at              REAL
);

CREATE TABLE IF NOT EXISTS counterfactual_analysis (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id        TEXT NOT NULL,
    scenario           TEXT,
    alternative_action TEXT,
    estimated_pnl      REAL,
    estimated_pnl_pct  REAL,
    probability        REAL,
    better_than_actual INTEGER DEFAULT 0,
    delta_pnl          REAL,
    notes              TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS decision_scores (
    decision_id  TEXT PRIMARY KEY,
    overall      REAL,
    grade        TEXT,
    sub_scores   TEXT DEFAULT '{}',
    notes        TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS decision_patterns (
    pattern_id   TEXT PRIMARY KEY,
    category     TEXT,
    description  TEXT,
    evidence     TEXT,
    win_rate     REAL,
    avg_pnl      REAL,
    sample_size  INTEGER,
    strength     TEXT,
    direction    TEXT,
    created_at   REAL
);

CREATE TABLE IF NOT EXISTS knowledge_base (
    entry_id     TEXT PRIMARY KEY,
    entry_type   TEXT,
    title        TEXT,
    description  TEXT,
    tags         TEXT DEFAULT '[]',
    links        TEXT DEFAULT '[]',
    decision_ids TEXT DEFAULT '[]',
    metadata     TEXT DEFAULT '{}',
    created_at   REAL
);

CREATE TABLE IF NOT EXISTS decision_lineage (
    lineage_id   TEXT PRIMARY KEY,
    decision_id  TEXT NOT NULL,
    nodes        TEXT DEFAULT '[]',
    created_at   REAL
);

CREATE INDEX IF NOT EXISTS idx_dm_instrument  ON decision_memory(instrument);
CREATE INDEX IF NOT EXISTS idx_dm_type        ON decision_memory(decision_type);
CREATE INDEX IF NOT EXISTS idx_dm_timestamp   ON decision_memory(timestamp);
CREATE INDEX IF NOT EXISTS idx_do_type        ON decision_outcomes(outcome_type);
CREATE INDEX IF NOT EXISTS idx_dl_decision    ON decision_lineage(decision_id);
"""


class MemoryDatabase:
    """SQLite-backed persistence for the Decision Memory system."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._persistent_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_TABLES)

    @contextlib.contextmanager
    def _conn(self):
        if self._db_path == ":memory:":
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:")
                self._persistent_conn.row_factory = sqlite3.Row
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_decision(self, record: DecisionRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO decision_memory
                   (decision_id, decision_type, instrument, timestamp, trade_id,
                    lineage_id, tags, notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    record.decision_id,
                    record.decision_type.value,
                    record.instrument,
                    record.timestamp,
                    record.trade_id,
                    record.lineage_id,
                    json.dumps(record.tags),
                    record.notes,
                    time.time(),
                ),
            )
            snap = record.snapshot
            conn.execute(
                """INSERT OR REPLACE INTO decision_snapshots
                   (decision_id, price, atr, spread, volatility, session,
                    liquidity_estimate, market_regime, volatility_regime, timeframe,
                    forecast_direction, forecast_confidence, expected_move,
                    expected_holding_h, tp_probability, sl_probability,
                    model_version, champion_model, research_score, ensemble_output,
                    portfolio_exposure, correlation_risk, risk_metrics,
                    feature_snapshot, expert_weights)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.decision_id,
                    snap.price,
                    snap.atr,
                    snap.spread,
                    snap.volatility,
                    snap.session,
                    snap.liquidity_estimate,
                    snap.market_regime,
                    snap.volatility_regime,
                    snap.timeframe,
                    snap.forecast_direction,
                    snap.forecast_confidence,
                    snap.expected_move,
                    snap.expected_holding_h,
                    snap.tp_probability,
                    snap.sl_probability,
                    snap.model_version,
                    snap.champion_model,
                    snap.research_score,
                    snap.ensemble_output,
                    snap.portfolio_exposure,
                    snap.correlation_risk,
                    json.dumps(snap.risk_metrics),
                    json.dumps(snap.feature_snapshot),
                    json.dumps(snap.expert_weights),
                ),
            )
            rea = record.reasoning
            conn.execute(
                """INSERT OR REPLACE INTO decision_reasoning
                   (decision_id, primary_reasons, supporting_factors,
                    contradicting_factors, risk_factors, uncertainty_sources,
                    alternatives_considered, final_rationale)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    record.decision_id,
                    json.dumps(rea.primary_reasons),
                    json.dumps(rea.supporting_factors),
                    json.dumps(rea.contradicting_factors),
                    json.dumps(rea.risk_factors),
                    json.dumps(rea.uncertainty_sources),
                    json.dumps(rea.alternatives_considered),
                    rea.final_rationale,
                ),
            )
            if record.outcome:
                self._save_outcome(conn, record.decision_id, record.outcome)
            if record.score:
                self._save_score(conn, record.decision_id, record.score)
            for cf in record.counterfactuals:
                self._save_counterfactual(conn, record.decision_id, cf)
            if self._db_path == ":memory:":
                conn.commit()

    def _save_outcome(self, conn, decision_id: str, out: DecisionOutcome) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO decision_outcomes
               (decision_id, outcome_type, actual_direction_correct,
                actual_pnl, actual_pnl_pct, hit_tp, hit_sl,
                holding_time_h, market_event, notes, recorded_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                decision_id,
                out.outcome_type.value,
                (
                    (1 if out.actual_direction_correct else 0)
                    if out.actual_direction_correct is not None
                    else None
                ),
                out.actual_pnl,
                out.actual_pnl_pct,
                int(out.hit_tp),
                int(out.hit_sl),
                out.holding_time_h,
                out.market_event,
                out.notes,
                out.recorded_at,
            ),
        )

    def _save_score(self, conn, decision_id: str, score: DecisionScore) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO decision_scores
               (decision_id, overall, grade, sub_scores, notes)
               VALUES (?,?,?,?,?)""",
            (
                decision_id,
                score.overall,
                score.grade.value,
                json.dumps(score.sub_scores),
                json.dumps(score.notes),
            ),
        )

    def _save_counterfactual(self, conn, decision_id: str, cf: CounterfactualResult) -> None:
        conn.execute(
            """INSERT INTO counterfactual_analysis
               (decision_id, scenario, alternative_action, estimated_pnl,
                estimated_pnl_pct, probability, better_than_actual, delta_pnl, notes)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                decision_id,
                cf.scenario,
                cf.alternative_action,
                cf.estimated_pnl,
                cf.estimated_pnl_pct,
                cf.probability,
                int(cf.better_than_actual),
                cf.delta_pnl,
                cf.notes,
            ),
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def load_decisions(
        self,
        instrument: Optional[str] = None,
        decision_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        clauses = []
        params: List[Any] = []
        if instrument:
            clauses.append("instrument = ?")
            params.append(instrument)
        if decision_type:
            clauses.append("decision_type = ?")
            params.append(decision_type)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM decision_memory {where} " f"ORDER BY timestamp DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_decisions(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM decision_memory").fetchone()[0]

    def count_outcomes(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM decision_outcomes").fetchone()[0]

    def count_scores(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM decision_scores").fetchone()[0]

    def count_counterfactuals(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM counterfactual_analysis").fetchone()[0]

    def decision_ids(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT decision_id FROM decision_memory").fetchall()
        return [r[0] for r in rows]

    def stats(self) -> Dict:
        return {
            "decisions": self.count_decisions(),
            "outcomes": self.count_outcomes(),
            "scores": self.count_scores(),
            "counterfactuals": self.count_counterfactuals(),
        }
