"""Phase 22 — Microstructure Database.

SQLite persistence for the MMIE engine.
6 tables: microstructure_features, microstructure_embeddings,
          market_states, microstructure_scores,
          liquidity_estimates, momentum_metrics.
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

from trainer.microstructure.micro_models import (
    MicrostructureFeatures,
    MicrostructureEmbedding,
    MarketStateRecord,
    MicrostructureScore,
)

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS microstructure_features (
    bar_id      TEXT PRIMARY KEY,
    timestamp   REAL NOT NULL,
    instrument  TEXT,
    timeframe   TEXT,
    granularity TEXT,
    features_json TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS microstructure_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,
    instrument  TEXT,
    timeframe   TEXT,
    embedding   TEXT NOT NULL,
    dim         INTEGER,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS market_states (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    instrument      TEXT,
    timeframe       TEXT,
    state           TEXT NOT NULL,
    confidence      REAL,
    momentum_state  TEXT,
    volatility_state TEXT,
    liquidity_state TEXT,
    trend_strength  TEXT,
    reasoning       TEXT DEFAULT '[]',
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS microstructure_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    instrument      TEXT,
    timeframe       TEXT,
    trend_score     REAL,
    liquidity_score REAL,
    momentum_score  REAL,
    efficiency_score REAL,
    volatility_score REAL,
    compression_score REAL,
    expansion_score REAL,
    overall_score   REAL,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS liquidity_estimates (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp              REAL NOT NULL,
    instrument             TEXT,
    timeframe              TEXT,
    spread_proxy           REAL,
    gap_frequency          REAL,
    large_candle_frequency REAL,
    breakout_strength      REAL,
    reversal_strength      REAL,
    liquidity_vacuum_prob  REAL,
    compression_zone_score REAL,
    liquidity_state        TEXT,
    created_at             REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS momentum_metrics (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                REAL NOT NULL,
    instrument               TEXT,
    timeframe                TEXT,
    momentum_acceleration    REAL,
    momentum_deceleration    REAL,
    momentum_exhaustion      REAL,
    directional_persistence  REAL,
    trend_energy             REAL,
    impulse_decay            REAL,
    momentum_state           TEXT,
    created_at               REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mf_instrument  ON microstructure_features(instrument);
CREATE INDEX IF NOT EXISTS idx_mf_timestamp   ON microstructure_features(timestamp);
CREATE INDEX IF NOT EXISTS idx_ms_state       ON market_states(state);
CREATE INDEX IF NOT EXISTS idx_ms_instrument  ON market_states(instrument);
CREATE INDEX IF NOT EXISTS idx_ms_timestamp   ON market_states(timestamp);
"""


class MicrostructureDatabase:
    """SQLite-backed persistence for the MMIE."""

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

    def _commit_if_memory(self, conn) -> None:
        if self._db_path == ":memory:":
            conn.commit()

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_features(self, features: MicrostructureFeatures) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO microstructure_features
                   (bar_id, timestamp, instrument, timeframe, granularity,
                    features_json, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    features.bar_id,
                    features.timestamp,
                    features.instrument,
                    features.timeframe,
                    features.granularity,
                    json.dumps(features.to_dict()),
                    time.time(),
                ),
            )
            self._commit_if_memory(conn)

    def save_embedding(self, emb: MicrostructureEmbedding) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO microstructure_embeddings
                   (timestamp, instrument, timeframe, embedding, dim, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    emb.timestamp,
                    emb.instrument,
                    emb.timeframe,
                    json.dumps(emb.embedding),
                    emb.dim,
                    time.time(),
                ),
            )
            self._commit_if_memory(conn)

    def save_market_state(
        self,
        state_record: MarketStateRecord,
        instrument: str = "",
        timeframe: str = "H1",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO market_states
                   (timestamp, instrument, timeframe, state, confidence,
                    momentum_state, volatility_state, liquidity_state,
                    trend_strength, reasoning, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    state_record.timestamp,
                    instrument,
                    timeframe,
                    state_record.state.value,
                    state_record.confidence,
                    state_record.momentum_state.value,
                    state_record.volatility_state.value,
                    state_record.liquidity_state.value,
                    state_record.trend_strength.value,
                    json.dumps(state_record.reasoning),
                    time.time(),
                ),
            )
            self._commit_if_memory(conn)

    def save_score(
        self,
        score: MicrostructureScore,
        timestamp: float,
        instrument: str = "",
        timeframe: str = "H1",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO microstructure_scores
                   (timestamp, instrument, timeframe, trend_score, liquidity_score,
                    momentum_score, efficiency_score, volatility_score,
                    compression_score, expansion_score, overall_score, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    timestamp,
                    instrument,
                    timeframe,
                    score.trend_score,
                    score.liquidity_score,
                    score.momentum_score,
                    score.efficiency_score,
                    score.volatility_score,
                    score.compression_score,
                    score.expansion_score,
                    score.overall_score,
                    time.time(),
                ),
            )
            self._commit_if_memory(conn)

    def save_liquidity(
        self,
        features: MicrostructureFeatures,
        liq_state: str = "NORMAL",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO liquidity_estimates
                   (timestamp, instrument, timeframe, spread_proxy, gap_frequency,
                    large_candle_frequency, breakout_strength, reversal_strength,
                    liquidity_vacuum_prob, compression_zone_score, liquidity_state, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    features.timestamp,
                    features.instrument,
                    features.timeframe,
                    features.spread_proxy,
                    features.gap_frequency,
                    features.large_candle_frequency,
                    features.breakout_strength,
                    features.reversal_strength,
                    features.liquidity_vacuum_prob,
                    features.compression_zone_score,
                    liq_state,
                    time.time(),
                ),
            )
            self._commit_if_memory(conn)

    def save_momentum(
        self,
        features: MicrostructureFeatures,
        mom_state: str = "NEUTRAL",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO momentum_metrics
                   (timestamp, instrument, timeframe, momentum_acceleration,
                    momentum_deceleration, momentum_exhaustion, directional_persistence,
                    trend_energy, impulse_decay, momentum_state, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    features.timestamp,
                    features.instrument,
                    features.timeframe,
                    features.momentum_acceleration,
                    features.momentum_deceleration,
                    features.momentum_exhaustion,
                    features.directional_persistence,
                    features.trend_energy,
                    features.impulse_decay,
                    mom_state,
                    time.time(),
                ),
            )
            self._commit_if_memory(conn)

    # ── Query ─────────────────────────────────────────────────────────────────

    def count_features(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM microstructure_features").fetchone()[0]

    def count_market_states(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM market_states").fetchone()[0]

    def count_scores(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM microstructure_scores").fetchone()[0]

    def count_embeddings(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM microstructure_embeddings").fetchone()[0]

    def recent_states(
        self,
        instrument: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        params: List[Any] = []
        where = ""
        if instrument:
            where = "WHERE instrument = ?"
            params.append(instrument)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM market_states {where} ORDER BY timestamp DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def state_distribution(self, instrument: Optional[str] = None) -> Dict[str, int]:
        params: List[Any] = []
        where = ""
        if instrument:
            where = "WHERE instrument = ?"
            params.append(instrument)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT state, COUNT(*) as cnt FROM market_states {where} GROUP BY state",
                params,
            ).fetchall()
        return {r["state"]: r["cnt"] for r in rows}

    def stats(self) -> Dict:
        return {
            "features": self.count_features(),
            "embeddings": self.count_embeddings(),
            "market_states": self.count_market_states(),
            "scores": self.count_scores(),
        }
