"""Phase 24.9 — MIE Database (SQLite, 10 tables, historical analytics)."""

from __future__ import annotations

import contextlib
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

_SCHEMA = """
-- 1. Master intelligence records
CREATE TABLE IF NOT EXISTS market_intelligence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id       TEXT    NOT NULL UNIQUE,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    tradeability    REAL,
    trade_grade     TEXT,
    market_quality  REAL,
    execution_quality REAL,
    regime          TEXT,
    risk_regime     TEXT,
    summary         TEXT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_mi_symbol_ts ON market_intelligence(symbol, timestamp);

-- 2. Market regime history
CREATE TABLE IF NOT EXISTS market_regime_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    regime          TEXT    NOT NULL,
    confidence      REAL,
    sub_regime      TEXT,
    adx             REAL,
    vol_ratio       REAL,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_mrh_symbol_ts ON market_regime_history(symbol, timestamp);

-- 3. Risk history
CREATE TABLE IF NOT EXISTS risk_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    risk_regime     TEXT    NOT NULL,
    risk_score      REAL,
    multiplier      REAL,
    vol_factor      REAL,
    spread_factor   REAL,
    liquidity_factor REAL,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_rh_symbol_ts ON risk_history(symbol, timestamp);

-- 4. Tradeability history
CREATE TABLE IF NOT EXISTS tradeability_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    score           REAL    NOT NULL,
    grade           TEXT    NOT NULL,
    confidence_f    REAL,
    rr_f            REAL,
    liquidity_f     REAL,
    risk_f          REAL,
    regime_f        REAL,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_th_symbol_ts ON tradeability_history(symbol, timestamp);

-- 5. Market quality scores
CREATE TABLE IF NOT EXISTS market_quality (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    market_quality  REAL,
    execution_quality REAL,
    prediction_quality REAL,
    liquidity_quality REAL,
    volatility_quality REAL,
    momentum_quality REAL,
    trend_quality   REAL,
    session_quality REAL,
    news_impact     REAL,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_mq_symbol_ts ON market_quality(symbol, timestamp);

-- 6. Execution quality records
CREATE TABLE IF NOT EXISTS execution_quality (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    mode            TEXT,
    entry_low       REAL,
    entry_high      REAL,
    optimal_stop    REAL,
    optimal_target  REAL,
    expected_dd     REAL,
    max_hold_bars   INTEGER,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_eq_symbol_ts ON execution_quality(symbol, timestamp);

-- 7. Prediction reliability
CREATE TABLE IF NOT EXISTS prediction_reliability (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    reliability     REAL,
    confidence_score REAL,
    consistency     REAL,
    model_health    REAL,
    win_rate_adj    REAL,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_pr_symbol_ts ON prediction_reliability(symbol, timestamp);

-- 8. Behavior similarity
CREATE TABLE IF NOT EXISTS behavior_similarity (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    similarity      REAL,
    historical_win  REAL,
    match_count     INTEGER,
    best_match      REAL,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_bs_symbol_ts ON behavior_similarity(symbol, timestamp);

-- 9. AI commentary
CREATE TABLE IF NOT EXISTS ai_commentary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    summary         TEXT,
    regime_note     TEXT,
    risk_note       TEXT,
    trade_note      TEXT,
    recommendation  TEXT,
    confidence      REAL,
    tags            TEXT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_ac_symbol_ts ON ai_commentary(symbol, timestamp);

-- 10. Composite market scores (denormalised for analytics)
CREATE TABLE IF NOT EXISTS market_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    tradeability    REAL,
    market_quality  REAL,
    risk_score      REAL,
    regime          TEXT,
    risk_regime     TEXT,
    session         TEXT,
    news_impact     REAL,
    volatility_state TEXT,
    momentum_state  TEXT,
    trend_state     TEXT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_ms_symbol_ts ON market_scores(symbol, timestamp);
"""


class MIEDatabase:
    """SQLite persistence layer for the Market Intelligence Engine."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._path = db_path
        if db_path == ":memory:":
            self._persistent_conn: Optional[sqlite3.Connection] = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.executescript(_SCHEMA)
            self._persistent_conn.commit()
        else:
            self._persistent_conn = None
            with self._conn() as conn:
                conn.executescript(_SCHEMA)
        logger.debug(f"[MIEDB] Initialised at {db_path!r}")

    # ── Context manager ────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._persistent_conn is not None:
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ── Write methods ──────────────────────────────────────────────────────

    def save_intelligence(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO market_intelligence
            (report_id, symbol, timestamp, tradeability, trade_grade,
             market_quality, execution_quality, regime, risk_regime, summary)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("report_id", ""),
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("tradeability"),
                rec.get("trade_grade"),
                rec.get("market_quality"),
                rec.get("execution_quality"),
                rec.get("regime"),
                rec.get("risk_regime"),
                rec.get("summary"),
            ))
            return cur.lastrowid or 0

    def save_regime(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO market_regime_history
            (symbol, timestamp, regime, confidence, sub_regime, adx, vol_ratio)
        VALUES (?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("regime", "UNKNOWN"),
                rec.get("confidence"),
                rec.get("sub_regime"),
                rec.get("adx"),
                rec.get("vol_ratio"),
            ))
            return cur.lastrowid or 0

    def save_risk(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO risk_history
            (symbol, timestamp, risk_regime, risk_score, multiplier,
             vol_factor, spread_factor, liquidity_factor)
        VALUES (?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("risk_regime", "NORMAL"),
                rec.get("risk_score"),
                rec.get("multiplier"),
                rec.get("vol_factor"),
                rec.get("spread_factor"),
                rec.get("liquidity_factor"),
            ))
            return cur.lastrowid or 0

    def save_tradeability(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO tradeability_history
            (symbol, timestamp, score, grade, confidence_f, rr_f,
             liquidity_f, risk_f, regime_f)
        VALUES (?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("score", 0.0),
                rec.get("grade", "Avoid"),
                rec.get("confidence_f"),
                rec.get("rr_f"),
                rec.get("liquidity_f"),
                rec.get("risk_f"),
                rec.get("regime_f"),
            ))
            return cur.lastrowid or 0

    def save_quality(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO market_quality
            (symbol, timestamp, market_quality, execution_quality,
             prediction_quality, liquidity_quality, volatility_quality,
             momentum_quality, trend_quality, session_quality, news_impact)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("market_quality"),
                rec.get("execution_quality"),
                rec.get("prediction_quality"),
                rec.get("liquidity_quality"),
                rec.get("volatility_quality"),
                rec.get("momentum_quality"),
                rec.get("trend_quality"),
                rec.get("session_quality"),
                rec.get("news_impact"),
            ))
            return cur.lastrowid or 0

    def save_execution(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO execution_quality
            (symbol, timestamp, mode, entry_low, entry_high,
             optimal_stop, optimal_target, expected_dd, max_hold_bars)
        VALUES (?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("mode"),
                rec.get("entry_low"),
                rec.get("entry_high"),
                rec.get("optimal_stop"),
                rec.get("optimal_target"),
                rec.get("expected_dd"),
                rec.get("max_hold_bars"),
            ))
            return cur.lastrowid or 0

    def save_reliability(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO prediction_reliability
            (symbol, timestamp, reliability, confidence_score,
             consistency, model_health, win_rate_adj)
        VALUES (?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("reliability"),
                rec.get("confidence_score"),
                rec.get("consistency"),
                rec.get("model_health"),
                rec.get("win_rate_adj"),
            ))
            return cur.lastrowid or 0

    def save_similarity(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO behavior_similarity
            (symbol, timestamp, similarity, historical_win, match_count, best_match)
        VALUES (?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("similarity"),
                rec.get("historical_win"),
                rec.get("match_count"),
                rec.get("best_match"),
            ))
            return cur.lastrowid or 0

    def save_commentary(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO ai_commentary
            (symbol, timestamp, summary, regime_note, risk_note,
             trade_note, recommendation, confidence, tags)
        VALUES (?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("summary"),
                rec.get("regime_note"),
                rec.get("risk_note"),
                rec.get("trade_note"),
                rec.get("recommendation"),
                rec.get("confidence"),
                ",".join(rec.get("tags", [])),
            ))
            return cur.lastrowid or 0

    def save_scores(self, rec: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO market_scores
            (symbol, timestamp, tradeability, market_quality, risk_score,
             regime, risk_regime, session, news_impact,
             volatility_state, momentum_state, trend_state)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                rec.get("symbol", ""),
                rec.get("timestamp", time.time()),
                rec.get("tradeability"),
                rec.get("market_quality"),
                rec.get("risk_score"),
                rec.get("regime"),
                rec.get("risk_regime"),
                rec.get("session"),
                rec.get("news_impact"),
                rec.get("volatility_state"),
                rec.get("momentum_state"),
                rec.get("trend_state"),
            ))
            return cur.lastrowid or 0

    # ── Read methods ───────────────────────────────────────────────────────

    def get_latest_intelligence(
        self, symbol: str, limit: int = 1
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM market_intelligence WHERE symbol=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_regime_history(
        self, symbol: str, since: float = 0.0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM market_regime_history WHERE symbol=? AND timestamp>=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (symbol, since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_tradeability_history(
        self, symbol: str, since: float = 0.0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tradeability_history WHERE symbol=? AND timestamp>=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (symbol, since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_risk_history(
        self, symbol: str, since: float = 0.0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM risk_history WHERE symbol=? AND timestamp>=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (symbol, since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_quality_history(
        self, symbol: str, since: float = 0.0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM market_quality WHERE symbol=? AND timestamp>=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (symbol, since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_commentary_history(
        self, symbol: str, since: float = 0.0, limit: int = 20
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM ai_commentary WHERE symbol=? AND timestamp>=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (symbol, since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_scores_history(
        self, symbol: str, since: float = 0.0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM market_scores WHERE symbol=? AND timestamp>=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (symbol, since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def table_count(self, table: str) -> int:
        with self._conn() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return row[0] if row else 0
