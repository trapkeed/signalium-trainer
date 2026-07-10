"""S33 — SQLite persistence for opportunity score history.

Follows the same conventions as ``trainer/memory/memory_database.py`` and
``trainer/setup_memory/store.py``. Kept as its own table (rather than
bolted onto ``decision_memory``) because a score can be computed many
times for the same setup as it matures — this is a time series, not a
one-row-per-decision record — but every row optionally carries a
``decision_id``/``lineage_id`` so it can be joined back into Decision
Memory / Decision Lineage for research use.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from trainer.qde.opportunity_scoring.models import (
    OpportunityBand,
    OpportunityScoreResult,
)

DEFAULT_DB_PATH = "data/opportunity_scores.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS opportunity_score_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol         TEXT NOT NULL,
    timeframe      TEXT NOT NULL,
    broker_symbol  TEXT NOT NULL DEFAULT '',
    setup_state    TEXT NOT NULL DEFAULT '',
    score          REAL NOT NULL,
    band           TEXT NOT NULL,
    sub_scores     TEXT NOT NULL DEFAULT '{}',
    weights        TEXT NOT NULL DEFAULT '{}',
    decision_id    TEXT NOT NULL DEFAULT '',
    lineage_id     TEXT NOT NULL DEFAULT '',
    scored_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_opportunity_score_history_key
    ON opportunity_score_history(symbol, timeframe, scored_at);
CREATE INDEX IF NOT EXISTS idx_opportunity_score_history_decision
    ON opportunity_score_history(decision_id);
"""


class OpportunityScoreStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def append(self, result: OpportunityScoreResult) -> int:
        cur = self._conn.execute(
            "INSERT INTO opportunity_score_history "
            "(symbol, timeframe, broker_symbol, setup_state, score, band, sub_scores, "
            "weights, decision_id, lineage_id, scored_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.symbol,
                result.timeframe,
                result.broker_symbol,
                result.setup_state,
                result.score,
                result.band.value,
                json.dumps(result.sub_scores),
                json.dumps(result.weights),
                result.decision_id,
                result.lineage_id,
                result.scored_at,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def history(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[OpportunityScoreResult]:
        cur = self._conn.execute(
            "SELECT symbol, timeframe, broker_symbol, setup_state, score, band, "
            "sub_scores, weights, decision_id, lineage_id, scored_at "
            "FROM opportunity_score_history WHERE symbol = ? AND timeframe = ? "
            "ORDER BY id DESC LIMIT ?",
            (symbol, timeframe, limit),
        )
        return [self._row_to_result(row) for row in cur.fetchall()]

    def by_decision(self, decision_id: str) -> list[OpportunityScoreResult]:
        cur = self._conn.execute(
            "SELECT symbol, timeframe, broker_symbol, setup_state, score, band, "
            "sub_scores, weights, decision_id, lineage_id, scored_at "
            "FROM opportunity_score_history WHERE decision_id = ? ORDER BY id ASC",
            (decision_id,),
        )
        return [self._row_to_result(row) for row in cur.fetchall()]

    def recent(self, limit: int = 100) -> list[OpportunityScoreResult]:
        cur = self._conn.execute(
            "SELECT symbol, timeframe, broker_symbol, setup_state, score, band, "
            "sub_scores, weights, decision_id, lineage_id, scored_at "
            "FROM opportunity_score_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_result(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_result(row: tuple[Any, ...]) -> OpportunityScoreResult:
        return OpportunityScoreResult(
            symbol=row[0],
            timeframe=row[1],
            broker_symbol=row[2],
            setup_state=row[3],
            score=row[4],
            band=OpportunityBand(row[5]),
            sub_scores=json.loads(row[6]),
            weights=json.loads(row[7]),
            decision_id=row[8],
            lineage_id=row[9],
            scored_at=row[10],
        )
