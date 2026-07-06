"""Phase 21.6 — Decision Memory.

Permanent in-process registry of every decision Signalium produces.
Every recommendation generates a DecisionRecord with a globally unique ID.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_types import DecisionType, OutcomeType, MarketRegime
from trainer.memory.decision_models import (
    DecisionRecord,
    DecisionSnapshot,
    DecisionReasoning,
    DecisionOutcome,
)


class DecisionMemory:
    """Permanent registry of all Signalium decisions.

    All decisions are kept in memory (self._records) and can be persisted
    to the MemoryDatabase separately.
    """

    def __init__(self):
        self._records: Dict[str, DecisionRecord] = {}

    # ── Storage ───────────────────────────────────────────────────────────────

    def store(self, record: DecisionRecord) -> str:
        """Store a decision record. Returns decision_id."""
        self._records[record.decision_id] = record
        logger.info(
            f"DecisionMemory: stored {record.decision_type.value} "
            f"[{record.instrument}] id={record.decision_id[:8]}"
        )
        return record.decision_id

    def record_decision(
        self,
        decision_type: DecisionType,
        instrument: str,
        snapshot: DecisionSnapshot,
        reasoning: DecisionReasoning,
        tags: Optional[List[str]] = None,
        trade_id: str = "",
        lineage_id: str = "",
    ) -> DecisionRecord:
        """Create and store a new decision record. Returns the record."""
        record = DecisionRecord.create(
            decision_type=decision_type,
            instrument=instrument,
            snapshot=snapshot,
            reasoning=reasoning,
            tags=tags,
            trade_id=trade_id,
            lineage_id=lineage_id,
        )
        self.store(record)
        return record

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, decision_id: str) -> Optional[DecisionRecord]:
        return self._records.get(decision_id)

    def all_records(self) -> List[DecisionRecord]:
        return list(self._records.values())

    def total(self) -> int:
        return len(self._records)

    # ── Outcome attachment ────────────────────────────────────────────────────

    def record_outcome(
        self,
        decision_id: str,
        outcome_type: OutcomeType,
        actual_pnl: float = 0.0,
        actual_pnl_pct: float = 0.0,
        hit_tp: bool = False,
        hit_sl: bool = False,
        holding_time_h: float = 0.0,
        direction_correct: Optional[bool] = None,
        market_event: str = "",
        notes: str = "",
    ) -> bool:
        """Attach an outcome to an existing decision. Returns True if found."""
        record = self._records.get(decision_id)
        if not record:
            logger.warning(f"DecisionMemory: outcome for unknown id {decision_id}")
            return False

        record.outcome = DecisionOutcome(
            outcome_type=outcome_type,
            actual_direction_correct=direction_correct,
            actual_pnl=actual_pnl,
            actual_pnl_pct=actual_pnl_pct,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
            holding_time_h=holding_time_h,
            market_event=market_event,
            notes=notes,
            recorded_at=time.time(),
        )
        logger.debug(f"DecisionMemory: outcome {outcome_type.value} attached to {decision_id[:8]}")
        return True

    # ── Filtering ─────────────────────────────────────────────────────────────

    def filter(
        self,
        instrument: Optional[str] = None,
        decision_type: Optional[DecisionType] = None,
        regime: Optional[str] = None,
        session: Optional[str] = None,
        has_outcome: Optional[bool] = None,
        min_confidence: float = 0.0,
        since: Optional[float] = None,
        limit: int = 500,
    ) -> List[DecisionRecord]:
        """Filter decisions by criteria. Returns matching records newest-first."""
        results = []
        for r in self._records.values():
            if instrument and r.instrument != instrument:
                continue
            if decision_type and r.decision_type != decision_type:
                continue
            if regime and r.snapshot.market_regime != regime:
                continue
            if session and r.snapshot.session != session:
                continue
            if has_outcome is not None and r.has_outcome != has_outcome:
                continue
            if r.snapshot.forecast_confidence < min_confidence:
                continue
            if since and r.timestamp < since:
                continue
            results.append(r)

        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]

    def decisions_with_outcomes(self) -> List[DecisionRecord]:
        return [r for r in self._records.values() if r.has_outcome]

    def decisions_without_outcomes(self) -> List[DecisionRecord]:
        return [r for r in self._records.values() if not r.has_outcome]

    def by_instrument(self, instrument: str) -> List[DecisionRecord]:
        return self.filter(instrument=instrument)

    def by_type(self, decision_type: DecisionType) -> List[DecisionRecord]:
        return self.filter(decision_type=decision_type)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        records = self.all_records()
        n = len(records)
        with_outcome = sum(1 for r in records if r.has_outcome)
        with_score = sum(1 for r in records if r.has_score)
        type_dist = {}
        for r in records:
            k = r.decision_type.value
            type_dist[k] = type_dist.get(k, 0) + 1
        return {
            "total_decisions": n,
            "with_outcome": with_outcome,
            "without_outcome": n - with_outcome,
            "with_score": with_score,
            "outcome_rate": round(with_outcome / n, 4) if n else 0.0,
            "decision_type_dist": type_dist,
        }
