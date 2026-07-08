"""Phase 21.6 — Outcome Tracker.

Records and analyses outcomes across all decisions in the Decision Memory.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_models import DecisionRecord
from trainer.memory.decision_types import NEGATIVE_OUTCOMES, POSITIVE_OUTCOMES, DecisionType


@dataclass
class OutcomeSummary:
    """Aggregate outcome statistics across a set of decisions."""

    total: int = 0
    with_outcome: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    avg_holding_h: float = 0.0
    hit_tp_count: int = 0
    hit_sl_count: int = 0
    outcome_distribution: dict[str, int] = field(default_factory=dict)
    by_instrument: dict[str, dict] = field(default_factory=dict)
    by_session: dict[str, dict] = field(default_factory=dict)
    by_regime: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "with_outcome": self.with_outcome,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "win_rate": round(self.win_rate, 4),
            "avg_pnl": round(self.avg_pnl, 5),
            "avg_holding_h": round(self.avg_holding_h, 2),
            "hit_tp_count": self.hit_tp_count,
            "hit_sl_count": self.hit_sl_count,
            "outcome_distribution": self.outcome_distribution,
            "by_instrument": self.by_instrument,
            "by_session": self.by_session,
            "by_regime": self.by_regime,
        }


class OutcomeTracker:
    """Analyses outcomes across a portfolio of decisions."""

    def summarise(
        self,
        records: list[DecisionRecord],
        decision_type: DecisionType | None = None,
    ) -> OutcomeSummary:
        if decision_type:
            records = [r for r in records if r.decision_type == decision_type]

        summary = OutcomeSummary(total=len(records))

        with_outcome = [r for r in records if r.has_outcome]
        summary.with_outcome = len(with_outcome)

        if not with_outcome:
            return summary

        # Outcome distribution
        dist: dict[str, int] = collections.Counter(
            r.outcome.outcome_type.value for r in with_outcome
        )
        summary.outcome_distribution = dict(dist)

        # Positive / negative / neutral
        for r in with_outcome:
            ot = r.outcome.outcome_type
            if ot in POSITIVE_OUTCOMES:
                summary.positive_count += 1
            elif ot in NEGATIVE_OUTCOMES:
                summary.negative_count += 1
            else:
                summary.neutral_count += 1

        n = len(with_outcome)
        summary.win_rate = round(summary.positive_count / n, 4) if n else 0.0
        summary.avg_pnl = round(sum(r.outcome.actual_pnl for r in with_outcome) / n, 5)
        summary.avg_holding_h = round(sum(r.outcome.holding_time_h for r in with_outcome) / n, 2)
        summary.hit_tp_count = sum(1 for r in with_outcome if r.outcome.hit_tp)
        summary.hit_sl_count = sum(1 for r in with_outcome if r.outcome.hit_sl)

        # By instrument
        by_instr: dict[str, list] = collections.defaultdict(list)
        for r in with_outcome:
            by_instr[r.instrument].append(r.outcome.actual_pnl)
        summary.by_instrument = {
            instr: {
                "count": len(pnls),
                "avg_pnl": round(sum(pnls) / len(pnls), 5),
                "total_pnl": round(sum(pnls), 5),
            }
            for instr, pnls in by_instr.items()
        }

        # By session
        by_session: dict[str, list] = collections.defaultdict(list)
        for r in with_outcome:
            by_session[r.snapshot.session].append(r.outcome.actual_pnl)
        summary.by_session = {
            sess: {
                "count": len(pnls),
                "avg_pnl": round(sum(pnls) / len(pnls), 5),
            }
            for sess, pnls in by_session.items()
        }

        # By regime
        by_regime: dict[str, list] = collections.defaultdict(list)
        for r in with_outcome:
            by_regime[r.snapshot.market_regime].append(r.outcome.actual_pnl)
        summary.by_regime = {
            reg: {
                "count": len(pnls),
                "avg_pnl": round(sum(pnls) / len(pnls), 5),
            }
            for reg, pnls in by_regime.items()
        }

        logger.debug(f"OutcomeTracker: {n} outcomes analysed, win_rate={summary.win_rate:.2%}")
        return summary

    def direction_accuracy(self, records: list[DecisionRecord]) -> float:
        """Fraction of entry decisions with correct direction."""
        entry = [
            r
            for r in records
            if r.has_outcome
            and r.is_entry_decision
            and r.outcome.actual_direction_correct is not None
        ]
        if not entry:
            return 0.0
        correct = sum(1 for r in entry if r.outcome.actual_direction_correct)
        return round(correct / len(entry), 4)

    def tp_hit_rate(self, records: list[DecisionRecord]) -> float:
        with_outcome = [r for r in records if r.has_outcome]
        if not with_outcome:
            return 0.0
        return round(sum(1 for r in with_outcome if r.outcome.hit_tp) / len(with_outcome), 4)

    def sl_hit_rate(self, records: list[DecisionRecord]) -> float:
        with_outcome = [r for r in records if r.has_outcome]
        if not with_outcome:
            return 0.0
        return round(sum(1 for r in with_outcome if r.outcome.hit_sl) / len(with_outcome), 4)
