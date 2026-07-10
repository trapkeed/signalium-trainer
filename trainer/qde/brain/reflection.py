"""QDE Brain — reflection.

Periodic self-review that reads from the existing decision/knowledge
infrastructure (`DecisionMemory`, `KnowledgeBase`) rather than keeping a
parallel ledger — this module is purely a synthesis layer that turns raw
decision history into an insight + lessons for a symbol's `BrainState`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from trainer.qde.brain.brain_models import ReflectionEntry

if TYPE_CHECKING:
    from trainer.memory.decision_memory import DecisionMemory
    from trainer.memory.knowledge_base import KnowledgeBase

DEFAULT_REFLECTION_WINDOW_SECONDS = 24 * 3600.0


class ReflectionEngine:
    def reflect(
        self,
        instrument: str,
        decision_memory: "DecisionMemory | None" = None,
        knowledge_base: "KnowledgeBase | None" = None,
        window_seconds: float = DEFAULT_REFLECTION_WINDOW_SECONDS,
        now: float | None = None,
    ) -> ReflectionEntry:
        now = now if now is not None else time.time()
        period_start = now - window_seconds

        records = []
        if decision_memory is not None:
            records = decision_memory.filter(instrument=instrument, since=period_start, limit=500)

        if not records:
            return ReflectionEntry(
                instrument=instrument,
                period_start=period_start,
                period_end=now,
                insight="No decisions in this window — nothing new to learn yet.",
                decisions_reviewed=0,
            )

        scored = [r for r in records if r.has_outcome]
        wins = [r for r in scored if r.outcome and r.outcome.actual_direction_correct]
        win_rate = (len(wins) / len(scored)) if scored else None

        lessons: list[str] = []
        if win_rate is not None:
            if win_rate >= 0.6:
                lessons.append(
                    f"Recent decisions on {instrument} have been directionally accurate "
                    f"({win_rate:.0%}) — current read of this symbol appears well-calibrated."
                )
            elif win_rate <= 0.4:
                lessons.append(
                    f"Recent decisions on {instrument} have underperformed "
                    f"({win_rate:.0%} accuracy) — treat current hypotheses with extra caution."
                )

        regimes = {r.snapshot.market_regime for r in records if r.snapshot}
        if len(regimes) > 1:
            lessons.append(
                f"{instrument} has traded across {len(regimes)} distinct regimes this window "
                "— narrative continuity should account for regime shifts, not assume stability."
            )

        insight = (
            f"Reviewed {len(records)} decisions on {instrument} "
            f"({len(scored)} with known outcomes"
            + (f", {win_rate:.0%} win rate)" if win_rate is not None else ")")
        )

        entry = ReflectionEntry(
            instrument=instrument,
            period_start=period_start,
            period_end=now,
            insight=insight,
            lessons=lessons,
            decisions_reviewed=len(records),
        )

        if knowledge_base is not None:
            knowledge_base.create_entry(
                entry_type="LIFECYCLE_EVENT",
                title=f"QDE Brain reflection — {instrument}",
                description=insight,
                tags=["BRAIN_REFLECTION", instrument],
                metadata=entry.to_dict(),
            )

        return entry
