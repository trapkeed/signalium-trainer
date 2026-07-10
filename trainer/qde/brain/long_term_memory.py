"""QDE Brain — long-term memory & symbol personality learning.

Consolidation is the mechanism that turns transient working-memory items
into durable knowledge: items that were reinforced repeatedly (survived
many ticks without decaying away) are promoted into a
`LongTermMemoryEntry`, which never decays and persists across sessions.

`SymbolPersonality` is updated here too — it's the cross-session "what has
this symbol taught QDE about itself" layer (symbol personality learning).
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain_models import (
    ExperienceRecord,
    LongTermMemoryEntry,
    SymbolPersonality,
    WorkingMemoryItem,
)

CONSOLIDATION_STRENGTH_THRESHOLD = 0.55
CONSOLIDATION_MIN_REINFORCEMENTS = 2
MAX_LONG_TERM_ENTRIES_PER_SYMBOL = 500


class LongTermMemory:
    """Consolidation + personality-learning operations.

    Stateless — operates on the `BrainState.long_term_memory` /
    `BrainState.personality` objects passed in, the same convention as
    `WorkingMemory`, so `QDEBrain` owns the actual state.
    """

    def consolidate(
        self,
        working_items: list[WorkingMemoryItem],
        long_term_entries: list[LongTermMemoryEntry],
        now: float | None = None,
    ) -> list[LongTermMemoryEntry]:
        """Promote strongly-reinforced working memory items into long-term
        entries. Returns the newly-created entries.
        """
        now = now if now is not None else time.time()
        already_consolidated = {eid for entry in long_term_entries for eid in entry.source_item_ids}
        newly_created: list[LongTermMemoryEntry] = []

        for item in working_items:
            if item.item_id in already_consolidated:
                continue
            if item.reinforcement_count < CONSOLIDATION_MIN_REINFORCEMENTS:
                continue
            if item.current_strength(now) < CONSOLIDATION_STRENGTH_THRESHOLD:
                continue

            entry = LongTermMemoryEntry(
                instrument=item.instrument,
                summary=self._summarize(item),
                tags=[item.kind, item.instrument],
                salience=item.current_strength(now),
                source_item_ids=[item.item_id],
                created_at=now,
                last_accessed_at=now,
            )
            long_term_entries.append(entry)
            newly_created.append(entry)

        if len(long_term_entries) > MAX_LONG_TERM_ENTRIES_PER_SYMBOL:
            long_term_entries.sort(key=lambda e: (e.access_count, e.salience))
            del long_term_entries[0 : len(long_term_entries) - MAX_LONG_TERM_ENTRIES_PER_SYMBOL]

        return newly_created

    def _summarize(self, item: WorkingMemoryItem) -> str:
        return (
            f"[{item.kind}] {item.instrument}: reinforced "
            f"{item.reinforcement_count}x — {item.content.get('summary', item.content)}"
        )

    # ── Symbol personality learning ─────────────────────────────────────

    def record_experience(
        self, personality: SymbolPersonality, experience: ExperienceRecord
    ) -> None:
        """Fold one experience into the symbol's learned personality via an
        exponential moving average, so personality adapts gradually instead
        of over-reacting to any single outcome.
        """
        personality.sample_count += 1

        if experience.kind == "HYPOTHESIS_CONFIRMED":
            personality.update_trait("hypothesis_reliability", 1.0)
        elif experience.kind == "HYPOTHESIS_INVALIDATED":
            personality.update_trait("hypothesis_reliability", 0.0)

        if experience.regime:
            personality.update_regime_quality(experience.regime, experience.outcome_quality)

        for tag in experience.tags:
            if tag in (
                "TREND_FOLLOWS",
                "MEAN_REVERTS",
                "HIGH_VOLATILITY_BIAS",
                "SESSION_SENSITIVE",
            ):
                personality.update_trait(tag.lower(), 1.0)
