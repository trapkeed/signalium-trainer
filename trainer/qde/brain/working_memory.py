"""QDE Brain — working memory.

Bounded, decaying, per-symbol short-term memory. This is the "what just
happened and how much does it still matter" layer — distinct from
`long_term_memory.LongTermMemory`, which holds durable, non-decaying
consolidated knowledge.
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain_models import (
    DEFAULT_WORKING_MEMORY_CAPACITY,
    WorkingMemoryItem,
)

MIN_STRENGTH_TO_KEEP = 0.02  # items decayed below this are pruned


class WorkingMemory:
    """Per-symbol bounded, salience-decaying item buffer.

    Not itself persisted independently — `BrainState.working_memory` holds
    the list this class operates on, so it can be serialized as part of
    the aggregate per-symbol state.
    """

    def __init__(self, capacity: int = DEFAULT_WORKING_MEMORY_CAPACITY) -> None:
        self.capacity = capacity

    def add(
        self,
        items: list[WorkingMemoryItem],
        instrument: str,
        kind: str,
        content: dict,
        salience: float = 0.5,
        now: float | None = None,
    ) -> WorkingMemoryItem:
        """Append a new item, evicting the weakest one if over capacity."""
        now = now if now is not None else time.time()
        item = WorkingMemoryItem(
            instrument=instrument,
            kind=kind,
            content=content,
            salience=salience,
            created_at=now,
            last_reinforced_at=now,
        )
        items.append(item)
        if len(items) > self.capacity:
            items.sort(key=lambda i: i.current_strength(now))
            del items[0 : len(items) - self.capacity]
        return item

    def prune_stale(self, items: list[WorkingMemoryItem], now: float | None = None) -> int:
        """Drop items whose decayed strength has fallen below threshold
        (context aging) — returns the number pruned.
        """
        now = now if now is not None else time.time()
        before = len(items)
        items[:] = [i for i in items if i.current_strength(now) >= MIN_STRENGTH_TO_KEEP]
        return before - len(items)

    def top(
        self, items: list[WorkingMemoryItem], n: int = 10, now: float | None = None
    ) -> list[WorkingMemoryItem]:
        now = now if now is not None else time.time()
        return sorted(items, key=lambda i: i.current_strength(now), reverse=True)[:n]

    def most_salient_by_kind(
        self, items: list[WorkingMemoryItem], kind: str, now: float | None = None
    ) -> WorkingMemoryItem | None:
        now = now if now is not None else time.time()
        candidates = [i for i in items if i.kind == kind]
        if not candidates:
            return None
        return max(candidates, key=lambda i: i.current_strength(now))
