"""Phase 21.6 — Decision Knowledge Base.

Searchable institutional knowledge base linking:
Decisions, Trades, Experiments, Models, Features,
Market Regimes, Research Reports, Lifecycle Events, Portfolio States.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_models import DecisionRecord


@dataclass
class KnowledgeEntry:
    """One entry in the institutional knowledge base."""

    entry_id: str
    entry_type: str  # DECISION | TRADE | EXPERIMENT | MODEL | FEATURE |
    # REGIME | RESEARCH | LIFECYCLE_EVENT | PORTFOLIO_STATE
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # entry_ids of related entries
    decision_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "links": self.links,
            "decision_ids": self.decision_ids,
            "metadata": self.metadata,
            "created_at": round(self.created_at, 3),
        }


class KnowledgeBase:
    """Searchable institutional knowledge base."""

    def __init__(self):
        self._entries: dict[str, KnowledgeEntry] = {}
        self._decision_index: dict[str, list[str]] = {}  # decision_id → [entry_ids]

    # ── Add / Link ────────────────────────────────────────────────────────────

    def add(self, entry: KnowledgeEntry) -> str:
        self._entries[entry.entry_id] = entry
        for did in entry.decision_ids:
            self._decision_index.setdefault(did, [])
            if entry.entry_id not in self._decision_index[did]:
                self._decision_index[did].append(entry.entry_id)
        logger.debug(
            f"KnowledgeBase: added [{entry.entry_type}] {entry.entry_id[:8]} '{entry.title}'"
        )
        return entry.entry_id

    def create_entry(
        self,
        entry_type: str,
        title: str,
        description: str,
        tags: list[str] | None = None,
        links: list[str] | None = None,
        decision_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeEntry:
        entry = KnowledgeEntry(
            entry_id=str(uuid.uuid4())[:12],
            entry_type=entry_type,
            title=title,
            description=description,
            tags=tags or [],
            links=links or [],
            decision_ids=decision_ids or [],
            metadata=metadata or {},
        )
        self.add(entry)
        return entry

    def link_entries(self, entry_id_a: str, entry_id_b: str) -> bool:
        """Bidirectional link between two entries."""
        a = self._entries.get(entry_id_a)
        b = self._entries.get(entry_id_b)
        if not a or not b:
            return False
        if entry_id_b not in a.links:
            a.links.append(entry_id_b)
        if entry_id_a not in b.links:
            b.links.append(entry_id_a)
        return True

    def index_decision(self, record: DecisionRecord) -> KnowledgeEntry:
        """Auto-index a DecisionRecord as a knowledge entry."""
        title = (
            f"{record.decision_type.value} {record.instrument} "
            f"@ {time.strftime('%Y-%m-%d %H:%M', time.gmtime(record.timestamp))}"
        )
        desc = (
            f"{record.decision_type.value} on {record.instrument}. "
            f"Regime: {record.snapshot.market_regime}. "
            f"Session: {record.snapshot.session}. "
            f"Confidence: {record.snapshot.forecast_confidence:.0%}. "
            f"Grade: {record.grade or 'unscored'}."
        )
        tags = [
            record.decision_type.value,
            record.instrument,
            record.snapshot.market_regime,
            record.snapshot.session,
            record.snapshot.volatility_regime,
        ] + record.tags

        return self.create_entry(
            entry_type="DECISION",
            title=title,
            description=desc,
            tags=tags,
            decision_ids=[record.decision_id],
            metadata=record.summary(),
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str = "",
        entry_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[KnowledgeEntry]:
        """Search the knowledge base by text query, type, and/or tags."""
        q = query.lower().strip()
        results = []

        for entry in self._entries.values():
            if entry_type and entry.entry_type != entry_type:
                continue
            if tags and not any(t in entry.tags for t in tags):
                continue
            if q:
                text = (entry.title + " " + entry.description + " " + " ".join(entry.tags)).lower()
                if q not in text:
                    continue
            results.append(entry)

        results.sort(key=lambda e: e.created_at, reverse=True)
        return results[:limit]

    def entries_for_decision(self, decision_id: str) -> list[KnowledgeEntry]:
        """Return all knowledge entries linked to a decision."""
        entry_ids = self._decision_index.get(decision_id, [])
        return [self._entries[eid] for eid in entry_ids if eid in self._entries]

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        return self._entries.get(entry_id)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def total(self) -> int:
        return len(self._entries)

    def by_type(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for e in self._entries.values():
            dist[e.entry_type] = dist.get(e.entry_type, 0) + 1
        return dist

    def all_tags(self) -> list[str]:
        tags = set()
        for e in self._entries.values():
            tags.update(e.tags)
        return sorted(tags)

    def stats(self) -> dict:
        return {
            "total_entries": self.total(),
            "by_type": self.by_type(),
            "indexed_decisions": len(self._decision_index),
        }
