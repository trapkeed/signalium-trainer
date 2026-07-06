"""Phase 19 — Institutional Knowledge Base.

Maintains the institutional memory of all research activity:

  - Research findings
  - Successful experiments
  - Failed experiments
  - Lessons learned
  - Feature history
  - Model history
  - Deployment history
  - Validation history

Queryable by type, tag, model, date, and relevance.

Usage::

    kb = KnowledgeBase()
    entry_id = kb.add(KnowledgeEntry(
        entry_type=KnowledgeEntryType.FINDING,
        title="Transformer backbone outperforms LSTM on trending regimes",
        body="...",
        tags=["architecture", "transformer", "trending"],
        model_id="model_02",
        impact_level="high",
    ))
    findings = kb.search("transformer")
    report = kb.knowledge_report()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Entry type
# ---------------------------------------------------------------------------


class KnowledgeEntryType:
    FINDING = "finding"
    SUCCESSFUL_EXPERIMENT = "successful_experiment"
    FAILED_EXPERIMENT = "failed_experiment"
    LESSON_LEARNED = "lesson_learned"
    FEATURE_HISTORY = "feature_history"
    MODEL_HISTORY = "model_history"
    DEPLOYMENT_HISTORY = "deployment_history"
    VALIDATION_HISTORY = "validation_history"
    MARKET_OBSERVATION = "market_observation"
    DATASET_NOTE = "dataset_note"
    RESEARCH_NOTE = "research_note"

    ALL = [
        FINDING,
        SUCCESSFUL_EXPERIMENT,
        FAILED_EXPERIMENT,
        LESSON_LEARNED,
        FEATURE_HISTORY,
        MODEL_HISTORY,
        DEPLOYMENT_HISTORY,
        VALIDATION_HISTORY,
        MARKET_OBSERVATION,
        DATASET_NOTE,
        RESEARCH_NOTE,
    ]


# ---------------------------------------------------------------------------
# Knowledge entry
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeEntry:
    """One institutional knowledge entry.

    Attributes:
        entry_id: Unique UUID.
        entry_type: KnowledgeEntryType value.
        title: Short title.
        body: Full content (free-form text or structured dict description).
        tags: List of topic tags for retrieval.
        model_id: Associated model (optional).
        feature_name: Associated feature (optional).
        dataset_version: Associated dataset (optional).
        impact_level: low / medium / high.
        author: Who authored this entry.
        created_at: UTC ISO-8601 timestamp.
        references: List of related entry_ids.
        metadata: Arbitrary structured metadata.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: str = KnowledgeEntryType.RESEARCH_NOTE
    title: str = ""
    body: str = ""
    tags: list[str] = field(default_factory=list)
    model_id: str | None = None
    feature_name: str | None = None
    dataset_version: str | None = None
    impact_level: str = "medium"  # low / medium / high
    author: str = "system"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type,
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
            "model_id": self.model_id,
            "feature_name": self.feature_name,
            "dataset_version": self.dataset_version,
            "impact_level": self.impact_level,
            "author": self.author,
            "created_at": self.created_at,
            "references": self.references,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """Institutional knowledge base for research intelligence.

    Stores all research findings, experiment outcomes, lessons learned,
    and historical records. Queryable by type, tag, model, and full-text.

    Args:
        max_entries: Maximum entries to hold in memory.
    """

    def __init__(self, max_entries: int = 50_000) -> None:
        self.max_entries = max_entries
        self._entries: dict[str, KnowledgeEntry] = {}
        self._order: list[str] = []

    def add(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry. Returns entry_id."""
        self._entries[entry.entry_id] = entry
        self._order.append(entry.entry_id)
        if len(self._order) > self.max_entries:
            oldest_id = self._order.pop(0)
            self._entries.pop(oldest_id, None)
        return entry.entry_id

    def log(
        self,
        entry_type: str,
        title: str,
        body: str = "",
        tags: list[str] | None = None,
        model_id: str | None = None,
        feature_name: str | None = None,
        dataset_version: str | None = None,
        impact_level: str = "medium",
        author: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeEntry:
        """Convenience method to create and add an entry."""
        entry = KnowledgeEntry(
            entry_type=entry_type,
            title=title,
            body=body,
            tags=tags or [],
            model_id=model_id,
            feature_name=feature_name,
            dataset_version=dataset_version,
            impact_level=impact_level,
            author=author,
            metadata=metadata or {},
        )
        self.add(entry)
        return entry

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        return self._entries.get(entry_id)

    def query(
        self,
        entry_type: str | None = None,
        model_id: str | None = None,
        feature_name: str | None = None,
        dataset_version: str | None = None,
        tag: str | None = None,
        impact_level: str | None = None,
        limit: int = 50,
        most_recent_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Query entries with optional filters."""
        entries = [self._entries[eid] for eid in self._order if eid in self._entries]
        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]
        if model_id:
            entries = [e for e in entries if e.model_id == model_id]
        if feature_name:
            entries = [e for e in entries if e.feature_name == feature_name]
        if dataset_version:
            entries = [e for e in entries if e.dataset_version == dataset_version]
        if tag:
            entries = [e for e in entries if tag in e.tags]
        if impact_level:
            entries = [e for e in entries if e.impact_level == impact_level]
        if most_recent_first:
            entries = list(reversed(entries))
        return [e.to_dict() for e in entries[:limit]]

    def search(self, query_text: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across title, body, and tags."""
        query_lower = query_text.lower()
        results = []
        for eid in reversed(self._order):
            entry = self._entries.get(eid)
            if entry is None:
                continue
            score = 0
            if query_lower in entry.title.lower():
                score += 3
            if query_lower in entry.body.lower():
                score += 1
            if any(query_lower in t.lower() for t in entry.tags):
                score += 2
            if score > 0:
                results.append((score, entry))
        results.sort(key=lambda x: -x[0])
        return [e.to_dict() for _, e in results[:limit]]

    def findings(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(entry_type=KnowledgeEntryType.FINDING, limit=limit)

    def lessons_learned(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(entry_type=KnowledgeEntryType.LESSON_LEARNED, limit=limit)

    def successful_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(entry_type=KnowledgeEntryType.SUCCESSFUL_EXPERIMENT, limit=limit)

    def failed_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(entry_type=KnowledgeEntryType.FAILED_EXPERIMENT, limit=limit)

    def high_impact_entries(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query(impact_level="high", limit=limit)

    def model_history(self, model_id: str) -> list[dict[str, Any]]:
        return self.query(model_id=model_id)

    def feature_history(self, feature_name: str) -> list[dict[str, Any]]:
        return self.query(feature_name=feature_name)

    def knowledge_report(self) -> dict[str, Any]:
        """Generate a summary knowledge report."""
        by_type: dict[str, int] = {}
        by_impact: dict[str, int] = {}
        for entry in self._entries.values():
            by_type[entry.entry_type] = by_type.get(entry.entry_type, 0) + 1
            by_impact[entry.impact_level] = by_impact.get(entry.impact_level, 0) + 1
        return {
            "report_type": "knowledge_report",
            "total_entries": len(self._entries),
            "by_type": by_type,
            "by_impact": by_impact,
            "recent_findings": self.findings(limit=5),
            "recent_lessons": self.lessons_learned(limit=5),
            "recent_successes": self.successful_experiments(limit=5),
            "generated_at": datetime.now(UTC).isoformat(),
        }

    def export_all(self) -> list[dict[str, Any]]:
        return [self._entries[eid].to_dict() for eid in self._order if eid in self._entries]

    @property
    def n_entries(self) -> int:
        return len(self._entries)
