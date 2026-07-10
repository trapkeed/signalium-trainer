"""S33 — Optional Knowledge Base integration for opportunity scores.

Kept as a standalone helper (rather than importing ``trainer.memory``
inside ``engine.py`` directly) so ``OpportunityScoringEngine`` has no hard
dependency on the memory package — callers wire this in via the engine's
``on_score`` callback only if/when they want it:

    from trainer.memory.knowledge_base import KnowledgeBase
    from trainer.qde.opportunity_scoring.knowledge_bridge import record_to_knowledge_base

    kb = KnowledgeBase(...)
    engine = OpportunityScoringEngine(on_score=lambda r: record_to_knowledge_base(kb, r))
"""

from __future__ import annotations

from typing import Any, Protocol

from trainer.qde.opportunity_scoring.models import OpportunityScoreResult


class KnowledgeBaseLike(Protocol):
    def create_entry(
        self,
        entry_type: str,
        title: str,
        description: str,
        tags: list[str] | None = None,
        links: list[str] | None = None,
        decision_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Any: ...


def record_to_knowledge_base(
    knowledge_base: KnowledgeBaseLike, result: OpportunityScoreResult
) -> Any:
    """Push one opportunity score into the Knowledge Base as a searchable
    RESEARCH entry, linked back to its decision (if any). Only meaningful
    scores (WATCH_CLOSELY and above) are indexed, to avoid flooding the
    Knowledge Base with routine IGNORE/OBSERVE noise."""
    if result.score < 60.0:
        return None
    return knowledge_base.create_entry(
        entry_type="RESEARCH",
        title=f"Opportunity score {result.score:.1f} ({result.band.value}) — "
        f"{result.symbol}/{result.timeframe}",
        description=(
            f"Setup state={result.setup_state or 'n/a'}; " f"sub_scores={result.sub_scores}"
        ),
        tags=["opportunity_score", result.band.value.lower(), result.symbol],
        decision_ids=[result.decision_id] if result.decision_id else None,
        metadata=result.to_dict(),
    )
