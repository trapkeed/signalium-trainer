"""Phase 23 — Recommendation Engine.

Generates evidence-based research recommendations.
All recommendations reference retrieved artifacts.
Never recommends live trading actions.

Recommendation types:
DATA_COLLECTION, FEATURE_ENGINEERING, ARCHITECTURE_CHANGE,
CALIBRATION, REGIME_SPECIALIZATION, SEQUENCE_LENGTH,
VALIDATION, STABILITY, DIVERSITY.
"""

from __future__ import annotations

import re

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_models import Citation, Recommendation, ResearchQuery
from trainer.analyst.analyst_types import (
    Permission,
    QueryType,
    RecommendationType,
    has_permission,
)
from trainer.analyst.citation_engine import CitationEngine
from trainer.analyst.knowledge_index import KnowledgeIndex
from trainer.analyst.llm_provider import LLMProvider, MockLLMProvider
from trainer.analyst.prompt_engine import PromptEngine
from trainer.analyst.retrieval_engine import RetrievalEngine

# Keywords that trigger each recommendation type
_TRIGGER_MAP = {
    RecommendationType.DATA_COLLECTION: [
        "insufficient data",
        "low data",
        "limited regime",
        "missing regime",
        "data diversity",
        "collect more",
    ],
    RecommendationType.FEATURE_ENGINEERING: [
        "feature",
        "redundant",
        "unstable feature",
        "feature importance",
        "feature drift",
        "correlated feature",
    ],
    RecommendationType.ARCHITECTURE_CHANGE: [
        "architecture",
        "transformer",
        "lstm",
        "sequence length",
        "model size",
        "attention",
        "head count",
    ],
    RecommendationType.CALIBRATION: [
        "calibration",
        "miscalibrated",
        "ece",
        "reliability",
        "confidence drift",
        "overconfident",
    ],
    RecommendationType.REGIME_SPECIALIZATION: [
        "regime",
        "specialist",
        "expert",
        "regime-specific",
        "ranging",
        "trending",
        "volatile",
    ],
    RecommendationType.SEQUENCE_LENGTH: [
        "sequence length",
        "lookback",
        "window size",
        "context window",
    ],
    RecommendationType.VALIDATION: [
        "validation",
        "walk-forward",
        "backtest",
        "holdout",
        "overfitting",
        "leakage",
    ],
    RecommendationType.STABILITY: [
        "stability",
        "unstable",
        "drift",
        "degrading",
        "performance drop",
        "regression",
    ],
    RecommendationType.DIVERSITY: [
        "diversity",
        "instrument",
        "asset class",
        "market",
        "gbpjpy",
        "specialist",
        "coverage",
    ],
}

_PRIORITY_SIGNALS = {
    "HIGH": ["critical", "severe", "urgent", "significant degradation", "failure"],
    "MEDIUM": ["moderate", "notable", "should", "consider", "improvement"],
    "LOW": ["minor", "optional", "future", "nice to have"],
}


class RecommendationEngine:
    """Generates evidence-based research recommendations from retrieved knowledge."""

    def __init__(
        self,
        index: KnowledgeIndex,
        provider: LLMProvider | None = None,
        top_k: int = 6,
    ):
        self._index = index
        self._provider = provider or MockLLMProvider()
        self._retriever = RetrievalEngine(index, top_k=top_k)
        self._prompt = PromptEngine()
        self._citation = CitationEngine()
        self._history: list[Recommendation] = []

    def recommend(
        self,
        query: ResearchQuery,
    ) -> list[Recommendation]:
        """Generate evidence-based recommendations for a query."""
        if not has_permission(query.role, Permission.READ):
            return []

        # Retrieve relevant context
        chunks, context = self._retriever.retrieve_and_build(query.text, top_k=6)

        # Generate LLM response
        prompt = self._prompt.build(QueryType.RECOMMENDATION, query.text, context)
        response = self._provider.generate(prompt, context)
        response.retrieved_chunks = chunks

        citations = self._citation.extract(response.content)
        citations = self._citation.add_from_chunks(citations, chunks)

        # Parse recommendations from the response
        recs = self._parse_recommendations(response.content, citations, query.text)

        # Store in history
        self._history.extend(recs)

        logger.info(f"RecommendationEngine: generated {len(recs)} recommendations")
        return recs

    def auto_recommend(self) -> list[Recommendation]:
        """Scan the entire knowledge index and auto-generate recommendations.

        Returns recommendations triggered by detected patterns in indexed content.
        """
        all_chunks = self._index.all_chunks()
        if not all_chunks:
            return []

        full_text = " ".join(c.content for c in all_chunks).lower()
        recs = []

        for rec_type, triggers in _TRIGGER_MAP.items():
            triggered = [t for t in triggers if t in full_text]
            if not triggered:
                continue

            # Find most relevant chunks for this trigger
            trigger_query = " ".join(triggered[:3])
            relevant = self._index.search(trigger_query, top_k=3)
            cits = [
                Citation(
                    citation_type=self._citation._source_to_citation(c),
                    artifact_id=c.artifact_id,
                    description=c.title,
                    relevance=0.75,
                )
                for c in relevant
            ]

            priority = self._infer_priority(full_text)
            rec = Recommendation(
                rec_type=rec_type,
                title=self._title_for_type(rec_type),
                description=self._description_for_type(rec_type, triggered),
                rationale=f"Triggered by: {', '.join(triggered[:3])}",
                citations=cits,
                priority=priority,
            )
            recs.append(rec)
            self._history.append(rec)

        return recs

    def _parse_recommendations(
        self,
        text: str,
        citations: list[Citation],
        query: str,
    ) -> list[Recommendation]:
        """Parse numbered recommendations from LLM response text."""
        recs = []

        # Match numbered items: "1. **Title**: description"
        items = re.findall(r"\d+\.\s+\*\*([^*]+)\*\*[:\-]?\s*(.+?)(?=\n\d+\.|\Z)", text, re.DOTALL)

        for title, desc in items[:6]:  # cap at 6
            rec_type = self._classify_type(title + " " + desc)
            priority = self._infer_priority(desc)
            rec = Recommendation(
                rec_type=rec_type,
                title=title.strip()[:120],
                description=desc.strip()[:400],
                rationale="Evidence from retrieved research artifacts",
                citations=citations[:3],
                priority=priority,
            )
            recs.append(rec)

        # Fallback if no structured items parsed
        if not recs:
            rec_type = self._classify_type(query)
            recs.append(
                Recommendation(
                    rec_type=rec_type,
                    title=self._title_for_type(rec_type),
                    description=text[:300].strip(),
                    rationale="Generated from retrieved research context",
                    citations=citations[:3],
                    priority="MEDIUM",
                )
            )

        return recs

    def _classify_type(self, text: str) -> RecommendationType:
        lower = text.lower()
        best = (0, RecommendationType.DATA_COLLECTION)
        for rtype, triggers in _TRIGGER_MAP.items():
            score = sum(1 for t in triggers if t in lower)
            if score > best[0]:
                best = (score, rtype)
        return best[1]

    def _infer_priority(self, text: str) -> str:
        lower = text.lower()
        for pri, signals in _PRIORITY_SIGNALS.items():
            if any(s in lower for s in signals):
                return pri
        return "MEDIUM"

    def _title_for_type(self, rec_type: RecommendationType) -> str:
        titles = {
            RecommendationType.DATA_COLLECTION: "Collect Additional Training Data",
            RecommendationType.FEATURE_ENGINEERING: "Review Feature Stability",
            RecommendationType.ARCHITECTURE_CHANGE: "Evaluate Architecture Change",
            RecommendationType.CALIBRATION: "Improve Model Calibration",
            RecommendationType.REGIME_SPECIALIZATION: "Train Regime-Specialist Model",
            RecommendationType.SEQUENCE_LENGTH: "Adjust Sequence Length",
            RecommendationType.VALIDATION: "Enhance Validation Coverage",
            RecommendationType.STABILITY: "Address Performance Instability",
            RecommendationType.DIVERSITY: "Expand Instrument Coverage",
        }
        return titles.get(rec_type, "Research Recommendation")

    def _description_for_type(self, rec_type: RecommendationType, triggers: list[str]) -> str:
        descs = {
            RecommendationType.DATA_COLLECTION: "Expand training dataset to include underrepresented market regimes.",
            RecommendationType.FEATURE_ENGINEERING: "Review and prune features showing signs of instability or redundancy.",
            RecommendationType.ARCHITECTURE_CHANGE: "Evaluate alternative architectures based on detected performance patterns.",
            RecommendationType.CALIBRATION: "Recalibrate model confidence estimates on recent validation windows.",
            RecommendationType.REGIME_SPECIALIZATION: "Consider training a specialist model for the identified regime gap.",
            RecommendationType.SEQUENCE_LENGTH: "Experiment with sequence length to improve temporal coverage.",
            RecommendationType.VALIDATION: "Extend walk-forward validation to cover additional market periods.",
            RecommendationType.STABILITY: "Investigate and address sources of detected performance instability.",
            RecommendationType.DIVERSITY: "Expand instrument coverage to improve model generalisability.",
        }
        return descs.get(rec_type, "See retrieved research artifacts for details.")

    def history(self, last_n: int = 20) -> list[Recommendation]:
        return self._history[-last_n:]

    def pending(self) -> list[Recommendation]:
        return [r for r in self._history if not r.implemented]

    def mark_implemented(self, rec_id: str) -> bool:
        for r in self._history:
            if r.rec_id == rec_id:
                r.implemented = True
                return True
        return False
