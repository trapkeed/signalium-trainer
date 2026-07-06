"""Phase 23 — Citation Engine.

Extracts, validates, and enriches citations in LLM responses.
Every SAQRA response must reference actual research artifacts.
"""

from __future__ import annotations

import re

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_models import Citation, KnowledgeChunk
from trainer.analyst.analyst_types import CitationType

_CITATION_PATTERNS = [
    (r"\[EXP[-:](\w+)\]", CitationType.EXPERIMENT),
    (r"\[MODEL[-:](\w+)\]", CitationType.MODEL),
    (r"\[DATASET[-:](\w+)\]", CitationType.DATASET),
    (r"\[FEATURE[-:](\w+)\]", CitationType.FEATURE),
    (r"\[VAL[-:](\w+)\]", CitationType.VALIDATION),
    (r"\[REPORT[-:](\w+)\]", CitationType.REPORT),
    (r"\[DECISION[-:](\w+)\]", CitationType.DECISION),
    (r"\[AUDIT[-:](\w+)\]", CitationType.AUDIT_LOG),
    (r"\[KB[-:](\w+)\]", CitationType.KNOWLEDGE),
]


class CitationEngine:
    """Extracts and validates citations in SAQRA responses."""

    def extract(self, text: str) -> list[Citation]:
        """Extract all [TYPE:ID] citations from response text."""
        citations = []
        seen = set()

        for pattern, ctype in _CITATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                full_ref = match.group(0).upper()
                artifact = match.group(1).upper()
                if full_ref not in seen:
                    seen.add(full_ref)
                    # Find the surrounding sentence for description
                    start = max(0, match.start() - 80)
                    end = min(len(text), match.end() + 80)
                    context_snippet = text[start:end].strip().replace("\n", " ")
                    citations.append(
                        Citation(
                            citation_type=ctype,
                            artifact_id=artifact,
                            description=context_snippet,
                            relevance=1.0,
                        )
                    )

        logger.debug(f"CitationEngine: extracted {len(citations)} citations")
        return citations

    def validate(
        self,
        citations: list[Citation],
        chunks: list[KnowledgeChunk],
    ) -> list[Citation]:
        """Validate citations against retrieved chunks.

        Validated citations have relevance boosted; unvalidated citations are
        flagged but kept (they may reference indexed artifacts not retrieved
        for this particular query).
        """
        chunk_artifact_ids = {c.artifact_id.upper() for c in chunks}

        validated = []
        for cit in citations:
            if cit.artifact_id.upper() in chunk_artifact_ids:
                validated.append(
                    Citation(
                        citation_type=cit.citation_type,
                        artifact_id=cit.artifact_id,
                        description=cit.description,
                        relevance=1.0,
                    )
                )
            else:
                validated.append(
                    Citation(
                        citation_type=cit.citation_type,
                        artifact_id=cit.artifact_id,
                        description=cit.description,
                        relevance=0.60,  # lower confidence for unvalidated
                    )
                )

        return validated

    def add_from_chunks(
        self,
        citations: list[Citation],
        chunks: list[KnowledgeChunk],
    ) -> list[Citation]:
        """Supplement citations with references from retrieved chunks.

        Ensures that all retrieved chunks are cited even if the LLM
        did not explicitly reference them.
        """
        existing_ids = {c.artifact_id.upper() for c in citations}
        extra = []

        for chunk in chunks:
            if chunk.artifact_id.upper() not in existing_ids:
                existing_ids.add(chunk.artifact_id.upper())
                # Map source type to citation type
                ct = self._source_to_citation(chunk)
                extra.append(
                    Citation(
                        citation_type=ct,
                        artifact_id=chunk.artifact_id,
                        description=f"Retrieved: {chunk.title}",
                        relevance=0.80,
                    )
                )

        return citations + extra

    def format_bibliography(self, citations: list[Citation]) -> str:
        """Format a bibliography section from a list of citations."""
        if not citations:
            return "No citations."
        lines = ["## References", ""]
        for i, c in enumerate(citations, 1):
            lines.append(
                f"{i}. [{c.citation_type.value}:{c.artifact_id}] — "
                f"{c.description[:80]}... (relevance: {c.relevance:.2f})"
            )
        return "\n".join(lines)

    def is_grounded(self, citations: list[Citation], min_citations: int = 1) -> bool:
        """Return True if the response has sufficient citations."""
        return len(citations) >= min_citations

    @staticmethod
    def _source_to_citation(chunk: KnowledgeChunk) -> CitationType:
        from trainer.analyst.analyst_types import KnowledgeSourceType

        mapping = {
            KnowledgeSourceType.EXPERIMENT: CitationType.EXPERIMENT,
            KnowledgeSourceType.MODEL_REGISTRY: CitationType.MODEL,
            KnowledgeSourceType.DATASET_VERSION: CitationType.DATASET,
            KnowledgeSourceType.FEATURE_IMPORTANCE: CitationType.FEATURE,
            KnowledgeSourceType.VALIDATION_REPORT: CitationType.VALIDATION,
            KnowledgeSourceType.RESEARCH_REPORT: CitationType.REPORT,
            KnowledgeSourceType.DECISION_MEMORY: CitationType.DECISION,
            KnowledgeSourceType.AUDIT_LOG: CitationType.AUDIT_LOG,
        }
        return mapping.get(chunk.source_type, CitationType.KNOWLEDGE)
