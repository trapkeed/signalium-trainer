"""Phase 23 — Research Search.

Natural language search across the knowledge base.
Supports queries such as:
  - "Show every experiment using Transformer models"
  - "Show experiments where Sharpe exceeded 2"
  - "Find all failed breakout models"
  - "Find models trained before March"
  - "Find experiments using Dataset v18"
"""

from __future__ import annotations

import re

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_models import KnowledgeChunk, ResearchQuery
from trainer.analyst.analyst_types import (
    KnowledgeSourceType,
    Permission,
    QueryType,
    has_permission,
)
from trainer.analyst.citation_engine import CitationEngine
from trainer.analyst.knowledge_index import KnowledgeIndex
from trainer.analyst.llm_provider import LLMProvider, MockLLMProvider
from trainer.analyst.prompt_engine import PromptEngine
from trainer.analyst.retrieval_engine import RetrievalEngine


class ResearchSearch:
    """Natural language search over the indexed knowledge base."""

    def __init__(
        self,
        index: KnowledgeIndex,
        provider: LLMProvider | None = None,
        top_k: int = 10,
    ):
        self._index = index
        self._provider = provider or MockLLMProvider()
        self._retriever = RetrievalEngine(index, top_k=top_k)
        self._prompt = PromptEngine()
        self._citation = CitationEngine()

    def search(
        self,
        query: ResearchQuery,
        source_type: KnowledgeSourceType | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> dict:
        """Run a natural language search and return structured results."""
        if not has_permission(query.role, Permission.READ):
            return {"error": "Access denied", "results": []}

        # Detect source_type from query if not provided
        if source_type is None:
            source_type = self._infer_source_type(query.text)

        # Retrieve
        chunks, context = self._retriever.retrieve_and_build(
            query.text,
            top_k=top_k,
            source_type=source_type,
            tags=tags,
        )

        # Apply metric filters from the query text
        filtered = self._apply_filters(chunks, query.text)

        # Build LLM response
        prompt = self._prompt.build(QueryType.SEARCH, query.text, context)
        response = self._provider.generate(prompt, context)
        response.retrieved_chunks = filtered

        citations = self._citation.extract(response.content)
        citations = self._citation.add_from_chunks(citations, filtered)
        response.citations = citations

        return {
            "query": query.text,
            "source_type": source_type.value if source_type else "ALL",
            "results": [c.to_dict() for c in filtered],
            "result_count": len(filtered),
            "summary": response.content,
            "citations": [c.to_dict() for c in citations],
        }

    def _infer_source_type(self, query: str) -> KnowledgeSourceType | None:
        """Heuristically detect the likely source type from the query."""
        lower = query.lower()
        # Dataset must be checked before experiment (query may mention both)
        if any(w in lower for w in ("dataset", "data version", "dataset v")):
            return KnowledgeSourceType.DATASET_VERSION
        if any(w in lower for w in ("experiment", "exp-", "training run")):
            return KnowledgeSourceType.EXPERIMENT
        if any(w in lower for w in ("model", "model-", "architecture")):
            return KnowledgeSourceType.MODEL_REGISTRY
        if any(w in lower for w in ("feature", "feature importance")):
            return KnowledgeSourceType.FEATURE_IMPORTANCE
        if any(w in lower for w in ("validation", "walk-forward", "val run")):
            return KnowledgeSourceType.VALIDATION_REPORT
        if any(w in lower for w in ("report", "research report")):
            return KnowledgeSourceType.RESEARCH_REPORT
        if any(w in lower for w in ("decision", "trade decision")):
            return KnowledgeSourceType.DECISION_MEMORY
        if any(w in lower for w in ("drift", "drift report")):
            return KnowledgeSourceType.DRIFT_REPORT
        return None

    def _apply_filters(self, chunks: list[KnowledgeChunk], query: str) -> list[KnowledgeChunk]:
        """Apply metric-based filters parsed from the query text."""
        # Detect "sharpe > X" or "sharpe exceeded X"
        sharpe_match = re.search(
            r"sharpe\s+(?:exceeded?|>|>=|above|over)\s+([\d.]+)", query, re.IGNORECASE
        )
        if sharpe_match:
            threshold = float(sharpe_match.group(1))
            chunks = [c for c in chunks if c.metadata.get("sharpe", 0) >= threshold]

        # Detect "drawdown < X%"
        dd_match = re.search(
            r"drawdown\s+(?:<|<=|below|under)\s+([\d.]+)\s*%?", query, re.IGNORECASE
        )
        if dd_match:
            threshold = float(dd_match.group(1))
            chunks = [c for c in chunks if c.metadata.get("max_drawdown", 100) <= threshold]

        # Detect "failed" filter
        if re.search(r"\b(failed|failure|unsuccessful)\b", query, re.IGNORECASE):
            chunks = [
                c
                for c in chunks
                if "fail" in c.content.lower() or c.metadata.get("status") == "FAILED"
            ]

        # Detect "successful" / "passing" filter
        if re.search(r"\b(successful|passed|best|top)\b", query, re.IGNORECASE):
            chunks = [
                c
                for c in chunks
                if c.metadata.get("status") in ("PASSED", "PROMOTED", "CHAMPION")
                or "success" in c.content.lower()
            ]

        return chunks

    def keyword_search(self, keyword: str, top_k: int = 10) -> list[KnowledgeChunk]:
        """Simple keyword search across the full index."""
        return self._index.search(keyword, top_k=top_k)

    def search_by_artifact(self, artifact_id: str) -> list[KnowledgeChunk]:
        """Find all chunks for a specific artifact ID."""
        return self._index.get_by_artifact(artifact_id)

    def search_by_source(
        self, source_type: KnowledgeSourceType, top_k: int = 20
    ) -> list[KnowledgeChunk]:
        """Get all indexed chunks for a specific knowledge source type."""
        return self._index.get_by_source(source_type)[:top_k]
