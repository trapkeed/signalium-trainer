"""Phase 23 — Model Comparator.

Compares models, experiments, datasets, features, and validation runs.
Generates structured comparison reports with citations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_models import Citation, LLMResponse, ResearchQuery
from trainer.analyst.analyst_types import CitationType, Permission, QueryType, Role, has_permission
from trainer.analyst.citation_engine import CitationEngine
from trainer.analyst.knowledge_index import KnowledgeIndex
from trainer.analyst.llm_provider import LLMProvider, MockLLMProvider
from trainer.analyst.prompt_engine import PromptEngine
from trainer.analyst.retrieval_engine import RetrievalEngine


class ModelComparator:
    """Generates structured comparison reports between research artifacts."""

    def __init__(
        self,
        index: KnowledgeIndex,
        provider: Optional[LLMProvider] = None,
        top_k: int = 6,
    ):
        self._index = index
        self._provider = provider or MockLLMProvider()
        self._retriever = RetrievalEngine(index, top_k=top_k)
        self._prompt = PromptEngine()
        self._citation = CitationEngine()

    def compare(
        self,
        query: ResearchQuery,
        artifact_a: str,
        artifact_b: str,
        metrics: Optional[List[str]] = None,
    ) -> Dict:
        """Compare two research artifacts and return a structured comparison report."""
        if not has_permission(query.role, Permission.COMPARE):
            return {"error": "Access denied. COMPARE permission required."}

        # Retrieve context for both artifacts
        chunks_a = self._index.get_by_artifact(artifact_a)
        chunks_b = self._index.get_by_artifact(artifact_b)
        all_chunks = chunks_a + chunks_b

        # Also do a semantic retrieval for the comparison context
        comparison_query = f"compare {artifact_a} {artifact_b} {query.text}"
        extra_chunks, _ = self._retriever.retrieve_and_build(comparison_query, top_k=4)
        all_chunks = list({c.chunk_id: c for c in all_chunks + extra_chunks}.values())

        context = self._retriever.build_context(all_chunks)

        # Build comparison prompt
        full_query = f"Compare {artifact_a} vs {artifact_b}. {query.text}"
        prompt = self._prompt.build(QueryType.COMPARISON, full_query, context)

        # Generate
        response = self._provider.generate(prompt, context)
        response.retrieved_chunks = all_chunks

        citations = self._citation.extract(response.content)
        citations = self._citation.add_from_chunks(citations, all_chunks)
        response.citations = citations

        # Build structured comparison table
        comparison_table = self._build_table(chunks_a, chunks_b, metrics)

        logger.info(f"ModelComparator: compared {artifact_a} vs {artifact_b}")

        return {
            "artifact_a": artifact_a,
            "artifact_b": artifact_b,
            "table": comparison_table,
            "analysis": response.content,
            "citations": [c.to_dict() for c in citations],
            "grounded": response.grounded,
        }

    def compare_batch(
        self,
        query: ResearchQuery,
        artifacts: List[str],
    ) -> Dict:
        """Compare multiple artifacts. Returns pairwise comparison matrix."""
        if not has_permission(query.role, Permission.COMPARE):
            return {"error": "Access denied."}

        all_chunks = []
        artifact_data: Dict[str, List] = {}
        for art in artifacts:
            chunks = self._index.get_by_artifact(art)
            artifact_data[art] = chunks
            all_chunks.extend(chunks)

        context = self._retriever.build_context(all_chunks)
        full_query = f"Compare {', '.join(artifacts)}. {query.text}"
        prompt = self._prompt.build(QueryType.COMPARISON, full_query, context)
        response = self._provider.generate(prompt, context)
        response.retrieved_chunks = all_chunks

        citations = self._citation.extract(response.content)
        citations = self._citation.add_from_chunks(citations, all_chunks)

        return {
            "artifacts": artifacts,
            "analysis": response.content,
            "citations": [c.to_dict() for c in citations],
            "grounded": response.grounded,
        }

    def _build_table(
        self,
        chunks_a: List,
        chunks_b: List,
        metrics: Optional[List[str]],
    ) -> Dict[str, Dict]:
        """Build a metric comparison table from chunk metadata."""
        default_metrics = metrics or ["sharpe", "max_drawdown", "accuracy", "calibration", "status"]
        table: Dict[str, Dict] = {}

        meta_a = {}
        for c in chunks_a:
            meta_a.update(c.metadata)

        meta_b = {}
        for c in chunks_b:
            meta_b.update(c.metadata)

        for metric in default_metrics:
            val_a = meta_a.get(metric, "N/A")
            val_b = meta_b.get(metric, "N/A")
            # Simple winner detection for numeric metrics
            winner = None
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                if metric == "max_drawdown":
                    winner = "A" if val_a < val_b else ("B" if val_b < val_a else "TIE")
                else:
                    winner = "A" if val_a > val_b else ("B" if val_b > val_a else "TIE")
            table[metric] = {"A": val_a, "B": val_b, "winner": winner}

        return table

    def ranking(
        self,
        query: ResearchQuery,
        artifacts: List[str],
        metric: str = "sharpe",
    ) -> List[Dict]:
        """Rank artifacts by a given metric (descending for sharpe, ascending for drawdown)."""
        if not has_permission(query.role, Permission.COMPARE):
            return []

        ranked = []
        for art in artifacts:
            chunks = self._index.get_by_artifact(art)
            meta: Dict[str, Any] = {}
            for c in chunks:
                meta.update(c.metadata)
            ranked.append(
                {
                    "artifact_id": art,
                    "metric": metric,
                    "value": meta.get(metric, None),
                }
            )

        # Sort — lower is better for drawdown, higher for everything else
        reverse = metric != "max_drawdown"
        ranked.sort(key=lambda x: (x["value"] is not None, x["value"] or 0), reverse=reverse)

        for i, r in enumerate(ranked, 1):
            r["rank"] = i

        return ranked
