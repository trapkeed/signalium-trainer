"""Phase 23 — Retrieval Engine.

Retrieves relevant KnowledgeChunks for a given query using BM25-like
keyword scoring. Builds the context string fed into the LLM prompt.
"""

from __future__ import annotations

import math
from typing import List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_types import KnowledgeSourceType
from trainer.analyst.analyst_models import KnowledgeChunk
from trainer.analyst.knowledge_index import KnowledgeIndex


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "and", "or",
    "of", "for", "with", "this", "that", "was", "are", "be", "by", "from",
    "as", "what", "which", "where", "when", "how", "did", "does", "do",
    "i", "we", "you", "they", "he", "she", "who",
})


def _tokenise(text: str) -> List[str]:
    """Lowercase tokenise, remove stopwords and short tokens."""
    tokens = text.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


class RetrievalEngine:
    """BM25-inspired retrieval over the KnowledgeIndex."""

    # BM25 parameters
    K1 = 1.5   # term-frequency saturation
    B  = 0.75  # length normalisation

    def __init__(self, index: KnowledgeIndex, top_k: int = 5, min_score: float = 0.0):
        self._index   = index
        self._top_k   = top_k
        self._min_score = min_score

    def retrieve(
        self,
        query:        str,
        top_k:        Optional[int]                 = None,
        source_type:  Optional[KnowledgeSourceType] = None,
        tags:         Optional[List[str]]           = None,
    ) -> List[KnowledgeChunk]:
        """Retrieve the most relevant chunks for a query."""
        k = top_k or self._top_k
        query_terms = _tokenise(query)

        if not query_terms:
            # Fallback: return recent chunks
            return self._index.all_chunks()[:k]

        candidates = (
            self._index.get_by_source(source_type)
            if source_type else self._index.all_chunks()
        )

        if not candidates:
            logger.debug("RetrievalEngine: no candidates in index")
            return []

        if tags:
            candidates = [
                c for c in candidates
                if any(t.lower() in (ct.lower() for ct in c.tags) for t in tags)
            ]

        # Compute average document length for BM25 normalisation
        avg_len = sum(len(_tokenise(c.content)) for c in candidates) / max(len(candidates), 1)

        scored = []
        for chunk in candidates:
            doc_terms   = _tokenise(chunk.title + " " + chunk.content)
            tag_terms   = [t.lower() for t in chunk.tags]
            doc_len     = max(len(doc_terms), 1)
            term_freq: dict = {}
            for t in doc_terms:
                term_freq[t] = term_freq.get(t, 0) + 1

            score = 0.0
            for term in query_terms:
                tf  = term_freq.get(term, 0)
                # Title + tag boosts
                tf += 3 * sum(1 for t in _tokenise(chunk.title) if t == term)
                tf += 2 * sum(1 for t in tag_terms if t == term)

                # BM25 score
                idf   = math.log(1 + (len(candidates) / (1 + sum(1 for c in candidates if term in _tokenise(c.content)))))
                score += idf * (tf * (self.K1 + 1)) / (tf + self.K1 * (1 - self.B + self.B * doc_len / avg_len))

            if score > self._min_score:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [c for _, c in scored[:k]]

        logger.debug(f"RetrievalEngine: query='{query}' retrieved {len(result)}/{len(candidates)} chunks")
        return result

    def build_context(self, chunks: List[KnowledgeChunk], max_chars: int = 3000) -> str:
        """Build a context string from retrieved chunks for the LLM prompt."""
        if not chunks:
            return ""

        parts = []
        used  = 0
        for chunk in chunks:
            entry = (
                f"[{chunk.source_type.value}:{chunk.artifact_id}] {chunk.title}\n"
                f"{chunk.content}"
            )
            if used + len(entry) > max_chars:
                remaining = max_chars - used
                if remaining > 100:
                    parts.append(entry[:remaining] + "...")
                break
            parts.append(entry)
            used += len(entry)

        return "\n---\n".join(parts)

    def retrieve_and_build(
        self,
        query:       str,
        top_k:       Optional[int]                 = None,
        source_type: Optional[KnowledgeSourceType] = None,
        tags:        Optional[List[str]]           = None,
        max_chars:   int                           = 3000,
    ) -> tuple:
        """Retrieve chunks and build context. Returns (chunks, context_str)."""
        chunks  = self.retrieve(query, top_k, source_type, tags)
        context = self.build_context(chunks, max_chars)
        return chunks, context
