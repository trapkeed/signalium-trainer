"""Phase 23 — Knowledge Index.

Indexes research artifacts into retrievable KnowledgeChunks.
Supports all 20 knowledge source types.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_types import KnowledgeSourceType
from trainer.analyst.analyst_models import KnowledgeChunk


class KnowledgeIndex:
    """In-memory knowledge index for RAG retrieval.

    Documents are stored as KnowledgeChunks and retrieved by keyword matching.
    """

    def __init__(self):
        self._chunks: Dict[str, KnowledgeChunk] = {}  # chunk_id → chunk
        self._by_source: Dict[str, List[str]] = {}  # source_type → [chunk_ids]
        self._by_artifact: Dict[str, List[str]] = {}  # artifact_id → [chunk_ids]

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index(self, chunk: KnowledgeChunk) -> str:
        """Index a KnowledgeChunk. Returns the chunk_id."""
        self._chunks[chunk.chunk_id] = chunk

        st = chunk.source_type.value
        self._by_source.setdefault(st, [])
        if chunk.chunk_id not in self._by_source[st]:
            self._by_source[st].append(chunk.chunk_id)

        self._by_artifact.setdefault(chunk.artifact_id, [])
        if chunk.chunk_id not in self._by_artifact[chunk.artifact_id]:
            self._by_artifact[chunk.artifact_id].append(chunk.chunk_id)

        logger.debug(f"KnowledgeIndex: indexed chunk_id={chunk.chunk_id} source={st}")
        return chunk.chunk_id

    def index_document(
        self,
        source_type: KnowledgeSourceType,
        artifact_id: str,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> KnowledgeChunk:
        """Create and index a KnowledgeChunk from raw document fields."""
        chunk = KnowledgeChunk(
            chunk_id=str(uuid.uuid4())[:12],
            source_type=source_type,
            artifact_id=artifact_id,
            title=title,
            content=content,
            tags=tags or [],
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self.index(chunk)
        return chunk

    def index_batch(self, chunks: List[KnowledgeChunk]) -> List[str]:
        """Index a batch of chunks. Returns list of chunk_ids."""
        return [self.index(c) for c in chunks]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        return self._chunks.get(chunk_id)

    def get_by_artifact(self, artifact_id: str) -> List[KnowledgeChunk]:
        ids = self._by_artifact.get(artifact_id, [])
        return [self._chunks[i] for i in ids if i in self._chunks]

    def get_by_source(self, source_type: KnowledgeSourceType) -> List[KnowledgeChunk]:
        ids = self._by_source.get(source_type.value, [])
        return [self._chunks[i] for i in ids if i in self._chunks]

    def all_chunks(self) -> List[KnowledgeChunk]:
        return list(self._chunks.values())

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[KnowledgeSourceType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[KnowledgeChunk]:
        """Keyword-based search across the knowledge index.

        Scores each chunk by term overlap with the query.
        Returns top_k chunks sorted by relevance descending.
        """
        query_terms = set(query.lower().split())
        candidates = self.get_by_source(source_type) if source_type else self.all_chunks()

        scored = []
        for chunk in candidates:
            # Filter by tags if specified
            if tags:
                if not any(t.lower() in (ct.lower() for ct in chunk.tags) for t in tags):
                    continue

            # TF-style score: count query term occurrences in title + content
            text = (chunk.title + " " + chunk.content).lower()
            words = set(text.split())
            score = len(query_terms & words)

            # Boost for title matches
            title_words = set(chunk.title.lower().split())
            score += len(query_terms & title_words) * 2

            # Boost for tag matches
            tag_words = set(t.lower() for t in chunk.tags)
            score += len(query_terms & tag_words)

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [c for _, c in scored[:top_k]]

        logger.debug(f"KnowledgeIndex.search: query='{query}' found {len(results)} chunks")
        return results

    # ── Management ────────────────────────────────────────────────────────────

    def remove(self, chunk_id: str) -> bool:
        chunk = self._chunks.pop(chunk_id, None)
        if not chunk:
            return False
        st = chunk.source_type.value
        if st in self._by_source:
            self._by_source[st] = [i for i in self._by_source[st] if i != chunk_id]
        art = chunk.artifact_id
        if art in self._by_artifact:
            self._by_artifact[art] = [i for i in self._by_artifact[art] if i != chunk_id]
        return True

    def clear(self) -> None:
        self._chunks.clear()
        self._by_source.clear()
        self._by_artifact.clear()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        source_counts = {st: len(ids) for st, ids in self._by_source.items()}
        return {
            "total_chunks": len(self._chunks),
            "artifact_count": len(self._by_artifact),
            "source_types": source_counts,
        }
