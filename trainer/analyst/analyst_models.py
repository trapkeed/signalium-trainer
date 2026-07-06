"""Phase 23 — SAQRA data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trainer.analyst.analyst_types import (
    CitationType,
    KnowledgeSourceType,
    LLMProviderType,
    Permission,
    QueryType,
    RecommendationType,
    ReportType,
    Role,
)

# ── KnowledgeChunk ─────────────────────────────────────────────────────────────


@dataclass
class KnowledgeChunk:
    """A single retrievable unit of research knowledge."""

    chunk_id: str
    source_type: KnowledgeSourceType
    artifact_id: str  # experiment ID, model ID, etc.
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "source_type": self.source_type.value,
            "artifact_id": self.artifact_id,
            "title": self.title,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "tags": self.tags,
            "timestamp": round(self.timestamp, 3),
            "word_count": self.word_count,
        }


# ── Citation ──────────────────────────────────────────────────────────────────


@dataclass
class Citation:
    """A reference to a specific research artifact."""

    citation_type: CitationType
    artifact_id: str
    description: str
    relevance: float = 1.0  # 0–1

    def to_dict(self) -> Dict:
        return {
            "citation_type": self.citation_type.value,
            "artifact_id": self.artifact_id,
            "description": self.description,
            "relevance": round(self.relevance, 3),
        }

    def __str__(self) -> str:
        return f"[{self.citation_type.value}:{self.artifact_id}]"


# ── LLMResponse ───────────────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    citations: List[Citation] = field(default_factory=list)
    provider: LLMProviderType = LLMProviderType.MOCK
    model_name: str = "mock"
    prompt_tokens: int = 0
    response_tokens: int = 0
    latency_ms: float = 0.0
    grounded: bool = True
    safe: bool = True
    retrieved_chunks: List[KnowledgeChunk] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "citations": [c.to_dict() for c in self.citations],
            "provider": self.provider.value,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "grounded": self.grounded,
            "safe": self.safe,
            "chunks_used": len(self.retrieved_chunks),
        }


# ── ResearchQuery ─────────────────────────────────────────────────────────────


@dataclass
class ResearchQuery:
    """An incoming research question or request."""

    query_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    text: str = ""
    query_type: QueryType = QueryType.CHAT
    role: Role = Role.RESEARCHER
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)
    filters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "query_type": self.query_type.value,
            "role": self.role.value,
            "session_id": self.session_id,
            "timestamp": round(self.timestamp, 3),
        }


# ── ChatMessage ───────────────────────────────────────────────────────────────


@dataclass
class ChatMessage:
    """One turn in a research chat session."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    role: str = "user"  # user | assistant
    content: str = ""
    citations: List[Citation] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0

    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "citations": [c.to_dict() for c in self.citations],
            "timestamp": round(self.timestamp, 3),
            "tokens": self.tokens,
        }


# ── ChatSession ───────────────────────────────────────────────────────────────


@dataclass
class ChatSession:
    """A research chat session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    researcher: str = ""
    role: Role = Role.RESEARCHER
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active: bool = True

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def add_message(self, role: str, content: str, citations: List[Citation] = None) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, citations=citations or [])
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def context_window(self, last_n: int = 6) -> List[ChatMessage]:
        """Return recent messages for context."""
        return self.messages[-last_n:]

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "researcher": self.researcher,
            "role": self.role.value,
            "message_count": self.message_count,
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "active": self.active,
        }


# ── ResearchReport ────────────────────────────────────────────────────────────


@dataclass
class ResearchReport:
    """A generated research report."""

    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    report_type: ReportType = ReportType.DAILY_SUMMARY
    title: str = ""
    content: str = ""
    sections: List[Dict] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)
    period: str = ""
    word_count: int = 0
    grounded: bool = True

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "title": self.title,
            "sections": [s.get("title", "") for s in self.sections],
            "citations": [c.to_dict() for c in self.citations],
            "generated_at": round(self.generated_at, 3),
            "period": self.period,
            "word_count": self.word_count,
            "grounded": self.grounded,
        }


# ── Recommendation ────────────────────────────────────────────────────────────


@dataclass
class Recommendation:
    """An evidence-based research recommendation."""

    rec_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    rec_type: RecommendationType = RecommendationType.DATA_COLLECTION
    title: str = ""
    description: str = ""
    rationale: str = ""
    citations: List[Citation] = field(default_factory=list)
    priority: str = "MEDIUM"  # HIGH | MEDIUM | LOW
    timestamp: float = field(default_factory=time.time)
    implemented: bool = False

    def to_dict(self) -> Dict:
        return {
            "rec_id": self.rec_id,
            "rec_type": self.rec_type.value,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "citations": [c.to_dict() for c in self.citations],
            "priority": self.priority,
            "timestamp": round(self.timestamp, 3),
            "implemented": self.implemented,
        }
