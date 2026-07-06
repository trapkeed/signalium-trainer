"""Phase 23 — Research Chat.

Conversational research mode. Researchers ask questions naturally;
responses reference actual research artifacts with citations.
Never generates trading signals.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_models import (
    ChatMessage,
    ChatSession,
    LLMResponse,
    ResearchQuery,
)
from trainer.analyst.analyst_types import Permission, QueryType, Role, has_permission
from trainer.analyst.citation_engine import CitationEngine
from trainer.analyst.knowledge_index import KnowledgeIndex
from trainer.analyst.llm_provider import LLMProvider, MockLLMProvider
from trainer.analyst.prompt_engine import PromptEngine
from trainer.analyst.retrieval_engine import RetrievalEngine


class ResearchChat:
    """Conversational research interface for SAQRA."""

    _PROHIBITED_SIGNAL_KEYWORDS = {
        "buy",
        "sell",
        "long",
        "short",
        "enter",
        "exit",
        "trade now",
        "place order",
        "execute",
        "signal",
    }

    def __init__(
        self,
        index: KnowledgeIndex,
        provider: Optional[LLMProvider] = None,
        top_k: int = 5,
    ):
        self._index = index
        self._provider = provider or MockLLMProvider()
        self._retriever = RetrievalEngine(index, top_k=top_k)
        self._prompt = PromptEngine()
        self._citation = CitationEngine()
        self._sessions: Dict[str, ChatSession] = {}

    # ── Session management ─────────────────────────────────────────────────────

    def new_session(self, researcher: str = "", role: Role = Role.RESEARCHER) -> ChatSession:
        session = ChatSession(researcher=researcher, role=role)
        self._sessions[session.session_id] = session
        logger.debug(f"ResearchChat: new session {session.session_id} for {researcher}")
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session:
            session.active = False
            return True
        return False

    def active_sessions(self) -> List[ChatSession]:
        return [s for s in self._sessions.values() if s.active]

    # ── Chat ───────────────────────────────────────────────────────────────────

    def chat(
        self,
        query: ResearchQuery,
        session_id: str,
    ) -> LLMResponse:
        """Process a research query and return a grounded LLM response."""
        # Permission check
        if not has_permission(query.role, Permission.READ):
            return LLMResponse(
                content="Access denied. Insufficient permissions.",
                safe=True,
                grounded=False,
            )

        # Safety check on incoming query
        if self._is_signal_request(query.text):
            return LLMResponse(
                content=(
                    "I cannot assist with trading signals. "
                    "SAQRA is a research analyst — please ask about model performance, "
                    "feature analysis, or experiment results."
                ),
                safe=True,
                grounded=False,
            )

        session = self._sessions.get(session_id)
        if not session:
            session = self.new_session(role=query.role)

        # Retrieve relevant context
        chunks, context = self._retriever.retrieve_and_build(
            query.text,
            top_k=5,
            source_type=query.filters.get("source_type"),
            tags=query.filters.get("tags"),
        )

        # Build prompt with history
        history = session.context_window(last_n=6)
        prompt = self._prompt.build(
            query_type=QueryType.CHAT,
            query=query.text,
            context=context,
            history=history,
        )

        # Generate
        response = self._provider.generate(prompt, context)
        response.retrieved_chunks = chunks

        # Extract and validate citations
        inline_cits = self._citation.extract(response.content)
        validated = self._citation.validate(inline_cits, chunks)
        enriched = self._citation.add_from_chunks(validated, chunks)
        response.citations = enriched

        # Record in session
        session.add_message("user", query.text)
        session.add_message("assistant", response.content, citations=response.citations)

        logger.info(
            f"ResearchChat: session={session_id} grounded={response.grounded} "
            f"citations={len(response.citations)}"
        )
        return response

    def _is_signal_request(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in self._PROHIBITED_SIGNAL_KEYWORDS)

    # ── Stats ───────────────────────────────────────────────────────────────────

    def session_stats(self) -> Dict:
        sessions = list(self._sessions.values())
        return {
            "total": len(sessions),
            "active": sum(1 for s in sessions if s.active),
            "messages": sum(s.message_count for s in sessions),
        }
