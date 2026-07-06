"""Phase 23 — Analyst API.

Unified API for SAQRA — Signalium's AI Quant Research Analyst.

Endpoints:
1. chat              — conversational research
2. search            — natural language knowledge search
3. compare           — structured model/experiment comparison
4. report            — professional report generation
5. recommend         — evidence-based recommendations
6. knowledge_search  — direct knowledge index search
7. get_dashboard     — dashboard views
8. system_stats      — platform statistics
9. index_document    — add document to knowledge base
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_dashboard import AnalystDashboard
from trainer.analyst.analyst_database import AnalystDatabase
from trainer.analyst.analyst_models import ResearchQuery
from trainer.analyst.analyst_types import (
    KnowledgeSourceType,
    LLMProviderType,
    Permission,
    QueryType,
    ReportType,
    Role,
    has_permission,
)
from trainer.analyst.knowledge_index import KnowledgeIndex
from trainer.analyst.llm_provider import LLMProvider, MockLLMProvider, create_provider
from trainer.analyst.model_comparator import ModelComparator
from trainer.analyst.recommendation_engine import RecommendationEngine
from trainer.analyst.report_generator import ReportGenerator
from trainer.analyst.research_chat import ResearchChat
from trainer.analyst.research_search import ResearchSearch


class AnalystAPI:
    """Unified SAQRA API — entry point for all research analyst operations."""

    def __init__(
        self,
        db_path: str = ":memory:",
        provider_type: LLMProviderType = LLMProviderType.MOCK,
        **provider_kwargs,
    ):
        self._index = KnowledgeIndex()
        self._provider: LLMProvider = create_provider(provider_type, **provider_kwargs)
        self._db = AnalystDatabase(db_path)
        self._dash = AnalystDashboard()

        # Sub-systems
        self._chat = ResearchChat(self._index, self._provider)
        self._search = ResearchSearch(self._index, self._provider)
        self._comparator = ModelComparator(self._index, self._provider)
        self._reporter = ReportGenerator(self._index, self._provider)
        self._recommender = RecommendationEngine(self._index, self._provider)

    # ── 1. Chat ───────────────────────────────────────────────────────────────

    def chat(
        self,
        text: str,
        session_id: Optional[str] = None,
        researcher: str = "",
        role: Role = Role.RESEARCHER,
        filters: Dict = None,
    ) -> Dict:
        """Start or continue a research chat session."""
        # Ensure session exists
        if session_id is None or self._chat.get_session(session_id) is None:
            session = self._chat.new_session(researcher=researcher, role=role)
            session_id = session.session_id

        query = ResearchQuery(
            text=text,
            query_type=QueryType.CHAT,
            role=role,
            session_id=session_id,
            filters=filters or {},
        )

        response = self._chat.chat(query, session_id)

        # Persist
        self._db.save_query(query, response_preview=response.content[:200])
        self._db.record_usage(
            provider=response.provider.value,
            model_name=response.model_name,
            prompt_tokens=response.prompt_tokens,
            response_tokens=response.response_tokens,
            latency_ms=response.latency_ms,
            grounded=response.grounded,
            safe=response.safe,
            query_type=QueryType.CHAT.value,
        )

        session = self._chat.get_session(session_id)
        if session:
            self._db.save_session(session)

        logger.info(f"AnalystAPI.chat: session={session_id} grounded={response.grounded}")

        return {
            "session_id": session_id,
            "response": response.to_dict(),
        }

    def new_session(self, researcher: str = "", role: Role = Role.RESEARCHER) -> str:
        """Create a new chat session and return its ID."""
        session = self._chat.new_session(researcher=researcher, role=role)
        self._db.save_session(session)
        return session.session_id

    # ── 2. Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        text: str,
        source_type: Optional[KnowledgeSourceType] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 10,
        role: Role = Role.RESEARCHER,
    ) -> Dict:
        """Natural language search over the knowledge base."""
        query = ResearchQuery(text=text, query_type=QueryType.SEARCH, role=role)
        result = self._search.search(query, source_type=source_type, tags=tags, top_k=top_k)
        self._db.save_query(query)
        return result

    # ── 3. Compare ────────────────────────────────────────────────────────────

    def compare(
        self,
        artifact_a: str,
        artifact_b: str,
        question: str = "",
        metrics: Optional[List[str]] = None,
        role: Role = Role.RESEARCHER,
    ) -> Dict:
        """Compare two research artifacts."""
        query = ResearchQuery(
            text=question or f"Compare {artifact_a} vs {artifact_b}",
            query_type=QueryType.COMPARISON,
            role=role,
        )
        result = self._comparator.compare(query, artifact_a, artifact_b, metrics)
        self._db.save_query(query)
        return result

    def rank(
        self,
        artifacts: List[str],
        metric: str = "sharpe",
        role: Role = Role.RESEARCHER,
    ) -> List[Dict]:
        """Rank multiple artifacts by a metric."""
        query = ResearchQuery(
            text=f"rank by {metric}",
            query_type=QueryType.COMPARISON,
            role=role,
        )
        return self._comparator.ranking(query, artifacts, metric)

    # ── 4. Report ─────────────────────────────────────────────────────────────

    def report(
        self,
        report_type: ReportType,
        question: str = "",
        period: str = "",
        role: Role = Role.RESEARCHER,
    ) -> Dict:
        """Generate a research report."""
        query = ResearchQuery(
            text=question or f"Generate {report_type.value}",
            query_type=QueryType.REPORT,
            role=role,
        )
        rep = self._reporter.generate(query, report_type, period)
        self._db.save_report(rep)
        self._db.save_query(query)
        return rep.to_dict()

    # ── 5. Recommend ──────────────────────────────────────────────────────────

    def recommend(
        self,
        question: str = "What research improvements should we prioritise?",
        role: Role = Role.RESEARCHER,
    ) -> List[Dict]:
        """Generate evidence-based research recommendations."""
        query = ResearchQuery(text=question, query_type=QueryType.RECOMMENDATION, role=role)
        recs = self._recommender.recommend(query)
        for r in recs:
            self._db.save_recommendation(r)
        self._db.save_query(query)
        return [r.to_dict() for r in recs]

    def auto_recommend(self) -> List[Dict]:
        """Scan knowledge base and auto-generate recommendations."""
        recs = self._recommender.auto_recommend()
        for r in recs:
            self._db.save_recommendation(r)
        return [r.to_dict() for r in recs]

    def pending_recommendations(self) -> List[Dict]:
        return [r.to_dict() for r in self._recommender.pending()]

    # ── 6. Knowledge search ───────────────────────────────────────────────────

    def knowledge_search(self, keyword: str, top_k: int = 10) -> List[Dict]:
        """Direct keyword search over the knowledge index."""
        chunks = self._search.keyword_search(keyword, top_k=top_k)
        return [c.to_dict() for c in chunks]

    # ── 7. Dashboard ──────────────────────────────────────────────────────────

    def get_dashboard(self, name: str = "all") -> Dict:
        sessions = self._chat.active_sessions()
        reports = self._reporter.list_reports()
        report_objs = [
            self._reporter.get_report(r["report_id"])
            for r in reports
            if self._reporter.get_report(r["report_id"])
        ]
        recs = self._recommender.history()
        chunks = self._index.all_chunks()

        views = {
            "research_chat": lambda: self._dash.research_chat(sessions),
            "experiment_explorer": lambda: self._dash.experiment_explorer(chunks),
            "knowledge_graph": lambda: self._dash.knowledge_graph(chunks),
            "recommendation_queue": lambda: self._dash.recommendation_queue(recs),
            "executive_dashboard": lambda: self._dash.executive_dashboard(
                sessions, report_objs, recs, chunks
            ),
        }

        if name == "all":
            return self._dash.all_dashboards(sessions, report_objs, recs, chunks)

        fn = views.get(name)
        if fn:
            return fn()
        return {"error": f"Unknown dashboard '{name}'"}

    # ── 8. System stats ───────────────────────────────────────────────────────

    def system_stats(self) -> Dict:
        return {
            "knowledge_index": self._index.stats(),
            "database": self._db.stats(),
            "chat": self._chat.session_stats(),
            "reports": len(self._reporter.list_reports()),
            "recommendations": len(self._recommender.history()),
            "provider": self._provider.provider_type().value,
            "provider_available": self._provider.is_available(),
        }

    # ── 9. Index document ─────────────────────────────────────────────────────

    def index_document(
        self,
        source_type: KnowledgeSourceType,
        artifact_id: str,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        role: Role = Role.ADMIN,
    ) -> Dict:
        """Add a document to the knowledge base."""
        if not has_permission(role, Permission.READ):
            return {"error": "Access denied"}
        chunk = self._index.index_document(
            source_type=source_type,
            artifact_id=artifact_id,
            title=title,
            content=content,
            tags=tags or [],
            metadata=metadata or {},
        )
        return {
            "chunk_id": chunk.chunk_id,
            "artifact_id": chunk.artifact_id,
            "source_type": chunk.source_type.value,
            "indexed": True,
        }

    def switch_provider(
        self,
        provider_type: LLMProviderType,
        **kwargs,
    ) -> Dict:
        """Switch the LLM provider without restarting the system."""
        new_provider = create_provider(provider_type, **kwargs)
        self._provider = new_provider
        self._chat._provider = new_provider
        self._search._provider = new_provider
        self._comparator._provider = new_provider
        self._reporter._provider = new_provider
        self._recommender._provider = new_provider

        return {
            "provider": provider_type.value,
            "available": new_provider.is_available(),
        }
