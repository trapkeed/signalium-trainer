"""Phase 23 — Analyst Dashboard.

5 dashboard views:
1. Research Chat
2. Experiment Explorer
3. Knowledge Graph
4. Recommendation Queue
5. Executive Dashboard
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
    ChatSession,
    ResearchReport,
    Recommendation,
    KnowledgeChunk,
)
from trainer.analyst.analyst_types import KnowledgeSourceType, RecommendationType


def _bucket(ts: float, granularity: str = "day") -> str:
    t = time.gmtime(ts)
    if granularity == "hour":
        return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d} {t.tm_hour:02d}:00"
    if granularity == "month":
        return f"{t.tm_year}-{t.tm_mon:02d}"
    return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"


class AnalystDashboard:
    """Generates the 5 SAQRA dashboard views."""

    # ── 1. Research Chat ──────────────────────────────────────────────────────

    def research_chat(
        self,
        sessions: List[ChatSession],
        granularity: str = "day",
    ) -> Dict:
        if not sessions:
            return {
                "view": "RESEARCH_CHAT",
                "total_sessions": 0,
                "active_sessions": 0,
                "total_messages": 0,
                "activity": {},
            }

        activity: Dict[str, int] = {}
        total_messages = 0

        for session in sessions:
            for msg in session.messages:
                if msg.role == "user":
                    b = _bucket(msg.timestamp, granularity)
                    activity[b] = activity.get(b, 0) + 1
            total_messages += session.message_count

        return {
            "view": "RESEARCH_CHAT",
            "total_sessions": len(sessions),
            "active_sessions": sum(1 for s in sessions if s.active),
            "total_messages": total_messages,
            "activity": dict(sorted(activity.items())),
            "researchers": list({s.researcher for s in sessions if s.researcher}),
        }

    # ── 2. Experiment Explorer ────────────────────────────────────────────────

    def experiment_explorer(
        self,
        chunks: List[KnowledgeChunk],
    ) -> Dict:
        """Explore indexed experiment and model artifacts."""
        if not chunks:
            return {"view": "EXPERIMENT_EXPLORER", "artifacts": [], "source_breakdown": {}}

        # Group by source type
        source_breakdown: Dict[str, int] = {}
        artifacts = []

        for chunk in chunks:
            st = chunk.source_type.value
            source_breakdown[st] = source_breakdown.get(st, 0) + 1

            if chunk.source_type in (
                KnowledgeSourceType.EXPERIMENT,
                KnowledgeSourceType.MODEL_REGISTRY,
                KnowledgeSourceType.VALIDATION_REPORT,
                KnowledgeSourceType.TRAINING_RUN,
            ):
                artifacts.append(
                    {
                        "artifact_id": chunk.artifact_id,
                        "source_type": st,
                        "title": chunk.title,
                        "tags": chunk.tags,
                        "timestamp": round(chunk.timestamp, 3),
                        **{
                            k: v
                            for k, v in chunk.metadata.items()
                            if k in ("sharpe", "max_drawdown", "accuracy", "status")
                        },
                    }
                )

        # Sort by timestamp desc
        artifacts.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "view": "EXPERIMENT_EXPLORER",
            "total_artifacts": len(artifacts),
            "artifacts": artifacts[:50],
            "source_breakdown": source_breakdown,
        }

    # ── 3. Knowledge Graph ────────────────────────────────────────────────────

    def knowledge_graph(
        self,
        chunks: List[KnowledgeChunk],
    ) -> Dict:
        """Map relationships between indexed knowledge artifacts."""
        if not chunks:
            return {"view": "KNOWLEDGE_GRAPH", "nodes": [], "edges": [], "clusters": {}}

        nodes = []
        edges = []
        artifact_tags: Dict[str, set] = {}

        for chunk in chunks:
            nodes.append(
                {
                    "id": chunk.chunk_id,
                    "artifact_id": chunk.artifact_id,
                    "source_type": chunk.source_type.value,
                    "title": chunk.title[:60],
                    "tags": chunk.tags,
                }
            )
            artifact_tags.setdefault(chunk.artifact_id, set()).update(chunk.tags)

        # Build edges: chunks sharing tags are related
        tag_to_chunks: Dict[str, List[str]] = {}
        for chunk in chunks:
            for tag in chunk.tags:
                tag_to_chunks.setdefault(tag, []).append(chunk.chunk_id)

        seen_edges: set = set()
        for tag, chunk_ids in tag_to_chunks.items():
            for i in range(len(chunk_ids)):
                for j in range(i + 1, len(chunk_ids)):
                    edge_key = tuple(sorted([chunk_ids[i], chunk_ids[j]]))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(
                            {
                                "from": chunk_ids[i],
                                "to": chunk_ids[j],
                                "via": tag,
                                "weight": 1,
                            }
                        )

        # Cluster by source type
        clusters: Dict[str, List[str]] = {}
        for chunk in chunks:
            st = chunk.source_type.value
            clusters.setdefault(st, []).append(chunk.artifact_id)

        return {
            "view": "KNOWLEDGE_GRAPH",
            "nodes": nodes,
            "edges": edges[:200],  # cap edges for display
            "clusters": {k: list(set(v)) for k, v in clusters.items()},
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    # ── 4. Recommendation Queue ───────────────────────────────────────────────

    def recommendation_queue(
        self,
        recommendations: List[Recommendation],
    ) -> Dict:
        if not recommendations:
            return {
                "view": "RECOMMENDATION_QUEUE",
                "pending": 0,
                "implemented": 0,
                "by_priority": {},
                "by_type": {},
                "items": [],
            }

        pending = [r for r in recommendations if not r.implemented]
        implemented = [r for r in recommendations if r.implemented]

        by_priority: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for r in pending:
            by_priority[r.priority] = by_priority.get(r.priority, 0) + 1
            by_type[r.rec_type.value] = by_type.get(r.rec_type.value, 0) + 1

        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        pending_sorted = sorted(pending, key=lambda r: priority_order.get(r.priority, 3))

        return {
            "view": "RECOMMENDATION_QUEUE",
            "pending": len(pending),
            "implemented": len(implemented),
            "by_priority": by_priority,
            "by_type": by_type,
            "items": [r.to_dict() for r in pending_sorted[:20]],
        }

    # ── 5. Executive Dashboard ────────────────────────────────────────────────

    def executive_dashboard(
        self,
        sessions: List[ChatSession],
        reports: List[ResearchReport],
        recommendations: List[Recommendation],
        chunks: List[KnowledgeChunk],
    ) -> Dict:
        pending_recs = [r for r in recommendations if not r.implemented]
        high_priority = [r for r in pending_recs if r.priority == "HIGH"]

        recent_reports = sorted(reports, key=lambda r: r.generated_at, reverse=True)[:3]

        return {
            "view": "EXECUTIVE_DASHBOARD",
            "knowledge_base": {
                "total_chunks": len(chunks),
                "source_types": len({c.source_type for c in chunks}),
            },
            "research_activity": {
                "total_sessions": len(sessions),
                "active_sessions": sum(1 for s in sessions if s.active),
                "total_queries": sum(s.message_count // 2 for s in sessions),
            },
            "reports": {
                "total_generated": len(reports),
                "recent": [{"title": r.title, "type": r.report_type.value} for r in recent_reports],
            },
            "recommendations": {
                "pending": len(pending_recs),
                "high_priority": len(high_priority),
                "implemented": len([r for r in recommendations if r.implemented]),
            },
            "health": {
                "knowledge_base_populated": len(chunks) > 0,
                "active_sessions": len(sessions) > 0,
                "recent_reports": len(recent_reports) > 0,
            },
        }

    def all_dashboards(
        self,
        sessions: List[ChatSession],
        reports: List[ResearchReport],
        recommendations: List[Recommendation],
        chunks: List[KnowledgeChunk],
    ) -> Dict:
        return {
            "research_chat": self.research_chat(sessions),
            "experiment_explorer": self.experiment_explorer(chunks),
            "knowledge_graph": self.knowledge_graph(chunks),
            "recommendation_queue": self.recommendation_queue(recommendations),
            "executive_dashboard": self.executive_dashboard(
                sessions, reports, recommendations, chunks
            ),
        }
