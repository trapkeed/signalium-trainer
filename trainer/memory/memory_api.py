"""Phase 21.6 — Memory API.

16 endpoints for the Decision Memory system:
 1. record_decision
 2. get_decision
 3. record_outcome
 4. score_decision
 5. score_all
 6. analyse_counterfactuals
 7. summarise_outcomes
 8. discover_patterns
 9. self_evaluate
10. search_decisions
11. knowledge_search
12. index_in_knowledge_base
13. build_lineage
14. benchmark
15. get_dashboard
16. system_stats
"""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.counterfactual_engine import CounterfactualEngine
from trainer.memory.decision_benchmarking import DecisionBenchmarking
from trainer.memory.decision_lineage import DecisionLineage
from trainer.memory.decision_memory import DecisionMemory
from trainer.memory.decision_models import DecisionReasoning, DecisionRecord, DecisionSnapshot
from trainer.memory.decision_types import DecisionType, OutcomeType
from trainer.memory.knowledge_base import KnowledgeBase
from trainer.memory.memory_dashboard import MemoryDashboard
from trainer.memory.memory_database import MemoryDatabase
from trainer.memory.memory_search import MemorySearch, SearchQuery
from trainer.memory.outcome_tracker import OutcomeTracker
from trainer.memory.pattern_discovery import PatternDiscovery
from trainer.memory.quality_engine import QualityEngine
from trainer.memory.self_evaluation import SelfEvaluationEngine


class MemoryAPI:
    """Unified API for the Decision Memory and Self-Evaluation Engine."""

    def __init__(self, db_path: str = ":memory:"):
        self.memory = DecisionMemory()
        self.tracker = OutcomeTracker()
        self.quality = QualityEngine()
        self.counterfact = CounterfactualEngine()
        self.patterns = PatternDiscovery()
        self.evaluator = SelfEvaluationEngine()
        self.knowledge = KnowledgeBase()
        self.lineage = DecisionLineage()
        self.benchmarking = DecisionBenchmarking()
        self.search = MemorySearch()
        self.dashboard = MemoryDashboard()
        self.db = MemoryDatabase(db_path)

    # ── 1. Record decision ────────────────────────────────────────────────────

    def record_decision(
        self,
        decision_type: DecisionType,
        instrument: str,
        snapshot: DecisionSnapshot,
        reasoning: DecisionReasoning,
        tags: list[str] | None = None,
        trade_id: str = "",
        lineage_id: str = "",
        persist: bool = True,
    ) -> DecisionRecord:
        record = self.memory.record_decision(
            decision_type=decision_type,
            instrument=instrument,
            snapshot=snapshot,
            reasoning=reasoning,
            tags=tags,
            trade_id=trade_id,
            lineage_id=lineage_id,
        )
        if persist:
            self.db.save_decision(record)
        return record

    # ── 2. Get decision ───────────────────────────────────────────────────────

    def get_decision(self, decision_id: str) -> dict | None:
        record = self.memory.get(decision_id)
        return record.to_dict() if record else None

    # ── 3. Record outcome ─────────────────────────────────────────────────────

    def record_outcome(
        self,
        decision_id: str,
        outcome_type: OutcomeType,
        actual_pnl: float = 0.0,
        actual_pnl_pct: float = 0.0,
        hit_tp: bool = False,
        hit_sl: bool = False,
        holding_time_h: float = 0.0,
        direction_correct: bool | None = None,
        market_event: str = "",
        notes: str = "",
        persist: bool = True,
    ) -> bool:
        ok = self.memory.record_outcome(
            decision_id=decision_id,
            outcome_type=outcome_type,
            actual_pnl=actual_pnl,
            actual_pnl_pct=actual_pnl_pct,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
            holding_time_h=holding_time_h,
            direction_correct=direction_correct,
            market_event=market_event,
            notes=notes,
        )
        if ok and persist:
            record = self.memory.get(decision_id)
            if record:
                self.db.save_decision(record)
        return ok

    # ── 4. Score decision ─────────────────────────────────────────────────────

    def score_decision(self, decision_id: str) -> dict | None:
        record = self.memory.get(decision_id)
        if not record:
            return None
        score = self.quality.score_and_attach(record)
        self.db.save_decision(record)
        return score.to_dict()

    # ── 5. Score all ──────────────────────────────────────────────────────────

    def score_all(self) -> dict:
        records = self.memory.all_records()
        scores = self.quality.batch_score(records)
        return {
            "scored": len(scores),
            "avg_quality": (
                round(sum(s.overall for s in scores) / len(scores), 4) if scores else 0.0
            ),
        }

    # ── 6. Analyse counterfactuals ────────────────────────────────────────────

    def analyse_counterfactuals(self, decision_id: str) -> list[dict]:
        record = self.memory.get(decision_id)
        if not record:
            return []
        results = self.counterfact.analyse(record)
        self.db.save_decision(record)
        return [c.to_dict() for c in results]

    # ── 7. Summarise outcomes ─────────────────────────────────────────────────

    def summarise_outcomes(
        self,
        instrument: str | None = None,
        decision_type: DecisionType | None = None,
    ) -> dict:
        records = self.memory.filter(instrument=instrument, decision_type=decision_type)
        summary = self.tracker.summarise(records, decision_type=decision_type)
        return summary.to_dict()

    # ── 8. Discover patterns ──────────────────────────────────────────────────

    def discover_patterns(self) -> dict:
        records = self.memory.all_records()
        report = self.patterns.discover(records)
        return report.to_dict()

    # ── 9. Self-evaluate ──────────────────────────────────────────────────────

    def self_evaluate(self, period: str = "CURRENT") -> dict:
        records = self.memory.all_records()
        assessment = self.evaluator.evaluate(records, period=period)
        return assessment.to_dict()

    # ── 10. Search decisions ──────────────────────────────────────────────────

    def search_decisions(self, query: SearchQuery) -> dict:
        records = self.memory.all_records()
        result = self.search.search(records, query)
        return result.to_dict()

    # ── 11. Knowledge search ──────────────────────────────────────────────────

    def knowledge_search(
        self,
        query: str = "",
        entry_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        entries = self.knowledge.search(query=query, entry_type=entry_type, tags=tags, limit=limit)
        return [e.to_dict() for e in entries]

    # ── 12. Index in knowledge base ───────────────────────────────────────────

    def index_decision_in_knowledge_base(self, decision_id: str) -> dict | None:
        record = self.memory.get(decision_id)
        if not record:
            return None
        entry = self.knowledge.index_decision(record)
        return entry.to_dict()

    # ── 13. Build lineage ─────────────────────────────────────────────────────

    def build_lineage(
        self,
        decision_id: str,
        model_version: str = "",
        dataset_name: str = "",
        feature_set: str = "",
        training_run: str = "",
        experiment: str = "",
        git_commit: str = "",
    ) -> dict | None:
        record = self.memory.get(decision_id)
        if not record:
            return None
        lin = self.lineage.create_lineage(
            decision_id=decision_id,
            model_version=model_version,
            dataset_name=dataset_name,
            feature_set=feature_set,
            training_run=training_run,
            experiment=experiment,
            git_commit=git_commit,
        )
        record.lineage_id = lin.lineage_id
        return lin.to_dict()

    # ── 14. Benchmark ─────────────────────────────────────────────────────────

    def benchmark(self, baselines: list[str] | None = None) -> dict:
        records = self.memory.all_records()
        report = self.benchmarking.benchmark(records, baselines=baselines)
        return report.to_dict()

    # ── 15. Get dashboard ─────────────────────────────────────────────────────

    def get_dashboard(self, name: str = "all") -> dict:
        records = self.memory.all_records()
        fns = {
            "decision_timeline": self.dashboard.decision_timeline,
            "decision_tree": self.dashboard.decision_tree,
            "quality_distribution": self.dashboard.quality_distribution,
            "failure_analysis": self.dashboard.failure_analysis,
            "counterfactual_explorer": self.dashboard.counterfactual_explorer,
            "knowledge_graph": self.dashboard.knowledge_graph,
            "learning_timeline": self.dashboard.learning_timeline,
            "decision_heatmaps": self.dashboard.decision_heatmaps,
        }
        if name == "all":
            return self.dashboard.all_dashboards(records)
        fn = fns.get(name)
        if fn:
            return fn(records)
        return {"error": f"Unknown dashboard '{name}'"}

    # ── 16. System stats ──────────────────────────────────────────────────────

    def system_stats(self) -> dict:
        mem_stats = self.memory.stats()
        db_stats = self.db.stats()
        kb_stats = self.knowledge.stats()
        lin_stats = self.lineage.stats()
        return {
            "memory": mem_stats,
            "database": db_stats,
            "knowledge": kb_stats,
            "lineage": lin_stats,
            "baselines": len(self.benchmarking.list_baselines()),
        }
