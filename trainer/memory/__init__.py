"""Phase 21.6 — Institutional Decision Memory & Self-Evaluation Engine."""

from trainer.memory.decision_types import (
    DecisionType,
    OutcomeType,
    QualityFactor,
    MarketRegime,
    VolatilityRegime,
    DecisionGrade,
    QUALITY_WEIGHTS,
    GRADE_THRESHOLDS,
    POSITIVE_OUTCOMES,
    NEGATIVE_OUTCOMES,
    grade_from_score,
)

from trainer.memory.decision_models import (
    DecisionSnapshot,
    DecisionReasoning,
    DecisionOutcome,
    DecisionScore,
    CounterfactualResult,
    DecisionRecord,
)

from trainer.memory.decision_memory import DecisionMemory
from trainer.memory.outcome_tracker import OutcomeTracker, OutcomeSummary
from trainer.memory.quality_engine import QualityEngine
from trainer.memory.counterfactual_engine import CounterfactualEngine
from trainer.memory.pattern_discovery import PatternDiscovery, DiscoveredPattern, PatternReport
from trainer.memory.self_evaluation import SelfEvaluationEngine, SelfAssessment
from trainer.memory.knowledge_base import KnowledgeBase, KnowledgeEntry
from trainer.memory.decision_lineage import DecisionLineage, LineageNode, LineageRecord
from trainer.memory.decision_benchmarking import (
    DecisionBenchmarking,
    BenchmarkBaseline,
    BenchmarkResult,
    BenchmarkReport,
)
from trainer.memory.memory_search import MemorySearch, SearchQuery, SearchResult
from trainer.memory.memory_database import MemoryDatabase
from trainer.memory.memory_dashboard import MemoryDashboard
from trainer.memory.memory_api import MemoryAPI

__all__ = [
    # Types
    "DecisionType",
    "OutcomeType",
    "QualityFactor",
    "MarketRegime",
    "VolatilityRegime",
    "DecisionGrade",
    "QUALITY_WEIGHTS",
    "GRADE_THRESHOLDS",
    "POSITIVE_OUTCOMES",
    "NEGATIVE_OUTCOMES",
    "grade_from_score",
    # Models
    "DecisionSnapshot",
    "DecisionReasoning",
    "DecisionOutcome",
    "DecisionScore",
    "CounterfactualResult",
    "DecisionRecord",
    # Engines
    "DecisionMemory",
    "OutcomeTracker",
    "OutcomeSummary",
    "QualityEngine",
    "CounterfactualEngine",
    "PatternDiscovery",
    "DiscoveredPattern",
    "PatternReport",
    "SelfEvaluationEngine",
    "SelfAssessment",
    "KnowledgeBase",
    "KnowledgeEntry",
    "DecisionLineage",
    "LineageNode",
    "LineageRecord",
    "DecisionBenchmarking",
    "BenchmarkBaseline",
    "BenchmarkResult",
    "BenchmarkReport",
    "MemorySearch",
    "SearchQuery",
    "SearchResult",
    "MemoryDatabase",
    "MemoryDashboard",
    # Unified API
    "MemoryAPI",
]
