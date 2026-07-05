"""Signalium Research & Benchmarking Framework.

A complete research environment for objective comparison and lifecycle
management of quantitative trading models.

Phase 19 — Research Intelligence Platform::

    from trainer.research import (
        ResearchObservatory, MarketEvolutionMonitor,
        FeatureLifecycleManager, LabelQualityMonitor,
        DatasetEvolutionTracker, ModelHealthEvolutionTracker,
        RetrainingAdvisor, build_default_advisor,
        ExperimentRecommendationEngine, KnowledgeBase,
        ResearchAssistant, ResearchEngine,
    )

Phase 19.1 — Experiment Lineage & Reproducibility::

    from trainer.research import (
        EnvironmentSnapshot,
        DatasetSnapshot, FeatureSnapshot, ModelSnapshot, ValidationSnapshot,
        LineageNode, LineageTree, ExperimentStatus,
        DatasetDiff, FeatureDiff, ModelDiff, ValidationDiff, ExperimentDiffer,
        KnowledgeGraph, NodeType, EdgeType,
        ResearchTimeline, TimelineEvent, TimelineEventType,
        CostTracker, ExperimentCost,
        DashboardBuilder,
        ResearchExporter,
        LineageDatabase,
        LineageAPI,
    )
"""

from trainer.research.benchmark import (
    BenchmarkConditions,
    BenchmarkResult,
    BenchmarkRunner,
    ModelRunResult,
)
from trainer.research.champion import ChampionRegistry, ModelEntry, ModelStatus
from trainer.research.experiment import ExperimentRecord, ExperimentTracker
from trainer.research.hyperparameter import HyperparameterRecord
from trainer.research.leaderboard import Leaderboard, LeaderboardEntry, ScoringWeights
from trainer.research.report import ReportConfig, ReportGenerator
from trainer.research.validation import (
    RollingFold,
    RollingWindowValidator,
    ValidationFold,
    WalkForwardValidator,
)

# Phase 19.1 — Experiment Lineage & Reproducibility
from trainer.research.reproducibility import EnvironmentSnapshot
from trainer.research.snapshots import (
    DatasetSnapshot,
    FeatureSnapshot,
    ModelSnapshot,
    ValidationSnapshot,
)
from trainer.research.lineage import (
    ExperimentStatus,
    LineageNode,
    LineageTree,
)
from trainer.research.differencing import (
    DatasetDiff,
    ExperimentDiff,
    ExperimentDiffer,
    FeatureDiff,
    ModelDiff,
    ValidationDiff,
)
from trainer.research.knowledge_graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    NodeType,
)
from trainer.research.timeline import (
    ResearchTimeline,
    TimelineEvent,
    TimelineEventType,
)
from trainer.research.cost_tracker import CostTracker, ExperimentCost
from trainer.research.dashboard import DashboardBuilder
from trainer.research.export import ResearchExporter
from trainer.research.lineage_database import LineageDatabase
from trainer.research.lineage_api import LineageAPI

__all__ = [
    # Experiment (legacy)
    "ExperimentRecord",
    "ExperimentTracker",
    # Benchmark
    "BenchmarkConditions",
    "BenchmarkRunner",
    "BenchmarkResult",
    "ModelRunResult",
    # Leaderboard
    "Leaderboard",
    "LeaderboardEntry",
    "ScoringWeights",
    # Validation (legacy)
    "WalkForwardValidator",
    "ValidationFold",
    "RollingWindowValidator",
    "RollingFold",
    # Hyperparameters
    "HyperparameterRecord",
    # Reports
    "ReportGenerator",
    "ReportConfig",
    # Champion/Challenger
    "ModelStatus",
    "ChampionRegistry",
    "ModelEntry",
    # Phase 19.1 — Reproducibility
    "EnvironmentSnapshot",
    # Phase 19.1 — Snapshots
    "DatasetSnapshot",
    "FeatureSnapshot",
    "ModelSnapshot",
    "ValidationSnapshot",
    # Phase 19.1 — Lineage
    "ExperimentStatus",
    "LineageNode",
    "LineageTree",
    # Phase 19.1 — Differencing
    "DatasetDiff",
    "ExperimentDiff",
    "ExperimentDiffer",
    "FeatureDiff",
    "ModelDiff",
    "ValidationDiff",
    # Phase 19.1 — Knowledge Graph
    "EdgeType",
    "GraphEdge",
    "GraphNode",
    "KnowledgeGraph",
    "NodeType",
    # Phase 19.1 — Timeline
    "ResearchTimeline",
    "TimelineEvent",
    "TimelineEventType",
    # Phase 19.1 — Cost
    "CostTracker",
    "ExperimentCost",
    # Phase 19.1 — Dashboard
    "DashboardBuilder",
    # Phase 19.1 — Export
    "ResearchExporter",
    # Phase 19.1 — Persistence
    "LineageDatabase",
    # Phase 19.1 — API
    "LineageAPI",
]
