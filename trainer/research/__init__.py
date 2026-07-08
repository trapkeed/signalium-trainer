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

Phase 19.2 — Institutional Research Scoring Engine (IRSE)::

    from trainer.research import (
        # Weights & Grades
        IRSEScoringWeights, GradeSystem, Grade, GradeResult,
        # Category inputs & scorers
        ValidationQualityInputs, ValidationQualityScorer,
        TradingPerformanceInputs, TradingPerformanceScorer,
        RobustnessInputs, RobustnessScorer,
        CalibrationInputs, CalibrationScorer,
        GeneralizationInputs, GeneralizationScorer,
        EfficiencyInputs, EfficiencyScorer,
        ExplainabilityInputs, ExplainabilityScorer,
        OperationalReadinessInputs, OperationalReadinessScorer,
        # Confidence
        ConfidenceInputs, ConfidenceResult, ConfidenceCalculator,
        # Core scorer
        ResearchScoreInputs, CategoryScore, ResearchScore, ResearchScorer,
        # Trends
        ScoreTrendTracker, ScoreSnapshot, ChampionChange, RegressionAlert,
        # Leaderboard
        IRSELeaderboard, IRSELeaderboardEntry,
        # Benchmarking
        IRSEBenchmarker, ModelRole, BenchmarkRecord, BenchmarkComparison,
        # Persistence
        IRSEScoreDatabase,
        # API / Dashboard / Export
        IRSEApi, IRSEScoreDashboard, IRSEScoreExporter,
    )
"""

from trainer.research.benchmark import (
    BenchmarkConditions,
    BenchmarkResult,
    BenchmarkRunner,
    ModelRunResult,
)
from trainer.research.benchmarking import (
    BenchmarkComparison,
    BenchmarkRecord,
    IRSEBenchmarker,
    ModelRole,
)
from trainer.research.categories import (
    CalibrationInputs,
    CalibrationScorer,
    EfficiencyInputs,
    EfficiencyScorer,
    ExplainabilityInputs,
    ExplainabilityScorer,
    GeneralizationInputs,
    GeneralizationScorer,
    OperationalReadinessInputs,
    OperationalReadinessScorer,
    RobustnessInputs,
    RobustnessScorer,
    TradingPerformanceInputs,
    TradingPerformanceScorer,
    ValidationQualityInputs,
    ValidationQualityScorer,
)
from trainer.research.champion import ChampionRegistry, ModelEntry, ModelStatus
from trainer.research.confidence import ConfidenceCalculator, ConfidenceInputs, ConfidenceResult
from trainer.research.cost_tracker import CostTracker, ExperimentCost
from trainer.research.dashboard import DashboardBuilder
from trainer.research.differencing import (
    DatasetDiff,
    ExperimentDiff,
    ExperimentDiffer,
    FeatureDiff,
    ModelDiff,
    ValidationDiff,
)
from trainer.research.experiment import ExperimentRecord, ExperimentTracker
from trainer.research.export import ResearchExporter
from trainer.research.grades import Grade, GradeResult, GradeSystem
from trainer.research.hrsa_api import HRSAApi
from trainer.research.hrsa_benchmarks import (
    BenchmarkDelta,
    BenchmarkRole,
    BenchmarkSuite,
    HRSABenchmarkEngine,
    HRSABenchmarkRecord,
)
from trainer.research.hrsa_categories import (
    CategoryDefinition,
    CategoryResult,
    HRSACategoryScorer,
    MetricMapping,
)
from trainer.research.hrsa_dashboard import HRSADashboard
from trainer.research.hrsa_database import HRSADatabase
from trainer.research.hrsa_explanations import (
    CategoryNarrative,
    HRSAExplainer,
    MetricContributor,
    Recommendation,
    ScoreExplanation,
)
from trainer.research.hrsa_grades import (
    FuturePotential,
    GradeAssessment,
    HRSAGrade,
    HRSAGradeDefinition,
    HRSAGradeEngine,
    PromotionRecommendation,
    ResearchGrade,
    ResearchGradeDefinition,
    RiskRating,
)
from trainer.research.hrsa_scorer import (
    HierarchicalResearchScorer,
    HierarchicalScore,
    SensitivityResult,
)
from trainer.research.hrsa_trends import HRSATrendEngine, RegressionEvent, TrendPoint, TrendResult
from trainer.research.hrsa_whatif import HRSAWhatIfSimulator, SimulationResult
from trainer.research.hyperparameter import HyperparameterRecord
from trainer.research.knowledge_graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    NodeType,
)
from trainer.research.leaderboard import Leaderboard, LeaderboardEntry, ScoringWeights
from trainer.research.leaderboard_irse import IRSELeaderboard, IRSELeaderboardEntry
from trainer.research.lineage import ExperimentStatus, LineageNode, LineageTree
from trainer.research.lineage_api import LineageAPI
from trainer.research.lineage_database import LineageDatabase
from trainer.research.normalization import (
    CategoryNormaliser,
    NormalisationResult,
    Normaliser,
    NormMethod,
)

# ── Phase 19.3 — Hierarchical Research Scoring Architecture (HRSA) ────────────
from trainer.research.raw_metrics import (
    MetricDefinition,
    MetricDirection,
    RawMetricsBundle,
    RawMetricsRegistry,
)
from trainer.research.report import ReportConfig, ReportGenerator

# Phase 19.1 — Experiment Lineage & Reproducibility
from trainer.research.reproducibility import EnvironmentSnapshot
from trainer.research.score_api import IRSEApi
from trainer.research.score_dashboard import IRSEScoreDashboard
from trainer.research.score_database import IRSEScoreDatabase
from trainer.research.score_export import IRSEScoreExporter
from trainer.research.scorer import (
    CategoryScore,
    ResearchScore,
    ResearchScoreInputs,
    ResearchScorer,
)
from trainer.research.snapshots import (
    DatasetSnapshot,
    FeatureSnapshot,
    ModelSnapshot,
    ValidationSnapshot,
)
from trainer.research.timeline import ResearchTimeline, TimelineEvent, TimelineEventType
from trainer.research.trends import (
    ChampionChange,
    RegressionAlert,
    ScoreSnapshot,
    ScoreTrendTracker,
)
from trainer.research.validation import (
    RollingFold,
    RollingWindowValidator,
    ValidationFold,
    WalkForwardValidator,
)

# Phase 19.2 — Institutional Research Scoring Engine (IRSE)
from trainer.research.weights import ScoringWeights as IRSEScoringWeights

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
    # Phase 19.2 — Weights & Grades
    "IRSEScoringWeights",
    "GradeSystem",
    "Grade",
    "GradeResult",
    # Phase 19.2 — Category Inputs
    "ValidationQualityInputs",
    "ValidationQualityScorer",
    "TradingPerformanceInputs",
    "TradingPerformanceScorer",
    "RobustnessInputs",
    "RobustnessScorer",
    "CalibrationInputs",
    "CalibrationScorer",
    "GeneralizationInputs",
    "GeneralizationScorer",
    "EfficiencyInputs",
    "EfficiencyScorer",
    "ExplainabilityInputs",
    "ExplainabilityScorer",
    "OperationalReadinessInputs",
    "OperationalReadinessScorer",
    # Phase 19.2 — Confidence
    "ConfidenceInputs",
    "ConfidenceResult",
    "ConfidenceCalculator",
    # Phase 19.2 — Core Scorer
    "ResearchScoreInputs",
    "CategoryScore",
    "ResearchScore",
    "ResearchScorer",
    # Phase 19.2 — Trends
    "ScoreTrendTracker",
    "ScoreSnapshot",
    "ChampionChange",
    "RegressionAlert",
    # Phase 19.2 — Leaderboard
    "IRSELeaderboard",
    "IRSELeaderboardEntry",
    # Phase 19.2 — Benchmarking
    "IRSEBenchmarker",
    "ModelRole",
    "BenchmarkRecord",
    "BenchmarkComparison",
    # Phase 19.2 — Persistence
    "IRSEScoreDatabase",
    # Phase 19.2 — API / Dashboard / Export
    "IRSEApi",
    "IRSEScoreDashboard",
    "IRSEScoreExporter",
    # ── Phase 19.3 — HRSA Raw Metrics ─────────────────────────────────────────
    "MetricDefinition",
    "MetricDirection",
    "RawMetricsRegistry",
    "RawMetricsBundle",
    # Phase 19.3 — Normalisation
    "NormMethod",
    "NormalisationResult",
    "Normaliser",
    "CategoryNormaliser",
    # Phase 19.3 — Grade Engine
    "HRSAGrade",
    "HRSAGradeDefinition",
    "ResearchGrade",
    "ResearchGradeDefinition",
    "PromotionRecommendation",
    "RiskRating",
    "FuturePotential",
    "GradeAssessment",
    "HRSAGradeEngine",
    # Phase 19.3 — Category Scorer
    "MetricMapping",
    "CategoryDefinition",
    "CategoryResult",
    "HRSACategoryScorer",
    # Phase 19.3 — Hierarchical Scorer
    "HierarchicalScore",
    "SensitivityResult",
    "HierarchicalResearchScorer",
    # Phase 19.3 — Benchmarks
    "BenchmarkRole",
    "HRSABenchmarkRecord",
    "BenchmarkDelta",
    "BenchmarkSuite",
    "HRSABenchmarkEngine",
    # Phase 19.3 — Trends
    "TrendPoint",
    "TrendResult",
    "RegressionEvent",
    "HRSATrendEngine",
    # Phase 19.3 — Explanations
    "MetricContributor",
    "Recommendation",
    "CategoryNarrative",
    "ScoreExplanation",
    "HRSAExplainer",
    # Phase 19.3 — What-If Simulation
    "SimulationResult",
    "HRSAWhatIfSimulator",
    # Phase 19.3 — Database / API / Dashboard
    "HRSADatabase",
    "HRSAApi",
    "HRSADashboard",
]
