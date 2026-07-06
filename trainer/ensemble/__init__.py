"""Phase 17 — Adaptive Ensemble Intelligence Platform (AEIP).

Public API for the ensemble package.

The existing Signalium Quant Model is Expert 1. Future models become
additional experts. This package does NOT modify any prior phase module.

Quickstart::

    from trainer.ensemble import EnsembleEngine, EnsembleConfig, ExpertModel

    engine = EnsembleEngine.from_config({
        "db_path": "experiments/ensemble/ensemble.db",
        "strategy": "soft_voting",
    })

    result = engine.predict(features={"close": [1.0, 1.1]},
                             context={"regime": "trending_up",
                                      "instrument": "EURUSD",
                                      "timeframe": "H1"})
    print(result.to_dict()["institutional_forecast"])
"""

from __future__ import annotations

# API
from trainer.ensemble.api import EnsembleAPI

# Calibration
from trainer.ensemble.calibration import (
    CalibrationReport,
    ExpertCalibrator,
)

# Database
from trainer.ensemble.database import EnsembleDatabase

# Engine
from trainer.ensemble.engine import (
    EnsembleConfig,
    EnsembleEngine,
    PredictionBundle,
)

# Expert
from trainer.ensemble.expert import (
    ExpertModel,
    ExpertPrediction,
    ExpertRegistry,
    ModelFamily,
    SpecializationType,
    build_default_registry,
)

# Export
from trainer.ensemble.export import EnsembleExporter

# Failure analysis
from trainer.ensemble.failure_analysis import (
    ExpertFailureAnalyzer,
    ExpertFailureReport,
    FailureEvent,
)

# Leaderboard
from trainer.ensemble.leaderboard import (
    EnsembleLeaderboard,
    LeaderboardEntry,
)

# Meta model
from trainer.ensemble.meta_model import (
    EnsembleMetaModel,
    MetaFeatureBuilder,
    MetaPrediction,
)

# Online scoring
from trainer.ensemble.online_scoring import (
    OnlineScorer,
    PredictionRecord,
    RollingMetrics,
)

# Routing
from trainer.ensemble.routing import (
    ExpertRouter,
    RouterConfig,
    RoutingDecision,
)

# Strategies
from trainer.ensemble.strategies import (
    AVAILABLE_STRATEGIES,
    BaseEnsembleStrategy,
    BlendingStrategy,
    EnsembleResult,
    HardVotingStrategy,
    MixtureOfExpertsStrategy,
    PerformanceWeightedStrategy,
    SimpleAveragingStrategy,
    SoftVotingStrategy,
    StackingStrategy,
    WeightedAveragingStrategy,
    build_strategy,
)

# Visualization
from trainer.ensemble.visualization import EnsembleVisualizationBuilder

# Weighting
from trainer.ensemble.weighting import (
    REGIME_FAMILY_AFFINITY,
    DynamicWeightEngine,
    WeightFactorConfig,
)

__all__ = [
    # Expert
    "SpecializationType",
    "ModelFamily",
    "ExpertPrediction",
    "ExpertModel",
    "ExpertRegistry",
    "build_default_registry",
    # Strategies
    "EnsembleResult",
    "BaseEnsembleStrategy",
    "SimpleAveragingStrategy",
    "WeightedAveragingStrategy",
    "PerformanceWeightedStrategy",
    "HardVotingStrategy",
    "SoftVotingStrategy",
    "MixtureOfExpertsStrategy",
    "StackingStrategy",
    "BlendingStrategy",
    "build_strategy",
    "AVAILABLE_STRATEGIES",
    # Weighting
    "WeightFactorConfig",
    "DynamicWeightEngine",
    "REGIME_FAMILY_AFFINITY",
    # Routing
    "RoutingDecision",
    "RouterConfig",
    "ExpertRouter",
    # Meta model
    "MetaPrediction",
    "MetaFeatureBuilder",
    "EnsembleMetaModel",
    # Calibration
    "CalibrationReport",
    "ExpertCalibrator",
    # Failure analysis
    "FailureEvent",
    "ExpertFailureReport",
    "ExpertFailureAnalyzer",
    # Online scoring
    "PredictionRecord",
    "RollingMetrics",
    "OnlineScorer",
    # Database
    "EnsembleDatabase",
    # Leaderboard
    "LeaderboardEntry",
    "EnsembleLeaderboard",
    # Visualization
    "EnsembleVisualizationBuilder",
    # API
    "EnsembleAPI",
    # Export
    "EnsembleExporter",
    # Engine
    "EnsembleConfig",
    "PredictionBundle",
    "EnsembleEngine",
]
