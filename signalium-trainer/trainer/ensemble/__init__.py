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

# Expert
from trainer.ensemble.expert import (
    SpecializationType,
    ModelFamily,
    ExpertPrediction,
    ExpertModel,
    ExpertRegistry,
    build_default_registry,
)

# Strategies
from trainer.ensemble.strategies import (
    EnsembleResult,
    BaseEnsembleStrategy,
    SimpleAveragingStrategy,
    WeightedAveragingStrategy,
    PerformanceWeightedStrategy,
    HardVotingStrategy,
    SoftVotingStrategy,
    MixtureOfExpertsStrategy,
    StackingStrategy,
    BlendingStrategy,
    build_strategy,
    AVAILABLE_STRATEGIES,
)

# Weighting
from trainer.ensemble.weighting import (
    WeightFactorConfig,
    DynamicWeightEngine,
    REGIME_FAMILY_AFFINITY,
)

# Routing
from trainer.ensemble.routing import (
    RoutingDecision,
    RouterConfig,
    ExpertRouter,
)

# Meta model
from trainer.ensemble.meta_model import (
    MetaPrediction,
    MetaFeatureBuilder,
    EnsembleMetaModel,
)

# Calibration
from trainer.ensemble.calibration import (
    CalibrationReport,
    ExpertCalibrator,
)

# Failure analysis
from trainer.ensemble.failure_analysis import (
    FailureEvent,
    ExpertFailureReport,
    ExpertFailureAnalyzer,
)

# Online scoring
from trainer.ensemble.online_scoring import (
    PredictionRecord,
    RollingMetrics,
    OnlineScorer,
)

# Database
from trainer.ensemble.database import EnsembleDatabase

# Leaderboard
from trainer.ensemble.leaderboard import (
    LeaderboardEntry,
    EnsembleLeaderboard,
)

# Visualization
from trainer.ensemble.visualization import EnsembleVisualizationBuilder

# API
from trainer.ensemble.api import EnsembleAPI

# Export
from trainer.ensemble.export import EnsembleExporter

# Engine
from trainer.ensemble.engine import (
    EnsembleConfig,
    PredictionBundle,
    EnsembleEngine,
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
