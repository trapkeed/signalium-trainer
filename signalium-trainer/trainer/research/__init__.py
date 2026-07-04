"""Signalium Research & Benchmarking Framework.

A complete research environment for objective comparison and lifecycle
management of quantitative trading models.

Public API::

    from trainer.research import (
        ExperimentRecord, ExperimentTracker,
        BenchmarkConditions, BenchmarkRunner, BenchmarkResult,
        Leaderboard, ScoringWeights,
        WalkForwardValidator, RollingWindowValidator,
        HyperparameterRecord,
        ReportGenerator, ReportConfig,
        ChampionRegistry, ModelStatus,
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

__all__ = [
    # Experiment
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
    # Validation
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
]
