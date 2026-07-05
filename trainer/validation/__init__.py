"""Phase 15 — Signalium Institutional Research Validation Platform.

The official validation framework for Signalium Quant.
No model may be deployed unless it successfully passes this platform.

Quick start::

    from trainer.validation import (
        InstitutionalValidationEngine,
        EngineConfig,
        PromotionPolicy,
        InstitutionalLeaderboard,
        ValidationDatabase,
    )

    policy = PromotionPolicy(
        min_accuracy=0.55,
        min_sharpe=1.0,
        max_drawdown=0.20,
        min_profit_factor=1.5,
        require_manual_approval=True,
    )

    engine = InstitutionalValidationEngine(
        db_path="experiments/validation.db",
        policy=policy,
    )

    result = engine.validate(model=my_model, x=x_data, y=y_labels, model_id="quant_v1")
    print(result.summary())
"""

from trainer.validation.windows import (
    WindowFold,
    NestedFold,
    SlidingWindowValidator,
    ExpandingWindowValidator,
    AnchoredWalkForward,
    NestedValidator,
)
from trainer.validation.leakage import (
    LeakageDetector,
    LeakageReport,
    LeakageDetected,
)
from trainer.validation.regime import (
    MarketRegime,
    MarketRegimeClassifier,
    RegimeDistribution,
)
from trainer.validation.instrument import (
    INSTRUMENTS,
    TIMEFRAMES,
    register_instrument,
    register_timeframe,
    get_all_instruments,
    get_all_timeframes,
    InstrumentResult,
    MultiAssetValidationResult,
    MultiAssetValidator,
)
from trainer.validation.statistical import (
    StatisticalMetrics,
    StatisticalMetricsEngine,
)
from trainer.validation.trading import (
    TradeResult,
    TradingMetrics,
    TradingMetricsEngine,
    TradeSimulator,
)
from trainer.validation.monte_carlo import (
    MonteCarloResult,
    MonteCarloValidator,
)
from trainer.validation.robustness import (
    StressScenarioResult,
    RobustnessReport,
    RobustnessTester,
)
from trainer.validation.calibration_validator import (
    CalibrationReport,
    CalibrationValidator,
)
from trainer.validation.drift import (
    FeatureDriftStats,
    DriftReport,
    DriftAnalyzer,
)
from trainer.validation.comparison import (
    ModelSnapshot,
    ComparisonMetricDelta,
    ComparisonReport,
    ModelComparison,
)
from trainer.validation.leaderboard import (
    LeaderboardEntry,
    InstitutionalLeaderboard,
)
from trainer.validation.promotion import (
    PromotionRule,
    PromotionDecision,
    PromotionPolicy,
)
from trainer.validation.failure_analysis import (
    WorstPerformer,
    FailureReport,
    FailureAnalyzer,
)
from trainer.validation.database import (
    ValidationRunRecord,
    ValidationDatabase,
)
from trainer.validation.head_validator import (
    HeadResult,
    PredictionHeadValidator,
)
from trainer.validation.visualization import VisualizationDataBuilder
from trainer.validation.export import ExportConfig, ReportExporter
from trainer.validation.api import ValidationAPI
from trainer.validation.engine import (
    EngineConfig,
    ValidationResult,
    InstitutionalValidationEngine,
)

__all__ = [
    # Windows
    "WindowFold",
    "NestedFold",
    "SlidingWindowValidator",
    "ExpandingWindowValidator",
    "AnchoredWalkForward",
    "NestedValidator",
    # Leakage
    "LeakageDetector",
    "LeakageReport",
    "LeakageDetected",
    # Regime
    "MarketRegime",
    "MarketRegimeClassifier",
    "RegimeDistribution",
    # Instruments
    "INSTRUMENTS",
    "TIMEFRAMES",
    "register_instrument",
    "register_timeframe",
    "get_all_instruments",
    "get_all_timeframes",
    "InstrumentResult",
    "MultiAssetValidationResult",
    "MultiAssetValidator",
    # Statistical
    "StatisticalMetrics",
    "StatisticalMetricsEngine",
    # Trading
    "TradeResult",
    "TradingMetrics",
    "TradingMetricsEngine",
    "TradeSimulator",
    # Monte Carlo
    "MonteCarloResult",
    "MonteCarloValidator",
    # Robustness
    "StressScenarioResult",
    "RobustnessReport",
    "RobustnessTester",
    # Calibration
    "CalibrationReport",
    "CalibrationValidator",
    # Drift
    "FeatureDriftStats",
    "DriftReport",
    "DriftAnalyzer",
    # Comparison
    "ModelSnapshot",
    "ComparisonMetricDelta",
    "ComparisonReport",
    "ModelComparison",
    # Leaderboard
    "LeaderboardEntry",
    "InstitutionalLeaderboard",
    # Promotion
    "PromotionRule",
    "PromotionDecision",
    "PromotionPolicy",
    # Failure Analysis
    "WorstPerformer",
    "FailureReport",
    "FailureAnalyzer",
    # Database
    "ValidationRunRecord",
    "ValidationDatabase",
    # Head Validator
    "HeadResult",
    "PredictionHeadValidator",
    # Visualization
    "VisualizationDataBuilder",
    # Export
    "ExportConfig",
    "ReportExporter",
    # API
    "ValidationAPI",
    # Engine (top-level)
    "EngineConfig",
    "ValidationResult",
    "InstitutionalValidationEngine",
]
