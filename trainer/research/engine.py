"""Phase 19 — Research Engine.

The ResearchEngine is the central orchestrator for the Institutional
Research Intelligence and Continuous Learning Governance Platform.

It coordinates:
  1. Research Observatory (production monitoring, signal outcomes)
  2. Market Evolution Monitor (structural market change detection)
  3. Feature Lifecycle Management (importance, drift, classification)
  4. Label Quality Monitor (noise, consistency, imbalance, drift)
  5. Dataset Evolution Tracker (versioned dataset registry)
  6. Model Health Evolution Tracker (long-term performance trends)
  7. Retraining Advisor (advisory recommendations only)
  8. Experiment Recommendation Engine (research experiment suggestions)
  9. Knowledge Base (institutional memory)
  10. Research Assistant (periodic report generation)
  11. Research API (structured endpoints)
  12. Visualization Builder (7 chart types)
  13. SQLite Database (8 tables)

No automatic retraining. No automatic deployment.
All recommendations are advisory.
Full backward compatibility with Phases 1–18.

Usage::

    engine = ResearchEngine.from_config({
        "db_path": "experiments/research/research.db",
    })
    engine.record_production_snapshot("model_01",
        accuracy=0.68, confidence=0.78, latency_ms=12.0)
    engine.record_health_point("model_01", accuracy=0.68, confidence=0.78)
    recs = engine.evaluate_retraining("model_01",
        {"performance_degradation": 0.07})
    report = engine.weekly_report()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from trainer.research.api import ResearchAPI
from trainer.research.database import ResearchDatabase
from trainer.research.dataset_evolution import DatasetEvolutionTracker, DatasetRecord
from trainer.research.experiment_engine import ExperimentRecommendationEngine
from trainer.research.feature_lifecycle import FeatureLifecycleManager, FeatureRecord
from trainer.research.knowledge_base import KnowledgeBase, KnowledgeEntryType
from trainer.research.label_quality import LabelQualityMonitor
from trainer.research.market_evolution import MarketEvolutionMonitor
from trainer.research.model_health_evolution import ModelHealthEvolutionTracker
from trainer.research.observatory import ResearchObservatory, ValidationRecord
from trainer.research.research_assistant import ResearchAssistant
from trainer.research.retraining_advisor import RetrainingAdvisor, build_default_advisor
from trainer.research.visualization import ResearchVisualizationBuilder

# ---------------------------------------------------------------------------
# Engine config
# ---------------------------------------------------------------------------


@dataclass
class ResearchConfig:
    """Configuration for the ResearchEngine.

    Attributes:
        db_path: SQLite database file path.
        observatory_window: Rolling observation window.
        market_baseline_window: Market evolution baseline window.
        market_recent_window: Market evolution recent window.
        volatility_shift_threshold: Volatility shift detection threshold.
        trend_shift_threshold: Trend change detection threshold.
        feature_obs_window: Feature observation rolling window.
        health_tracker_window: Health point rolling window.
        label_batch_window: Label batch rolling window.
        enable_default_advisor: Whether to load default retraining rules.
    """

    db_path: str = "experiments/research/research.db"
    observatory_window: int = 200
    market_baseline_window: int = 100
    market_recent_window: int = 20
    volatility_shift_threshold: float = 0.05
    trend_shift_threshold: float = 0.15
    liquidity_shift_threshold: float = 0.20
    feature_obs_window: int = 100
    feature_importance_promote_threshold: float = 0.03
    feature_stability_retire_threshold: float = 0.50
    feature_drift_deprecate_threshold: float = 0.30
    health_tracker_window: int = 50
    label_batch_window: int = 50
    enable_default_advisor: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ResearchConfig:
        cfg = ResearchConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Research engine
# ---------------------------------------------------------------------------


class ResearchEngine:
    """Central orchestrator for the Research Intelligence Platform.

    No neural network changes. No automatic retraining.
    Full backward compatibility with Phases 1–18.

    Args:
        config: ResearchConfig.
    """

    def __init__(self, config: ResearchConfig | None = None) -> None:
        self.config = config or ResearchConfig()
        self.db = ResearchDatabase(self.config.db_path)
        self.observatory = ResearchObservatory(window=self.config.observatory_window)
        self.market_monitor = MarketEvolutionMonitor(
            baseline_window=self.config.market_baseline_window,
            recent_window=self.config.market_recent_window,
            volatility_shift_threshold=self.config.volatility_shift_threshold,
            trend_shift_threshold=self.config.trend_shift_threshold,
            liquidity_shift_threshold=self.config.liquidity_shift_threshold,
        )
        self.feature_manager = FeatureLifecycleManager(
            obs_window=self.config.feature_obs_window,
            importance_promote_threshold=self.config.feature_importance_promote_threshold,
            stability_retire_threshold=self.config.feature_stability_retire_threshold,
            drift_deprecate_threshold=self.config.feature_drift_deprecate_threshold,
        )
        self.label_monitor = LabelQualityMonitor(
            batch_window=self.config.label_batch_window,
        )
        self.dataset_tracker = DatasetEvolutionTracker()
        self.health_tracker = ModelHealthEvolutionTracker(
            window=self.config.health_tracker_window,
        )
        self.advisor = (
            build_default_advisor() if self.config.enable_default_advisor else RetrainingAdvisor()
        )
        self.experiment_engine = ExperimentRecommendationEngine()
        self.knowledge_base = KnowledgeBase()
        self.viz = ResearchVisualizationBuilder()
        self.assistant = ResearchAssistant(
            observatory=self.observatory,
            health_tracker=self.health_tracker,
            feature_manager=self.feature_manager,
            label_monitor=self.label_monitor,
            dataset_tracker=self.dataset_tracker,
            advisor=self.advisor,
            experiment_engine=self.experiment_engine,
            knowledge_base=self.knowledge_base,
            market_monitor=self.market_monitor,
        )
        self.api = ResearchAPI(
            db=self.db,
            observatory=self.observatory,
            market_monitor=self.market_monitor,
            feature_manager=self.feature_manager,
            label_monitor=self.label_monitor,
            dataset_tracker=self.dataset_tracker,
            health_tracker=self.health_tracker,
            advisor=self.advisor,
            experiment_engine=self.experiment_engine,
            knowledge_base=self.knowledge_base,
            assistant=self.assistant,
        )
        self.knowledge_base.log(
            KnowledgeEntryType.RESEARCH_NOTE,
            title="ResearchEngine initialized",
            body=f"Phase 19 Research Intelligence Platform started. " f"DB: {self.config.db_path}",
            author="engine",
            metadata=self.config.to_dict(),
        )
        logger.info(f"ResearchEngine initialized: db={self.config.db_path}")

    # -----------------------------------------------------------------------
    # Observatory
    # -----------------------------------------------------------------------

    def record_production_snapshot(self, model_id: str, **kwargs: Any) -> None:
        """Record one production monitoring snapshot."""
        self.observatory.record_production_snapshot(model_id, **kwargs)

    def record_signal_outcome(
        self, model_id: str, signal: str, outcome: str, pnl: float = 0.0, **kwargs: Any
    ) -> None:
        """Record one signal outcome."""
        self.observatory.record_signal_outcome(
            model_id, signal=signal, outcome=outcome, pnl=pnl, **kwargs
        )

    def record_validation(self, record: ValidationRecord) -> None:
        self.observatory.record_validation(record)

    # -----------------------------------------------------------------------
    # Market evolution
    # -----------------------------------------------------------------------

    def record_regime_snapshot(self, instrument: str, timeframe: str, **kwargs: Any) -> None:
        """Record one market regime snapshot."""
        snap = self.market_monitor.record_regime_snapshot(instrument, timeframe, **kwargs)
        self.db.save_market_snapshot(snap.to_dict())

    def detect_market_evolution(self, instrument: str, timeframe: str) -> list[dict[str, Any]]:
        """Detect market structural changes."""
        detections = self.market_monitor.detect_evolution(instrument, timeframe)
        return [d.to_dict() for d in detections]

    # -----------------------------------------------------------------------
    # Feature lifecycle
    # -----------------------------------------------------------------------

    def register_feature(
        self, name: str, category: str = "technical", description: str = ""
    ) -> None:
        """Register a feature in the lifecycle manager."""
        self.feature_manager.register_feature(
            FeatureRecord(name=name, category=category, description=description)
        )

    def record_feature_observation(self, name: str, **kwargs: Any) -> None:
        """Record one feature quality observation."""
        obs = self.feature_manager.record_observation(name, **kwargs)
        self.db.save_feature_observation(obs.to_dict())

    def evaluate_feature(self, name: str) -> dict[str, Any]:
        return self.feature_manager.evaluate(name).to_dict()

    def evaluate_all_features(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.feature_manager.evaluate_all()]

    # -----------------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------------

    def record_label_batch(self, label_version: str, **kwargs: Any) -> None:
        """Record one label quality batch."""
        batch = self.label_monitor.record_batch(label_version, **kwargs)
        self.db.save_label_batch(batch.to_dict())

    def get_label_quality(self, label_version: str) -> dict[str, Any]:
        return self.label_monitor.get_quality_report(label_version).to_dict()

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------

    def register_dataset(self, record: DatasetRecord) -> str:
        did = self.dataset_tracker.register(record)
        self.db.save_dataset(record.to_dict())
        return did

    def get_dataset_evolution(self) -> dict[str, Any]:
        return self.dataset_tracker.evolution_report()

    # -----------------------------------------------------------------------
    # Model health
    # -----------------------------------------------------------------------

    def record_health_point(self, model_id: str, **kwargs: Any) -> None:
        """Record one model health point."""
        pt = self.health_tracker.record_health_point(model_id, **kwargs)
        self.db.save_health_point(pt.to_dict())

    def get_health_trend(self, model_id: str) -> dict[str, Any]:
        return self.health_tracker.get_trend(model_id).to_dict()

    # -----------------------------------------------------------------------
    # Advisor
    # -----------------------------------------------------------------------

    def evaluate_retraining(
        self, model_id: str, context: dict[str, Any], force_manual: bool = False
    ) -> dict[str, Any]:
        """Evaluate retraining policies — advisory only."""
        return self.api.evaluate_retraining(model_id, context, force_manual=force_manual)

    # -----------------------------------------------------------------------
    # Experiments
    # -----------------------------------------------------------------------

    def generate_experiments(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate experiment suggestions — advisory only."""
        return self.api.generate_experiments(context)

    # -----------------------------------------------------------------------
    # Knowledge base
    # -----------------------------------------------------------------------

    def add_finding(
        self,
        title: str,
        body: str = "",
        tags: list[str] | None = None,
        impact_level: str = "medium",
        model_id: str | None = None,
        author: str = "researcher",
    ) -> str:
        """Add a research finding to the knowledge base."""
        entry = self.knowledge_base.log(
            KnowledgeEntryType.FINDING,
            title=title,
            body=body,
            tags=tags or [],
            model_id=model_id,
            impact_level=impact_level,
            author=author,
        )
        self.db.save_knowledge_entry(entry.to_dict())
        return entry.entry_id

    def add_knowledge(self, entry_type: str, title: str, body: str = "", **kwargs: Any) -> str:
        entry = self.knowledge_base.log(entry_type, title=title, body=body, **kwargs)
        self.db.save_knowledge_entry(entry.to_dict())
        return entry.entry_id

    def search_knowledge(self, query: str) -> list[dict[str, Any]]:
        return self.knowledge_base.search(query)

    # -----------------------------------------------------------------------
    # Reports
    # -----------------------------------------------------------------------

    def weekly_report(self) -> dict[str, Any]:
        return self.api.weekly_report()

    def monthly_health_report(self, model_id: str | None = None) -> dict[str, Any]:
        return self.api.monthly_health_report(model_id)

    def research_roadmap(self, horizon_months: int = 3) -> dict[str, Any]:
        return self.api.research_roadmap(horizon_months)

    def full_intelligence_report(self) -> dict[str, Any]:
        return self.api.full_intelligence_report()

    def get_stats(self) -> dict[str, Any]:
        return self.api.get_stats()

    def summary(self) -> str:
        db_stats = self.db.stats()
        return (
            f"=== Signalium Research Intelligence Platform — Phase 19 ===\n"
            f"  DB: {self.config.db_path}\n"
            f"  Knowledge Base: {db_stats['total_knowledge_entries']} entries\n"
            f"  Feature Observations: {db_stats['total_feature_observations']}\n"
            f"  Datasets: {db_stats['total_datasets']} versions\n"
            f"  Market Snapshots: {db_stats['total_market_snapshots']}\n"
            f"  Health Points: {db_stats['total_health_points']}\n"
            f"  Pending Recommendations: {db_stats['pending_recommendations']}\n"
            f"  Pending Experiments: {db_stats['pending_experiments']}\n"
            f"  Auto-retrain: DISABLED (advisory only)\n"
        )

    def close(self) -> None:
        self.db.close()

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @staticmethod
    def from_config(config_dict: dict[str, Any]) -> ResearchEngine:
        return ResearchEngine(config=ResearchConfig.from_dict(config_dict))

    @staticmethod
    def from_yaml(yaml_path: str) -> ResearchEngine:
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        cfg_data = data.get("research", data)
        return ResearchEngine.from_config(cfg_data)
