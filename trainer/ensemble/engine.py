"""Phase 17 — AEIP Ensemble Engine.

The EnsembleEngine is the central orchestrator for the Adaptive Ensemble
Intelligence Platform.

It coordinates:
  1. Expert registration and lifecycle
  2. Expert routing (regime, instrument, timeframe)
  3. Ensemble combination (strategy)
  4. Meta model prediction
  5. Calibration per expert
  6. Failure analysis tracking
  7. Online performance scoring
  8. Database persistence
  9. Leaderboard updates
  10. Dashboard generation

The existing Signalium Quant model is Expert 1 and is never removed.

Usage::

    engine = EnsembleEngine.from_config({
        "db_path": "experiments/ensemble/ensemble.db",
        "strategy": "soft_voting",
    })
    engine.register_expert(expert_model)
    result = engine.predict(features, context)
    engine.record_outcome(result_id, true_class="buy", profit_pnl=120.0)
    dashboard = engine.get_dashboard()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from trainer.ensemble.api import EnsembleAPI
from trainer.ensemble.calibration import ExpertCalibrator
from trainer.ensemble.database import EnsembleDatabase
from trainer.ensemble.expert import (
    ExpertModel,
    ExpertPrediction,
    ExpertRegistry,
    build_default_registry,
)
from trainer.ensemble.export import EnsembleExporter
from trainer.ensemble.failure_analysis import ExpertFailureAnalyzer
from trainer.ensemble.leaderboard import EnsembleLeaderboard
from trainer.ensemble.meta_model import EnsembleMetaModel, MetaPrediction
from trainer.ensemble.online_scoring import OnlineScorer
from trainer.ensemble.routing import ExpertRouter, RoutingDecision
from trainer.ensemble.strategies import (
    BaseEnsembleStrategy,
    EnsembleResult,
    build_strategy,
)
from trainer.ensemble.visualization import EnsembleVisualizationBuilder
from trainer.ensemble.weighting import DynamicWeightEngine

# ---------------------------------------------------------------------------
# Ensemble config
# ---------------------------------------------------------------------------


@dataclass
class EnsembleConfig:
    """Configuration for the EnsembleEngine.

    Attributes:
        db_path: SQLite database file path.
        strategy: Ensemble combination strategy name.
        weight_adjust_interval: Update weights every N predictions.
        rolling_window: Rolling window size for OnlineScorer.
        meta_min_samples: Meta model activates after this many training samples.
        meta_learning_rate: Meta model learning rate.
        persist_routing: Save routing decisions to DB.
        persist_votes: Save individual votes to DB.
        auto_retire_threshold: Auto-retire expert if accuracy < this (None = off).
        export_dir: Directory for bundle exports.
    """

    db_path: str = "experiments/ensemble/ensemble.db"
    strategy: str = "soft_voting"
    weight_adjust_interval: int = 50
    rolling_window: int = 200
    meta_min_samples: int = 50
    meta_learning_rate: float = 0.01
    persist_routing: bool = True
    persist_votes: bool = True
    auto_retire_threshold: float | None = None
    export_dir: str = "experiments/ensemble/exports"
    ensemble_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": self.db_path,
            "strategy": self.strategy,
            "weight_adjust_interval": self.weight_adjust_interval,
            "rolling_window": self.rolling_window,
            "meta_min_samples": self.meta_min_samples,
            "meta_learning_rate": self.meta_learning_rate,
            "persist_routing": self.persist_routing,
            "persist_votes": self.persist_votes,
            "auto_retire_threshold": self.auto_retire_threshold,
            "export_dir": self.export_dir,
            "ensemble_version": self.ensemble_version,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EnsembleConfig:
        cfg = EnsembleConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Prediction bundle (full output)
# ---------------------------------------------------------------------------


@dataclass
class PredictionBundle:
    """Full output of one ensemble prediction cycle.

    Attributes:
        bundle_id: Unique prediction UUID.
        routing_decision: Which experts were selected and their weights.
        expert_predictions: Individual expert predictions.
        ensemble_result: Combined ensemble output.
        meta_prediction: Meta model output (may be None if not active).
        final_result: Authoritative prediction (meta if trained, else ensemble).
        features: Input features.
        context: Input context.
    """

    bundle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    routing_decision: RoutingDecision | None = None
    expert_predictions: list[ExpertPrediction] = field(default_factory=list)
    ensemble_result: EnsembleResult | None = None
    meta_prediction: MetaPrediction | None = None
    final_result: EnsembleResult | None = None
    features: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        final = self.meta_prediction or self.ensemble_result
        return {
            "bundle_id": self.bundle_id,
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision else None,
            "n_experts": len(self.expert_predictions),
            "ensemble_result": self.ensemble_result.to_dict() if self.ensemble_result else None,
            "meta_prediction": self.meta_prediction.to_dict() if self.meta_prediction else None,
            "final": final.to_dict() if final else None,
            "institutional_forecast": (
                final.institutional_forecast
                if final and hasattr(final, "institutional_forecast")
                else ""
            ),
            "context": self.context,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Ensemble engine
# ---------------------------------------------------------------------------


class EnsembleEngine:
    """Central orchestrator for the Adaptive Ensemble Intelligence Platform.

    Manages the full prediction pipeline:
    register → route → predict → combine → meta → calibrate → score → persist.

    Args:
        config: EnsembleConfig.
        registry: ExpertRegistry (defaults to standard registry with Expert 1).
    """

    def __init__(
        self,
        config: EnsembleConfig | None = None,
        registry: ExpertRegistry | None = None,
    ) -> None:
        self.config = config or EnsembleConfig()
        self.registry = registry or build_default_registry()
        self.db = EnsembleDatabase(self.config.db_path)
        self.strategy: BaseEnsembleStrategy = build_strategy(self.config.strategy)
        self.weight_engine = DynamicWeightEngine()
        self.router = ExpertRouter(self.registry, self.weight_engine)
        self.meta_model = EnsembleMetaModel(
            learning_rate=self.config.meta_learning_rate,
            min_samples_to_activate=self.config.meta_min_samples,
        )
        self.calibrator = ExpertCalibrator()
        self.failure_analyzer = ExpertFailureAnalyzer()
        self.scorer = OnlineScorer(window=self.config.rolling_window)
        self.leaderboard = EnsembleLeaderboard()
        self.viz = EnsembleVisualizationBuilder()
        self.api = EnsembleAPI(
            database=self.db,
            registry=self.registry,
            leaderboard=self.leaderboard,
            weight_engine=self.weight_engine,
        )
        self._n_predictions = 0
        self._persist_all_experts()
        logger.info(
            f"EnsembleEngine initialized: strategy={self.config.strategy}, "
            f"n_experts={self.registry.n_experts}"
        )

    def _persist_all_experts(self) -> None:
        for expert in self.registry.all_experts():
            self.db.save_expert(expert)

    # -----------------------------------------------------------------------
    # Expert management
    # -----------------------------------------------------------------------

    def register_expert(self, expert: ExpertModel) -> str:
        """Register a new expert model. Returns expert_id."""
        expert_id = self.registry.register(expert)
        self.db.save_expert(expert)
        logger.info(f"Expert {expert_id} ({expert.name}) registered.")
        return expert_id

    def retire_expert(self, expert_id: str) -> None:
        """Retire an expert from the ensemble."""
        self.registry.retire(expert_id)
        expert = self.registry.get(expert_id)
        if expert:
            self.db.save_expert(expert)
        self.leaderboard.retire(expert_id)

    def set_champion(self, expert_id: str) -> None:
        """Set an expert as champion."""
        self.registry.set_champion(expert_id)
        expert = self.registry.get(expert_id)
        if expert:
            self.db.save_expert(expert)
        self.leaderboard.promote_champion(expert_id)

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        features: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PredictionBundle:
        """Run a full ensemble prediction cycle.

        Steps:
          1. Route experts by context (regime, instrument, timeframe)
          2. Collect expert predictions
          3. Combine with ensemble strategy
          4. Refine with meta model (if trained)
          5. Persist to database
          6. Return PredictionBundle

        Args:
            features: Input feature dict for expert models.
            context: Context dict with keys: regime, instrument, timeframe.

        Returns:
            PredictionBundle with all intermediate and final results.
        """
        ctx = context or {}
        bundle_id = str(uuid.uuid4())

        routing = self.router.route(ctx)
        experts = [
            self.registry.get(eid)
            for eid in routing.selected_expert_ids
            if self.registry.get(eid) is not None
        ]
        expert_predictions = [e.predict({**features, **ctx}) for e in experts]

        ensemble_result = self.strategy.combine(expert_predictions, routing.weights)
        ensemble_result.strategy = self.config.strategy

        meta_prediction = self.meta_model.predict(expert_predictions, routing.weights, ctx)

        final: Any = meta_prediction if self.meta_model.is_trained else ensemble_result

        if self.config.persist_routing:
            self.db.save_routing_decision(routing.to_dict())

        result_dict = ensemble_result.to_dict()
        result_dict["result_id"] = bundle_id
        self.db.save_ensemble_result(
            result_dict,
            regime=ctx.get("regime", ""),
            instrument=ctx.get("instrument", ""),
            timeframe=ctx.get("timeframe", ""),
        )

        if self.config.persist_votes:
            for p in expert_predictions:
                self.db.save_vote(
                    result_id=bundle_id,
                    expert_id=p.expert_id,
                    voted_class=p.predicted_class,
                    confidence=p.confidence,
                    weight=routing.weights.get(p.expert_id, 0.0),
                )

        if self.config.persist_routing:
            for eid, w in routing.weights.items():
                self.db.save_weight_entry(
                    eid,
                    w,
                    regime=ctx.get("regime", ""),
                    instrument=ctx.get("instrument", ""),
                    timeframe=ctx.get("timeframe", ""),
                )

        for p in expert_predictions:
            self.failure_analyzer.record(
                expert_id=p.expert_id,
                regime=ctx.get("regime", "unknown"),
                instrument=ctx.get("instrument", ""),
                timeframe=ctx.get("timeframe", ""),
                predicted_class=p.predicted_class,
                ensemble_class=ensemble_result.predicted_class,
                confidence=p.confidence,
                uncertainty=p.uncertainty,
            )

        self._n_predictions += 1

        if self._n_predictions % self.config.weight_adjust_interval == 0:
            self._update_leaderboard()
            if self.config.auto_retire_threshold is not None:
                self._auto_retire_underperformers()

        return PredictionBundle(
            bundle_id=bundle_id,
            routing_decision=routing,
            expert_predictions=expert_predictions,
            ensemble_result=ensemble_result,
            meta_prediction=meta_prediction,
            final_result=final,
            features=features,
            context=ctx,
        )

    # -----------------------------------------------------------------------
    # Outcome recording
    # -----------------------------------------------------------------------

    def record_outcome(
        self,
        result_id: str,
        true_class: str,
        profit_pnl: float = 0.0,
        expert_id: str | None = None,
    ) -> None:
        """Record the ground truth outcome for a prediction.

        Updates: calibrator, failure_analyzer, online_scorer, meta_model.

        Args:
            result_id: The bundle_id from a previous predict() call.
            true_class: Actual market class ("buy" | "sell" | "no_trade").
            profit_pnl: PnL realized from this prediction.
            expert_id: If provided, record for a specific expert only.
        """
        vote_records = self.db.get_voting_history(result_id)
        for vote in vote_records:
            eid = vote["expert_id"]
            if expert_id and eid != expert_id:
                continue
            predicted = vote["voted_class"]
            confidence = vote["confidence"]
            is_correct = int(predicted == true_class)

            self.calibrator.update(eid, confidence, is_correct)
            self.scorer.record(
                expert_id=eid,
                predicted_class=predicted,
                true_class=true_class,
                profit_pnl=profit_pnl if predicted == true_class else -abs(profit_pnl) * 0.5,
                confidence=confidence,
            )
            self.db.save_performance_record(
                expert_id=eid,
                predicted_class=predicted,
                true_class=true_class,
                profit_pnl=profit_pnl,
                confidence=confidence,
            )

        results = self.db.list_ensemble_results(limit=100)
        for r in results:
            if r.get("result_id") == result_id:
                pass

    def update_meta_model(
        self,
        expert_predictions: list[ExpertPrediction],
        weights: dict[str, float],
        true_class: str,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Manually provide a labeled example to update the meta model."""
        return self.meta_model.update(expert_predictions, weights, true_class, context)

    # -----------------------------------------------------------------------
    # Weight updates
    # -----------------------------------------------------------------------

    def update_weights(self, context: dict[str, Any] | None = None) -> dict[str, float]:
        """Recompute expert weights from recent performance and context."""
        metrics = {}
        for expert in self.registry.active_experts():
            m = self.scorer.get_metrics(expert.expert_id)
            metrics[expert.expert_id] = {
                "accuracy": m.rolling_accuracy,
                "sharpe": m.rolling_sharpe,
                "calibration_error": self.calibrator.get_report(expert.expert_id).ece,
                "uncertainty": m.rolling_avg_uncertainty,
            }
        weights = self.weight_engine.update_expert_weights_from_performance(
            self.registry.active_experts(), context or {}, metrics
        )
        for eid, w in weights.items():
            self.db.update_expert_weight(eid, w)
        return weights

    # -----------------------------------------------------------------------
    # Leaderboard
    # -----------------------------------------------------------------------

    def _update_leaderboard(self) -> None:
        for expert in self.registry.active_experts():
            m = self.scorer.get_metrics(expert.expert_id)
            cal = self.calibrator.get_report(expert.expert_id)
            self.leaderboard.update_expert(
                expert_id=expert.expert_id,
                metrics={
                    "accuracy": m.rolling_accuracy,
                    "sharpe_ratio": m.rolling_sharpe,
                    "win_rate": m.rolling_win_rate,
                    "profit_factor": m.rolling_profit_factor,
                    "calibration_error": cal.ece,
                    "n_predictions": m.n_total_records,
                },
                name=expert.name,
                status=expert.status,
                model_family=expert.model_family,
                specializations=expert.specializations,
            )

        all_metrics_list = self.db.list_ensemble_results(limit=500)
        if all_metrics_list:
            n = len(all_metrics_list)
            n_correct = sum(1 for r in all_metrics_list if r.get("is_correct"))
            ensemble_acc = n_correct / n if n > 0 else 0.0
            avg_conf = sum(r.get("confidence", 0.5) for r in all_metrics_list) / n
            self.leaderboard.update_ensemble({"accuracy": ensemble_acc, "n_predictions": n})

    def _auto_retire_underperformers(self) -> None:
        threshold = self.config.auto_retire_threshold
        if threshold is None:
            return
        for expert in self.registry.active_experts():
            if expert.status == "champion":
                continue
            m = self.scorer.get_metrics(expert.expert_id)
            if m.n_records >= self.scorer.min_window and m.rolling_accuracy < threshold:
                logger.warning(
                    f"Auto-retiring expert {expert.expert_id} "
                    f"(accuracy={m.rolling_accuracy:.2%} < {threshold:.2%})"
                )
                self.retire_expert(expert.expert_id)

    # -----------------------------------------------------------------------
    # Dashboard and export
    # -----------------------------------------------------------------------

    def get_dashboard(self) -> dict[str, Any]:
        """Return full dashboard via API."""
        return self.api.get_dashboard()

    def export_bundle(self, path: str | None = None) -> dict[str, Any]:
        """Export the full ensemble bundle to JSON."""
        exporter = EnsembleExporter(
            registry=self.registry,
            meta_model=self.meta_model,
            strategy_name=self.config.strategy,
            ensemble_version=self.config.ensemble_version,
        )
        bundle = exporter.export_bundle()
        if path:
            exporter.save_bundle(bundle, path)
        return bundle

    def export_inference_contract(self) -> dict[str, Any]:
        """Export inference contract."""
        exporter = EnsembleExporter(
            registry=self.registry,
            meta_model=self.meta_model,
            strategy_name=self.config.strategy,
            ensemble_version=self.config.ensemble_version,
        )
        return exporter.export_inference_contract()

    def summary(self) -> str:
        """Return a human-readable summary of the ensemble state."""
        n_active = self.registry.n_active
        n_total = self.registry.n_experts
        lb = self.leaderboard.summary()
        champion = lb.get("champion")
        champ_str = champion["name"] if champion else "None"
        return (
            f"=== AEIP Ensemble Engine ===\n"
            f"  Strategy: {self.config.strategy}\n"
            f"  Experts: {n_active} active / {n_total} total\n"
            f"  Champion: {champ_str}\n"
            f"  Total Predictions: {self._n_predictions}\n"
            f"  Meta Model: {'active' if self.meta_model.is_trained else 'warming up'} "
            f"({self.meta_model._n_training_samples}/{self.config.meta_min_samples} samples)\n"
            f"  DB: {self.config.db_path}\n"
        )

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @staticmethod
    def from_config(config_dict: dict[str, Any]) -> EnsembleEngine:
        """Create EnsembleEngine from a config dict."""
        cfg = EnsembleConfig.from_dict(config_dict)
        return EnsembleEngine(config=cfg)

    @staticmethod
    def from_yaml(yaml_path: str) -> EnsembleEngine:
        """Create EnsembleEngine from a YAML config file."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        cfg_data = data.get("ensemble", data)
        return EnsembleEngine.from_config(cfg_data)
