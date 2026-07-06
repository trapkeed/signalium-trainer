"""Phase 17 — AEIP Ensemble API.

Structured endpoint layer for the Adaptive Ensemble Intelligence Platform.

All methods return::

    {
      "status": "ok" | "error",
      "code": "...",          # only on error
      "message": "...",
      "data": { ... },
      "timestamp": "2026-07-05T12:00:00Z"
    }

Endpoints:
  Expert:
    get_expert, list_experts, register_expert,
    retire_expert, promote_champion
  Prediction:
    predict, get_ensemble_prediction,
    get_expert_predictions, get_meta_prediction
  Weights:
    get_weights, get_weight_history, get_weight_evolution
  Routing:
    get_routing_decision, get_routing_history, get_routing_stats
  Leaderboard:
    get_leaderboard, get_champion, get_retired, get_experimental
  Performance:
    get_performance_history, get_metrics
  Dashboard:
    get_dashboard, get_stats
"""

from __future__ import annotations

import traceback
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from trainer.ensemble.database import EnsembleDatabase
from trainer.ensemble.expert import ExpertModel, ExpertRegistry
from trainer.ensemble.leaderboard import EnsembleLeaderboard
from trainer.ensemble.visualization import EnsembleVisualizationBuilder
from trainer.ensemble.weighting import DynamicWeightEngine


def _ts() -> str:
    return datetime.now(UTC).isoformat()


def _ok(data: Any, message: str = "ok") -> dict[str, Any]:
    return {"status": "ok", "message": message, "data": data, "timestamp": _ts()}


def _err(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "code": code, "message": message, "data": None, "timestamp": _ts()}


class EnsembleAPI:
    """Structured API for the AEIP.

    Args:
        database: EnsembleDatabase instance.
        registry: ExpertRegistry instance (optional, built fresh if not given).
        leaderboard: EnsembleLeaderboard instance.
        weight_engine: DynamicWeightEngine instance.
    """

    def __init__(
        self,
        database: EnsembleDatabase,
        registry: ExpertRegistry | None = None,
        leaderboard: EnsembleLeaderboard | None = None,
        weight_engine: DynamicWeightEngine | None = None,
    ) -> None:
        self.db = database
        self.registry = registry or ExpertRegistry()
        self.leaderboard = leaderboard or EnsembleLeaderboard()
        self.weight_engine = weight_engine or DynamicWeightEngine()
        self.viz = EnsembleVisualizationBuilder()

    # -----------------------------------------------------------------------
    # Expert endpoints
    # -----------------------------------------------------------------------

    def get_expert(self, expert_id: str) -> dict[str, Any]:
        try:
            expert = self.db.get_expert(expert_id)
            if expert is None:
                return _err("NOT_FOUND", f"Expert '{expert_id}' not found.")
            return _ok(expert)
        except Exception as e:
            logger.error(f"get_expert: {e}")
            return _err("GET_EXPERT_ERROR", str(e))

    def list_experts(self, status: str | None = None) -> dict[str, Any]:
        try:
            experts = self.db.list_experts(status=status)
            return _ok({"experts": experts, "count": len(experts)})
        except Exception as e:
            logger.error(f"list_experts: {e}")
            return _err("LIST_EXPERTS_ERROR", str(e))

    def register_expert(self, expert_config: dict[str, Any]) -> dict[str, Any]:
        try:
            expert = ExpertModel.from_dict(expert_config)
            self.registry.register(expert)
            self.db.save_expert(expert)
            return _ok(expert.to_dict(), "Expert registered.")
        except Exception as e:
            logger.error(f"register_expert: {e}")
            return _err("REGISTER_ERROR", str(e))

    def retire_expert(self, expert_id: str) -> dict[str, Any]:
        try:
            expert = self.registry.get(expert_id)
            if expert:
                expert.retire()
                self.db.save_expert(expert)
            self.db.update_expert_status(expert_id, "retired")
            self.leaderboard.retire(expert_id)
            return _ok({"expert_id": expert_id, "status": "retired"})
        except Exception as e:
            logger.error(f"retire_expert: {e}")
            return _err("RETIRE_ERROR", str(e))

    def promote_champion(self, expert_id: str) -> dict[str, Any]:
        try:
            expert = self.registry.get(expert_id)
            if expert:
                self.registry.set_champion(expert_id)
                self.db.save_expert(expert)
            self.db.update_expert_status(expert_id, "champion")
            self.leaderboard.promote_champion(expert_id)
            return _ok({"expert_id": expert_id, "status": "champion"})
        except Exception as e:
            logger.error(f"promote_champion: {e}")
            return _err("PROMOTE_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Prediction endpoints
    # -----------------------------------------------------------------------

    def get_ensemble_prediction(self, result_id: str) -> dict[str, Any]:
        try:
            results = self.db.list_ensemble_results(limit=1000)
            for r in results:
                if r.get("result_id") == result_id:
                    return _ok(r)
            return _err("NOT_FOUND", f"Result '{result_id}' not found.")
        except Exception as e:
            logger.error(f"get_ensemble_prediction: {e}")
            return _err("GET_RESULT_ERROR", str(e))

    def list_predictions(self, strategy: str | None = None, limit: int = 50) -> dict[str, Any]:
        try:
            results = self.db.list_ensemble_results(strategy=strategy, limit=limit)
            return _ok({"predictions": results, "count": len(results)})
        except Exception as e:
            logger.error(f"list_predictions: {e}")
            return _err("LIST_PREDICTIONS_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Weight endpoints
    # -----------------------------------------------------------------------

    def get_weights(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            experts = self.registry.active_experts()
            weights = self.weight_engine.compute_weights(experts, context or {})
            return _ok(
                {
                    "weights": weights,
                    "context": context or {},
                    "n_experts": len(weights),
                }
            )
        except Exception as e:
            logger.error(f"get_weights: {e}")
            return _err("GET_WEIGHTS_ERROR", str(e))

    def get_weight_history(self, expert_id: str, limit: int = 100) -> dict[str, Any]:
        try:
            history = self.db.get_weight_history(expert_id, limit=limit)
            return _ok({"expert_id": expert_id, "history": history, "count": len(history)})
        except Exception as e:
            logger.error(f"get_weight_history: {e}")
            return _err("WEIGHT_HISTORY_ERROR", str(e))

    def get_weight_evolution(self) -> dict[str, Any]:
        try:
            evolution = self.weight_engine.all_weight_evolution()
            chart = self.viz.weight_evolution(evolution)
            return _ok({"chart": chart, "n_experts": len(evolution)})
        except Exception as e:
            logger.error(f"get_weight_evolution: {e}")
            return _err("WEIGHT_EVOLUTION_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Routing endpoints
    # -----------------------------------------------------------------------

    def get_routing_history(self, regime: str | None = None, limit: int = 50) -> dict[str, Any]:
        try:
            history = self.db.list_routing_decisions(regime=regime, limit=limit)
            return _ok({"history": history, "count": len(history)})
        except Exception as e:
            logger.error(f"get_routing_history: {e}")
            return _err("ROUTING_HISTORY_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Leaderboard endpoints
    # -----------------------------------------------------------------------

    def get_leaderboard(
        self, include_retired: bool = False, entity_type: str | None = None
    ) -> dict[str, Any]:
        try:
            rankings = self.leaderboard.get_rankings(
                include_retired=include_retired,
                entity_type=entity_type,
            )
            return _ok({"rankings": rankings, "total": len(rankings)})
        except Exception as e:
            logger.error(f"get_leaderboard: {e}")
            return _err("LEADERBOARD_ERROR", str(e))

    def get_champion(self) -> dict[str, Any]:
        try:
            champion = self.leaderboard.get_champion()
            return _ok({"champion": champion})
        except Exception as e:
            logger.error(f"get_champion: {e}")
            return _err("CHAMPION_ERROR", str(e))

    def get_retired(self) -> dict[str, Any]:
        try:
            retired = self.leaderboard.retired()
            return _ok({"retired": retired, "count": len(retired)})
        except Exception as e:
            return _err("RETIRED_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Performance endpoints
    # -----------------------------------------------------------------------

    def get_performance_history(self, expert_id: str, limit: int = 100) -> dict[str, Any]:
        try:
            history = self.db.get_performance_history(expert_id, limit=limit)
            return _ok({"expert_id": expert_id, "history": history, "count": len(history)})
        except Exception as e:
            logger.error(f"get_performance_history: {e}")
            return _err("PERF_HISTORY_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Dashboard and stats
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        try:
            stats = self.db.stats()
            lb_summary = self.leaderboard.summary()
            reg_summary = self.registry.summary()
            return _ok({**stats, "leaderboard": lb_summary, "registry": reg_summary})
        except Exception as e:
            logger.error(f"get_stats: {e}")
            return _err("STATS_ERROR", str(e))

    def get_dashboard(self) -> dict[str, Any]:
        try:
            experts = self.registry.active_experts()
            weights = self.weight_engine.compute_weights(experts, {})
            weight_evolution = self.weight_engine.all_weight_evolution()
            ensemble_results = self.db.list_ensemble_results(limit=200)
            vote_records = []
            for r in ensemble_results:
                for eid, w in r.get("weight_map", {}).items():
                    vote_records.append(
                        {
                            "expert_id": eid,
                            "voted_class": r.get("predicted_class", "no_trade"),
                            "weight": w,
                            "confidence": r.get("confidence", 0.5),
                        }
                    )
            agreement_data = {
                "expert_ids": [e.expert_id for e in experts],
                "matrix": {
                    e.expert_id: {
                        e2.expert_id: 1.0 if e.expert_id == e2.expert_id else None for e2 in experts
                    }
                    for e in experts
                },
            }
            performance_records = []
            for expert in experts:
                perf = self.db.get_performance_history(expert.expert_id, limit=100)
                performance_records.extend(perf)
            charts = self.viz.full_dashboard(
                current_weights=weights,
                weight_history=weight_evolution,
                vote_records=vote_records,
                agreement_matrix_data=agreement_data,
                ensemble_results=ensemble_results,
                performance_records=performance_records,
                expert_names={e.expert_id: e.name for e in experts},
            )
            return _ok(
                {
                    "charts": charts,
                    "stats": self.db.stats(),
                    "leaderboard_summary": self.leaderboard.summary(),
                }
            )
        except Exception as e:
            logger.error(f"get_dashboard: {traceback.format_exc()}")
            return _err("DASHBOARD_ERROR", str(e))
