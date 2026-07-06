"""Phase 17 — AEIP Expert Routing.

Implements intelligent expert routing: selects which experts contribute
to a given prediction based on current market context.

Routing decisions are driven by:
  - Market regime
  - Instrument
  - Timeframe
  - Expert specializations
  - Expert performance

Example routing decisions:
  Trending Market  → Transformer receives highest weight
  Ranging Market   → LSTM receives highest weight
  High Volatility  → TCN receives highest weight

The routing architecture supports future expert types without code changes.

Usage::

    router = ExpertRouter(registry, weight_engine)
    route = router.route(context={"regime": "trending_up",
                                   "instrument": "EURUSD", "timeframe": "H1"})
    predictions = [expert.predict(features) for expert in route.selected_experts]
    weights = route.weights
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from trainer.ensemble.expert import ExpertRegistry
from trainer.ensemble.weighting import DynamicWeightEngine

# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """The output of one routing evaluation.

    Attributes:
        decision_id: Unique UUID for this routing decision.
        regime: Market regime used for routing.
        instrument: Instrument context.
        timeframe: Timeframe context.
        selected_expert_ids: Ordered list of selected expert IDs.
        weights: Dict of expert_id → assigned weight.
        routing_reason: Explanation of the routing logic applied.
        n_experts_available: Total active experts considered.
        n_experts_selected: Number of experts selected.
        timestamp: UTC ISO-8601 timestamp.
    """

    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    regime: str = "unknown"
    instrument: str = ""
    timeframe: str = ""
    selected_expert_ids: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    routing_reason: str = ""
    n_experts_available: int = 0
    n_experts_selected: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "regime": self.regime,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "selected_expert_ids": self.selected_expert_ids,
            "weights": self.weights,
            "routing_reason": self.routing_reason,
            "n_experts_available": self.n_experts_available,
            "n_experts_selected": self.n_experts_selected,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Routing config
# ---------------------------------------------------------------------------


@dataclass
class RouterConfig:
    """Configuration for the ExpertRouter.

    Attributes:
        min_experts: Minimum experts required in a route (default 1).
        max_experts: Maximum experts per route (None = all active).
        min_weight_threshold: Experts with weight below this are excluded.
        use_specialization_filter: Pre-filter by regime specialization.
        use_instrument_filter: Pre-filter by instrument specialization.
        use_timeframe_filter: Pre-filter by timeframe specialization.
        fallback_to_all: If no specialized expert found, use all experts.
    """

    min_experts: int = 1
    max_experts: int | None = None
    min_weight_threshold: float = 0.0
    use_specialization_filter: bool = True
    use_instrument_filter: bool = True
    use_timeframe_filter: bool = True
    fallback_to_all: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_experts": self.min_experts,
            "max_experts": self.max_experts,
            "min_weight_threshold": self.min_weight_threshold,
            "use_specialization_filter": self.use_specialization_filter,
            "use_instrument_filter": self.use_instrument_filter,
            "use_timeframe_filter": self.use_timeframe_filter,
            "fallback_to_all": self.fallback_to_all,
        }


# ---------------------------------------------------------------------------
# Expert router
# ---------------------------------------------------------------------------


class ExpertRouter:
    """Routes prediction requests to the most appropriate expert subset.

    Uses the DynamicWeightEngine to compute weights and optionally filters
    experts by specialization, instrument, and timeframe.

    Args:
        registry: ExpertRegistry with all registered experts.
        weight_engine: DynamicWeightEngine for weight computation.
        config: RouterConfig for routing behavior.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        weight_engine: DynamicWeightEngine | None = None,
        config: RouterConfig | None = None,
    ) -> None:
        self.registry = registry
        self.weight_engine = weight_engine or DynamicWeightEngine()
        self.config = config or RouterConfig()
        self._routing_history: list[RoutingDecision] = []

    def route(
        self,
        context: dict[str, Any] | None = None,
        recent_metrics: dict[str, dict[str, float]] | None = None,
    ) -> RoutingDecision:
        """Select experts and compute weights for a prediction request.

        Args:
            context: Dict with keys: regime, instrument, timeframe.
            recent_metrics: Optional per-expert recent performance metrics.

        Returns:
            RoutingDecision with selected experts and their weights.
        """
        ctx = context or {}
        regime = ctx.get("regime", "unknown")
        instrument = ctx.get("instrument", "")
        timeframe = ctx.get("timeframe", "")
        cfg = self.config

        candidates = self.registry.active_experts()
        n_available = len(candidates)
        reasons = []

        if cfg.use_instrument_filter and instrument:
            instr_filtered = [
                e for e in candidates if not e.instruments or instrument in e.instruments
            ]
            if instr_filtered or not cfg.fallback_to_all:
                candidates = instr_filtered
                reasons.append(f"instrument={instrument}")

        if cfg.use_timeframe_filter and timeframe:
            tf_filtered = [e for e in candidates if not e.timeframes or timeframe in e.timeframes]
            if tf_filtered or not cfg.fallback_to_all:
                candidates = tf_filtered
                reasons.append(f"timeframe={timeframe}")

        if not candidates and cfg.fallback_to_all:
            candidates = self.registry.active_experts()
            reasons.append("fallback_to_all")

        weights = self.weight_engine.compute_weights(
            candidates, context=ctx, recent_metrics=recent_metrics
        )

        if cfg.min_weight_threshold > 0:
            candidates = [
                e for e in candidates if weights.get(e.expert_id, 0.0) >= cfg.min_weight_threshold
            ]

        if cfg.max_experts is not None:
            sorted_experts = sorted(
                candidates,
                key=lambda e: weights.get(e.expert_id, 0.0),
                reverse=True,
            )
            candidates = sorted_experts[: cfg.max_experts]
            reasons.append(f"top_{cfg.max_experts}")

        if len(candidates) < cfg.min_experts and cfg.fallback_to_all:
            candidates = self.registry.active_experts()[: max(cfg.min_experts, 1)]
            reasons.append("fallback_min_experts")

        total = sum(weights.get(e.expert_id, 1.0) for e in candidates)
        selected_weights = {
            e.expert_id: weights.get(e.expert_id, 1.0) / max(total, 1e-9) for e in candidates
        }

        decision = RoutingDecision(
            regime=regime,
            instrument=instrument,
            timeframe=timeframe,
            selected_expert_ids=[e.expert_id for e in candidates],
            weights=selected_weights,
            routing_reason=f"regime={regime}; filters=[{', '.join(reasons)}]",
            n_experts_available=n_available,
            n_experts_selected=len(candidates),
        )
        self._routing_history.append(decision)
        logger.debug(f"Router: regime={regime} → {len(candidates)} experts selected")
        return decision

    def route_and_predict(
        self, features: dict[str, Any], context: dict[str, Any] | None = None
    ) -> tuple:
        """Route and collect expert predictions in one call.

        Returns:
            Tuple of (RoutingDecision, List[ExpertPrediction]).
        """
        decision = self.route(context)
        experts = [
            self.registry.get(eid)
            for eid in decision.selected_expert_ids
            if self.registry.get(eid) is not None
        ]
        predictions = [e.predict(features) for e in experts]
        return decision, predictions

    def routing_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self._routing_history[-limit:]]

    def routing_stats(self) -> dict[str, Any]:
        if not self._routing_history:
            return {"total_routes": 0}
        regimes = [d.regime for d in self._routing_history]
        unique_regimes = list(set(regimes))
        return {
            "total_routes": len(self._routing_history),
            "regime_distribution": {r: regimes.count(r) for r in unique_regimes},
            "avg_experts_selected": sum(d.n_experts_selected for d in self._routing_history)
            / len(self._routing_history),
        }
