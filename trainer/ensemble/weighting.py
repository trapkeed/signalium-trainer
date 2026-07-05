"""Phase 17 — AEIP Dynamic Expert Weighting.

Computes adaptive weights for each expert based on multiple factors:

  - Recent performance (rolling accuracy, Sharpe)
  - Current market regime
  - Instrument specialization
  - Timeframe specialization
  - Prediction calibration
  - Prediction uncertainty
  - Historical accuracy
  - Inference latency

Weights adapt automatically as market conditions change.

Usage::

    weighter = DynamicWeightEngine()
    weights = weighter.compute_weights(
        experts=registry.active_experts(),
        context={
            "regime": "trending_up",
            "instrument": "EURUSD",
            "timeframe": "H1",
        },
    )
    # weights: {expert_id: float}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trainer.ensemble.expert import ExpertModel, SpecializationType


# ---------------------------------------------------------------------------
# Weight factor configuration
# ---------------------------------------------------------------------------


@dataclass
class WeightFactorConfig:
    """Configures the relative importance of each weighting factor.

    All factors are applied multiplicatively and then normalized.

    Attributes:
        recent_performance_factor: Weight given to recent accuracy/Sharpe.
        regime_match_factor: Boost for experts specialized in current regime.
        instrument_match_factor: Boost for instrument-specialized experts.
        timeframe_match_factor: Boost for timeframe-specialized experts.
        calibration_factor: Weight given to calibration quality.
        uncertainty_factor: Penalty for high-uncertainty experts.
        latency_factor: Penalty for high-latency experts (0 = ignore).
        history_factor: Weight given to lifetime accuracy.
        regime_boost: Multiplier for regime-matched experts (default 2.0).
        instrument_boost: Multiplier for instrument-matched experts.
        timeframe_boost: Multiplier for timeframe-matched experts.
        min_weight: Minimum weight floor (prevents zero weights).
        max_weight_ratio: Maximum ratio between highest and lowest weight.
    """

    recent_performance_factor: float = 0.35
    regime_match_factor: float = 0.25
    instrument_match_factor: float = 0.15
    timeframe_match_factor: float = 0.10
    calibration_factor: float = 0.10
    uncertainty_factor: float = 0.05
    history_factor: float = 0.05
    latency_factor: float = 0.0
    regime_boost: float = 2.0
    instrument_boost: float = 1.5
    timeframe_boost: float = 1.3
    min_weight: float = 0.05
    max_weight_ratio: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recent_performance_factor": self.recent_performance_factor,
            "regime_match_factor": self.regime_match_factor,
            "instrument_match_factor": self.instrument_match_factor,
            "timeframe_match_factor": self.timeframe_match_factor,
            "calibration_factor": self.calibration_factor,
            "uncertainty_factor": self.uncertainty_factor,
            "history_factor": self.history_factor,
            "latency_factor": self.latency_factor,
            "regime_boost": self.regime_boost,
            "instrument_boost": self.instrument_boost,
            "timeframe_boost": self.timeframe_boost,
            "min_weight": self.min_weight,
            "max_weight_ratio": self.max_weight_ratio,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WeightFactorConfig":
        cfg = WeightFactorConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Regime→family affinity table
# ---------------------------------------------------------------------------

# Maps regime keywords to preferred model families.
# Router uses these to apply regime_boost to matched experts.
REGIME_FAMILY_AFFINITY: Dict[str, List[str]] = {
    "trending_up":    ["transformer_encoder", "hybrid_lstm_transformer"],
    "trending_down":  ["transformer_encoder", "hybrid_lstm_transformer"],
    "ranging":        ["lstm", "gru"],
    "volatile":       ["temporal_cnn", "hybrid_gru_transformer"],
    "high_volatility": ["temporal_cnn"],
    "low_volatility": ["lstm"],
    "breakout":       ["transformer_encoder", "temporal_cnn"],
    "pullback":       ["lstm", "gru"],
    "crisis":         ["temporal_cnn"],
    "unknown":        [],
}

REGIME_SPECIALIZATION_AFFINITY: Dict[str, List[str]] = {
    "trending_up":    [SpecializationType.TRENDING],
    "trending_down":  [SpecializationType.TRENDING],
    "ranging":        [SpecializationType.RANGING],
    "volatile":       [SpecializationType.HIGH_VOLATILITY],
    "high_volatility": [SpecializationType.HIGH_VOLATILITY],
    "low_volatility": [SpecializationType.LOW_VOLATILITY],
    "breakout":       [SpecializationType.BREAKOUT],
    "pullback":       [SpecializationType.PULLBACK],
    "crisis":         [SpecializationType.HIGH_VOLATILITY],
}


# ---------------------------------------------------------------------------
# Weight engine
# ---------------------------------------------------------------------------


class DynamicWeightEngine:
    """Computes regime-, instrument-, and performance-aware expert weights.

    Args:
        config: WeightFactorConfig controlling factor importance.
    """

    def __init__(self, config: Optional[WeightFactorConfig] = None) -> None:
        self.config = config or WeightFactorConfig()
        self._weight_history: Dict[str, List[float]] = {}

    def compute_weights(self, experts: List[ExpertModel],
                        context: Optional[Dict[str, Any]] = None,
                        recent_metrics: Optional[Dict[str, Dict[str, float]]] = None
                        ) -> Dict[str, float]:
        """Compute normalized weights for a list of active experts.

        Args:
            experts: List of ExpertModel objects.
            context: Dict with keys: regime, instrument, timeframe.
            recent_metrics: Optional override for per-expert recent metrics.
                            Keys: expert_id → {accuracy, sharpe, calibration_error, uncertainty}.

        Returns:
            Dict of {expert_id: weight} with weights summing to 1.0.
        """
        if not experts:
            return {}

        ctx = context or {}
        regime = ctx.get("regime", "unknown")
        instrument = ctx.get("instrument", "")
        timeframe = ctx.get("timeframe", "")

        cfg = self.config
        raw_weights: Dict[str, float] = {}

        for expert in experts:
            met = (recent_metrics or {}).get(expert.expert_id, {})
            accuracy = met.get("accuracy", expert.recent_accuracy)
            sharpe = met.get("sharpe", expert.recent_sharpe)
            cal_err = met.get("calibration_error", expert.calibration_error)
            uncertainty = met.get("uncertainty", 0.5)
            latency = met.get("latency_ms", 0.0)
            history_acc = expert.accuracy

            perf_score = (
                cfg.recent_performance_factor * max(0.0, accuracy) +
                cfg.recent_performance_factor * max(0.0, (sharpe + 2) / 7)
            )

            regime_affinity_families = REGIME_FAMILY_AFFINITY.get(regime, [])
            regime_affinity_specs = REGIME_SPECIALIZATION_AFFINITY.get(regime, [])
            family_match = expert.model_family in regime_affinity_families
            spec_match = any(s in regime_affinity_specs for s in expert.specializations)
            regime_score = cfg.regime_match_factor * (
                cfg.regime_boost if (family_match or spec_match) else 1.0
            )

            instr_match = (not expert.instruments or instrument in expert.instruments)
            instr_score = cfg.instrument_match_factor * (
                cfg.instrument_boost if instr_match else 1.0
            )

            tf_match = (not expert.timeframes or timeframe in expert.timeframes)
            tf_score = cfg.timeframe_match_factor * (
                cfg.timeframe_boost if tf_match else 1.0
            )

            cal_quality = 1.0 - min(1.0, cal_err)
            cal_score = cfg.calibration_factor * cal_quality

            unc_penalty = cfg.uncertainty_factor * (1.0 - min(1.0, uncertainty))

            history_score = cfg.history_factor * max(0.0, history_acc)

            latency_penalty = 0.0
            if cfg.latency_factor > 0 and latency > 0:
                latency_penalty = cfg.latency_factor * min(1.0, latency / 100.0)

            raw = max(
                cfg.min_weight,
                perf_score + regime_score + instr_score + tf_score +
                cal_score + unc_penalty + history_score - latency_penalty,
            )
            raw_weights[expert.expert_id] = raw

        weights = self._apply_ratio_cap(raw_weights)
        total = sum(weights.values())
        if total <= 0:
            return {eid: 1.0 / len(experts) for e in experts for eid in [e.expert_id]}
        normalized = {eid: w / total for eid, w in weights.items()}

        for eid, w in normalized.items():
            if eid not in self._weight_history:
                self._weight_history[eid] = []
            self._weight_history[eid].append(w)

        return normalized

    def _apply_ratio_cap(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Cap the ratio between max and min weights."""
        if not weights:
            return weights
        max_w = max(weights.values())
        min_floor = max_w / self.config.max_weight_ratio
        return {k: max(v, min_floor) for k, v in weights.items()}

    def weight_evolution(self, expert_id: str) -> List[float]:
        """Return weight history for an expert."""
        return self._weight_history.get(expert_id, [])

    def all_weight_evolution(self) -> Dict[str, List[float]]:
        """Return weight history for all experts."""
        return dict(self._weight_history)

    def update_expert_weights_from_performance(
            self, experts: List[ExpertModel],
            context: Optional[Dict[str, Any]] = None,
            recent_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """Compute weights and update expert.weight field in-place."""
        weights = self.compute_weights(experts, context, recent_metrics)
        for expert in experts:
            if expert.expert_id in weights:
                expert.weight = weights[expert.expert_id]
        return weights
