"""Phase 21 — Take Profit Engine.

Recommends structured take profit levels: TP1, TP2, TP3, Runner Target,
and an Adaptive Target with expected hit probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.execution.execution_types import ExecutionForecast, _clamp
from trainer.execution.entry_zones import EntryZones
from trainer.execution.stop_loss_engine import StopLossRecommendation


@dataclass
class TakeProfitRecommendation:
    """Structured take profit levels for one trade."""

    instrument:     str
    direction:      str

    tp1:            float   # 1:1 RR target (partial close ~50%)
    tp2:            float   # 1.5:1 RR (reduce further ~30%)
    tp3:            float   # 2:1 RR (reduce to runner)
    runner_target:  float   # full expected-return target
    adaptive_target: float  # dynamically scaled by confidence + vol

    # Probabilities
    tp1_hit_prob:   float = 0.0
    tp2_hit_prob:   float = 0.0
    tp3_hit_prob:   float = 0.0
    runner_hit_prob: float = 0.0

    # Risk/reward at each level
    rr_tp1:         float = 1.0
    rr_tp2:         float = 1.5
    rr_tp3:         float = 2.0
    rr_runner:      float = 3.0

    def best_expectancy_level(self) -> str:
        evs = {
            "TP1":    self.rr_tp1   * self.tp1_hit_prob,
            "TP2":    self.rr_tp2   * self.tp2_hit_prob,
            "TP3":    self.rr_tp3   * self.tp3_hit_prob,
            "RUNNER": self.rr_runner * self.runner_hit_prob,
        }
        return max(evs, key=evs.get)

    def to_dict(self) -> Dict:
        return {
            "instrument":     self.instrument,
            "direction":      self.direction,
            "tp1":            round(self.tp1, 5),
            "tp2":            round(self.tp2, 5),
            "tp3":            round(self.tp3, 5),
            "runner_target":  round(self.runner_target, 5),
            "adaptive_target": round(self.adaptive_target, 5),
            "probabilities": {
                "tp1":    round(self.tp1_hit_prob, 4),
                "tp2":    round(self.tp2_hit_prob, 4),
                "tp3":    round(self.tp3_hit_prob, 4),
                "runner": round(self.runner_hit_prob, 4),
            },
            "risk_reward": {
                "tp1":    self.rr_tp1,
                "tp2":    self.rr_tp2,
                "tp3":    self.rr_tp3,
                "runner": self.rr_runner,
            },
            "best_expectancy_level": self.best_expectancy_level(),
        }


class TakeProfitEngine:
    """Computes take profit levels based on stop distance and expected return.

    TP geometry:
      TP1:    entry + 1.0 × stop_distance (1:1 RR)
      TP2:    entry + 1.5 × stop_distance (1.5:1 RR)
      TP3:    entry + 2.0 × stop_distance (2:1 RR)
      Runner: entry + expected_return_price_move
      Adaptive: blends TP3 and Runner by confidence
    """

    def recommend(
        self,
        forecast:  ExecutionForecast,
        zones:     Optional[EntryZones] = None,
        stop_loss: Optional[StopLossRecommendation] = None,
    ) -> TakeProfitRecommendation:

        entry = zones.primary_entry if zones else forecast.current_price
        sign  = 1.0 if forecast.is_buy else -1.0

        # Stop distance
        if stop_loss is not None:
            stop_dist = abs(entry - stop_loss.initial_stop)
        else:
            stop_dist = forecast.atr * 1.5

        stop_dist = max(stop_dist, forecast.atr * 0.5)   # floor

        # Expected return in price units
        ret_move = forecast.current_price * forecast.expected_return

        # TP levels
        tp1    = round(entry + sign * 1.0 * stop_dist, 5)
        tp2    = round(entry + sign * 1.5 * stop_dist, 5)
        tp3    = round(entry + sign * 2.0 * stop_dist, 5)
        runner = round(entry + sign * ret_move, 5)

        # Adaptive: interpolate between TP3 and Runner by confidence
        c = forecast.confidence
        adaptive = round(tp3 * (1 - c) + runner * c, 5)

        # RR ratios
        rr_runner = round(ret_move / max(stop_dist, 1e-9), 2)
        rr_runner = max(rr_runner, 2.0)

        # Hit probabilities (decreasing with distance)
        # Use logistic decay: P(TPn) = confidence * decay^n
        base  = forecast.confidence
        decay = 0.70
        p_tp1 = _clamp(base * 0.90)
        p_tp2 = _clamp(base * decay)
        p_tp3 = _clamp(base * decay ** 2)
        p_run = _clamp(base * decay ** 3)

        logger.debug(
            f"TakeProfit {forecast.instrument}: TP1={tp1} TP2={tp2} "
            f"TP3={tp3} Runner={runner} RR={rr_runner}"
        )

        return TakeProfitRecommendation(
            instrument      = forecast.instrument,
            direction       = forecast.direction,
            tp1             = tp1,
            tp2             = tp2,
            tp3             = tp3,
            runner_target   = runner,
            adaptive_target = adaptive,
            tp1_hit_prob    = round(p_tp1, 4),
            tp2_hit_prob    = round(p_tp2, 4),
            tp3_hit_prob    = round(p_tp3, 4),
            runner_hit_prob = round(p_run, 4),
            rr_tp1          = 1.0,
            rr_tp2          = 1.5,
            rr_tp3          = 2.0,
            rr_runner       = round(rr_runner, 2),
        )
