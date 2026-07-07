"""Phase 21 — Risk Engine.

Estimates full trade risk profile:
  Probability of Success / Failure, Risk/Reward, Expected Value,
  Tail Risk, Worst/Best Case, Expected Drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.execution_types import ExecutionForecast, _clamp
from trainer.execution.stop_loss_engine import StopLossRecommendation
from trainer.execution.take_profit_engine import TakeProfitRecommendation


@dataclass
class RiskProfile:
    """Complete risk assessment for one trade."""

    instrument: str
    direction: str

    prob_success: float  # P(reach TP2 or better)
    prob_failure: float  # P(hit stop)
    risk_reward_ratio: float  # TP2 distance / stop distance
    expected_value: float  # prob_success*reward - prob_failure*risk (in price units)
    expected_value_pct: float  # EV as % of account

    tail_risk: float  # 95th percentile loss
    worst_case: float  # max realistic loss (emergency stop hit)
    best_case: float  # runner target reached
    expected_drawdown: float  # blended avg adverse excursion

    # Sortino-style metrics
    downside_deviation: float = 0.0
    upside_potential: float = 0.0

    def is_positive_ev(self) -> bool:
        return self.expected_value > 0.0

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "prob_success": round(self.prob_success, 4),
            "prob_failure": round(self.prob_failure, 4),
            "risk_reward_ratio": round(self.risk_reward_ratio, 3),
            "expected_value": round(self.expected_value, 5),
            "expected_value_pct": round(self.expected_value_pct, 4),
            "tail_risk": round(self.tail_risk, 5),
            "worst_case": round(self.worst_case, 5),
            "best_case": round(self.best_case, 5),
            "expected_drawdown": round(self.expected_drawdown, 5),
            "positive_ev": self.is_positive_ev(),
            "downside_deviation": round(self.downside_deviation, 5),
            "upside_potential": round(self.upside_potential, 5),
        }


class RiskEngine:
    """Computes full risk profile from forecast + stop/TP recommendations."""

    def analyze(
        self,
        forecast: ExecutionForecast,
        stop_loss: StopLossRecommendation | None = None,
        take_profit: TakeProfitRecommendation | None = None,
        lot_size: float = 1.0,
    ) -> RiskProfile:

        entry = forecast.current_price
        sign = 1.0 if forecast.is_buy else -1.0

        # Stop and TP distances
        if stop_loss:
            stop_dist = abs(entry - stop_loss.initial_stop)
            struct_dist = abs(entry - stop_loss.structure_stop)
            emerg_dist = abs(entry - stop_loss.emergency_stop)
        else:
            stop_dist = struct_dist = emerg_dist = forecast.atr * 1.5

        if take_profit:
            tp2_dist = abs(entry - take_profit.tp2)
            runner_dist = abs(entry - take_profit.runner_target)
            p_success = take_profit.tp2_hit_prob
            p_fail = stop_loss.expected_hit_probability if stop_loss else (1 - forecast.confidence)
        else:
            tp2_dist = stop_dist * 1.5
            runner_dist = stop_dist * 2.5
            p_success = forecast.confidence * 0.75
            p_fail = _clamp(1.0 - p_success)

        # Risk/Reward
        rr = round(tp2_dist / max(stop_dist, 1e-9), 3)

        # Expected Value (price units, per lot)
        ev_raw = p_success * tp2_dist - p_fail * stop_dist
        ev_pct = ev_raw / max(entry, 1e-9)

        # Tail risk: 95th percentile adverse move (vol-based)
        sigma = forecast.volatility * entry
        tail_risk = round(min(1.645 * sigma, struct_dist), 5)

        # Worst / best case
        worst_case = round(emerg_dist, 5)
        best_case = round(runner_dist, 5)

        # Expected drawdown: weighted average of tail risk and stop distance
        expected_dd = round(0.6 * stop_dist + 0.4 * tail_risk, 5)

        # Downside deviation / upside potential
        downside = round(stop_dist * (1 + forecast.volatility), 5)
        upside = round(runner_dist * forecast.confidence, 5)

        logger.debug(
            f"RiskEngine {forecast.instrument}: RR={rr} EV={ev_raw:.5f} "
            f"P_success={p_success:.3f} P_fail={p_fail:.3f}"
        )

        return RiskProfile(
            instrument=forecast.instrument,
            direction=forecast.direction,
            prob_success=round(p_success, 4),
            prob_failure=round(p_fail, 4),
            risk_reward_ratio=rr,
            expected_value=round(ev_raw, 5),
            expected_value_pct=round(ev_pct, 4),
            tail_risk=tail_risk,
            worst_case=worst_case,
            best_case=best_case,
            expected_drawdown=expected_dd,
            downside_deviation=downside,
            upside_potential=upside,
        )
