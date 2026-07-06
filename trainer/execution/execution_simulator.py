"""Phase 21 — Execution Simulator.

Estimates expected fill quality, slippage, and alternative scenarios
without placing any real orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.entry_zones import EntryZones
from trainer.execution.execution_types import (
    SESSION_QUALITY,
    BrokerConstraints,
    ExecutionForecast,
    _clamp,
    default_broker,
)
from trainer.execution.stop_loss_engine import StopLossRecommendation
from trainer.execution.take_profit_engine import TakeProfitRecommendation


@dataclass
class ScenarioOutcome:
    """One simulated trade outcome."""

    label: str
    probability: float
    pnl_pct: float  # gain/loss as fraction of account
    description: str

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "probability": round(self.probability, 4),
            "pnl_pct": round(self.pnl_pct, 4),
            "description": self.description,
        }


@dataclass
class SimulationResult:
    """Complete simulation output for one trade."""

    instrument: str
    direction: str

    expected_fill: float  # estimated actual fill price
    expected_slippage: float  # in price units
    fill_quality: str  # "EXCELLENT", "GOOD", "FAIR", "POOR"

    alternative_entries: List[float] = field(default_factory=list)
    alternative_stops: List[float] = field(default_factory=list)
    alternative_targets: List[float] = field(default_factory=list)

    expected_outcomes: List[ScenarioOutcome] = field(default_factory=list)

    # Summary statistics
    expected_pnl_pct: float = 0.0
    pnl_std_dev: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favourable_excursion: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "expected_fill": round(self.expected_fill, 5),
            "expected_slippage": round(self.expected_slippage, 5),
            "fill_quality": self.fill_quality,
            "alternative_entries": [round(x, 5) for x in self.alternative_entries],
            "alternative_stops": [round(x, 5) for x in self.alternative_stops],
            "alternative_targets": [round(x, 5) for x in self.alternative_targets],
            "expected_outcomes": [o.to_dict() for o in self.expected_outcomes],
            "summary": {
                "expected_pnl_pct": round(self.expected_pnl_pct, 4),
                "pnl_std_dev": round(self.pnl_std_dev, 4),
                "max_adverse_excursion": round(self.max_adverse_excursion, 5),
                "max_favourable_excursion": round(self.max_favourable_excursion, 5),
            },
        }


class ExecutionSimulator:
    """Simulates execution outcomes for a single trade without placing orders.

    Slippage model:
      slippage = spread × (1.0 + session_penalty) × (1.0 + vol_factor)
    """

    def __init__(self, broker: Optional[BrokerConstraints] = None):
        self._broker = broker or default_broker()

    def simulate(
        self,
        forecast: ExecutionForecast,
        zones: Optional[EntryZones] = None,
        stop_loss: Optional[StopLossRecommendation] = None,
        take_profit: Optional[TakeProfitRecommendation] = None,
    ) -> SimulationResult:

        entry = zones.primary_entry if zones else forecast.current_price
        sign = 1.0 if forecast.is_buy else -1.0
        atr = forecast.atr

        # Session penalty on slippage
        sess_q = SESSION_QUALITY.get(forecast.session, 0.5)
        sess_penalty = 1.0 - sess_q  # worse session → more slippage

        # Volatility factor
        vol_factor = _clamp(forecast.volatility / 0.01)

        expected_slippage = round(
            forecast.spread * (1.0 + sess_penalty) * (1.0 + vol_factor * 0.5),
            5,
        )

        # Expected fill (slippage is against us)
        expected_fill = round(entry + sign * expected_slippage, 5)

        # Fill quality
        slip_ratio = expected_slippage / max(atr, 1e-9)
        if slip_ratio < 0.05:
            fill_quality = "EXCELLENT"
        elif slip_ratio < 0.15:
            fill_quality = "GOOD"
        elif slip_ratio < 0.30:
            fill_quality = "FAIR"
        else:
            fill_quality = "POOR"

        # Alternative entries (pullback scenarios)
        alt_entries = [
            round(entry + sign * -0.5 * atr, 5),  # earlier
            round(entry + sign * -1.0 * atr, 5),  # much earlier
            round(entry + sign * 0.5 * atr, 5),  # deeper pullback
        ]

        # Alternative stops
        stop_base = stop_loss.initial_stop if stop_loss else round(entry - sign * atr * 1.5, 5)
        alt_stops = [
            round(stop_base - sign * atr * 0.25, 5),  # tighter
            round(stop_base + sign * atr * 0.50, 5),  # wider
            round(stop_base + sign * atr * 1.00, 5),  # structure stop
        ]

        # Alternative targets
        if take_profit:
            tp_base = take_profit.tp2
        else:
            tp_base = round(entry + sign * atr * 2.0, 5)
        alt_targets = [
            take_profit.tp1 if take_profit else round(entry + sign * atr * 1.0, 5),
            take_profit.tp3 if take_profit else round(entry + sign * atr * 3.0, 5),
            take_profit.runner_target if take_profit else round(entry + sign * atr * 4.0, 5),
        ]

        # Scenario outcomes
        p_win = forecast.confidence
        p_loss = 1.0 - p_win
        stop_d = abs(entry - (stop_loss.initial_stop if stop_loss else entry - sign * atr * 1.5))
        tp2_d = abs(entry - (take_profit.tp2 if take_profit else entry + sign * atr * 2.0))
        runner_d = abs(
            entry - (take_profit.runner_target if take_profit else entry + sign * atr * 4.0)
        )

        # Risk as fraction of account (approximate)
        risk_lot = 1.0
        risk_pct = round(risk_lot * stop_d / max(forecast.account_size, 1.0) * 0.01, 4)

        outcomes = [
            ScenarioOutcome(
                "Stop Hit",
                round(p_loss * 0.85, 4),
                round(-risk_pct, 4),
                "Stop loss triggered — full risk realised",
            ),
            ScenarioOutcome(
                "TP1 Hit (partial)",
                round(p_win * 0.90, 4),
                round(risk_pct * 1.0, 4),
                "50% closed at TP1; remainder managed",
            ),
            ScenarioOutcome(
                "TP2 Hit",
                round(p_win * 0.70, 4),
                round(risk_pct * 1.5, 4),
                "Reached primary target — good outcome",
            ),
            ScenarioOutcome(
                "Runner Target",
                round(p_win * 0.40, 4),
                round(risk_pct * 3.0, 4),
                "Full runner — excellent institutional outcome",
            ),
            ScenarioOutcome(
                "Breakeven",
                round(0.10, 4),
                0.0,
                "Stop moved to breakeven — trade flat",
            ),
            ScenarioOutcome(
                "Early Exit",
                round(0.05, 4),
                round(-risk_pct * 0.3, 4),
                "News/event forced early management exit",
            ),
        ]

        # Expected PnL
        ev_pct = sum(o.probability * o.pnl_pct for o in outcomes)
        std_dev = round(
            (sum(o.probability * (o.pnl_pct - ev_pct) ** 2 for o in outcomes)) ** 0.5, 4
        )

        mae = round(stop_d, 5)
        mfe = round(runner_d, 5)

        logger.debug(
            f"Simulator {forecast.instrument}: fill={expected_fill} "
            f"slippage={expected_slippage} quality={fill_quality}"
        )

        return SimulationResult(
            instrument=forecast.instrument,
            direction=forecast.direction,
            expected_fill=expected_fill,
            expected_slippage=expected_slippage,
            fill_quality=fill_quality,
            alternative_entries=alt_entries,
            alternative_stops=alt_stops,
            alternative_targets=alt_targets,
            expected_outcomes=outcomes,
            expected_pnl_pct=round(ev_pct, 4),
            pnl_std_dev=std_dev,
            max_adverse_excursion=mae,
            max_favourable_excursion=mfe,
        )
