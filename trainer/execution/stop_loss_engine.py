"""Phase 21 — Stop Loss Engine.

Recommends a full suite of stop loss levels for every execution plan:
  Initial, Adaptive, ATR, Structure, Volatility, Emergency.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.entry_zones import EntryZones
from trainer.execution.execution_types import ExecutionForecast, _clamp


@dataclass
class StopLossRecommendation:
    """Full suite of stop loss recommendations for one trade."""

    instrument: str
    direction: str

    # Stop levels (in price units)
    initial_stop: float  # default recommended stop
    adaptive_stop: float  # adjusts as trade matures
    atr_stop: float  # 2× ATR from entry
    structure_stop: float  # beyond nearest S/R (invalidation level)
    volatility_stop: float  # based on realised volatility
    emergency_stop: float  # absolute worst case

    # Metadata
    stop_confidence: float = 0.0  # 0-1
    expected_hit_probability: float = 0.0  # P(stop hit before target)

    # Distance metrics
    pips_initial: float = 0.0
    pips_structure: float = 0.0

    def tightest(self) -> float:
        candidates = [
            self.initial_stop,
            self.atr_stop,
            self.structure_stop,
            self.volatility_stop,
        ]
        if self.direction.upper() == "BUY":
            return max(candidates)  # furthest from zero = closest to price for BUY
        return min(candidates)

    def widest(self) -> float:
        return self.emergency_stop

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "initial_stop": round(self.initial_stop, 5),
            "adaptive_stop": round(self.adaptive_stop, 5),
            "atr_stop": round(self.atr_stop, 5),
            "structure_stop": round(self.structure_stop, 5),
            "volatility_stop": round(self.volatility_stop, 5),
            "emergency_stop": round(self.emergency_stop, 5),
            "stop_confidence": round(self.stop_confidence, 4),
            "expected_hit_probability": round(self.expected_hit_probability, 4),
            "pips_initial": round(self.pips_initial, 5),
            "pips_structure": round(self.pips_structure, 5),
        }


class StopLossEngine:
    """Computes all stop loss variants from forecast and entry zones.

    Stop placement rules:
      ATR stop:       primary_entry ± 2.0 × ATR
      Initial:        primary_entry ± 1.5 × ATR
      Adaptive:       primary_entry ± 1.2 × ATR  (tighter, adjusts intra-trade)
      Volatility:     primary_entry ± 2.5 × vol-scaled ATR
      Structure:      invalidation level from EntryZones
      Emergency:      primary_entry ± 3.5 × ATR
    """

    _INITIAL_MULT = 1.50
    _ADAPTIVE_MULT = 1.20
    _ATR_MULT = 2.00
    _VOLATILITY_MULT = 2.50
    _EMERGENCY_MULT = 3.50

    def recommend(
        self,
        forecast: ExecutionForecast,
        zones: EntryZones | None = None,
    ) -> StopLossRecommendation:

        entry = zones.primary_entry if zones else forecast.current_price
        atr = forecast.atr
        sign = 1.0 if forecast.is_buy else -1.0  # BUY: stop below entry

        # All stops are BELOW entry for BUY, ABOVE entry for SELL
        def stop(mult: float) -> float:
            return round(entry - sign * mult * atr, 5)

        initial_stop = stop(self._INITIAL_MULT)
        adaptive_stop = stop(self._ADAPTIVE_MULT)
        atr_stop = stop(self._ATR_MULT)
        volatility_stop = stop(self._VOLATILITY_MULT * (1 + forecast.volatility * 5))
        emergency_stop = stop(self._EMERGENCY_MULT)

        # Structure stop comes from invalidation level in zones
        structure_stop = round(zones.invalidation, 5) if zones else stop(self._ATR_MULT * 1.3)

        # Distance metrics
        pips_initial = round(abs(entry - initial_stop), 5)
        pips_structure = round(abs(entry - structure_stop), 5)

        # Confidence: higher confidence when stops are well-separated from structure
        # and ATR is meaningful relative to spread
        atr_spread_ratio = atr / max(forecast.spread, 1e-9)
        stop_confidence = _clamp(0.5 + min(atr_spread_ratio / 20, 0.4) + forecast.confidence * 0.1)

        # P(stop hit): inverse of confidence, adjusted by risk/return
        rr_raw = forecast.risk_reward_raw
        expected_hit_prob = _clamp(1.0 / (1.0 + rr_raw) * (1 - forecast.confidence * 0.3))

        logger.debug(
            f"StopLoss {forecast.instrument}: initial={initial_stop} "
            f"structure={structure_stop} hit_prob={expected_hit_prob:.3f}"
        )

        return StopLossRecommendation(
            instrument=forecast.instrument,
            direction=forecast.direction,
            initial_stop=initial_stop,
            adaptive_stop=adaptive_stop,
            atr_stop=atr_stop,
            structure_stop=structure_stop,
            volatility_stop=volatility_stop,
            emergency_stop=emergency_stop,
            stop_confidence=round(stop_confidence, 4),
            expected_hit_probability=round(expected_hit_prob, 4),
            pips_initial=pips_initial,
            pips_structure=pips_structure,
        )
