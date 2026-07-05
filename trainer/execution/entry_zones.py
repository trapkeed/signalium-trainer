"""Phase 21 — Entry Zone Calculator.

Predicts structured entry zones for every execution forecast:
  Primary, Secondary, Aggressive, Conservative, Late Entry Threshold,
  Invalidation Level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.execution.execution_types import ExecutionForecast


@dataclass
class EntryZones:
    """Structured entry levels for a single trade."""

    instrument:         str
    direction:          str

    primary_entry:      float   # ideal limit entry
    secondary_entry:    float   # second pullback level
    aggressive_entry:   float   # earlier, less confirmed
    conservative_entry: float   # deeper pullback, higher probability
    late_entry:         float   # last acceptable entry level
    invalidation:       float   # stop idea invalidated beyond here

    atr_reference:      float   = 0.0

    # Quality metadata ───────────────────────────────────────────────────────
    zone_width:         float   = 0.0   # price range primary→conservative
    risk_to_primary:    float   = 0.0   # distance primary→invalidation

    def contains(self, price: float) -> bool:
        """True if price falls within the primary-to-conservative trade zone."""
        lo = min(self.primary_entry, self.conservative_entry)
        hi = max(self.primary_entry, self.conservative_entry)
        return lo <= price <= hi

    def to_dict(self) -> Dict:
        return {
            "instrument":         self.instrument,
            "direction":          self.direction,
            "primary_entry":      round(self.primary_entry, 5),
            "secondary_entry":    round(self.secondary_entry, 5),
            "aggressive_entry":   round(self.aggressive_entry, 5),
            "conservative_entry": round(self.conservative_entry, 5),
            "late_entry":         round(self.late_entry, 5),
            "invalidation":       round(self.invalidation, 5),
            "zone_width":         round(self.zone_width, 5),
            "risk_to_primary":    round(self.risk_to_primary, 5),
        }


class EntryZoneEngine:
    """Calculates multi-level entry zones from an ExecutionForecast.

    Zone geometry is ATR-based:
      - Aggressive:   current_price + sign * 0.1 * ATR
      - Primary:      current_price + sign * 0.25 * ATR
      - Secondary:    current_price + sign * 0.40 * ATR
      - Conservative: current_price + sign * 0.60 * ATR
      - Late entry:   current_price + sign * 0.80 * ATR
      - Invalidation: current_price - sign * 1.5  * ATR  (opposite direction)
    """

    # ATR multipliers (pullback offsets for BUY; reverse for SELL)
    _AGG_MULT  = 0.10
    _PRI_MULT  = 0.25
    _SEC_MULT  = 0.40
    _CON_MULT  = 0.60
    _LATE_MULT = 0.80
    _INVAL_MULT = 1.50

    def calculate(self, forecast: ExecutionForecast) -> EntryZones:
        p   = forecast.current_price
        atr = forecast.atr
        # For BUY: zones are below current price (pullback entries)
        # For SELL: zones are above current price (pullback entries)
        sign = -1.0 if forecast.is_buy else 1.0

        agg  = round(p + sign * self._AGG_MULT  * atr, 5)
        pri  = round(p + sign * self._PRI_MULT  * atr, 5)
        sec  = round(p + sign * self._SEC_MULT  * atr, 5)
        con  = round(p + sign * self._CON_MULT  * atr, 5)
        late = round(p + sign * self._LATE_MULT * atr, 5)
        # Invalidation is beyond the conservative entry — further in the pullback direction
        inval = round(pri + sign * self._INVAL_MULT * atr, 5)

        zone_width       = round(abs(con - agg), 5)
        risk_to_primary  = round(abs(pri - inval), 5)

        logger.debug(
            f"EntryZones {forecast.instrument} {forecast.direction}: "
            f"primary={pri} conservative={con} invalidation={inval}"
        )

        return EntryZones(
            instrument         = forecast.instrument,
            direction          = forecast.direction,
            primary_entry      = pri,
            secondary_entry    = sec,
            aggressive_entry   = agg,
            conservative_entry = con,
            late_entry         = late,
            invalidation       = inval,
            atr_reference      = atr,
            zone_width         = zone_width,
            risk_to_primary    = risk_to_primary,
        )
