"""Phase 24 — Position Sizing Engine.

Supports 10 sizing methodologies:
Fixed Fractional, Fixed Risk, ATR Adjusted, Volatility Adjusted,
Expected Value Adjusted, Confidence Adjusted, Risk Budget Adjusted,
Kelly Criterion, Fractional Kelly, Custom Formula.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.icari.icari_types import SizingMethod
from trainer.icari.icari_models import AllocationRequest, PositionSize

_MIN_UNITS = 0.001
_MAX_RISK_FRAC = 0.05  # never risk more than 5% per trade


class PositionSizingEngine:
    """Multi-method position sizing engine."""

    def __init__(
        self,
        method: SizingMethod = SizingMethod.FIXED_RISK,
        risk_frac: float = 0.01,  # default 1% risk
        kelly_frac: float = 0.5,  # fractional kelly multiplier
        atr_mult: float = 2.0,  # stop = atr * atr_mult
        custom_fn: Optional[Callable] = None,
    ):
        self._method = method
        self._risk_frac = min(risk_frac, _MAX_RISK_FRAC)
        self._kelly_frac = kelly_frac
        self._atr_mult = atr_mult
        self._custom_fn = custom_fn

    def size(
        self,
        req: AllocationRequest,
        method: Optional[SizingMethod] = None,
    ) -> PositionSize:
        """Calculate position size using the configured or specified method."""
        m = method or self._method

        dispatch = {
            SizingMethod.FIXED_FRACTIONAL: self._fixed_fractional,
            SizingMethod.FIXED_RISK: self._fixed_risk,
            SizingMethod.ATR_ADJUSTED: self._atr_adjusted,
            SizingMethod.VOLATILITY_ADJUSTED: self._volatility_adjusted,
            SizingMethod.EXPECTED_VALUE: self._expected_value_adjusted,
            SizingMethod.CONFIDENCE_ADJUSTED: self._confidence_adjusted,
            SizingMethod.RISK_BUDGET_ADJUSTED: self._risk_budget_adjusted,
            SizingMethod.KELLY: self._kelly,
            SizingMethod.FRACTIONAL_KELLY: self._fractional_kelly,
            SizingMethod.CUSTOM: self._custom,
        }

        fn = dispatch.get(m, self._fixed_risk)
        result = fn(req)
        result.method = m

        logger.debug(f"PositionSizing: {req.instrument} method={m.value} units={result.units:.4f}")
        return result

    # ── Methods ───────────────────────────────────────────────────────────────

    def _fixed_fractional(self, req: AllocationRequest) -> PositionSize:
        """Allocate a fixed fraction of capital."""
        units = (req.capital * self._risk_frac) / max(req.atr, 1e-9)
        return PositionSize(
            instrument=req.instrument,
            method=SizingMethod.FIXED_FRACTIONAL,
            units=max(_MIN_UNITS, units),
            risk_amount=req.capital * self._risk_frac,
            risk_frac=self._risk_frac,
            stop_distance=req.atr,
        )

    def _fixed_risk(self, req: AllocationRequest) -> PositionSize:
        """Risk exactly risk_frac of capital per trade."""
        risk_amount = req.capital * self._risk_frac
        stop_dist = req.atr * self._atr_mult
        units = risk_amount / max(stop_dist, 1e-9)
        return PositionSize(
            instrument=req.instrument,
            method=SizingMethod.FIXED_RISK,
            units=max(_MIN_UNITS, units),
            risk_amount=risk_amount,
            risk_frac=self._risk_frac,
            stop_distance=stop_dist,
        )

    def _atr_adjusted(self, req: AllocationRequest) -> PositionSize:
        """Size based on ATR; tighter ATR → larger position."""
        risk_amount = req.capital * self._risk_frac
        stop_dist = req.atr * self._atr_mult
        units = risk_amount / max(stop_dist, 1e-9)
        return PositionSize(
            instrument=req.instrument,
            method=SizingMethod.ATR_ADJUSTED,
            units=max(_MIN_UNITS, units),
            risk_amount=risk_amount,
            risk_frac=self._risk_frac,
            stop_distance=stop_dist,
        )

    def _volatility_adjusted(self, req: AllocationRequest) -> PositionSize:
        """Reduce size when volatility is high."""
        vol_mult = max(0.25, 1.0 - req.volatility * 10)
        risk_amount = req.capital * self._risk_frac * vol_mult
        stop_dist = req.atr * self._atr_mult
        units = risk_amount / max(stop_dist, 1e-9)
        return PositionSize(
            instrument=req.instrument,
            method=SizingMethod.VOLATILITY_ADJUSTED,
            units=max(_MIN_UNITS, units),
            risk_amount=risk_amount,
            risk_frac=risk_amount / max(req.capital, 1.0),
            stop_distance=stop_dist,
        )

    def _expected_value_adjusted(self, req: AllocationRequest) -> PositionSize:
        """Scale size by expected value; zero/negative EV → minimal size."""
        ev_mult = max(0.1, min(2.0, 1.0 + req.expected_value * 50))
        risk_amount = req.capital * self._risk_frac * ev_mult
        stop_dist = req.atr * self._atr_mult
        units = risk_amount / max(stop_dist, 1e-9)
        return PositionSize(
            instrument=req.instrument,
            method=SizingMethod.EXPECTED_VALUE,
            units=max(_MIN_UNITS, units),
            risk_amount=risk_amount,
            risk_frac=min(_MAX_RISK_FRAC, risk_amount / max(req.capital, 1.0)),
            stop_distance=stop_dist,
        )

    def _confidence_adjusted(self, req: AllocationRequest) -> PositionSize:
        """Scale size by composite forecast confidence."""
        conf = (req.forecast_confidence + req.model_confidence + req.ensemble_confidence) / 3.0
        risk_amount = req.capital * self._risk_frac * conf
        stop_dist = req.atr * self._atr_mult
        units = risk_amount / max(stop_dist, 1e-9)
        return PositionSize(
            instrument=req.instrument,
            method=SizingMethod.CONFIDENCE_ADJUSTED,
            units=max(_MIN_UNITS, units),
            risk_amount=risk_amount,
            risk_frac=min(_MAX_RISK_FRAC, risk_amount / max(req.capital, 1.0)),
            stop_distance=stop_dist,
        )

    def _risk_budget_adjusted(self, req: AllocationRequest) -> PositionSize:
        """Reduce size if portfolio exposure is near budget."""
        budget_remaining = max(0.0, 1.0 - req.portfolio_exposure)
        rf = min(self._risk_frac, self._risk_frac * budget_remaining)
        risk_amount = req.capital * rf
        stop_dist = req.atr * self._atr_mult
        units = risk_amount / max(stop_dist, 1e-9)
        return PositionSize(
            instrument=req.instrument,
            method=SizingMethod.RISK_BUDGET_ADJUSTED,
            units=max(_MIN_UNITS, units),
            risk_amount=risk_amount,
            risk_frac=rf,
            stop_distance=stop_dist,
        )

    def _kelly(self, req: AllocationRequest) -> PositionSize:
        """Full Kelly criterion: f* = (p*(b+1) - 1) / b.
        p = win probability ≈ forecast_confidence
        b = win/loss ratio ≈ |expected_return / expected_drawdown|
        """
        p = max(0.01, min(0.99, req.forecast_confidence))
        b = abs(req.expected_return) / max(abs(req.expected_drawdown), 0.001)
        kelly_frac = (p * (b + 1) - 1) / max(b, 1e-9)
        kelly_frac = max(0.0, min(_MAX_RISK_FRAC, kelly_frac))
        risk_amount = req.capital * kelly_frac
        stop_dist = req.atr * self._atr_mult
        units = risk_amount / max(stop_dist, 1e-9)
        return PositionSize(
            instrument=req.instrument,
            method=SizingMethod.KELLY,
            units=max(_MIN_UNITS, units),
            risk_amount=risk_amount,
            risk_frac=kelly_frac,
            stop_distance=stop_dist,
        )

    def _fractional_kelly(self, req: AllocationRequest) -> PositionSize:
        """Fractional Kelly: apply kelly_frac multiplier to full Kelly."""
        full = self._kelly(req)
        frac = full.risk_frac * self._kelly_frac
        stop_dist = req.atr * self._atr_mult
        risk_amount = req.capital * frac
        units = risk_amount / max(stop_dist, 1e-9)
        return PositionSize(
            instrument=req.instrument,
            method=SizingMethod.FRACTIONAL_KELLY,
            units=max(_MIN_UNITS, units),
            risk_amount=risk_amount,
            risk_frac=frac,
            stop_distance=stop_dist,
        )

    def _custom(self, req: AllocationRequest) -> PositionSize:
        """Call the administrator-provided custom formula."""
        if self._custom_fn:
            return self._custom_fn(req)
        # Fallback to fixed_risk
        logger.warning("Custom sizing function not provided; falling back to fixed_risk")
        return self._fixed_risk(req)

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare_methods(self, req: AllocationRequest) -> Dict[str, PositionSize]:
        """Run all methods and return a comparison dict."""
        results = {}
        for m in SizingMethod:
            if m == SizingMethod.CUSTOM:
                continue
            results[m.value] = self.size(req, method=m)
        return results
