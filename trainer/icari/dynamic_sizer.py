"""Phase 24 — Dynamic Position Sizer.

Adapts position size based on:
forecast_quality, portfolio_exposure, current_drawdown,
market_volatility, liquidity, correlation, spread, session,
historical_strategy_performance, model_health.
"""

from __future__ import annotations

from typing import Dict, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.icari.icari_models import AllocationRequest, PositionSize
from trainer.icari.icari_types import SizingMethod
from trainer.icari.position_sizing_engine import PositionSizingEngine

_ADJUSTMENT_BOUNDS = (0.10, 2.0)  # min/max adjustment multiplier


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class DynamicSizer:
    """Applies adaptive multipliers to position sizes based on portfolio state."""

    def __init__(
        self,
        base_engine: PositionSizingEngine = None,
        drawdown_soft_limit: float = 0.05,
        drawdown_hard_limit: float = 0.10,
        max_correlation: float = 0.70,
        session_multipliers: Optional[Dict[str, float]] = None,
    ):
        self._engine = base_engine or PositionSizingEngine()
        self._dd_soft = drawdown_soft_limit
        self._dd_hard = drawdown_hard_limit
        self._max_corr = max_correlation
        self._session_mult = session_multipliers or {
            "LONDON": 1.00,
            "NEW_YORK": 1.00,
            "TOKYO": 0.80,
            "SYDNEY": 0.70,
            "OVERLAP": 1.10,
        }

    def size(
        self,
        req: AllocationRequest,
        current_drawdown: float = 0.0,
        session: str = "LONDON",
        historical_perf: float = 0.5,  # 0–1 strategy performance score
        model_health: float = 1.0,  # 0–1
        spread_pips: float = 0.0,
        spread_threshold: float = 3.0,
    ) -> PositionSize:
        """Size the position then apply all dynamic adjustments."""
        base = self._engine.size(req)

        adjustments: Dict[str, float] = {}

        # 1. Drawdown adjustment
        dd_adj = self._drawdown_adj(current_drawdown)
        adjustments["drawdown"] = dd_adj

        # 2. Exposure adjustment
        exp_adj = self._exposure_adj(req.portfolio_exposure)
        adjustments["exposure"] = exp_adj

        # 3. Volatility adjustment
        vol_adj = self._volatility_adj(req.volatility)
        adjustments["volatility"] = vol_adj

        # 4. Liquidity adjustment
        liq_adj = _clamp(req.liquidity, 0.1, 1.0)
        adjustments["liquidity"] = liq_adj

        # 5. Correlation adjustment
        corr_adj = self._correlation_adj(req.correlation)
        adjustments["correlation"] = corr_adj

        # 6. Spread adjustment
        spread_adj = (
            1.0
            if spread_pips <= spread_threshold
            else max(0.3, 1.0 - (spread_pips - spread_threshold) * 0.1)
        )
        adjustments["spread"] = spread_adj

        # 7. Session adjustment
        sess_adj = self._session_mult.get(session.upper(), 0.90)
        adjustments["session"] = sess_adj

        # 8. Historical performance
        hist_adj = _clamp(0.5 + historical_perf * 0.5, 0.5, 1.5)
        adjustments["historical_perf"] = hist_adj

        # 9. Model health
        mh_adj = _clamp(model_health, 0.3, 1.0)
        adjustments["model_health"] = mh_adj

        # 10. Forecast quality (from request)
        fq_adj = _clamp(req.forecast_confidence, 0.2, 1.0)
        adjustments["forecast_quality"] = fq_adj

        # Composite multiplier — geometric mean preserves monotonicity
        composite = 1.0
        for v in adjustments.values():
            composite *= v
        composite = composite ** (1.0 / len(adjustments))
        composite = _clamp(composite, *_ADJUSTMENT_BOUNDS)

        # Apply
        new_units = max(0.001, base.units * composite)
        new_risk_amount = base.risk_amount * composite
        new_risk_frac = min(0.05, base.risk_frac * composite)

        result = PositionSize(
            instrument=base.instrument,
            method=base.method,
            units=round(new_units, 4),
            risk_amount=round(new_risk_amount, 2),
            risk_frac=round(new_risk_frac, 4),
            stop_distance=base.stop_distance,
            adjusted=True,
            adjustments={k: round(v, 4) for k, v in adjustments.items()},
        )

        logger.debug(
            f"DynamicSizer: {req.instrument} composite_mult={composite:.3f} "
            f"base_units={base.units:.4f} → {result.units:.4f}"
        )
        return result

    # ── Adjustment functions ──────────────────────────────────────────────────

    def _drawdown_adj(self, dd: float) -> float:
        if dd >= self._dd_hard:
            return 0.25  # emergency reduction
        if dd >= self._dd_soft:
            # Linear reduction from 1.0 at soft to 0.25 at hard
            span = max(self._dd_hard - self._dd_soft, 1e-9)
            frac = (dd - self._dd_soft) / span
            return 1.0 - frac * 0.75
        return 1.0

    def _exposure_adj(self, exposure: float) -> float:
        if exposure >= 0.90:
            return 0.25
        if exposure >= 0.70:
            return 0.60
        if exposure >= 0.50:
            return 0.85
        return 1.0

    def _volatility_adj(self, vol: float) -> float:
        # vol is daily fraction; above 2% start reducing
        if vol >= 0.05:
            return 0.40
        if vol >= 0.02:
            return _clamp(1.0 - (vol - 0.01) * 15, 0.40, 1.0)
        return 1.0

    def _correlation_adj(self, corr: float) -> float:
        if corr >= self._max_corr:
            return 0.50
        if corr >= 0.50:
            return 0.75
        return 1.0

    def drawdown_mode(self, drawdown: float) -> str:
        if drawdown >= self._dd_hard * 2:
            return "EMERGENCY_LOCKDOWN"
        if drawdown >= self._dd_hard:
            return "HARD_LIMIT"
        if drawdown >= self._dd_soft:
            return "SOFT_LIMIT"
        return "NORMAL"
