"""Phase 24.9 — Volatility Intelligence."""

from __future__ import annotations

import statistics
from typing import List

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import MarketInput, VolatilityResult
from trainer.mie.mie_types import VolatilityState, clamp


def _stdev(vals: List[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def _mean(vals: List[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _percentile_rank(values: List[float], target: float) -> float:
    """Return percentile rank of target in values list (0-100)."""
    if not values:
        return 50.0
    below = sum(1 for v in values if v <= target)
    return 100.0 * below / len(values)


class VolatilityIntelligenceEngine:
    """
    Computes VolatilityState and a quality score 0-100 where:
    - 100 = ideal volatility for trading (not too low, not too high)
    - 0   = either completely flat or wildly extreme

    Uses:
    - Returns series standard deviation (current vs. baseline)
    - ATR magnitude and percentile
    - Volatility ratio relative to long-run baseline
    """

    def __init__(
        self,
        optimal_vol_low: float = 0.5,  # ratio below baseline = too low
        optimal_vol_high: float = 2.0,  # ratio above baseline = too high
        extreme_ratio: float = 4.0,
    ) -> None:
        self._opt_low = optimal_vol_low
        self._opt_high = optimal_vol_high
        self._extreme = extreme_ratio

    # ── Public API ─────────────────────────────────────────────────────────

    def analyse(self, inp: MarketInput) -> VolatilityResult:
        returns = inp.returns or []
        curr_vol = _stdev(returns[-20:]) if len(returns) >= 2 else 0.0
        base_vol = _stdev(returns) if len(returns) >= 10 else max(curr_vol, 1e-9)
        vol_ratio = curr_vol / max(base_vol, 1e-9)

        # ATR-based percentile
        atr_series = [h - l for h, l in zip(inp.highs, inp.lows)] if inp.highs and inp.lows else []
        atr_pct = _percentile_rank(atr_series, inp.atr) if atr_series else 50.0

        # Absolute vol relative to ATR (detects extreme vol even when ratio ≈ 1)
        atr_vol_ratio = curr_vol / max(inp.atr, 1e-9)

        state = self._classify_state(vol_ratio, atr_vol_ratio)
        quality = self._compute_quality(vol_ratio, atr_pct, atr_vol_ratio)

        logger.debug(
            f"[VI] {inp.symbol} vol_ratio={vol_ratio:.3f} atr_ratio={atr_vol_ratio:.1f} → {state.value} q={quality:.1f}"
        )
        return VolatilityResult(
            state=state,
            quality=quality,
            current_vol=curr_vol,
            baseline_vol=base_vol,
            vol_ratio=vol_ratio,
            atr_percentile=atr_pct,
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _classify_state(self, vol_ratio: float, atr_vol_ratio: float = 0.0) -> VolatilityState:
        # Absolute-vol override: if current vol >> ATR, it's extreme regardless of ratio
        if vol_ratio >= self._extreme or atr_vol_ratio > 20:
            return VolatilityState.EXTREME
        if vol_ratio >= self._opt_high or atr_vol_ratio > 10:
            return VolatilityState.HIGH
        if vol_ratio >= self._opt_low:
            return VolatilityState.NORMAL
        if vol_ratio >= 0.25:
            return VolatilityState.LOW
        return VolatilityState.VERY_LOW

    def _compute_quality(
        self, vol_ratio: float, atr_pct: float, atr_vol_ratio: float = 0.0
    ) -> float:
        """
        Quality is maximised when vol_ratio ≈ 1.0 (normal conditions).
        Penalty applied for both very low and very high volatility.
        """
        # Extreme absolute volatility → very poor quality for trading
        if atr_vol_ratio > 20:
            return 10.0
        if atr_vol_ratio > 10:
            return clamp(30.0, 0.0, 100.0)

        # Ideal target = 1.0 (current vol == baseline)
        distance = abs(vol_ratio - 1.0)
        # Map 0 distance → 100, 3+ distance → 0
        vol_quality = clamp(100.0 - distance * 33.0, 0.0, 100.0)

        # ATR percentile blending: 30th–70th percentile = optimal
        if 30 <= atr_pct <= 70:
            atr_quality = 100.0
        elif atr_pct < 30:
            atr_quality = atr_pct / 30.0 * 100.0
        else:
            atr_quality = (100.0 - atr_pct) / 30.0 * 100.0

        return clamp(vol_quality * 0.70 + atr_quality * 0.30, 0.0, 100.0)
