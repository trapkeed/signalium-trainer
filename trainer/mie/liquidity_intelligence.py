"""Phase 24.9 — Liquidity Intelligence."""

from __future__ import annotations

import statistics
from typing import List

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import LiquidityResult, MarketInput
from trainer.mie.mie_types import LiquidityState, clamp


def _mean(vals: List[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


class LiquidityIntelligenceEngine:
    """
    Computes LiquidityState and a quality score 0-100 from:
    - Volume relative to historical average  (40%)
    - Bid-ask spread as fraction of ATR      (40%)
    - Depth estimate from volume consistency (20%)
    """

    def __init__(
        self,
        vol_lookback: int = 20,
        illiquid_vol: float = 0.1,  # volume < 10% of avg → ILLIQUID
        poor_vol: float = 0.3,
        spread_illiquid: float = 0.5,  # spread > 50% of ATR → ILLIQUID
        spread_poor: float = 0.20,
    ) -> None:
        self._vol_lb = vol_lookback
        self._illiquid_vol = illiquid_vol
        self._poor_vol = poor_vol
        self._spread_ill = spread_illiquid
        self._spread_poor = spread_poor

    # ── Public API ─────────────────────────────────────────────────────────

    def analyse(self, inp: MarketInput) -> LiquidityResult:
        volumes = inp.volumes or []
        avg_vol = _mean(volumes[-self._vol_lb :]) if len(volumes) >= 3 else inp.volume
        vol_ratio = inp.volume / max(avg_vol, 1.0)

        spread_atr_ratio = inp.spread / max(inp.atr, 1e-9)

        vol_score = self._score_volume(vol_ratio)
        spread_score = self._score_spread(spread_atr_ratio)
        depth_score = self._score_depth(volumes)

        quality = vol_score * 0.40 + spread_score * 0.40 + depth_score * 0.20
        state = self._classify_state(vol_ratio, spread_atr_ratio)

        logger.debug(
            f"[LIQ] {inp.symbol} vol_r={vol_ratio:.2f} spr_r={spread_atr_ratio:.3f} "
            f"→ {state.value} q={quality:.1f}"
        )
        return LiquidityResult(
            state=state,
            quality=clamp(quality, 0.0, 100.0),
            volume_score=clamp(vol_score, 0.0, 100.0),
            spread_score=clamp(spread_score, 0.0, 100.0),
            depth_estimate=clamp(depth_score, 0.0, 100.0),
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _score_volume(self, vol_ratio: float) -> float:
        if vol_ratio >= 2.0:
            return 100.0
        if vol_ratio >= 1.0:
            return 80.0 + (vol_ratio - 1.0) * 20.0
        if vol_ratio >= 0.5:
            return 50.0 + (vol_ratio - 0.5) * 60.0
        if vol_ratio >= 0.1:
            return (vol_ratio - 0.1) / 0.4 * 50.0
        return 0.0

    def _score_spread(self, spread_atr_ratio: float) -> float:
        if spread_atr_ratio >= self._spread_ill:
            return 0.0
        if spread_atr_ratio >= self._spread_poor:
            return clamp(
                (self._spread_ill - spread_atr_ratio)
                / (self._spread_ill - self._spread_poor)
                * 30.0,
                0.0,
                30.0,
            )
        return clamp(100.0 - spread_atr_ratio / self._spread_poor * 70.0, 30.0, 100.0)

    def _score_depth(self, volumes: List[float]) -> float:
        """Consistency of volume = proxy for market depth."""
        if len(volumes) < 5:
            return 50.0
        avg = _mean(volumes[-20:])
        if avg <= 0:
            return 0.0
        cv = statistics.stdev(volumes[-20:]) / avg if len(volumes) >= 2 else 0.0
        # Low coefficient of variation = consistent depth = good
        return clamp(100.0 - cv * 50.0, 0.0, 100.0)

    def _classify_state(self, vol_ratio: float, spread_atr_ratio: float) -> LiquidityState:
        if vol_ratio < self._illiquid_vol or spread_atr_ratio > self._spread_ill:
            return LiquidityState.ILLIQUID
        if vol_ratio < self._poor_vol or spread_atr_ratio > self._spread_poor:
            return LiquidityState.POOR
        if vol_ratio >= 1.5 and spread_atr_ratio < 0.05:
            return LiquidityState.EXCELLENT
        if vol_ratio >= 0.8 and spread_atr_ratio < 0.15:
            return LiquidityState.GOOD
        return LiquidityState.FAIR
