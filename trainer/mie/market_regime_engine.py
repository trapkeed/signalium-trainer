"""Phase 24.9 — Market Regime Intelligence (16 regimes)."""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_types import MarketRegime, clamp
from trainer.mie.mie_models import MarketInput, MarketRegimeResult


def _stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return statistics.mean(values)


def _ema(values: List[float], period: int) -> float:
    """Exponential moving average of the last `period` values."""
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    result = values[0]
    for v in values[1:]:
        result = v * k + result * (1 - k)
    return result


def _has_higher_highs(highs: List[float], lookback: int = 5) -> bool:
    if len(highs) < lookback:
        return False
    segment = highs[-lookback:]
    return all(segment[i] >= segment[i - 1] for i in range(1, len(segment)))


def _has_lower_lows(lows: List[float], lookback: int = 5) -> bool:
    if len(lows) < lookback:
        return False
    segment = lows[-lookback:]
    return all(segment[i] <= segment[i - 1] for i in range(1, len(segment)))


def _volume_spike(volumes: List[float], threshold: float = 2.5) -> bool:
    if len(volumes) < 5:
        return False
    avg_vol = _mean(volumes[:-1])
    return avg_vol > 0 and volumes[-1] > threshold * avg_vol


class MarketRegimeEngine:
    """
    Classifies the current market into one of 16 regimes using technical
    indicators, price action patterns, and volume / news context.

    Decision hierarchy:
    1. Black Swan / Illiquid (overrides all)
    2. News Driven
    3. Flash patterns (Breakout / False Breakout / Reversal)
    4. Trend strength (Strong / Trending / Weak / Expansion)
    5. Range patterns (Accumulation / Distribution / Compression / Range)
    6. Volatile
    7. Unknown
    """

    def __init__(
        self,
        adx_strong: float = 40.0,
        adx_trending: float = 25.0,
        adx_weak: float = 15.0,
        vol_extreme: float = 4.0,
        vol_high: float = 2.5,
        vol_compress: float = 0.5,
    ) -> None:
        self._adx_strong = adx_strong
        self._adx_trending = adx_trending
        self._adx_weak = adx_weak
        self._vol_extreme = vol_extreme
        self._vol_high = vol_high
        self._vol_compress = vol_compress

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(self, inp: MarketInput) -> MarketRegimeResult:
        signals: Dict = {}
        regime, confidence, sub = self._classify(inp, signals)
        logger.debug(f"[MRE] {inp.symbol} → {regime.value} ({confidence:.2f})")
        return MarketRegimeResult(
            regime=regime,
            confidence=clamp(confidence, 0.0, 1.0),
            sub_regime=sub,
            signals=signals,
        )

    # ── Internal classification ────────────────────────────────────────────

    def _classify(
        self,
        inp: MarketInput,
        signals: Dict,
    ) -> tuple:
        returns = inp.returns or []
        closes = inp.closes or []
        highs = inp.highs or []
        lows = inp.lows or []
        volumes = inp.volumes or []

        # Derived metrics
        current_vol = _stdev(returns[-20:]) if len(returns) >= 2 else 0.0
        base_vol = _stdev(returns[-60:]) if len(returns) >= 10 else max(current_vol, 1e-9)
        vol_ratio = current_vol / max(base_vol, 1e-9)
        adx = inp.adx
        rsi = inp.rsi
        atr = inp.atr
        spread = inp.spread
        news = inp.news_events or []

        signals["adx"] = round(adx, 2)
        signals["vol_ratio"] = round(vol_ratio, 3)
        signals["rsi"] = round(rsi, 2)
        signals["spread"] = round(spread, 6)

        # 1. Black Swan
        if vol_ratio > self._vol_extreme * 1.5 or (atr > 0 and spread > atr * 0.8):
            return MarketRegime.BLACK_SWAN, 0.90, "extreme_vol_or_spread"

        # 2. Illiquid
        avg_vol = _mean(volumes[-20:]) if len(volumes) >= 5 else 0.0
        if avg_vol > 0 and (inp.volume < avg_vol * 0.10 or spread > atr * 0.5):
            return MarketRegime.ILLIQUID, 0.80, "low_volume_wide_spread"

        # 3. News Driven — HIGH-impact news always triggers regime regardless of vol ratio
        high_news = [e for e in news if e.get("impact", "LOW") == "HIGH"]
        if len(high_news) >= 1:
            return MarketRegime.NEWS_DRIVEN, 0.85, f"{len(high_news)}_high_impact_events"

        # 4. Volatile — vol ratio spike OR absolute vol is high vs ATR
        abs_vol_ratio = current_vol / max(inp.atr, 1e-9)
        if vol_ratio > self._vol_high or abs_vol_ratio > 10.0:
            return MarketRegime.VOLATILE, 0.75, f"vol_ratio={vol_ratio:.2f} abs={abs_vol_ratio:.1f}"

        # 5. Breakout / False Breakout / Reversal
        if len(closes) >= 10 and _volume_spike(volumes):
            price_move = (closes[-1] - closes[-5]) / max(closes[-5], 1e-9)
            if abs(price_move) > 2 * atr / max(inp.close, 1e-9):
                # Direction reversal after a prior trend?
                prior_direction = closes[-5] - closes[-10] if len(closes) >= 10 else 0
                if prior_direction * price_move < 0:
                    return MarketRegime.REVERSAL, 0.72, "vol_spike_reversal"
                # High ADX breakout vs. range breakout
                if adx > self._adx_trending:
                    return MarketRegime.BREAKOUT, 0.78, "trend_breakout"
                else:
                    return MarketRegime.FALSE_BREAKOUT, 0.65, "range_breakout"

        # 6. Strong Trend
        if adx >= self._adx_strong:
            direction = "UP" if rsi > 55 else "DOWN"
            return MarketRegime.STRONG_TREND, 0.88, direction

        # 7. Trending
        if adx >= self._adx_trending:
            # Expansion: ATR expanding?
            if len(inp.highs) >= 5 and len(inp.lows) >= 5:
                recent_range = _mean([h - l for h, l in zip(highs[-5:], lows[-5:])])
                older_range = (
                    _mean([h - l for h, l in zip(highs[-10:-5], lows[-10:-5])])
                    if len(highs) >= 10
                    else recent_range
                )
                if recent_range > older_range * 1.3:
                    return MarketRegime.EXPANSION, 0.75, "atr_expanding"
            return MarketRegime.TRENDING, 0.80, "adx_trending"

        # 8. Weak Trend
        if adx >= self._adx_weak:
            # Check for Accumulation (bullish) or Distribution (bearish)
            if len(closes) >= 10:
                if _has_higher_highs(highs) and rsi < 60:
                    return MarketRegime.ACCUMULATION, 0.68, "hh_rising"
                if _has_lower_lows(lows) and rsi > 40:
                    return MarketRegime.DISTRIBUTION, 0.68, "ll_falling"
            return MarketRegime.WEAK_TREND, 0.70, "adx_weak_trend"

        # 9. Compression — vol ratio OR absolute vol extremely low vs ATR
        abs_vol_atr = current_vol / max(inp.atr, 1e-9)
        if vol_ratio < self._vol_compress or abs_vol_atr < 0.05:
            return (
                MarketRegime.COMPRESSION,
                0.74,
                f"vol_ratio={vol_ratio:.3f} abs_atr={abs_vol_atr:.4f}",
            )

        # 10. Range
        if adx < self._adx_weak:
            return MarketRegime.RANGE, 0.72, "low_adx"

        return MarketRegime.UNKNOWN, 0.40, "no_clear_signal"
