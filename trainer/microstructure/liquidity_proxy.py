"""Phase 22 — Liquidity Proxies.

Estimates liquidity using OHLC data:
Spread Proxy, Gap Frequency, Large Candle Frequency,
Compression Zones, Reaction Zones, Breakout Strength,
Reversal Strength, Liquidity Vacuum Probability.
"""

from __future__ import annotations

import statistics
from typing import List

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.microstructure.micro_models import OHLCBar, MicrostructureFeatures


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-10 else default


class LiquidityProxy:
    """Estimates liquidity from OHLC data using proxy measures."""

    # A large candle is one whose range exceeds this multiple of the avg range
    LARGE_CANDLE_MULTIPLE = 1.5
    # A gap is when open is outside the prior bar's high/low
    GAP_THRESHOLD = 0.10   # 10% of avg range

    def analyse_window(
        self,
        bars:     List[OHLCBar],
        features: MicrostructureFeatures,
    ) -> MicrostructureFeatures:
        if not bars:
            return features

        ranges = [b.range for b in bars]
        avg_range = statistics.mean(ranges) if ranges else 1e-10

        # ── Spread Proxy ──────────────────────────────────────────────────────
        # Low-liquidity bars tend to have larger low-body / high-wick ratios
        # Proxy: avg wick ratio as spread indicator
        wick_ratios = [
            (1.0 - abs(b.close - b.open) / b.range) if b.range > 1e-10 else 0.5
            for b in bars
        ]
        features.spread_proxy = _clamp(statistics.mean(wick_ratios) if wick_ratios else 0.0)

        # ── Gap Frequency ─────────────────────────────────────────────────────
        if len(bars) >= 2 and avg_range > 1e-10:
            gap_threshold = avg_range * self.GAP_THRESHOLD
            gaps = sum(
                1 for i in range(1, len(bars))
                if abs(bars[i].open - bars[i - 1].close) > gap_threshold
            )
            features.gap_frequency = _clamp(gaps / (len(bars) - 1))
        else:
            features.gap_frequency = 0.0

        # ── Large Candle Frequency ────────────────────────────────────────────
        if avg_range > 1e-10:
            large_count = sum(1 for r in ranges if r > avg_range * self.LARGE_CANDLE_MULTIPLE)
            features.large_candle_frequency = _clamp(large_count / len(ranges))
        else:
            features.large_candle_frequency = 0.0

        # ── Compression Zone Score ────────────────────────────────────────────
        # Bars that are small relative to ATR form compression zones
        compression_count = sum(1 for r in ranges if r < avg_range * 0.60)
        features.compression_zone_score = _clamp(compression_count / len(ranges))

        # ── Reaction Zone Score ───────────────────────────────────────────────
        # Bars that close near support/resistance (extreme closes)
        extreme_closes = sum(
            1 for b in bars
            if b.range > 1e-10 and (
                abs(b.close - b.low) / b.range < 0.15 or
                abs(b.close - b.high) / b.range < 0.15
            )
        )
        features.reaction_zone_score = _clamp(extreme_closes / len(bars))

        # ── Breakout Strength ─────────────────────────────────────────────────
        # Breakout = large bullish/bearish candle with close near high/low
        if bars and avg_range > 1e-10:
            last = bars[-1]
            is_large = last.range > avg_range * self.LARGE_CANDLE_MULTIPLE
            close_at_extreme = (
                _safe_div(last.close - last.low, last.range, 0.5) > 0.80 or
                _safe_div(last.close - last.low, last.range, 0.5) < 0.20
            )
            body_strong = abs(last.close - last.open) / last.range > 0.60 if last.range > 1e-10 else False
            strength = 0.0
            if is_large:  strength += 0.40
            if close_at_extreme: strength += 0.35
            if body_strong: strength += 0.25
            features.breakout_strength = _clamp(strength)

        # ── Reversal Strength ─────────────────────────────────────────────────
        if len(bars) >= 2 and avg_range > 1e-10:
            last  = bars[-1]
            prev  = bars[-2]
            # Reversal: previous and last bars move in opposite directions
            # AND last bar closes strongly in the reversal direction
            opposite_dir = (last.close > last.open) != (prev.close > prev.open)
            last_body_ratio = _safe_div(abs(last.close - last.open), last.range, 0.0)
            rev_close = (
                _safe_div(last.close - last.low, last.range, 0.5) > 0.70 or
                _safe_div(last.close - last.low, last.range, 0.5) < 0.30
            )
            strength = 0.0
            if opposite_dir:  strength += 0.40
            if last_body_ratio > 0.50: strength += 0.35
            if rev_close: strength += 0.25
            features.reversal_strength = _clamp(strength)

        # ── Liquidity Vacuum Probability ──────────────────────────────────────
        # High vacuum probability = gaps + large candles + high gap frequency
        features.liquidity_vacuum_prob = _clamp(
            features.gap_frequency * 0.40 +
            features.large_candle_frequency * 0.35 +
            (1.0 - features.compression_zone_score) * 0.25
        )

        logger.debug(
            f"LiquidityProxy: spread={features.spread_proxy:.3f} "
            f"breakout={features.breakout_strength:.3f} "
            f"vacuum={features.liquidity_vacuum_prob:.3f}"
        )
        return features
