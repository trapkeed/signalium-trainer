"""Phase 22 — Market Efficiency.

Estimates:
Trend Efficiency, Mean Reversion Probability, Randomness Index,
Fractal Behaviour (fractal dimension), Directional Stability, Noise Level.
"""

from __future__ import annotations

import math
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


class MarketEfficiency:
    """Estimates market efficiency from OHLC price data."""

    def analyse_window(
        self,
        bars:     List[OHLCBar],
        features: MicrostructureFeatures,
    ) -> MicrostructureFeatures:
        if len(bars) < 3:
            return features

        closes = [b.close for b in bars]
        moves  = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        # ── Trend Efficiency (Efficiency Ratio / ER) ──────────────────────────
        # ER = directional move / sum(abs moves)
        net_move   = abs(closes[-1] - closes[0])
        total_move = sum(abs(m) for m in moves)
        features.trend_efficiency = _clamp(_safe_div(net_move, total_move, 0.0))

        # ── Mean Reversion Probability ────────────────────────────────────────
        # High MRP when market oscillates (moves frequently reverse)
        reversals = sum(
            1 for i in range(1, len(moves))
            if (moves[i] >= 0) != (moves[i - 1] >= 0)
        )
        reversal_rate = _safe_div(reversals, len(moves) - 1, 0.0) if len(moves) > 1 else 0.5
        features.mean_reversion_probability = _clamp(reversal_rate)

        # ── Randomness Index ──────────────────────────────────────────────────
        # Trend efficiency close to 0 = random walk, close to 1 = trending
        features.randomness_index = _clamp(1.0 - features.trend_efficiency)

        # ── Fractal Dimension ─────────────────────────────────────────────────
        # Simplified Hurst-like estimate via variance method
        # D ≈ 1 (trending) to D ≈ 2 (random walk)
        features.fractal_dimension = self._estimate_fractal_dimension(closes)

        # ── Directional Stability ─────────────────────────────────────────────
        # High stability = consistent direction across the window
        bullish = sum(1 for m in moves if m > 0)
        stability = abs(bullish / len(moves) - 0.5) * 2.0   # 0=equal split, 1=all same dir
        features.directional_stability = _clamp(stability)

        # ── Noise Level ───────────────────────────────────────────────────────
        # Noise = mean abs(move) / range of whole window
        mean_abs_move = statistics.mean(abs(m) for m in moves) if moves else 0.0
        total_range   = max(closes) - min(closes)
        features.noise_level = _clamp(_safe_div(mean_abs_move * len(moves), total_range + mean_abs_move, 0.5))

        logger.debug(
            f"MarketEfficiency: ER={features.trend_efficiency:.3f} "
            f"MRP={features.mean_reversion_probability:.3f} "
            f"fractal={features.fractal_dimension:.3f}"
        )
        return features

    @staticmethod
    def _estimate_fractal_dimension(closes: List[float]) -> float:
        """Simplified fractal dimension estimate.

        D ≈ 1 = pure trend, D ≈ 2 = pure random walk.
        Based on range-to-scale ratio.
        """
        n = len(closes)
        if n < 4:
            return 1.5

        full_range = max(closes) - min(closes)
        if full_range < 1e-10:
            return 1.5

        # Split into halves
        half = n // 2
        range1 = max(closes[:half]) - min(closes[:half])
        range2 = max(closes[half:]) - min(closes[half:])
        sum_half_ranges = range1 + range2

        if sum_half_ranges < 1e-10:
            return 1.5

        # D = log(N) / log(N/r) where r = range ratio
        try:
            ratio = _safe_div(sum_half_ranges, full_range, 1.0)
            if ratio <= 1e-10:
                return 1.5
            d = 1.0 + math.log(2.0) / math.log(2.0 / max(ratio, 0.001))
            return max(1.0, min(2.0, d))
        except (ValueError, ZeroDivisionError):
            return 1.5
