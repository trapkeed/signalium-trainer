"""Phase 22 — Volatility Microstructure.

Generates:
ATR Expansion, ATR Compression, Volatility Clustering, Volatility Persistence,
Volatility Shock Detection, Noise Ratio, Range Compression, Range Expansion.
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

from trainer.microstructure.micro_models import MicrostructureFeatures, OHLCBar


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-10 else default


class VolatilityMicrostructure:
    """Derives volatility microstructure features from a window of OHLC bars."""

    # Thresholds
    EXPANSION_THRESHOLD = 1.30  # 30% above avg = expansion
    COMPRESSION_THRESHOLD = 0.70  # 30% below avg = compression
    SHOCK_Z_SCORE = 2.50  # Z-score for shock detection
    LARGE_CANDLE_MULTIPLE = 1.50  # range > 1.5× avg = large candle

    def analyse_window(
        self,
        bars: List[OHLCBar],
        features: MicrostructureFeatures,
        atr_period: int = 14,
    ) -> MicrostructureFeatures:
        if len(bars) < 2:
            return features

        ranges = [b.range for b in bars]
        current_range = ranges[-1]

        # Average range (ATR proxy)
        window_ranges = ranges[-atr_period:] if len(ranges) >= atr_period else ranges
        avg_range = statistics.mean(window_ranges) if window_ranges else current_range

        if avg_range < 1e-10:
            return features

        ratio = current_range / avg_range

        # ATR Expansion / Compression (0–1 scale)
        features.atr_expansion = _clamp((ratio - 1.0) / (self.EXPANSION_THRESHOLD - 1.0))
        features.atr_compression = _clamp((1.0 - ratio) / (1.0 - self.COMPRESSION_THRESHOLD))
        features.range_expansion = 1.0 if ratio >= self.EXPANSION_THRESHOLD else 0.0
        features.range_compression = 1.0 if ratio <= self.COMPRESSION_THRESHOLD else 0.0

        # Volatility Clustering — std(ranges) / mean(ranges)
        if len(window_ranges) >= 3:
            std_range = statistics.stdev(window_ranges)
            features.volatility_clustering = _clamp(_safe_div(std_range, avg_range))
        else:
            features.volatility_clustering = 0.0

        # Volatility Persistence — autocorrelation (lag-1) of ranges
        if len(window_ranges) >= 4:
            features.volatility_persistence = _clamp(abs(self._autocorr_lag1(window_ranges)))

        # Volatility Shock — Z-score of current range
        if len(window_ranges) >= 4:
            std_r = statistics.stdev(window_ranges)
            if std_r > 1e-10:
                z = (current_range - avg_range) / std_r
                features.volatility_shock = (
                    1.0 if z > self.SHOCK_Z_SCORE else _clamp(z / self.SHOCK_Z_SCORE)
                )

        # Noise Ratio — 1 - avg(body/range) = proportion of wicks
        body_ratios = [
            abs(b.close - b.open) / b.range if b.range > 1e-10 else 0.0 for b in bars[-atr_period:]
        ]
        avg_body_ratio = statistics.mean(body_ratios) if body_ratios else 0.5
        features.noise_ratio = _clamp(1.0 - avg_body_ratio)

        logger.debug(
            f"VolatilityMicro: ratio={ratio:.2f} expansion={features.atr_expansion:.3f} "
            f"compression={features.atr_compression:.3f} shock={features.volatility_shock:.3f}"
        )
        return features

    @staticmethod
    def _autocorr_lag1(series: List[float]) -> float:
        """Pearson autocorrelation at lag 1."""
        n = len(series)
        if n < 3:
            return 0.0
        mean = sum(series) / n
        shifted = series[:-1]
        original = series[1:]
        cov = sum((a - mean) * (b - mean) for a, b in zip(shifted, original)) / (n - 1)
        var = sum((x - mean) ** 2 for x in series) / n
        return cov / var if var > 1e-10 else 0.0
