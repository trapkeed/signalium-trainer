"""Phase 22 — Price Action Analyzer.

Derives advanced price-action metrics from OHLC bars:
Body Ratio, Wick Ratios, Close/Open Position, Candle Efficiency,
Directional Efficiency, Compression, Expansion, Impulse Strength,
Retracement Quality, Continuation Probability.
"""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.microstructure.micro_models import MicrostructureFeatures, OHLCBar


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-10 else default


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class PriceActionAnalyzer:
    """Computes price-action microstructure features from OHLC data."""

    def analyse_bar(self, bar: OHLCBar) -> dict:
        """Compute single-bar price action features."""
        rng = bar.range
        body = bar.body

        body_ratio = _clamp(_safe_div(body, rng, 0.0))
        upper_wick_ratio = _clamp(_safe_div(bar.high - max(bar.open, bar.close), rng, 0.0))
        lower_wick_ratio = _clamp(_safe_div(min(bar.open, bar.close) - bar.low, rng, 0.0))
        close_position = _clamp(_safe_div(bar.close - bar.low, rng, 0.5))
        open_position = _clamp(_safe_div(bar.open - bar.low, rng, 0.5))
        candle_efficiency = body_ratio  # efficiency = body / full range
        directional_efficiency = (1.0 if bar.is_bullish else -1.0) * body_ratio

        # Compression: small range vs body (bar "compressed" = small body large wicks)
        compression = _clamp(1.0 - body_ratio)
        # Expansion: large clean body = expansion
        expansion = body_ratio

        impulse_strength = body_ratio  # 0–1, size of directional move

        return {
            "body_ratio": round(body_ratio, 4),
            "upper_wick_ratio": round(upper_wick_ratio, 4),
            "lower_wick_ratio": round(lower_wick_ratio, 4),
            "close_position": round(close_position, 4),
            "open_position": round(open_position, 4),
            "candle_efficiency": round(candle_efficiency, 4),
            "directional_efficiency": round(directional_efficiency, 4),
            "compression": round(compression, 4),
            "expansion": round(expansion, 4),
            "impulse_strength": round(impulse_strength, 4),
        }

    def retracement_quality(self, bars: list[OHLCBar]) -> float:
        """Measure quality of a retracement within a sequence of bars.

        High value = clean, shallow pullback (good quality).
        Low  value = deep, choppy pullback (poor quality).
        """
        if len(bars) < 2:
            return 0.5
        # Direction of the dominant move (from first to penultimate bar)
        initial_direction = 1.0 if bars[0].close < bars[-2].close else -1.0
        # Retracement: how much did the last bar retrace?
        prev_move = abs(bars[-2].close - bars[0].close)
        last_move = (bars[-1].close - bars[-2].close) * initial_direction
        # A positive last_move (in same direction) = continuation, not retracement
        # Retracement quality = 1 when the pullback is shallow
        if prev_move < 1e-10:
            return 0.5
        retracement_depth = max(-last_move / prev_move, 0.0)  # positive = pullback
        quality = _clamp(1.0 - retracement_depth)
        return round(quality, 4)

    def continuation_probability(self, bars: list[OHLCBar]) -> float:
        """Estimate probability of trend continuation.

        Based on: close position in recent bars, directional persistence,
        and retracement quality.
        """
        if len(bars) < 3:
            return 0.5
        # Directional persistence
        bullish = sum(1 for b in bars if b.is_bullish)
        persistence = bullish / len(bars)

        # Close position of most recent bar (high close = bullish continuation)
        last = bars[-1]
        cp = _clamp(_safe_div(last.close - last.low, last.range, 0.5))

        # Blend
        bullish_signal = (persistence + cp) / 2.0
        # Above 0.5 = bullish continuation, below = bearish continuation
        # Return distance from 0.5, scaled to 0-1
        prob = _clamp(bullish_signal)
        return round(prob, 4)

    def analyse_window(
        self,
        bars: list[OHLCBar],
        features: MicrostructureFeatures,
    ) -> MicrostructureFeatures:
        """Compute all price-action features and write into the features object."""
        if not bars:
            return features

        current = bars[-1]
        bar_features = self.analyse_bar(current)

        features.body_ratio = bar_features["body_ratio"]
        features.upper_wick_ratio = bar_features["upper_wick_ratio"]
        features.lower_wick_ratio = bar_features["lower_wick_ratio"]
        features.close_position = bar_features["close_position"]
        features.open_position = bar_features["open_position"]
        features.candle_efficiency = bar_features["candle_efficiency"]
        features.directional_efficiency = bar_features["directional_efficiency"]
        features.compression = bar_features["compression"]
        features.expansion = bar_features["expansion"]
        features.impulse_strength = bar_features["impulse_strength"]

        if len(bars) >= 2:
            features.retracement_quality = self.retracement_quality(bars)
            features.continuation_probability = self.continuation_probability(bars)

        return features
