"""Phase S9 — Timeframe synchronization pipeline.

Pure alignment + feature computation, no cross-timeframe conditionals. For
a given "as-of" M1 timestamp, `TimeframeSynchronizer` finds the latest
*closed* bar on every timeframe (never a bar that closes after `as_of` —
that would leak future information into a supposedly-live decision) and
turns each timeframe's recent bar history into a `TimeframeFeatureVector`.

The output, a `MultiTimeframeSnapshot`, says nothing about how the
timeframes relate — it is strictly "here is what each timeframe currently
looks like, on its own terms". Relating them is the attention model's job.
"""

from __future__ import annotations

import math

from trainer.qde.qde_timeframe_models import (
    MultiTimeframeSnapshot,
    TimeframeBar,
    TimeframeFeatureVector,
)
from trainer.qde.qde_timeframe_types import ALL_TIMEFRAMES, Timeframe

_LOOKBACK_BARS = 20  # bars used for realized vol / range / volume baselines
_MOMENTUM_LOOKBACK = 10
_SWING_LOOKBACK = 20


class TimeframeSynchronizer:
    """Aligns per-timeframe bar histories to a common M1 timestamp."""

    def compute_feature_vector(
        self,
        instrument: str,
        timeframe: Timeframe,
        bars: list[TimeframeBar],
    ) -> TimeframeFeatureVector:
        """`bars` must be ordered oldest -> newest and already filtered to
        only bars with `close_time <= as_of` (the caller/`align()` enforces
        this). The most recent element is the "current" bar this vector
        describes.
        """
        if not bars:
            return TimeframeFeatureVector(instrument=instrument, timeframe=timeframe)

        current = bars[-1]
        window = bars[-_LOOKBACK_BARS:]

        log_return_1 = 0.0
        if len(bars) >= 2 and bars[-2].close > 0 and current.close > 0:
            log_return_1 = math.log(current.close / bars[-2].close)

        log_returns = []
        for i in range(1, len(window)):
            prev_c, cur_c = window[i - 1].close, window[i].close
            if prev_c > 0 and cur_c > 0:
                log_returns.append(math.log(cur_c / prev_c))
        realized_vol = _stdev(log_returns) if len(log_returns) >= 2 else 0.0

        momentum_n = 0.0
        if len(bars) > _MOMENTUM_LOOKBACK:
            ref = bars[-_MOMENTUM_LOOKBACK - 1].close
            if ref > 0 and current.close > 0:
                raw_momentum = math.log(current.close / ref)
                momentum_n = raw_momentum / realized_vol if realized_vol > 1e-9 else raw_momentum

        swing_window = bars[-_SWING_LOOKBACK:]
        swing_high = max(b.high for b in swing_window)
        swing_low = min(b.low for b in swing_window)
        range_span = swing_high - swing_low
        range_position = ((current.close - swing_low) / range_span) if range_span > 1e-12 else 0.5
        distance_to_swing_high = (
            (swing_high - current.close) / current.close if current.close > 0 else 0.0
        )
        distance_to_swing_low = (
            (current.close - swing_low) / current.close if current.close > 0 else 0.0
        )

        candle_range = current.high - current.low
        body = abs(current.close - current.open)
        body_ratio = (body / candle_range) if candle_range > 1e-12 else 0.0
        upper_wick = current.high - max(current.open, current.close)
        lower_wick = min(current.open, current.close) - current.low
        upper_wick_ratio = (upper_wick / candle_range) if candle_range > 1e-12 else 0.0
        lower_wick_ratio = (lower_wick / candle_range) if candle_range > 1e-12 else 0.0

        volumes = [b.volume for b in window]
        volume_zscore = 0.0
        if len(volumes) >= 2:
            avg_vol = sum(volumes) / len(volumes)
            vol_std = _stdev(volumes)
            if vol_std > 1e-9:
                volume_zscore = (current.volume - avg_vol) / vol_std

        return TimeframeFeatureVector(
            instrument=instrument,
            timeframe=timeframe,
            as_of=current.close_time,
            log_return_1=log_return_1,
            momentum_n=momentum_n,
            realized_vol=realized_vol,
            range_position=range_position,
            distance_to_swing_high=distance_to_swing_high,
            distance_to_swing_low=distance_to_swing_low,
            body_ratio=body_ratio,
            upper_wick_ratio=upper_wick_ratio,
            lower_wick_ratio=lower_wick_ratio,
            volume_zscore=volume_zscore,
            bars_available=len(bars),
        )

    def align(
        self,
        instrument: str,
        as_of: float,
        bars_by_timeframe: dict[Timeframe, list[TimeframeBar]],
    ) -> MultiTimeframeSnapshot:
        """`bars_by_timeframe[tf]` may contain bars up to and beyond `as_of`
        (e.g. a full historical series) — this method is responsible for
        the no-lookahead filtering, so callers don't have to pre-trim.
        """
        snapshot = MultiTimeframeSnapshot(instrument=instrument, as_of=as_of)
        for tf in ALL_TIMEFRAMES:
            all_bars = bars_by_timeframe.get(tf, [])
            closed_bars = [b for b in all_bars if b.close_time <= as_of]
            closed_bars.sort(key=lambda b: b.close_time)
            snapshot.features[tf] = self.compute_feature_vector(instrument, tf, closed_bars)
        return snapshot


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)
