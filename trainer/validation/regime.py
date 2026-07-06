"""Phase 15 — Market Regime Classification.

Classifies market regimes from OHLCV-like features without any look-ahead.
All classification uses only information available at the start of each bar.

Regimes classified:
  TRENDING_STRONG, TRENDING_WEAK
  RANGING
  BREAKOUT, PULLBACK
  HIGH_VOLATILITY, LOW_VOLATILITY
  COMPRESSION, EXPANSION
  ACCUMULATION, DISTRIBUTION
  NEWS_PERIOD (when external label supplied)
  SESSION_OPEN, SESSION_CLOSE
  WEEKEND_GAP

Usage::

    clf = MarketRegimeClassifier()
    regimes = clf.classify(ohlcv_tensor, timestamps=ts)
    dist = RegimeDistribution.from_labels(regimes)
    print(dist.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch


class MarketRegime(str, Enum):
    """All supported market regimes."""

    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    NEWS_PERIOD = "news_period"
    SESSION_OPEN = "session_open"
    SESSION_CLOSE = "session_close"
    WEEKEND_GAP = "weekend_gap"
    UNKNOWN = "unknown"


@dataclass
class RegimeDistribution:
    """Distribution of regime labels across a dataset.

    Attributes:
        counts: Dict mapping regime name → count.
        total: Total number of classified bars.
    """

    counts: dict[str, int]
    total: int

    @classmethod
    def from_labels(cls, labels: list[MarketRegime]) -> "RegimeDistribution":
        counts: dict[str, int] = {}
        for lbl in labels:
            counts[lbl.value] = counts.get(lbl.value, 0) + 1
        return cls(counts=counts, total=len(labels))

    def fraction(self, regime: str) -> float:
        if self.total == 0:
            return 0.0
        return self.counts.get(regime, 0) / self.total

    def most_common(self, k: int = 5) -> list[tuple[str, float]]:
        return sorted(
            [(r, self.counts[r] / self.total) for r in self.counts],
            key=lambda x: x[1],
            reverse=True,
        )[:k]

    def summary(self) -> str:
        lines = [f"Regime Distribution ({self.total} bars):"]
        for regime, pct in self.most_common(k=len(self.counts)):
            bar = "#" * max(1, int(pct * 30))
            lines.append(f"  {regime:<24s} {bar} ({pct:.1%})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "total": self.total,
            "fractions": {k: round(v / self.total, 4) for k, v in self.counts.items()},
        }


class MarketRegimeClassifier:
    """Classifies market regimes from OHLCV data using rule-based heuristics.

    No machine learning — all rules are deterministic and interpretable.
    Designed to be fast and run at validation time without GPU.

    Args:
        trend_window: Lookback for trend direction (bars).
        volatility_window: Lookback for ATR-based volatility (bars).
        compression_threshold: Fraction of average range to detect compression.
        strong_trend_adx_proxy: Slope magnitude above which trend is "strong".
        session_open_bars: Bars after session open to flag as SESSION_OPEN.
        session_close_bars: Bars before session close to flag as SESSION_CLOSE.

    Input format:
        ``ohlcv_tensor``: ``(N, 5)`` or ``(N, ≥5)`` — [open, high, low, close, volume]
        ``timestamps``: Optional ``(N,)`` Unix timestamps (seconds).
        ``news_mask``: Optional ``(N,)`` boolean mask for known news periods.
    """

    def __init__(
        self,
        trend_window: int = 20,
        volatility_window: int = 14,
        compression_threshold: float = 0.40,
        strong_trend_adx_proxy: float = 0.02,
        session_open_bars: int = 3,
        session_close_bars: int = 3,
    ) -> None:
        self._trend_w = trend_window
        self._vol_w = volatility_window
        self._comp_thresh = compression_threshold
        self._strong_thresh = strong_trend_adx_proxy
        self._open_bars = session_open_bars
        self._close_bars = session_close_bars

    def classify(
        self,
        ohlcv: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        news_mask: Optional[torch.Tensor] = None,
    ) -> list[MarketRegime]:
        """Classify every bar into a market regime.

        Args:
            ohlcv: ``(N, ≥5)`` OHLCV tensor.
            timestamps: Optional ``(N,)`` Unix timestamps (seconds).
            news_mask: Optional ``(N,)`` boolean mask for news periods.

        Returns:
            List of ``MarketRegime`` labels, one per bar.
        """
        n = ohlcv.shape[0]
        if n < 2:
            return [MarketRegime.UNKNOWN] * n

        o = ohlcv[:, 0].cpu().float().numpy()
        h = ohlcv[:, 1].cpu().float().numpy()
        l = ohlcv[:, 2].cpu().float().numpy()
        c = ohlcv[:, 3].cpu().float().numpy()
        v = ohlcv[:, 4].cpu().float().numpy() if ohlcv.shape[1] > 4 else np.ones(n)

        ts = timestamps.cpu().numpy() if timestamps is not None else None
        news = news_mask.cpu().numpy().astype(bool) if news_mask is not None else None

        # Derived series
        true_range = self._true_range(h, l, c)
        atr = self._rolling_mean(true_range, self._vol_w)
        avg_atr = (
            float(np.mean(atr[self._vol_w :])) if n > self._vol_w else float(np.mean(true_range))
        )
        slope = self._price_slope(c, self._trend_w)
        avg_range = self._rolling_mean(h - l, self._trend_w)

        labels: list[MarketRegime] = []
        for i in range(n):
            lbl = self._classify_bar(
                i,
                n,
                c,
                h,
                l,
                v,
                true_range,
                atr,
                avg_atr,
                slope,
                avg_range,
                ts,
                news,
            )
            labels.append(lbl)
        return labels

    def _classify_bar(
        self,
        i: int,
        n: int,
        c,
        h,
        l,
        v,
        true_range,
        atr,
        avg_atr,
        slope,
        avg_range,
        ts,
        news,
    ) -> MarketRegime:
        # News period (highest priority)
        if news is not None and news[i]:
            return MarketRegime.NEWS_PERIOD

        # Session open / close (requires timestamps)
        if ts is not None:
            # Seconds in day
            #   session_sec = int(ts[i]) % 86400
            bar_in_day = int(ts[i]) % 86400
            if bar_in_day < self._open_bars * 3600:
                return MarketRegime.SESSION_OPEN
            if bar_in_day > 86400 - self._close_bars * 3600:
                return MarketRegime.SESSION_CLOSE

        # Weekend gap
        if ts is not None and i > 0:
            gap = ts[i] - ts[i - 1]
            if gap > 43200 and (abs(c[i] - c[i - 1]) / max(atr[i], 1e-9)) > 1.0:
                return MarketRegime.WEEKEND_GAP

        # Volatility regimes (use ATR relative to recent average)
        current_atr = float(atr[i]) if i >= self._vol_w else float(true_range[i])
        if avg_atr > 0:
            rel_vol = current_atr / avg_atr
        else:
            rel_vol = 1.0

        # Compression / expansion
        if i >= self._trend_w:
            recent_range = float(avg_range[i])
            overall_avg_range = float(np.mean(avg_range[: i + 1]))
            if overall_avg_range > 0:
                range_ratio = recent_range / overall_avg_range
                if range_ratio < self._comp_thresh:
                    return MarketRegime.COMPRESSION
                if range_ratio > (1.0 / self._comp_thresh):
                    return MarketRegime.EXPANSION

        if rel_vol > 1.8:
            return MarketRegime.HIGH_VOLATILITY
        if rel_vol < 0.5:
            return MarketRegime.LOW_VOLATILITY

        # Trend regimes
        s = float(slope[i]) if i >= self._trend_w else 0.0
        abs_slope = abs(s)

        if abs_slope > self._strong_thresh:
            # Check for breakout vs pullback vs trend
            if i > 0:
                prev_high = float(np.max(h[max(0, i - self._trend_w) : i]))
                prev_low = float(np.min(l[max(0, i - self._trend_w) : i]))
                if c[i] > prev_high:
                    return MarketRegime.BREAKOUT
                if c[i] < prev_low:
                    return MarketRegime.BREAKOUT
            return (
                MarketRegime.TRENDING_STRONG
                if abs_slope > self._strong_thresh * 1.5
                else MarketRegime.TRENDING_WEAK
            )

        # Accumulation / distribution (volume-based)
        if i > 0 and v is not None:
            avg_vol = float(np.mean(v[max(0, i - self._trend_w) : i + 1]))
            if avg_vol > 0:
                vol_ratio = float(v[i]) / avg_vol
                price_change = c[i] - c[i - 1] if i > 0 else 0
                if vol_ratio > 1.5 and price_change > 0:
                    return MarketRegime.ACCUMULATION
                if vol_ratio > 1.5 and price_change < 0:
                    return MarketRegime.DISTRIBUTION

        # Pullback detection (within established trend)
        if i >= self._trend_w and abs_slope > 0.005:
            recent_slope = float(slope[max(0, i - 3)])
            if s * recent_slope < 0:  # direction reversal
                return MarketRegime.PULLBACK

        # Default: ranging
        return MarketRegime.RANGING

    @staticmethod
    def _true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
        tr = np.zeros(len(h))
        tr[0] = h[0] - l[0]
        for i in range(1, len(h)):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        return tr

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result[i] = arr[start : i + 1].mean()
        return result

    @staticmethod
    def _price_slope(c: np.ndarray, window: int) -> np.ndarray:
        """Normalised price slope (regression coefficient / mean price)."""
        slopes = np.zeros(len(c))
        for i in range(window, len(c)):
            y = c[i - window : i]
            x = np.arange(window, dtype=float)
            if y.mean() > 0:
                slopes[i] = float(np.polyfit(x, y, 1)[0]) / y.mean()
        return slopes
