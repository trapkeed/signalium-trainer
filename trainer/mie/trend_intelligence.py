"""Phase 24.9 — Trend Intelligence."""

from __future__ import annotations

import statistics
from typing import List

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_types import TrendState, clamp
from trainer.mie.mie_models import MarketInput, TrendResult


def _mean(vals: List[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _ema(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    result = values[0]
    for v in values[1:]:
        result = v * k + result * (1 - k)
    return result


class TrendIntelligenceEngine:
    """
    Computes TrendState and quality score 0-100 using ADX, EMA crossovers,
    and higher-high / lower-low pattern detection.

    Quality is highest when:
    - ADX is strong and rising
    - Price is well above / below the EMA
    - Clear HH/HL or LL/LH structure exists
    """

    def __init__(
        self,
        adx_strong: float = 40.0,
        adx_moderate: float = 25.0,
        adx_weak: float = 15.0,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self._adx_strong = adx_strong
        self._adx_moderate = adx_moderate
        self._adx_weak = adx_weak
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow

    # ── Public API ─────────────────────────────────────────────────────────

    def analyse(self, inp: MarketInput) -> TrendResult:
        closes = inp.closes or []
        highs = inp.highs or []
        lows = inp.lows or []
        adx = inp.adx

        direction = self._detect_direction(closes, highs, lows, inp.rsi)
        state = self._classify_state(adx, direction)
        quality = self._compute_quality(adx, direction, closes, highs, lows)
        strength = clamp(adx, 0.0, 100.0)

        logger.debug(
            f"[TI] {inp.symbol} adx={adx:.1f} dir={direction} → {state.value} q={quality:.1f}"
        )
        return TrendResult(
            state=state,
            quality=quality,
            direction=direction,
            strength=strength,
            adx_value=adx,
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _detect_direction(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        rsi: float,
    ) -> str:
        if not closes:
            return "NEUTRAL"
        # RSI extremes take priority — strong oversold/overbought signal
        if rsi < 40:
            return "DOWN"
        if rsi > 60:
            return "UP"
        # EMA crossover signal
        if len(closes) >= self._ema_slow:
            ema_f = _ema(closes[-self._ema_fast :], self._ema_fast)
            ema_s = _ema(closes[-self._ema_slow :], self._ema_slow)
            if ema_f > ema_s * 1.001:
                return "UP"
            if ema_f < ema_s * 0.999:
                return "DOWN"
        # RSI moderate fallback
        if rsi > 55:
            return "UP"
        if rsi < 45:
            return "DOWN"
        return "NEUTRAL"

    def _classify_state(self, adx: float, direction: str) -> TrendState:
        if adx >= self._adx_strong:
            return TrendState.STRONG
        if adx >= self._adx_moderate:
            return TrendState.MODERATE
        if adx >= self._adx_weak:
            return TrendState.WEAK
        if direction == "NEUTRAL":
            return TrendState.SIDEWAYS
        return TrendState.REVERSING

    def _has_hh_hl(self, highs: List[float], lows: List[float], n: int = 5) -> bool:
        if len(highs) < n or len(lows) < n:
            return False
        hs = highs[-n:]
        ls = lows[-n:]
        hh = all(hs[i] >= hs[i - 1] for i in range(1, n))
        hl = all(ls[i] >= ls[i - 1] for i in range(1, n))
        return hh and hl

    def _has_ll_lh(self, highs: List[float], lows: List[float], n: int = 5) -> bool:
        if len(highs) < n or len(lows) < n:
            return False
        hs = highs[-n:]
        ls = lows[-n:]
        ll = all(ls[i] <= ls[i - 1] for i in range(1, n))
        lh = all(hs[i] <= hs[i - 1] for i in range(1, n))
        return ll and lh

    def _compute_quality(
        self,
        adx: float,
        direction: str,
        closes: List[float],
        highs: List[float],
        lows: List[float],
    ) -> float:
        score = 0.0
        # ADX contribution (0-60)
        score += clamp(adx / self._adx_strong * 60.0, 0.0, 60.0)
        # Structure confirmation (0-30)
        if direction == "UP" and self._has_hh_hl(highs, lows):
            score += 30.0
        elif direction == "DOWN" and self._has_ll_lh(highs, lows):
            score += 30.0
        elif direction != "NEUTRAL":
            score += 10.0
        # EMA gap bonus (0-10)
        if len(closes) >= self._ema_slow:
            ema_f = _ema(closes[-self._ema_fast :], self._ema_fast)
            ema_s = _ema(closes[-self._ema_slow :], self._ema_slow)
            gap = abs(ema_f - ema_s) / max(ema_s, 1e-9)
            score += clamp(gap * 1000.0, 0.0, 10.0)
        return clamp(score, 0.0, 100.0)
