"""Phase 24.9 — Momentum Intelligence."""

from __future__ import annotations

import statistics
from typing import Dict

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import MarketInput, MomentumResult
from trainer.mie.mie_types import MomentumState, clamp


def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _stdev(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


class MomentumIntelligenceEngine:
    """
    Computes MomentumState and quality score 0-100.

    Strength  = weighted sum of recent returns (exponential decay, most-recent highest).
    Acceleration = rate of change of strength (second derivative).
    RSI signal normalised to [-1, +1].

    Quality scoring:
    - ACCELERATING or STEADY  → high quality
    - DECELERATING            → medium quality
    - REVERSING or FLAT       → low quality
    """

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 20,
        rsi_ob: float = 70.0,
        rsi_os: float = 30.0,
    ) -> None:
        self._fast = fast_period
        self._slow = slow_period
        self._rsi_ob = rsi_ob
        self._rsi_os = rsi_os

    # ── Public API ─────────────────────────────────────────────────────────

    def analyse(self, inp: MarketInput) -> MomentumResult:
        returns = inp.returns or []
        strength = self._momentum_strength(returns)
        acceleration = self._momentum_acceleration(returns)
        rsi_signal = (inp.rsi - 50.0) / 50.0  # -1 to +1

        state = self._classify_state(strength, acceleration, inp.rsi)
        quality = self._compute_quality(state, strength, inp.rsi)

        logger.debug(
            f"[MI] {inp.symbol} str={strength:.4f} acc={acceleration:.4f} → {state.value} q={quality:.1f}"
        )
        return MomentumResult(
            state=state,
            quality=quality,
            strength=strength,
            acceleration=acceleration,
            rsi_signal=inp.rsi,
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _momentum_strength(self, returns: list[float]) -> float:
        """Exponentially-weighted return over fast period."""
        window = returns[-self._fast :] if len(returns) >= self._fast else returns
        if not window:
            return 0.0
        weights = [2**i for i in range(len(window))]
        total_w = sum(weights)
        return sum(r * w for r, w in zip(window, weights)) / total_w

    def _momentum_acceleration(self, returns: list[float]) -> float:
        """Difference of two fast-period momentum windows (second derivative)."""
        if len(returns) < self._fast * 2:
            return 0.0
        recent = self._momentum_strength(returns[-self._fast :])
        prior = self._momentum_strength(returns[-self._fast * 2 : -self._fast])
        return recent - prior

    def _classify_state(
        self,
        strength: float,
        acceleration: float,
        rsi: float,
    ) -> MomentumState:
        abs_str = abs(strength)
        if abs_str < 0.0001:
            return MomentumState.FLAT
        # Reversal: momentum direction opposite to RSI signal
        if (strength > 0 and rsi < 40) or (strength < 0 and rsi > 60):
            return MomentumState.REVERSING
        if acceleration > 0.0001:
            return MomentumState.ACCELERATING
        if acceleration < -0.0001 and abs_str > 0.0005:
            return MomentumState.DECELERATING
        return MomentumState.STEADY

    def _compute_quality(
        self,
        state: MomentumState,
        strength: float,
        rsi: float,
    ) -> float:
        base: Dict[MomentumState, float] = {  # type: ignore[type-arg]
            MomentumState.ACCELERATING: 85.0,
            MomentumState.STEADY: 70.0,
            MomentumState.DECELERATING: 45.0,
            MomentumState.REVERSING: 25.0,
            MomentumState.FLAT: 20.0,
        }
        score = base.get(state, 50.0)
        # Overbought/oversold RSI reduces quality (mean-reversion risk)
        if rsi > self._rsi_ob or rsi < self._rsi_os:
            score -= 15.0
        # Strong strength boosts quality slightly
        score += min(abs(strength) * 500.0, 10.0)
        return clamp(score, 0.0, 100.0)
