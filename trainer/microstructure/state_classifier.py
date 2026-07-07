"""Phase 22 — Market State Classifier.

Classifies market into one of 12 states:
TRENDING, STRONG_TREND, WEAK_TREND, RANGE, COMPRESSION, EXPANSION,
BREAKOUT, FAKE_BREAKOUT, REVERSAL, HIGH_VOLATILITY, LOW_VOLATILITY, TRANSITION.
"""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.microstructure.micro_models import (
    MarketStateRecord,
    MicrostructureFeatures,
    MicrostructureScore,
)
from trainer.microstructure.micro_types import (
    LiquidityState,
    MarketState,
    MomentumState,
    VolatilityState,
    trend_strength_from_score,
)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class MarketStateClassifier:
    """Classifies market microstructure state from computed features and scores."""

    # Classification thresholds
    HIGH_VOL_THRESHOLD = 0.75
    LOW_VOL_THRESHOLD = 0.25
    STRONG_TREND_THRESH = 0.75
    TREND_THRESH = 0.55
    WEAK_TREND_THRESH = 0.35
    COMPRESSION_THRESH = 0.65
    EXPANSION_THRESH = 0.65
    BREAKOUT_THRESH = 0.70
    REVERSAL_THRESH = 0.60

    def classify(
        self,
        features: MicrostructureFeatures,
        score: MicrostructureScore,
    ) -> MarketStateRecord:
        """Classify market state from features and scores."""

        state, confidence, reasoning = self._determine_state(features, score)
        momentum_state = self._momentum_state(features)
        volatility_state = self._volatility_state(features, score)
        liquidity_state = self._liquidity_state(features)
        trend_strength = trend_strength_from_score(score.trend_score)

        record = MarketStateRecord(
            state=state,
            confidence=round(confidence, 4),
            momentum_state=momentum_state,
            volatility_state=volatility_state,
            liquidity_state=liquidity_state,
            trend_strength=trend_strength,
            score=score,
            reasoning=reasoning,
        )

        logger.debug(
            f"StateClassifier: {state.value} confidence={confidence:.3f} "
            f"trend={trend_strength.value}"
        )
        return record

    def _determine_state(
        self,
        features: MicrostructureFeatures,
        score: MicrostructureScore,
    ):
        """Returns (MarketState, confidence, reasoning)."""
        reasoning: list[str] = []

        ts = score.trend_score
        vs = score.volatility_score
        cs = score.compression_score
        es = score.expansion_score
        ls = score.liquidity_score
        ms = score.momentum_score

        # Priority 1: HIGH VOLATILITY
        if vs >= self.HIGH_VOL_THRESHOLD:
            reasoning.append(
                f"Volatility score {vs:.2f} exceeds HIGH threshold {self.HIGH_VOL_THRESHOLD}"
            )
            return (
                MarketState.HIGH_VOLATILITY,
                _clamp((vs - self.HIGH_VOL_THRESHOLD) / 0.25 + 0.70),
                reasoning,
            )

        # Priority 2: LOW VOLATILITY
        if vs <= self.LOW_VOL_THRESHOLD and cs >= self.COMPRESSION_THRESH:
            reasoning.append(
                f"Volatility score {vs:.2f} low, compression {cs:.2f} high — LOW_VOLATILITY"
            )
            return MarketState.LOW_VOLATILITY, _clamp(1.0 - vs), reasoning

        # Priority 3: STRONG TREND
        if ts >= self.STRONG_TREND_THRESH and features.directional_persistence >= 0.70:
            reasoning.append(
                f"Trend score {ts:.2f} and persistence {features.directional_persistence:.2f} — STRONG_TREND"
            )
            return (
                MarketState.STRONG_TREND,
                _clamp(ts * 0.7 + features.directional_persistence * 0.3),
                reasoning,
            )

        # Priority 4: BREAKOUT
        if features.breakout_strength >= self.BREAKOUT_THRESH and es >= self.EXPANSION_THRESH:
            # Check if fake breakout: reversal follows expansion
            if (
                features.reversal_strength >= self.REVERSAL_THRESH
                and features.directional_persistence < 0.45
            ):
                reasoning.append(
                    f"Breakout ({features.breakout_strength:.2f}) followed by reversal ({features.reversal_strength:.2f}) — FAKE_BREAKOUT"
                )
                return MarketState.FAKE_BREAKOUT, _clamp(features.reversal_strength), reasoning
            reasoning.append(
                f"Breakout strength {features.breakout_strength:.2f} + expansion {es:.2f} — BREAKOUT"
            )
            return (
                MarketState.BREAKOUT,
                _clamp(features.breakout_strength * 0.6 + es * 0.4),
                reasoning,
            )

        # Priority 5: REVERSAL
        if (
            features.reversal_strength >= self.REVERSAL_THRESH
            and features.momentum_exhaustion >= 0.65
        ):
            reasoning.append(
                f"Reversal strength {features.reversal_strength:.2f} + exhaustion {features.momentum_exhaustion:.2f} — REVERSAL"
            )
            return (
                MarketState.REVERSAL,
                _clamp((features.reversal_strength + features.momentum_exhaustion) / 2.0),
                reasoning,
            )

        # Priority 6: TRENDING
        if ts >= self.TREND_THRESH:
            reasoning.append(f"Trend score {ts:.2f} — TRENDING")
            return MarketState.TRENDING, _clamp(ts), reasoning

        # Priority 7: WEAK TREND
        if ts >= self.WEAK_TREND_THRESH:
            reasoning.append(f"Trend score {ts:.2f} — WEAK_TREND")
            return MarketState.WEAK_TREND, _clamp(ts), reasoning

        # Priority 8: COMPRESSION
        if cs >= self.COMPRESSION_THRESH:
            reasoning.append(f"Compression score {cs:.2f} — COMPRESSION")
            return MarketState.COMPRESSION, _clamp(cs), reasoning

        # Priority 9: EXPANSION
        if es >= self.EXPANSION_THRESH:
            reasoning.append(f"Expansion score {es:.2f} — EXPANSION")
            return MarketState.EXPANSION, _clamp(es), reasoning

        # Priority 10: RANGE
        if ts < self.WEAK_TREND_THRESH and vs < 0.50:
            reasoning.append(f"Low trend {ts:.2f} + low volatility {vs:.2f} — RANGE")
            return MarketState.RANGE, _clamp(1.0 - ts + (0.50 - vs)), reasoning

        # Default: TRANSITION
        reasoning.append("Mixed signals — TRANSITION state")
        return MarketState.TRANSITION, 0.50, reasoning

    def _momentum_state(self, features: MicrostructureFeatures) -> MomentumState:
        if features.momentum_exhaustion >= 0.70:
            return MomentumState.EXHAUSTED
        if features.momentum_acceleration >= 0.50:
            return MomentumState.ACCELERATING
        if features.momentum_deceleration >= 0.50:
            return MomentumState.DECELERATING
        if features.reversal_strength >= 0.60:
            return MomentumState.REVERSING
        return MomentumState.NEUTRAL

    def _volatility_state(
        self, features: MicrostructureFeatures, score: MicrostructureScore
    ) -> VolatilityState:
        if features.volatility_shock >= 0.80:
            return VolatilityState.SHOCK
        if features.atr_expansion >= 0.60:
            return VolatilityState.EXPANDING
        if features.atr_compression >= 0.60:
            return VolatilityState.CONTRACTING
        if features.volatility_clustering >= 0.60:
            return VolatilityState.CLUSTERING
        return VolatilityState.NORMAL

    def _liquidity_state(self, features: MicrostructureFeatures) -> LiquidityState:
        if features.liquidity_vacuum_prob >= 0.70:
            return LiquidityState.VACUUM
        if features.spread_proxy <= 0.20 and features.large_candle_frequency <= 0.15:
            return LiquidityState.DEEP
        if features.spread_proxy >= 0.55:
            return LiquidityState.THIN
        return LiquidityState.NORMAL
