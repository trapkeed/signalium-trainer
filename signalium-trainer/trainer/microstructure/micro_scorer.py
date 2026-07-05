"""Phase 22 — Microstructure Scorer.

Generates:
Trend Score, Liquidity Score, Momentum Score, Efficiency Score,
Volatility Score, Compression Score, Expansion Score,
Overall Microstructure Score.
"""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.microstructure.micro_models import MicrostructureFeatures, MicrostructureScore


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class MicrostructureScorer:
    """Computes composite microstructure scores from raw features."""

    # Overall score weights
    _WEIGHTS = {
        "trend":       0.25,
        "momentum":    0.20,
        "efficiency":  0.20,
        "volatility":  0.15,
        "liquidity":   0.10,
        "compression": 0.05,
        "expansion":   0.05,
    }

    def score(self, features: MicrostructureFeatures) -> MicrostructureScore:
        s = MicrostructureScore()

        # ── Trend Score ───────────────────────────────────────────────────────
        # High trend score = strong, clean directional movement
        s.trend_score = _clamp(
            features.trend_energy         * 0.35 +
            features.directional_persistence * 0.30 +
            features.candle_efficiency    * 0.20 +
            abs(features.directional_efficiency) * 0.15
        )

        # ── Liquidity Score ───────────────────────────────────────────────────
        # High score = deep liquid market
        s.liquidity_score = _clamp(
            (1.0 - features.spread_proxy)         * 0.30 +
            (1.0 - features.gap_frequency)         * 0.25 +
            (1.0 - features.large_candle_frequency) * 0.20 +
            (1.0 - features.liquidity_vacuum_prob) * 0.25
        )

        # ── Momentum Score ────────────────────────────────────────────────────
        # High score = strong, accelerating momentum
        s.momentum_score = _clamp(
            features.trend_energy             * 0.30 +
            features.momentum_acceleration    * 0.25 +
            features.directional_persistence  * 0.25 +
            (1.0 - features.momentum_exhaustion) * 0.20
        )

        # ── Efficiency Score ──────────────────────────────────────────────────
        # High score = efficient, non-noisy market
        s.efficiency_score = _clamp(
            features.trend_efficiency         * 0.35 +
            features.directional_stability    * 0.25 +
            (1.0 - features.noise_level)      * 0.25 +
            (1.0 - features.randomness_index) * 0.15
        )

        # ── Volatility Score ──────────────────────────────────────────────────
        # High score = elevated, expanding volatility environment
        s.volatility_score = _clamp(
            features.atr_expansion          * 0.30 +
            features.volatility_clustering   * 0.20 +
            features.volatility_shock        * 0.25 +
            features.noise_ratio             * 0.15 +
            features.range_expansion         * 0.10
        )

        # ── Compression Score ─────────────────────────────────────────────────
        # High score = compressed, coiling market
        s.compression_score = _clamp(
            features.atr_compression         * 0.35 +
            features.compression_zone_score  * 0.30 +
            features.range_compression       * 0.20 +
            (1.0 - features.trend_energy)    * 0.15
        )

        # ── Expansion Score ───────────────────────────────────────────────────
        # High score = expanding, impulsive market
        s.expansion_score = _clamp(
            features.atr_expansion       * 0.30 +
            features.range_expansion     * 0.25 +
            features.breakout_strength   * 0.25 +
            features.impulse_strength    * 0.20
        )

        # ── Overall Microstructure Score ──────────────────────────────────────
        s.overall_score = _clamp(
            s.trend_score       * self._WEIGHTS["trend"] +
            s.momentum_score    * self._WEIGHTS["momentum"] +
            s.efficiency_score  * self._WEIGHTS["efficiency"] +
            s.volatility_score  * self._WEIGHTS["volatility"] +
            s.liquidity_score   * self._WEIGHTS["liquidity"] +
            s.compression_score * self._WEIGHTS["compression"] +
            s.expansion_score   * self._WEIGHTS["expansion"]
        )

        logger.debug(
            f"MicrostructureScorer: trend={s.trend_score:.3f} "
            f"momentum={s.momentum_score:.3f} "
            f"efficiency={s.efficiency_score:.3f} "
            f"overall={s.overall_score:.3f}"
        )
        return s

    def weights(self) -> dict:
        return dict(self._WEIGHTS)
