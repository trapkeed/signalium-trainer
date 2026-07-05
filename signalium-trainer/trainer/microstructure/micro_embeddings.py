"""Phase 22 — Microstructure Embeddings.

Generates dense 32-dimensional embedding vectors representing:
- Price Action State (8 dims)
- Liquidity State (6 dims)
- Momentum State (6 dims)
- Volatility State (6 dims)
- Efficiency State (6 dims)

Embeddings are fed into forecasting models as an enriched representation
of the current microstructure environment.
"""

from __future__ import annotations

import math
import time
from typing import List

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.microstructure.micro_models import MicrostructureFeatures, MicrostructureEmbedding
from trainer.microstructure.micro_types import EMBEDDING_DIMS, EMBEDDING_TOTAL


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _normalise(values: List[float]) -> List[float]:
    """Clamp all values to [0, 1]."""
    return [_clamp(v) for v in values]


def _sigmoid(x: float) -> float:
    """Sigmoid activation for unbounded inputs."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class MicrostructureEmbedder:
    """Encodes microstructure features into dense embedding vectors."""

    def embed(
        self,
        features:   MicrostructureFeatures,
        instrument: str = "",
        timeframe:  str = "H1",
    ) -> MicrostructureEmbedding:
        """Encode features into a 32-dimensional embedding."""

        # ── Price Action State (8 dims) ───────────────────────────────────────
        pa = _normalise([
            features.body_ratio,
            features.upper_wick_ratio,
            features.lower_wick_ratio,
            features.close_position,
            features.open_position,
            features.candle_efficiency,
            features.impulse_strength,
            features.continuation_probability,
        ])

        # ── Liquidity State (6 dims) ──────────────────────────────────────────
        liq = _normalise([
            features.spread_proxy,
            features.gap_frequency,
            features.breakout_strength,
            features.reversal_strength,
            features.liquidity_vacuum_prob,
            features.large_candle_frequency,
        ])

        # ── Momentum State (6 dims) ───────────────────────────────────────────
        # directional_efficiency is -1 to +1, map to 0–1
        dir_eff_norm = (features.directional_efficiency + 1.0) / 2.0
        mom = _normalise([
            features.trend_energy,
            features.momentum_acceleration,
            features.directional_persistence,
            features.impulse_decay,
            features.momentum_exhaustion,
            dir_eff_norm,
        ])

        # ── Volatility State (6 dims) ─────────────────────────────────────────
        vol = _normalise([
            features.atr_expansion,
            features.atr_compression,
            features.volatility_clustering,
            features.volatility_persistence,
            features.noise_ratio,
            features.volatility_shock,
        ])

        # ── Efficiency State (6 dims) ─────────────────────────────────────────
        # fractal_dimension is 1–2; map to 0–1 (1=trend, 2=random)
        fd_norm = (features.fractal_dimension - 1.0)   # 0–1
        eff = _normalise([
            features.trend_efficiency,
            features.mean_reversion_probability,
            features.randomness_index,
            features.directional_stability,
            1.0 - features.noise_level,   # high efficiency = low noise
            1.0 - fd_norm,                # trending (D~1) = high efficiency
        ])

        # Full embedding = concatenation
        embedding = pa + liq + mom + vol + eff

        assert len(embedding) == EMBEDDING_TOTAL, (
            f"Embedding dim {len(embedding)} != expected {EMBEDDING_TOTAL}"
        )

        logger.debug(
            f"MicrostructureEmbedder: generated {EMBEDDING_TOTAL}-dim embedding "
            f"for {instrument} {timeframe}"
        )

        return MicrostructureEmbedding(
            timestamp          = features.timestamp,
            instrument         = instrument or features.instrument,
            timeframe          = timeframe,
            embedding          = [round(v, 5) for v in embedding],
            price_action_state = [round(v, 5) for v in pa],
            liquidity_state    = [round(v, 5) for v in liq],
            momentum_state     = [round(v, 5) for v in mom],
            volatility_state   = [round(v, 5) for v in vol],
            efficiency_state   = [round(v, 5) for v in eff],
        )

    def batch_embed(
        self,
        feature_list: List[MicrostructureFeatures],
        instrument:   str = "",
        timeframe:    str = "H1",
    ) -> List[MicrostructureEmbedding]:
        return [self.embed(f, instrument, timeframe) for f in feature_list]

    @staticmethod
    def embedding_dim() -> int:
        return EMBEDDING_TOTAL

    @staticmethod
    def sub_dims() -> dict:
        return dict(EMBEDDING_DIMS)
