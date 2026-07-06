"""Phase 22 — Momentum Microstructure.

Measures:
Momentum Acceleration, Deceleration, Exhaustion,
Directional Persistence, Trend Energy, Impulse Decay.
"""

from __future__ import annotations

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


class MomentumMicrostructure:
    """Derives momentum microstructure features from a window of OHLC bars."""

    def analyse_window(
        self,
        bars: List[OHLCBar],
        features: MicrostructureFeatures,
    ) -> MicrostructureFeatures:
        if len(bars) < 3:
            return features

        # Raw close-to-close moves
        moves = [bars[i].close - bars[i - 1].close for i in range(1, len(bars))]
        if not moves:
            return features

        n = len(moves)

        # ── Directional Persistence ───────────────────────────────────────────
        last_dir = 1 if moves[-1] >= 0 else -1
        same_dir = sum(1 for m in moves if (1 if m >= 0 else -1) == last_dir)
        features.directional_persistence = _clamp(same_dir / n)

        # ── Momentum Acceleration / Deceleration ──────────────────────────────
        # Compare recent half vs earlier half
        half = max(n // 2, 1)
        recent_avg_abs = statistics.mean(abs(m) for m in moves[-half:]) if moves[-half:] else 0.0
        earlier_avg_abs = (
            statistics.mean(abs(m) for m in moves[:half]) if moves[:half] else recent_avg_abs
        )

        if earlier_avg_abs > 1e-10:
            ratio = recent_avg_abs / earlier_avg_abs
            if ratio > 1.0:
                features.momentum_acceleration = _clamp((ratio - 1.0) / 1.0)
                features.momentum_deceleration = 0.0
            else:
                features.momentum_deceleration = _clamp((1.0 - ratio) / 1.0)
                features.momentum_acceleration = 0.0
        else:
            features.momentum_acceleration = 0.0
            features.momentum_deceleration = 0.0

        # ── Momentum Exhaustion ───────────────────────────────────────────────
        # Exhaustion: recent bars have large range but small body (wicks dominating)
        recent_bars = bars[-half:]
        avg_wick_ratio = statistics.mean(
            1.0 - abs(b.close - b.open) / b.range if b.range > 1e-10 else 0.5 for b in recent_bars
        )
        features.momentum_exhaustion = _clamp(avg_wick_ratio)

        # ── Trend Energy ──────────────────────────────────────────────────────
        # Directional move from first to last bar / sum of absolute moves
        net_move = abs(bars[-1].close - bars[0].close)
        total_move = sum(abs(m) for m in moves) if moves else 1e-10
        features.trend_energy = _clamp(_safe_div(net_move, total_move, 0.0))

        # ── Impulse Decay ─────────────────────────────────────────────────────
        # How much has the impulse (max move) decayed vs current momentum?
        abs_moves = [abs(m) for m in moves]
        peak_move = max(abs_moves) if abs_moves else 0.0
        latest_move = abs(moves[-1]) if moves else 0.0
        features.impulse_decay = _clamp(1.0 - _safe_div(latest_move, peak_move, 0.0))

        logger.debug(
            f"MomentumMicro: persistence={features.directional_persistence:.3f} "
            f"energy={features.trend_energy:.3f} exhaustion={features.momentum_exhaustion:.3f}"
        )
        return features
