"""Phase 24.9 — Behavior Similarity Engine."""

from __future__ import annotations

import math
import statistics

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import BehaviorSimilarityResult, MarketInput
from trainer.mie.mie_types import clamp

# Feature keys used when comparing current market to historical records.
# Each historical record in `inp.historical_similar` should carry these keys.
_FEATURE_KEYS = [
    "adx",
    "rsi",
    "vol_ratio",
    "spread_frac",
    "momentum_strength",
    "trend_quality",
    "session_quality",
]

# Feature weights for weighted Euclidean distance
_FEATURE_WEIGHTS: dict[str, float] = {
    "adx": 0.20,
    "rsi": 0.15,
    "vol_ratio": 0.20,
    "spread_frac": 0.15,
    "momentum_strength": 0.15,
    "trend_quality": 0.10,
    "session_quality": 0.05,
}

# Absolute normalisation scales (fallback when historical stdev = 0).
# These represent domain-knowledge estimates of typical feature spread.
_ABSOLUTE_SCALES: dict[str, float] = {
    "adx": 25.0,
    "rsi": 20.0,
    "vol_ratio": 1.0,
    "spread_frac": 0.5,
    "momentum_strength": 1.0,
    "trend_quality": 30.0,
    "session_quality": 25.0,
}


def _extract_current(inp: MarketInput) -> dict[str, float]:
    returns = inp.returns or []
    curr_vol = statistics.stdev(returns[-20:]) if len(returns) >= 2 else 0.0
    base_vol = statistics.stdev(returns) if len(returns) >= 10 else max(curr_vol, 1e-9)
    vol_ratio = curr_vol / max(base_vol, 1e-9)
    spread_frac = inp.spread / max(inp.atr, 1e-9)
    # Simple proxy for momentum strength from returns
    recent = returns[-5:] if len(returns) >= 5 else returns
    mom = sum(recent) / max(len(recent), 1)
    return {
        "adx": inp.adx,
        "rsi": inp.rsi,
        "vol_ratio": vol_ratio,
        "spread_frac": spread_frac,
        "momentum_strength": mom * 1000.0,
        "trend_quality": clamp(inp.adx / 50.0 * 100.0, 0.0, 100.0),
        "session_quality": 50.0,  # placeholder; caller may override
    }


def _weighted_distance(
    a: dict[str, float],
    b: dict[str, float],
    scales: dict[str, float],
) -> float:
    """Weighted Euclidean distance between two feature dicts."""
    dist = 0.0
    for key, w in _FEATURE_WEIGHTS.items():
        va = a.get(key, 0.0)
        vb = b.get(key, 0.0)
        sc = scales.get(key, 1.0)
        dist += w * ((va - vb) / max(sc, 1e-9)) ** 2
    return math.sqrt(dist)


def _normalise_scales(
    records: list[dict[str, float]],
) -> dict[str, float]:
    """Return per-feature normalisation scales.

    Uses the max of (historical stdev, absolute domain-knowledge scale) to
    ensure non-zero scales even when all records have identical feature values.
    """
    scales: dict[str, float] = {}
    for key in _FEATURE_KEYS:
        vals = [r.get(key, 0.0) for r in records]
        hist_std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        abs_scale = _ABSOLUTE_SCALES.get(key, 1.0)
        scales[key] = max(hist_std, abs_scale, 1e-9)
    return scales


def _similarity_from_distance(dist: float, decay: float = 3.0) -> float:
    """Convert distance → similarity score 0-100 using exponential decay."""
    return 100.0 * math.exp(-decay * dist)


class BehaviorSimilarityEngine:
    """
    Compares the current market feature vector against a library of
    historical market states, returning a similarity score 0-100 and the
    historical win rate on the most-similar past conditions.

    Historical records are supplied via `inp.historical_similar`.
    Each record is a dict with:
        - Feature keys (adx, rsi, vol_ratio, …)
        - "outcome": float  (+1.0 win, -1.0 loss)   [optional]
        - "label"  : str    [optional, for logging]
    """

    def __init__(self, min_matches: int = 3, decay: float = 2.0) -> None:
        self._min_matches = min_matches
        self._decay = decay

    # ── Public API ─────────────────────────────────────────────────────────

    def analyse(self, inp: MarketInput) -> BehaviorSimilarityResult:
        records = inp.historical_similar or []
        current = _extract_current(inp)

        if not records:
            return BehaviorSimilarityResult(
                similarity_score=0.0,
                historical_success=inp.historical_win_rate,
                match_count=0,
                features_used=_FEATURE_KEYS,
            )

        scales = _normalise_scales([{k: r.get(k, 0.0) for k in _FEATURE_KEYS} for r in records])

        scored: list[tuple[float, dict]] = []
        for record in records:
            feat = {k: float(record.get(k, 0.0)) for k in _FEATURE_KEYS}
            dist = _weighted_distance(current, feat, scales)
            sim = _similarity_from_distance(dist, self._decay)
            scored.append((sim, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(self._min_matches, 3)]

        best_score = top[0][0] if top else 0.0
        avg_score = statistics.mean(s for s, _ in top) if top else 0.0

        outcomes = [float(r.get("outcome", 0.0)) for _, r in top if "outcome" in r]
        if outcomes:
            hist_success = clamp((statistics.mean(outcomes) + 1.0) / 2.0, 0.0, 1.0)
        else:
            hist_success = inp.historical_win_rate

        logger.debug(
            f"[BSE] {inp.symbol} matches={len(scored)} best={best_score:.1f} "
            f"hist_wr={hist_success:.3f}"
        )
        return BehaviorSimilarityResult(
            similarity_score=clamp(avg_score, 0.0, 100.0),
            historical_success=hist_success,
            match_count=len(scored),
            best_match_score=clamp(best_score, 0.0, 100.0),
            features_used=_FEATURE_KEYS,
        )
