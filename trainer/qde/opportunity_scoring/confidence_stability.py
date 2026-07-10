"""S33 — Confidence-stability sub-score, bridged from the Setup Memory
Engine's own transition/confidence history.

This is the direct integration point with the Setup Memory Engine
(S32): rather than re-deriving "how stable has confidence been", we read
the confidence values already persisted in
``trainer.setup_memory.store.SetupMemoryStore`` and turn their volatility
into a 0-100 quality contribution (stable/rising confidence -> high
score; choppy/collapsing confidence -> low score).
"""

from __future__ import annotations

from trainer.setup_memory.engine import SetupMemoryEngine


def confidence_stability_score(
    engine: SetupMemoryEngine, symbol: str, timeframe: str, lookback: int = 20
) -> float:
    """0-100 score: high when recent confidence readings have been stable
    or steadily improving, low when they have been volatile or trending
    down. Returns a neutral 50.0 when there isn't enough history yet."""
    events = engine.history(symbol, timeframe, limit=lookback)
    confidences = [e["confidence_after"] for e in reversed(events)]  # oldest -> newest
    if len(confidences) < 2:
        return 50.0

    diffs = [confidences[i + 1] - confidences[i] for i in range(len(confidences) - 1)]
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
    volatility = variance**0.5

    # Volatility penalty: 0 volatility -> no penalty, 0.5+ swings -> heavy penalty.
    volatility_penalty = min(100.0, volatility * 200.0)
    # Trend bonus: consistently non-negative diffs (strengthening) nudge the
    # score up; consistently negative diffs (weakening) pull it down.
    trend_bonus = max(-20.0, min(20.0, mean_diff * 100.0))

    score = 70.0 - volatility_penalty + trend_bonus
    return max(0.0, min(100.0, score))
