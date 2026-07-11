"""QDE Brain — regime transition forecasting package.

Three modules that take the QDE beyond regime *classification* toward
regime *forecasting*: learning which sequences of market-feature patterns
historically precede each of six transition types, then estimating
forward-looking probabilities and propagating them into the QDE's
attention, conviction, and risk-appetite signals.

How it differs from existing regime work
-----------------------------------------
``MarketRegimeEngine`` (MIE layer) classifies the *current* regime from
ADX / vol-ratio / RSI / volume — it answers "what is the market doing
right now?".  ``SymbolPersonality.regime_outcome_quality`` accumulates
per-regime outcome quality EMA — it answers "how well did QDE trade in
each regime historically?".

This package answers a third question: "what kind of regime shift is
likely to occur *next*, given the recent sequence of market features?".

Modules
-------
``transition_memory``
    Data models and rolling per-symbol storage.

    * ``TransitionTarget`` — the six forecasted shift directions:
      ``TREND_FORMATION``, ``TREND_EXHAUSTION``, ``VOL_EXPANSION``,
      ``VOL_CONTRACTION``, ``BREAKOUT``, ``RANGE_FORMATION``.
    * ``RegimeSnapshot`` — one tick's observable features
      (ATR, spread, volatility, momentum, vol-rank, trend_valid,
      microstructure quality, volatility quality, current regime label).
    * ``TransitionRecord`` — a confirmed regime change with the N
      pre-transition snapshots and a cached feature vector.
    * ``TransitionMemory`` — rolling snapshot deque + transition list
      with persistence (``to_snapshot`` / ``load_snapshot``).

``transition_predictor``
    Stateless prediction from memory.

    * ``TransitionFeatureVector`` — 10-dimensional: slopes of ATR,
      spread, momentum, volatility, vol-rank; trend-valid ratio;
      microstructure and vol-quality slopes; ATR CV; window fill.
    * ``TransitionProbabilityMap`` — output distribution (6 values
      summing to 1.0) plus confidence, entropy, and sample count.
    * ``TransitionPredictor`` — cosine-similarity matching of the
      current feature vector against all historical pre-transition
      vectors, weighted by recency.  No hardcoded thresholds.

``regime_transition_engine``
    Top-level orchestrator and integration point for the Brain.

    * ``RegimeTransitionEngine`` — five integration points:
      ``on_observation``          → ingest a ``SymbolObservationInput``
      ``record_transition``       → notify of a confirmed regime change
      ``context_for_inference``   → ``regime_*`` brain_context keys
      ``conviction_modifier``     → ±0.20 float
      ``attention_signal``        → 0-1 float
      ``risk_appetite_modifier``  → −0.40..+0.20 float

Design rules
------------
* No hardcoded transition thresholds — signal strengths emerge from
  cosine-similarity between current and historical feature patterns.
* Does not call ``MarketRegimeEngine`` — consumes the already-classified
  ``market_regime`` string from ``SymbolObservationInput``.
* Does not import ``QDEDecision`` or any execution-layer type.
* No duplicate of ``SymbolPersonality.regime_outcome_quality`` or
  ``OpportunitySignals.unusual_behavior`` — this layer adds *sequence
  memory and forward probability* on top of those point-in-time signals.
"""

from __future__ import annotations

# ── Engine ────────────────────────────────────────────────────────────────────
from trainer.qde.brain.regime.regime_transition_engine import (
    RegimeTransitionEngine,
)

# ── Memory / data models ──────────────────────────────────────────────────────
from trainer.qde.brain.regime.transition_memory import (
    RegimeSnapshot,
    TransitionMemory,
    TransitionRecord,
    TransitionTarget,
    infer_target,
    vol_state_rank,
)

# ── Predictor ─────────────────────────────────────────────────────────────────
from trainer.qde.brain.regime.transition_predictor import (
    TransitionFeatureVector,
    TransitionPredictor,
    TransitionProbabilityMap,
)

__all__ = [
    # data models
    "TransitionTarget",
    "RegimeSnapshot",
    "TransitionRecord",
    "TransitionMemory",
    "infer_target",
    "vol_state_rank",
    # predictor
    "TransitionFeatureVector",
    "TransitionProbabilityMap",
    "TransitionPredictor",
    # engine
    "RegimeTransitionEngine",
]
