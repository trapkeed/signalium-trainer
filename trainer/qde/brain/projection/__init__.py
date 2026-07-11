"""QDE Brain — projection: forward estimates of move, duration, and risk.

Three modules that give the QDE quantitative forward estimates for every
opportunity, learned from historical (conditions → outcome) pairs rather
than hardcoded pip targets.

What this adds
--------------
The existing Brain reasons about *qualitative* setup quality (conviction,
maturity, narrative alignment).  This package adds *quantitative* forward
estimates:

  * Expected move size (in ATR multiples — instrument-agnostic)
  * Expected duration (seconds the trade is likely to be open)
  * Expected volatility (during the trade)
  * Expected reward (signed, in ATR multiples)
  * Expected adverse excursion / MAE (in ATR multiples)

All five are learned from accumulated (conditions, outcome) pairs.  There
are no hardcoded pip targets or minimum RR thresholds — "sufficient
reward" is always expressed as a percentile rank within the instrument's
own historical distribution.

Modules
-------
``projection_memory``
    Data models and rolling store.

    * ``ProjectionConditions`` — nine-dimensional market fingerprint
      (atr_pct, spread_atr_ratio, volatility, momentum, vol_state_rank_norm,
      volatility_quality, microstructure_quality, session_quality,
      trend_valid_f).  Built via ``from_observation(instrument, inp)``.
    * ``ProjectionOutcome`` — what actually happened: realized return, MAE,
      MFE, duration, realized vol — all in ATR multiples.  Bridge via
      ``from_outcome_path(path, atr_at_entry)``.
    * ``ProjectionRecord`` — one (conditions → outcome) pair.
    * ``ProjectionMemory`` — bounded per-symbol rolling store with
      ``to_snapshot`` / ``load_snapshot`` persistence.

``expected_duration_engine``
    Stateless analytics for duration and volatility estimation.

    * ``DurationProjection`` — expected_duration_seconds, duration_std,
      expected_volatility, volatility_std, volatility_percentile,
      confidence.
    * ``ExpectedDurationEngine`` — cosine-similarity nearest-neighbor
      estimation over ``ProjectionMemory``.

``expected_move_engine``
    Move / reward / MAE estimation and top-level integration engine.

    * ``ExpectedProjection`` — all five estimates plus four pre-computed
      integration signals.
    * ``ExpectedMoveEngine`` — four integration points:
      ``context_for_inference``   → 16 ``proj_*`` brain_context keys
      ``opportunity_rank_signal`` → 0-1 float (trade selection)
      ``position_size_modifier``  → 0.5..2.0 float (position sizing)
      ``risk_appetite_modifier``  → −0.50..+0.25 float (risk appetite)

Learning loop
-------------
``ExpectedMoveEngine.record_outcome(instrument, direction, conditions,
outcome)`` is the single write entry point.  Call it once per completed
trade with the realized ``ProjectionOutcome`` from
``TradeOutcomePath.from_outcome_path()``.

Design rules
------------
* No hardcoded pip / ATR targets anywhere.
* ``reward_sufficiency_score`` is a percentile rank — relative to the
  instrument's own history, never an absolute floor.
* Does not import ``QDEDecision`` or any execution-layer type.
* Does not re-derive conviction, regime classification, or narrative.
"""

from __future__ import annotations

# ── Duration engine ───────────────────────────────────────────────────────────
from trainer.qde.brain.projection.expected_duration_engine import (
    DurationProjection,
    ExpectedDurationEngine,
)

# ── Move / reward / MAE engine (top-level orchestrator) ───────────────────────
from trainer.qde.brain.projection.expected_move_engine import (
    ExpectedMoveEngine,
    ExpectedProjection,
)

# ── Memory / data models ──────────────────────────────────────────────────────
from trainer.qde.brain.projection.projection_memory import (
    DEFAULT_RECENCY_TAU_DAYS,
    ProjectionConditions,
    ProjectionMemory,
    ProjectionOutcome,
    ProjectionRecord,
)

__all__ = [
    # memory / data models
    "ProjectionConditions",
    "ProjectionOutcome",
    "ProjectionRecord",
    "ProjectionMemory",
    "DEFAULT_RECENCY_TAU_DAYS",
    # duration engine
    "DurationProjection",
    "ExpectedDurationEngine",
    # move / reward engine
    "ExpectedProjection",
    "ExpectedMoveEngine",
]
