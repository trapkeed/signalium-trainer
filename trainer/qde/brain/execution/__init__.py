"""QDE Brain — execution: learned execution quality analysis.

Three modules that give the QDE instrument-specific awareness of execution
risk — the kind of risk that is invisible to the conviction, projection,
and exposure engines because each measures market opportunity independently
of the cost and probability of actually getting into the trade cleanly.

Problem being solved
--------------------
A setup with 0.80 conviction and a 2.5 ATR expected move is genuinely
different depending on whether the current spread is 0.05 ATR (trivial
friction) or 0.60 ATR (the spread alone consumes 24% of the expected
move). Similarly, a LIMIT order placed 1.5 ATR away has an uncertain fill
probability that depends on volatility, session quality, and the
instrument's historical fill behaviour — none of which the per-signal
engines model.

What this package adds
-----------------------
Four execution-quality outputs for downstream consumers:

  ``timing_recommendation``      EXECUTE_NOW / WAIT / AVOID
  ``order_type_recommendation``  MARKET / LIMIT / STOP_LIMIT
  ``conviction_modifier``        −0.25..0
  ``expected_reward_modifier``   −0.30..0

All four are monotone functions of the combined structural + learned
execution quality signal — no hardcoded broker assumptions, no hardcoded
spread limits, no hardcoded per-instrument thresholds.

Modules
-------
``slippage_predictor``
    Stateless structural slippage model.

    * ``SlippageInput`` — conditions bag (spread_atr_ratio, volatility,
      session_quality, order_type, microstructure_quality).
    * ``SlippageEstimate`` — expected + worst-case slippage in ATR units,
      spread_cost_atr, favorable_probability, spread_risk.
    * ``SlippagePredictor.estimate(inp)`` → ``SlippageEstimate``.

    No internal state; historical correction is applied by
    ``ExecutionQualityEngine``.

``fill_probability_engine``
    Stateless structural fill model.

    * ``FillContext`` — per-order context (order_type, distance_to_target_atr,
      spread_atr_ratio, volatility, session_quality, momentum, direction).
    * ``FillProbabilitySignal`` — fill_probability, expected_fill_quality,
      estimated_wait_seconds, recommended_order_type.
    * ``FillProbabilityEngine.estimate(ctx)`` → ``FillProbabilitySignal``.
    * ``FillProbabilityEngine.recommend_order_type(spread_atr_ratio,
      expected_slippage_atr, timing_budget_seconds)`` → ``OrderType``.

    No internal state; structural model only.

``execution_quality_engine``
    Learned per-symbol execution quality engine.

    * ``ExecutionConditions`` — live market conditions bag (instrument,
      spread_atr_ratio, atr_pct, volatility, session_quality, momentum,
      microstructure_quality).
    * ``ExecutionObservation`` — conditions + order characteristics captured
      at submission time; 6-dim ``as_vector()`` for L2-distance lookup.
    * ``ExecutionOutcome`` — realized fill results + ``compute_quality()``
      static helper (normalised to structural prediction, no fixed target).
    * ``ExecutionRecord`` — (observation, outcome, timestamp).
    * ``ExecutionMemory`` — bounded FIFO, ``to_snapshot`` / ``load_snapshot``.
    * ``ExecutionQualitySignal`` — frozen dataclass with all 16 fields.
    * ``ExecutionQualityEngine(max_records, k, recency_tau)`` — per-symbol.
      ``score(conditions, order_type, distance, direction)``
      → ``ExecutionQualitySignal``.
      ``context_for_inference(...)`` → 16 ``exec_*`` keys for brain_context.
      ``build_observation(...)`` → ``ExecutionObservation`` (store at entry).
      ``record_outcome(observation, outcome)`` → learning loop write point.
      ``memory_snapshot()`` / ``load_memory(snapshot)`` → persistence.

Design rules
------------
* No ``QDEDecision`` import anywhere in this package.
* No hardcoded broker constraints or instrument-specific thresholds.
* Reuses ``OrderType``, ``TimingRecommendation``, ``ExecutionGrade``
  from ``trainer.execution.execution_types``.
* Slippage and fill estimates are in ATR units (instrument-agnostic).
* One ``ExecutionQualityEngine`` instance per watched symbol.
* Learning is per-symbol; EURUSD history does not influence XAUUSD scores.
"""

from __future__ import annotations

# ── Slippage predictor ────────────────────────────────────────────────────────
from trainer.qde.brain.execution.slippage_predictor import (
    SlippageEstimate,
    SlippageInput,
    SlippagePredictor,
)

# ── Fill probability engine ───────────────────────────────────────────────────
from trainer.qde.brain.execution.fill_probability_engine import (
    FillContext,
    FillProbabilityEngine,
    FillProbabilitySignal,
)

# ── Execution quality engine ──────────────────────────────────────────────────
from trainer.qde.brain.execution.execution_quality_engine import (
    ExecutionConditions,
    ExecutionMemory,
    ExecutionObservation,
    ExecutionOutcome,
    ExecutionQualityEngine,
    ExecutionQualitySignal,
    ExecutionRecord,
)

__all__ = [
    # slippage
    "SlippageInput",
    "SlippageEstimate",
    "SlippagePredictor",
    # fill probability
    "FillContext",
    "FillProbabilitySignal",
    "FillProbabilityEngine",
    # quality engine
    "ExecutionConditions",
    "ExecutionObservation",
    "ExecutionOutcome",
    "ExecutionRecord",
    "ExecutionMemory",
    "ExecutionQualitySignal",
    "ExecutionQualityEngine",
]
