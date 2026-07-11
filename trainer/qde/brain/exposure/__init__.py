"""QDE Brain — exposure: hidden portfolio concentration analysis.

Three modules that give the QDE a portfolio-manager view of hidden
concentration risk — the kind that is invisible to per-symbol engines
because each instrument looks independent, but is actually a single
large factor bet distributed across many positions.

Problem being solved
--------------------
EURUSD long + GBPUSD long + XAUUSD long + USDCHF short each look
individually reasonable.  But together they represent one massive short-
USD position.  The per-symbol engines (conviction, projection, attention)
cannot see this — only a cross-instrument computation can.

What this package adds
-----------------------
Four concentration dimensions — all quantified, none as hardcoded limits:

  1. **Currency** — net exposure per currency, derived from
     ``InstrumentInfo.currency_exposures()`` (the existing authoritative
     source, never redefined here).

  2. **Sector / asset class** — net weight per asset class (FOREX,
     COMMODITY, INDEX, CRYPTO), using ``InstrumentInfo.asset_class``.

  3. **Structural correlation** — inferred from ``InstrumentInfo.tags``
     without a static correlation table (``safe_haven``, ``risk_on``,
     ``commodity_currency`` groups; optional live ``CorrelationMatrix``
     when available).

  4. **Regional / macro** — net weight per ``InstrumentInfo.region``
     (NORTH_AMERICA, EUROPE, ASIA_PACIFIC, GLOBAL).

All four are expressed as Herfindahl-Hirschman Index values (0-1, same
formula as ``PortfolioScorer`` but applied to different data).  A
composite concentration blends the four for a single top-level signal.

No hardcoded limits
--------------------
``TradeSaturationEngine`` learns what level of concentration is
acceptable from ``ExposureMemory``.  Early in operation (thin history)
the structural marginal-HHI signal governs.  As the model accumulates
``(concentration → outcome)`` pairs, it progressively trusts its own
learned thresholds over the structural defaults.

Modules
-------
``exposure_memory``
    Bounded rolling store for learning.

    * ``ExposureObservation`` — portfolio concentration state at trade-
      entry time plus marginal change the trade would introduce.
    * ``ExposureOutcome`` — portfolio quality over the trade's lifetime.
      ``compute_quality(return, drawdown)`` derives the 0-1 learning
      target from raw account metrics — no hardcoded return floor.
    * ``ExposureRecord`` — (observation, outcome, timestamp).
    * ``ExposureMemory`` — bounded FIFO, ``to_snapshot``/``load_snapshot``.

``exposure_cluster_engine``
    Stateless four-dimensional concentration engine.

    * ``PositionExposure`` — (instrument, direction, weight) input bag.
    * ``ExposureCluster`` — four HHIs + dominant currency + composite.
    * ``ExposureClusterEngine.compute(positions)`` — O(N) pass.
    * ``ExposureClusterEngine.marginal(candidate, positions)`` — (before,
      after) cluster pair; delta = marginal concentration cost.
    * ``ExposureClusterEngine.marginal_deltas(candidate, positions)`` —
      convenience dict of all five deltas.

``trade_saturation_engine``
    Learned saturation signal + integration points.

    * ``SaturationSignal`` — all intermediate values plus the four
      integration signals (``size_modifier``, ``conviction_modifier``,
      ``rank_penalty``, ``risk_appetite_modifier``).
    * ``TradeSaturationEngine.score(candidate, current_positions)``
      → ``SaturationSignal``.
    * ``TradeSaturationEngine.context_for_inference(candidate,
      current_positions)`` → 14 ``exp_*`` keys for
      ``QDEInferenceInput.brain_context``.
    * ``TradeSaturationEngine.build_observation(candidate, positions)``
      → ``ExposureObservation`` (store at entry, pass back on close).
    * ``TradeSaturationEngine.record_outcome(observation, outcome)``
      → learning loop write entry point.

Design rules
------------
* No ``QDEDecision`` import anywhere in this package.
* Reuses ``InstrumentInfo.currency_exposures()`` — never re-derives it.
* Reuses ``InstrumentRegistry`` — never re-declares instrument metadata.
* Does not duplicate ``PortfolioScorer`` — the HHI is applied to net
  exposure vectors, not to position weights.
* All three engines are callable without a running scheduler or network.
"""

from __future__ import annotations

# ── Memory / data models ──────────────────────────────────────────────────────
from trainer.qde.brain.exposure.exposure_memory import (
    ExposureMemory,
    ExposureObservation,
    ExposureOutcome,
    ExposureRecord,
)

# ── Cluster engine ────────────────────────────────────────────────────────────
from trainer.qde.brain.exposure.exposure_cluster_engine import (
    ExposureCluster,
    ExposureClusterEngine,
    PositionExposure,
)

# ── Saturation engine ─────────────────────────────────────────────────────────
from trainer.qde.brain.exposure.trade_saturation_engine import (
    SaturationSignal,
    TradeSaturationEngine,
)

__all__ = [
    # memory
    "ExposureObservation",
    "ExposureOutcome",
    "ExposureRecord",
    "ExposureMemory",
    # cluster engine
    "PositionExposure",
    "ExposureCluster",
    "ExposureClusterEngine",
    # saturation engine
    "SaturationSignal",
    "TradeSaturationEngine",
]
