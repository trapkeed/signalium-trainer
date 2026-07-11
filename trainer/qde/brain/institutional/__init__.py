"""QDE Brain — institutional: probabilistic institutional intent inference.

This package gives the QDE learned, probabilistic beliefs about what
large market participants are doing — without asserting certainty and
without ICT-specific rules.

Problem being solved
--------------------
Price action is shaped by participants of vastly different size and
intent.  A retail participant and an institutional participant occupying
the same price zone are doing structurally different things:

  * Retail: reacting to current price, placing directional bets, placing
    stops near obvious structural levels.
  * Institutional: absorbing flow to build/exit positions, engineering
    conditions that allow large-size execution, acquiring liquidity.

If the QDE can learn to recognise *which type of participant behaviour
is currently dominant*, it can:

  * Contextualise its narrative ("price is compressing as large orders
    absorb sell flow" vs. "price is distributing at resistance")
  * Strengthen or weaken conviction for setups that align or conflict
    with the inferred intent
  * Rank opportunities higher when aligned institutional activity is
    detected
  * Improve execution timing by identifying the inflection points that
    follow institutional accumulation or stop hunts

The six beliefs
---------------
The engine produces six independent probabilistic beliefs:

  ``accumulation_prob``    P(gradual position building — long)
  ``distribution_prob``    P(gradual position exit — preparing downside)
  ``transfer_prob``        P(inventory transfer from institutional to retail)
  ``stop_hunt_prob``       P(engineered sweep of structural stops)
  ``liquidity_acq_prob``   P(directional move toward resting-order zones)
  ``rebalance_prob``       P(portfolio rebalancing — no strong directional intent)

All six are independent priors in [0, 1], not a probability distribution
over mutually exclusive states.  Multiple intents can simultaneously be
high (a stop hunt is often *also* a liquidity acquisition for the party
engineering it).

Architecture (four modules)
----------------------------
``accumulation_detector``
    Stateless structural model for accumulation probability.

    * ``BarSnapshot`` — minimal OHLCV bar shared across the package
    * ``AccumulationInput`` — conditions bag (optionally with raw bars)
    * ``AccumulationEstimate`` — probability + structural components
    * ``AccumulationDetector.detect(inp)`` → ``AccumulationEstimate``
    * ``AccumulationDetector.from_bars(bars, **ctx)`` → convenience

    Observes: range compression, support absorption, volume buy
    dominance, price-momentum divergence.

``distribution_detector``
    Symmetric structural model for distribution probability.

    * ``DistributionInput``, ``DistributionEstimate``, ``DistributionDetector``

    Observes: range compression at highs, resistance rejection, volume
    sell dominance, downside momentum divergence.

``inventory_transfer_detector``
    Structural model for transfer-related patterns.

    * ``SweepEvent`` — minimal descriptor of a structural sweep from
      the liquidity layer (no liquidity package import)
    * ``InventoryTransferInput``, ``InventoryTransferEstimate``
    * ``InventoryTransferDetector.detect(inp)`` → four probabilities:
      stop_hunt, liquidity_acq, transfer, rebalance

``institutional_intent_engine``
    Learned multi-symbol orchestrator.

    * ``InstitutionalConditions`` — 8-dim conditions fingerprint
    * ``InstitutionalOutcome`` — ground-truth labels after trade closes
    * ``InstitutionalRecord``, ``InstitutionalMemory`` — learning store
    * ``InstitutionalBeliefs`` — frozen dataclass with all 18 ``inst_*`` keys
    * ``InstitutionalIntentEngine``
        ``score(conditions, direction)``               → ``InstitutionalBeliefs``
        ``context_for_inference(conditions, direction)`` → 18 ``inst_*`` keys
        ``record_outcome(conditions, outcome)``        → learning loop
        ``to_snapshot(instrument)`` / ``load_snapshot(instrument, data)``

Integration signals (18 ``inst_*`` keys)
-----------------------------------------
These keys are merged into ``QDEInferenceInput.brain_context`` alongside
``proj_*``, ``rr_*``, ``exec_*``, ``exp_*``, ``regime_*``:

    inst_accumulation_prob        0-1: P(accumulation in progress)
    inst_distribution_prob        0-1: P(distribution in progress)
    inst_transfer_prob            0-1: P(inventory transfer)
    inst_stop_hunt_prob           0-1: P(stop hunt)
    inst_liquidity_acq_prob       0-1: P(liquidity acquisition move)
    inst_rebalance_prob           0-1: P(position rebalancing)
    inst_dominant_intent_id       0=accum 1=dist 2=xfer 3=hunt 4=liq 5=rebal 6=neutral
    inst_intent_certainty         max_prob − second_max (separation, 0-1)
    inst_direction_alignment      0=opposing 0.5=neutral 1=aligned with direction
    inst_conviction_modifier      −0.25..+0.25 (added to setup conviction)
    inst_opportunity_boost        0-1 (additive to opportunity ranking)
    inst_timing_confidence        0-1 (execution timing quality)
    inst_absorption_strength      structural absorption signal 0-1
    inst_range_compression        structural range compression 0-1
    inst_volume_signature         volume anomaly signal 0-1
    inst_sweep_recency            recency of last structural sweep 0-1
    inst_confidence               0-1 based on similar-record count
    inst_is_structural            1.0=pure structural; 0.0=history used

How each named concern is influenced
--------------------------------------
**Narratives**
    ``inst_dominant_intent_id`` and ``inst_intent_certainty`` let the
    narrative engine label the current market context without
    hardcoding patterns.  High certainty + clear dominant intent →
    the narrative gains an institutional-context thread.

**Conviction**
    ``inst_conviction_modifier`` (−0.25..+0.25) is added to the per-setup
    conviction score.  Aligned institutional intent boosts conviction;
    opposing intent reduces it.

**Opportunity ranking**
    ``inst_opportunity_boost`` (0-1) is an additive component the
    opportunity ranker can incorporate.  Highest when a confirmed stop
    hunt + direction alignment is present (the classic "hunt-and-reverse"
    entry timing).

**Execution timing**
    ``inst_timing_confidence`` is highest immediately after a confirmed
    stop-hunt reversal or at peak absorption.  Low during transfer and
    rebalancing phases (no clear entry inflection point).

Design rules
------------
* No ``QDEDecision`` imports.
* No imports from any other brain sub-package (independent package).
* No ICT-specific rules — all signal names are structurally neutral.
* All magnitude outputs are normalised to [0, 1] or defined float ranges.
* ``InstitutionalIntentEngine`` is multi-symbol.  Do not instantiate
  per-symbol.
"""

from __future__ import annotations

# ── Shared bar type ───────────────────────────────────────────────────────────
from trainer.qde.brain.institutional.accumulation_detector import (
    AccumulationDetector,
    AccumulationEstimate,
    AccumulationInput,
    BarSnapshot,
)

# ── Structural detectors ──────────────────────────────────────────────────────
from trainer.qde.brain.institutional.distribution_detector import (
    DistributionDetector,
    DistributionEstimate,
    DistributionInput,
)

# ── Learned engine ────────────────────────────────────────────────────────────
from trainer.qde.brain.institutional.institutional_intent_engine import (
    InstitutionalBeliefs,
    InstitutionalConditions,
    InstitutionalIntentEngine,
    InstitutionalMemory,
    InstitutionalOutcome,
    InstitutionalRecord,
)
from trainer.qde.brain.institutional.inventory_transfer_detector import (
    InventoryTransferDetector,
    InventoryTransferEstimate,
    InventoryTransferInput,
    SweepEvent,
)

__all__ = [
    # Shared bar type
    "BarSnapshot",
    # Accumulation
    "AccumulationInput",
    "AccumulationEstimate",
    "AccumulationDetector",
    # Distribution
    "DistributionInput",
    "DistributionEstimate",
    "DistributionDetector",
    # Inventory transfer
    "SweepEvent",
    "InventoryTransferInput",
    "InventoryTransferEstimate",
    "InventoryTransferDetector",
    # Intent engine
    "InstitutionalConditions",
    "InstitutionalOutcome",
    "InstitutionalRecord",
    "InstitutionalMemory",
    "InstitutionalBeliefs",
    "InstitutionalIntentEngine",
]
