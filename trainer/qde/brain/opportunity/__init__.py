"""QDE Brain — opportunity: cross-symbol comparison and capital allocation.

Three modules that give the QDE a portfolio-manager view of all active
opportunities simultaneously, rather than treating each symbol's signal
in isolation.

What this adds
--------------
The existing per-symbol engines (conviction, uncertainty, projection,
attention) each score their own symbol independently.  This package
aggregates those scores across all monitored symbols in one pass and
produces three additional outputs per opportunity:

  * **Capital efficiency** — expected reward (ATR multiples) per hour
    of deployment.  Duration becomes a first-class concern: a fast,
    clean setup that frees capital quickly beats a slow, comparable one.

  * **Opportunity cost** — what fraction of the best available
    candidate's efficiency is forgone by choosing this one instead.
    0 = this IS the best; 1 = taking this means missing the entire
    value of the best alternative.

  * **Priority tier** — TIER_1 / TIER_2 / TIER_3 / PASS assignment
    that enforces portfolio-level constraints: max concurrent positions,
    aggregate capital budget, minimum score threshold.

The result is a ranked list ordered by net desirability plus concrete
size-modifier and attention-fraction recommendations per candidate.

Modules
-------
``opportunity_cost_engine``
    Stateless cross-symbol capital-efficiency and opportunity-cost
    computation.

    * ``OpportunityInput`` — the input bag populated by callers from
      existing per-symbol engine outputs (no re-derivation).
    * ``OpportunityCost`` — capital_efficiency, opportunity_cost_score,
      forgone_efficiency, efficiency_rank.
    * ``OpportunityCostEngine.compute(candidates)`` — O(N) pass.

``opportunity_ranker``
    Stateless cross-symbol ranking.

    * ``RankedOpportunity`` — ranked candidate with all component
      scores plus passthrough fields for downstream consumers.
    * ``OpportunityRanker.rank(candidates)`` — composite score:
      conviction_joint (30 %) + ev_score (25 %) + uncertainty_discount
      (25 %) + capital_efficiency_score (20 %), minus opportunity-cost
      penalty, boosted by attention weight.

``capital_priority_engine``
    Stateless capital and attention allocation.

    * ``PriorityTier`` — TIER_1 / TIER_2 / TIER_3 / PASS enum.
    * ``CapitalAllocation`` — tier, suggested_size_modifier,
      attention_allocation, rank, net_score, reasoning.
    * ``CapitalPriorityEngine.allocate(ranked, …)`` — enforce
      portfolio constraints and distribute size + attention budget.
    * ``CapitalPriorityEngine.allocate_from_inputs(candidates, …)``
      — convenience wrapper (ranks then allocates).
    * ``CapitalPriorityEngine.context_for_inference(instrument, …)``
      → 8 ``opp_*`` keys for ``QDEInferenceInput.brain_context``.

Design rules
------------
* Does not recompute conviction, uncertainty, or projection estimates.
* Does not import ``QDEDecision`` or any execution-layer type.
* All three engines are stateless — safe to call on every brain tick
  with a fresh candidate set.
* Does not duplicate ``AttentionScoreAllocator`` (per-symbol attention
  weighting) or ``ExpectedMoveEngine`` (per-symbol projection).
"""

from __future__ import annotations

# ── Input bag ─────────────────────────────────────────────────────────────────
from trainer.qde.brain.opportunity.opportunity_cost_engine import (
    OpportunityInput,
    OpportunityCost,
    OpportunityCostEngine,
)

# ── Cross-symbol ranker ───────────────────────────────────────────────────────
from trainer.qde.brain.opportunity.opportunity_ranker import (
    RankedOpportunity,
    OpportunityRanker,
)

# ── Capital / attention allocator ─────────────────────────────────────────────
from trainer.qde.brain.opportunity.capital_priority_engine import (
    PriorityTier,
    CapitalAllocation,
    CapitalPriorityEngine,
)

# ── C11: expected-utility scoring, classification, and ranking ────────────────
from trainer.qde.brain.opportunity.opportunity_score import (
    OpportunityClass,
    OpportunityScore,
    classify_opportunity,
    score_opportunity,
    rank_opportunities,
)

__all__ = [
    # input bag
    "OpportunityInput",
    # cost engine
    "OpportunityCost",
    "OpportunityCostEngine",
    # ranker
    "RankedOpportunity",
    "OpportunityRanker",
    # capital allocator
    "PriorityTier",
    "CapitalAllocation",
    "CapitalPriorityEngine",
    # ── C11 ──────────────────────────────────────────────────────────────────
    "OpportunityClass",
    "OpportunityScore",
    "classify_opportunity",
    "score_opportunity",
    "rank_opportunities",
]
