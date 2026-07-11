"""QDE Brain — opportunity: capital and attention allocation.

``CapitalPriorityEngine`` is the final arbitration layer that converts a
ranked opportunity list into concrete capital and attention allocation
recommendations.  It enforces portfolio-level constraints that neither
the per-symbol engines nor the ranker can enforce in isolation:

  * **Max concurrent positions** — the portfolio cannot absorb unlimited
    open trades simultaneously; lower-ranked candidates are deferred.
  * **Capital budget** — the aggregate of all ``suggested_size_modifier``
    values must not push total deployment beyond a portfolio-level cap.
  * **Minimum score threshold** — setups below a net_score floor receive
    PASS regardless of available budget.

Priority tiers
--------------
::

    TIER_1  Top-ranked actionable setups below max_concurrent;
            full projection-engine size modifier applied.

    TIER_2  Good setups that fit within a looser budget;
            size modifier scaled to 60 % of the projection suggestion.

    TIER_3  Below-par setups worth watching but not deploying full
            capital; size modifier scaled to 30 %.

    PASS    Insufficient score, budget exhausted, or not yet actionable.

Attention allocation
--------------------
Attention is a separate resource from capital.  The engine allocates a
0-1 attention fraction per candidate — TIER_1 setups share the top 60 %
of monitoring bandwidth proportionally to their net score; TIER_2 share
the next 30 %; TIER_3 get the remaining 10 %; PASS setups receive 0.

Why attention and capital are allocated separately
--------------------------------------------------
A CONFIRMING setup not yet ready for capital may still deserve a large
fraction of monitoring attention because it is one tick away from READY.
Conversely, an EXECUTING position already open requires very little
incremental attention (the trade is live; monitoring is a position-
management responsibility, not an entry-decision one).  Separating the
two resources lets the Brain behave rationally in both cases.

Design rules
------------
* Stateless: accepts the ranked list and constraints, returns allocations.
* Does not import ``QDEDecision`` or any execution-layer type.
* Does not re-rank; always operates on the ``RankedOpportunity`` list
  produced by ``OpportunityRanker.rank()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from trainer.qde.brain.opportunity.opportunity_ranker import (
    OpportunityRanker,
    RankedOpportunity,
    OpportunityInput,
)

# ── Tier thresholds and defaults ──────────────────────────────────────────────

# Net-score thresholds for tier eligibility.  Below TIER_3_MIN → PASS.
_TIER_1_MIN    = 0.55
_TIER_2_MIN    = 0.38
_TIER_3_MIN    = 0.20

# Size modifier scaling per tier relative to the projection engine's suggestion
_TIER_1_SCALE  = 1.00   # full suggestion
_TIER_2_SCALE  = 0.60
_TIER_3_SCALE  = 0.30

# Fraction of total attention budget allocated to each tier group
_ATTN_TIER1    = 0.60
_ATTN_TIER2    = 0.30
_ATTN_TIER3    = 0.10

# Default portfolio constraints
DEFAULT_MAX_CONCURRENT  = 3
DEFAULT_MAX_CAPITAL_PCT = 1.0   # fraction of baseline (1.0 = 100 % of normal sizing)
DEFAULT_MIN_SCORE       = _TIER_3_MIN


# ── PriorityTier ──────────────────────────────────────────────────────────────


class PriorityTier(str, Enum):
    """Allocated priority for one opportunity."""

    TIER_1 = "TIER_1"   # full allocation — deploy capital now
    TIER_2 = "TIER_2"   # reduced allocation — queue with smaller size
    TIER_3 = "TIER_3"   # monitoring only — minimal capital if at all
    PASS   = "PASS"     # skip — insufficient quality or budget exhausted


# ── CapitalAllocation ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CapitalAllocation:
    """Allocation recommendation for one opportunity.

    ``suggested_size_modifier``
        0-1 or higher multiplier relative to the per-symbol projection
        engine's baseline.  0 = deploy nothing.  Callers multiply this
        by the strategy's base lot size / risk fraction.

        Note: unlike ``ExpectedProjection.position_size_modifier`` (which
        is a per-symbol signal on a 0.5..2.0 scale), this field starts
        from the projection modifier and then scales it by the tier factor
        — so it reflects *both* the setup quality *and* the portfolio
        context.

    ``attention_allocation``
        0-1 fraction of total monitoring attention to allocate to this
        symbol for this tick.  Fractions across the full portfolio sum
        to ≤ 1.0 (may be < 1.0 if low-scoring symbols absorb no
        attention).

    ``rank``
        Ordinal rank from ``OpportunityRanker`` (1 = best).

    ``net_score``
        Net score from ``RankedOpportunity.net_score``.
    """

    instrument:              str          = ""
    opportunity_id:          str          = ""
    tier:                    PriorityTier = PriorityTier.PASS
    suggested_size_modifier: float        = 0.0
    attention_allocation:    float        = 0.0
    rank:                    int          = 0
    net_score:               float        = 0.0
    composite_score:         float        = 0.0
    is_actionable:           bool         = False
    reasoning:               str          = ""

    def to_dict(self) -> dict:
        return {
            "instrument":              self.instrument,
            "opportunity_id":          self.opportunity_id,
            "tier":                    self.tier.value,
            "suggested_size_modifier": round(self.suggested_size_modifier, 4),
            "attention_allocation":    round(self.attention_allocation, 4),
            "rank":                    self.rank,
            "net_score":               round(self.net_score, 4),
            "composite_score":         round(self.composite_score, 4),
            "is_actionable":           self.is_actionable,
            "reasoning":               self.reasoning,
        }


# ── CapitalPriorityEngine ─────────────────────────────────────────────────────


class CapitalPriorityEngine:
    """Stateless capital and attention allocator.

    ``allocate()`` accepts a pre-ranked list from ``OpportunityRanker``
    and returns one ``CapitalAllocation`` per candidate.

    ``allocate_from_inputs()`` is a convenience wrapper that runs the
    ranker internally so callers need only pass raw ``OpportunityInput``
    records.
    """

    def __init__(self) -> None:
        self._ranker = OpportunityRanker()

    # ── Public API ────────────────────────────────────────────────────────

    def allocate(
        self,
        ranked:           list[RankedOpportunity],
        max_concurrent:   int   = DEFAULT_MAX_CONCURRENT,
        max_capital_pct:  float = DEFAULT_MAX_CAPITAL_PCT,
        min_score:        float = DEFAULT_MIN_SCORE,
        now:              float | None = None,
    ) -> dict[str, CapitalAllocation]:
        """Produce ``CapitalAllocation`` for each ranked opportunity.

        Parameters
        ----------
        ranked:
            Output of ``OpportunityRanker.rank()`` sorted best-first.
        max_concurrent:
            Hard cap on TIER_1 + TIER_2 positions that may receive
            non-zero capital.  Candidates beyond this cap are PASS
            regardless of score.
        max_capital_pct:
            Soft cap: if the running sum of ``suggested_size_modifier``
            across already-allocated candidates exceeds this value,
            subsequent candidates are demoted one tier.
        min_score:
            Net-score floor; candidates below this receive PASS.
        """
        if not ranked:
            return {}

        now = now if now is not None else time.time()

        # ── Pass 1: assign tiers and size modifiers ────────────────────
        tier_map: dict[str, PriorityTier] = {}
        size_map: dict[str, float]        = {}
        running_capital                   = 0.0
        deployed_count                    = 0

        for row in ranked:
            tier, size, reason = self._assign_tier(
                row=row,
                deployed_count=deployed_count,
                running_capital=running_capital,
                max_concurrent=max_concurrent,
                max_capital_pct=max_capital_pct,
                min_score=min_score,
            )
            tier_map[row.instrument] = tier
            size_map[row.instrument] = size
            if tier in (PriorityTier.TIER_1, PriorityTier.TIER_2):
                deployed_count  += 1
                running_capital += size

        # ── Pass 2: distribute attention budget ───────────────────────
        attn_map = self._allocate_attention(ranked, tier_map)

        # ── Pass 3: assemble results ───────────────────────────────────
        result: dict[str, CapitalAllocation] = {}
        for row in ranked:
            tier   = tier_map[row.instrument]
            size   = size_map[row.instrument]
            reason = self._reasoning(row, tier, size)
            result[row.instrument] = CapitalAllocation(
                instrument=row.instrument,
                opportunity_id=row.opportunity_id,
                tier=tier,
                suggested_size_modifier=round(size, 4),
                attention_allocation=round(attn_map.get(row.instrument, 0.0), 4),
                rank=row.rank,
                net_score=row.net_score,
                composite_score=row.composite_score,
                is_actionable=row.is_actionable,
                reasoning=reason,
            )

        return result

    def allocate_from_inputs(
        self,
        candidates:      list[OpportunityInput],
        max_concurrent:  int   = DEFAULT_MAX_CONCURRENT,
        max_capital_pct: float = DEFAULT_MAX_CAPITAL_PCT,
        min_score:       float = DEFAULT_MIN_SCORE,
        now:             float | None = None,
    ) -> dict[str, CapitalAllocation]:
        """Convenience wrapper: rank then allocate in one call."""
        now    = now if now is not None else time.time()
        ranked = self._ranker.rank(candidates, now=now)
        return self.allocate(
            ranked=ranked,
            max_concurrent=max_concurrent,
            max_capital_pct=max_capital_pct,
            min_score=min_score,
            now=now,
        )

    def top_tier(
        self,
        candidates:     list[OpportunityInput],
        max_concurrent: int   = DEFAULT_MAX_CONCURRENT,
        now:            float | None = None,
    ) -> list[CapitalAllocation]:
        """Return only TIER_1 allocations, sorted by rank."""
        allocs  = self.allocate_from_inputs(candidates, max_concurrent=max_concurrent, now=now)
        tier1   = [a for a in allocs.values() if a.tier == PriorityTier.TIER_1]
        return sorted(tier1, key=lambda a: a.rank)

    def context_for_inference(
        self,
        instrument:     str,
        candidates:     list[OpportunityInput],
        max_concurrent: int   = DEFAULT_MAX_CONCURRENT,
        now:            float | None = None,
    ) -> dict:
        """Return ``opp_*`` keys for ``QDEInferenceInput.brain_context``.

        Keys
        ----
        opp_tier                  — PriorityTier.value
        opp_rank                  — ordinal rank (1 = best)
        opp_net_score             — net composite score 0-1
        opp_suggested_size_mod    — suggested size multiplier
        opp_attention_allocation  — 0-1 attention fraction
        opp_is_actionable         — bool
        opp_total_candidates      — count of all candidates
        opp_concurrent_budget_pct — deployed / max_concurrent (0-1)
        """
        now    = now if now is not None else time.time()
        allocs = self.allocate_from_inputs(candidates, max_concurrent=max_concurrent, now=now)
        total  = len(candidates)
        t1_cnt = sum(1 for a in allocs.values() if a.tier == PriorityTier.TIER_1)
        t2_cnt = sum(1 for a in allocs.values() if a.tier == PriorityTier.TIER_2)
        used   = t1_cnt + t2_cnt

        alloc  = allocs.get(instrument, CapitalAllocation(instrument=instrument))
        return {
            "opp_tier":                  alloc.tier.value,
            "opp_rank":                  alloc.rank,
            "opp_net_score":             round(alloc.net_score, 4),
            "opp_suggested_size_mod":    round(alloc.suggested_size_modifier, 4),
            "opp_attention_allocation":  round(alloc.attention_allocation, 4),
            "opp_is_actionable":         alloc.is_actionable,
            "opp_total_candidates":      total,
            "opp_concurrent_budget_pct": round(used / max(max_concurrent, 1), 4),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _assign_tier(
        self,
        row:             RankedOpportunity,
        deployed_count:  int,
        running_capital: float,
        max_concurrent:  int,
        max_capital_pct: float,
        min_score:       float,
    ) -> tuple[PriorityTier, float, str]:
        """Return (tier, suggested_size_modifier, reasoning) for one row."""

        if row.net_score < min_score:
            return PriorityTier.PASS, 0.0, f"net_score={row.net_score:.3f} < floor {min_score}"

        if not row.is_actionable:
            # Score it but don't deploy — it may be in WATCHING / ARMED
            return PriorityTier.TIER_3, 0.0, f"state={row.setup_state} not yet actionable"

        base_size = row.proj_position_size_modifier   # from ExpectedMoveEngine

        # TIER_1: top-ranked, within budget, above threshold
        if (
            row.net_score >= _TIER_1_MIN
            and deployed_count < max_concurrent
            and running_capital + base_size * _TIER_1_SCALE <= max_capital_pct + 0.1
        ):
            size = base_size * _TIER_1_SCALE
            return PriorityTier.TIER_1, size, (
                f"rank={row.rank} net_score={row.net_score:.3f} → TIER_1"
            )

        # TIER_2: decent score, some budget remaining
        if (
            row.net_score >= _TIER_2_MIN
            and deployed_count < max_concurrent
            and running_capital + base_size * _TIER_2_SCALE <= max_capital_pct + 0.2
        ):
            size = base_size * _TIER_2_SCALE
            return PriorityTier.TIER_2, size, (
                f"rank={row.rank} net_score={row.net_score:.3f} → TIER_2 (budget partial)"
            )

        # TIER_3: below TIER_2 threshold or budget exhausted — watch, don't size
        if row.net_score >= _TIER_3_MIN:
            return PriorityTier.TIER_3, 0.0, (
                f"rank={row.rank} net_score={row.net_score:.3f} → TIER_3 (monitor only)"
            )

        return PriorityTier.PASS, 0.0, (
            f"rank={row.rank} net_score={row.net_score:.3f} → PASS"
        )

    @staticmethod
    def _allocate_attention(
        ranked:   list[RankedOpportunity],
        tier_map: dict[str, PriorityTier],
    ) -> dict[str, float]:
        """Distribute 0-1 attention budget across tiers.

        TIER_1 setups share 60 % of attention, proportional to net_score.
        TIER_2 share the next 30 %, TIER_3 the final 10 %.
        PASS setups receive 0.
        """
        buckets: dict[PriorityTier, list[RankedOpportunity]] = {
            PriorityTier.TIER_1: [],
            PriorityTier.TIER_2: [],
            PriorityTier.TIER_3: [],
        }
        for row in ranked:
            t = tier_map.get(row.instrument, PriorityTier.PASS)
            if t in buckets:
                buckets[t].append(row)

        result: dict[str, float] = {}

        def _distribute(rows: list[RankedOpportunity], budget: float) -> None:
            if not rows:
                return
            total_score = sum(r.net_score for r in rows) or 1e-9
            for r in rows:
                result[r.instrument] = budget * (r.net_score / total_score)

        _distribute(buckets[PriorityTier.TIER_1], _ATTN_TIER1)
        _distribute(buckets[PriorityTier.TIER_2], _ATTN_TIER2)
        _distribute(buckets[PriorityTier.TIER_3], _ATTN_TIER3)
        return result

    @staticmethod
    def _reasoning(row: RankedOpportunity, tier: PriorityTier, size: float) -> str:
        if tier == PriorityTier.PASS:
            return (
                f"{row.instrument}: PASS — net_score={row.net_score:.3f}, "
                f"state={row.setup_state}"
            )
        return (
            f"{row.instrument}: {tier.value} rank={row.rank} "
            f"net={row.net_score:.3f} "
            f"conv={row.conviction:.2f}×conf={row.confidence:.2f} "
            f"unc={row.uncertainty:.2f} "
            f"size_mod={size:.2f}"
        )
