"""QDE Brain — opportunity: cross-symbol ranking.

``OpportunityRanker`` is the portfolio-manager layer that compares every
active opportunity *against every other* and produces a ranked list
ordered by net desirability.  It never re-derives conviction,
uncertainty, or projections — it reads them from ``OpportunityInput``
fields that callers already have in hand from the existing per-symbol
engines.

Why cross-symbol ranking matters
----------------------------------
Each per-symbol engine (conviction, projection, attention) scores its
own symbol in isolation.  Two independent signals can both look
"strong" by their own standards while pointing in different directions
or competing for the same capital.  The ranker resolves that tension
by establishing a shared ordinal: within this specific set of
candidates, *right now*, which setup is actually superior?

Composite score formula
-----------------------
Four components, each normalized to 0-1 *within the current candidate
set* (relative ranking) so the weights are stable regardless of the
raw units:

  * **conviction_joint** (weight 0.30)
    ``conviction × confidence`` — the joint signal that the setup is
    both high-quality (conviction) and probably correct (confidence).
    Separating these follows the existing ``ConvictionProfile``
    vocabulary: HIGH_CONFIDENCE_LOW_CONVICTION is a real, distinct case.

  * **expected_value_score** (weight 0.25)
    Percentile rank of ``expected_reward_atr × reward_sufficiency_score``
    within the current candidate set.  ``reward_sufficiency_score`` is
    already a learned relative signal (from ``ExpectedMoveEngine``), so
    multiplying by it further discounts setups where the projection
    engine has thin historical evidence.

  * **uncertainty_discount** (weight 0.25)
    ``1 - uncertainty`` — lower ambiguity = higher score.  Uncertainty
    is market-conditions ambiguity (``UncertaintyScore.composite``), not
    conviction; a high-conviction setup in an ambiguous market still
    deserves a penalty.

  * **capital_efficiency_score** (weight 0.20)
    Percentile rank of capital_efficiency within the current set.
    Duration is a first-class concern: a fast, clean setup that frees
    capital quickly beats a slow, comparable one.

Opportunity cost penalty
------------------------
The raw composite is reduced by ``opp_cost_weight × opportunity_cost_score``
where opportunity_cost_score comes from ``OpportunityCostEngine``.  This
is the "portfolio manager" adjustment: even a genuinely good setup loses
ground if a clearly superior alternative is also available.

Attention amplifier
-------------------
Attention weight (from the normalized ``AttentionScoreAllocator``) is
applied as a *mild multiplicative boost* (not an additive component)
so that high-attention symbols already in focus get a small lift without
overwhelming the fundamental quality signals.

Actionability gate
------------------
Opportunities in WATCHING or ARMED state are scored but flagged
``is_actionable = False``.  ``CapitalPriorityEngine`` treats these as
ineligible for TIER_1 / TIER_2 capital allocation even if their score
is high — the setup has not yet confirmed enough to deploy capital.

Design rules
------------
* Stateless: no per-symbol memory, safe to call on every brain tick.
* Does not import ``QDEDecision`` or any execution-layer type.
* Does not recompute conviction/uncertainty/projection — reads from
  ``OpportunityInput``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from trainer.qde.brain.opportunity.opportunity_cost_engine import (
    OpportunityCost,
    OpportunityCostEngine,
    OpportunityInput,
)

# ── Weight constants ──────────────────────────────────────────────────────────

_W_CONVICTION    = 0.30
_W_EV            = 0.25
_W_UNCERTAINTY   = 0.25
_W_EFFICIENCY    = 0.20
_W_OPP_COST      = 0.15   # penalty weight applied to opportunity_cost_score
_ATTN_MAX_BOOST  = 0.10   # max multiplicative boost from attention weight

assert abs(_W_CONVICTION + _W_EV + _W_UNCERTAINTY + _W_EFFICIENCY - 1.0) < 1e-9


# ── Helpers ───────────────────────────────────────────────────────────────────


def _percentile_rank(value: float, population: list[float]) -> float:
    """Fraction of population values <= value (0-1)."""
    if not population:
        return 0.5
    return sum(1 for x in population if x <= value) / len(population)


def _minmax_norm(value: float, lo: float, hi: float) -> float:
    """Min-max normalise ``value`` to [0, 1].  Returns 0.5 when lo == hi."""
    span = hi - lo
    if span < 1e-12:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / span))


# ── RankedOpportunity ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RankedOpportunity:
    """One ranked candidate from the cross-symbol comparison.

    ``composite_score``   — raw 0-1 before opportunity-cost penalty
    ``net_score``         — composite_score minus opportunity-cost penalty
    ``is_actionable``     — False when setup_state is WATCHING or ARMED;
                           see module docstring for why this matters

    All intermediate component scores are retained so callers can inspect
    what drove the ranking (useful for logging and tests).
    """

    rank:                   int   = 0
    instrument:             str   = ""
    direction:              int   = 1
    opportunity_id:         str   = ""
    setup_state:            str   = ""

    # ── Intermediate scores (all 0-1) ─────────────────────────────────────
    conviction_joint:       float = 0.0   # conviction × confidence
    ev_score:               float = 0.0   # percentile of weighted expected value
    uncertainty_discount:   float = 0.5   # 1 - uncertainty
    efficiency_score:       float = 0.0   # percentile of capital_efficiency
    attention_boost:        float = 1.0   # multiplicative, 1.0 = no boost

    composite_score:        float = 0.0   # before opp-cost penalty
    opportunity_cost_score: float = 0.0   # 0-1 cost (from OpportunityCostEngine)
    net_score:              float = 0.0   # composite − cost_penalty

    # ── Capital-efficiency metadata ────────────────────────────────────────
    capital_efficiency:     float = 0.0   # ATR/hour
    efficiency_rank:        int   = 1

    is_actionable:          bool  = False   # CONFIRMING / READY / EXECUTING

    # ── Passthrough fields for downstream consumers ────────────────────────
    expected_reward_atr:    float = 0.0
    expected_duration_seconds: float = 0.0
    expected_mae_atr:       float = 0.0
    conviction:             float = 0.0
    confidence:             float = 0.0
    uncertainty:            float = 0.5
    proj_position_size_modifier: float = 1.0
    proj_risk_appetite_modifier: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rank":                   self.rank,
            "instrument":             self.instrument,
            "direction":              self.direction,
            "opportunity_id":         self.opportunity_id,
            "setup_state":            self.setup_state,
            "conviction_joint":       round(self.conviction_joint, 4),
            "ev_score":               round(self.ev_score, 4),
            "uncertainty_discount":   round(self.uncertainty_discount, 4),
            "efficiency_score":       round(self.efficiency_score, 4),
            "attention_boost":        round(self.attention_boost, 4),
            "composite_score":        round(self.composite_score, 4),
            "opportunity_cost_score": round(self.opportunity_cost_score, 4),
            "net_score":              round(self.net_score, 4),
            "capital_efficiency":     round(self.capital_efficiency, 4),
            "efficiency_rank":        self.efficiency_rank,
            "is_actionable":          self.is_actionable,
            "expected_reward_atr":    round(self.expected_reward_atr, 4),
            "expected_duration_seconds": round(self.expected_duration_seconds, 1),
            "expected_mae_atr":       round(self.expected_mae_atr, 4),
            "conviction":             round(self.conviction, 4),
            "confidence":             round(self.confidence, 4),
            "uncertainty":            round(self.uncertainty, 4),
            "proj_position_size_modifier": round(self.proj_position_size_modifier, 4),
            "proj_risk_appetite_modifier": round(self.proj_risk_appetite_modifier, 4),
        }


# ── OpportunityRanker ─────────────────────────────────────────────────────────


class OpportunityRanker:
    """Stateless cross-symbol opportunity ranker.

    ``rank()`` accepts the full current candidate set, delegates
    opportunity-cost computation to ``OpportunityCostEngine``, then
    scores and sorts all candidates into a ranked list in a single pass.

    The result is a portfolio-manager view of the opportunity set: not
    "is this setup good?" (that's the per-symbol engines) but "given
    everything currently on the table, which setups should we prioritise?"
    """

    def __init__(self) -> None:
        self._cost_engine = OpportunityCostEngine()

    def rank(
        self,
        candidates: list[OpportunityInput],
        now: float | None = None,
    ) -> list[RankedOpportunity]:
        """Rank all candidates best-first.

        Returns an empty list when ``candidates`` is empty.  A
        single-candidate set returns a list of one; in that case
        opportunity cost is zero and the candidate's net_score equals
        its composite_score.

        Parameters
        ----------
        candidates:
            All active ``OpportunityInput`` records for this brain tick,
            across all monitored symbols.

        Returns
        -------
        ``list[RankedOpportunity]`` sorted by ``net_score`` descending
        (rank=1 is best).
        """
        if not candidates:
            return []

        now = now if now is not None else time.time()

        # ── Step 1: opportunity costs ──────────────────────────────────────
        costs: dict[str, OpportunityCost] = self._cost_engine.compute(candidates)

        # ── Step 2: per-candidate raw signals ─────────────────────────────
        conviction_joints   = [c.conviction * c.confidence for c in candidates]
        ev_values           = [c.expected_reward_atr * c.reward_sufficiency_score for c in candidates]
        eff_values          = [
            costs.get(c.instrument, OpportunityCost(instrument=c.instrument)).capital_efficiency
            for c in candidates
        ]

        lo_conv, hi_conv  = min(conviction_joints), max(conviction_joints)
        lo_ev,   hi_ev    = min(ev_values),         max(ev_values)
        lo_eff,  hi_eff   = min(eff_values),        max(eff_values)

        # ── Step 3: score and collect ──────────────────────────────────────
        scored: list[tuple[float, RankedOpportunity]] = []

        for inp, cj, ev, eff in zip(candidates, conviction_joints, ev_values, eff_values):
            cost = costs.get(inp.instrument, OpportunityCost(
                instrument=inp.instrument, opportunity_id=inp.opportunity_id
            ))

            # Normalize each component within the current candidate set
            cj_score  = _minmax_norm(cj,  lo_conv, hi_conv)
            ev_score  = _minmax_norm(ev,  lo_ev,   hi_ev)
            eff_score = _minmax_norm(eff, lo_eff,  hi_eff)
            unc_disc  = max(0.0, min(1.0, 1.0 - inp.uncertainty))

            composite = (
                _W_CONVICTION  * cj_score
                + _W_EV        * ev_score
                + _W_UNCERTAINTY * unc_disc
                + _W_EFFICIENCY * eff_score
            )

            # Attention: mild multiplicative boost (1.0 at weight=0, up to 1+max_boost)
            attn_boost = 1.0 + _ATTN_MAX_BOOST * inp.attention_weight
            composite  = min(1.0, composite * attn_boost)

            # Opportunity-cost penalty (subtractive, proportional to cost weight)
            opp_cost_penalty = _W_OPP_COST * cost.opportunity_cost_score
            net_score        = max(0.0, composite - opp_cost_penalty)

            row = RankedOpportunity(
                rank=0,   # filled below after sort
                instrument=inp.instrument,
                direction=inp.direction,
                opportunity_id=inp.opportunity_id,
                setup_state=inp.setup_state,
                conviction_joint=round(cj, 4),
                ev_score=round(ev_score, 4),
                uncertainty_discount=round(unc_disc, 4),
                efficiency_score=round(eff_score, 4),
                attention_boost=round(attn_boost, 4),
                composite_score=round(composite, 4),
                opportunity_cost_score=round(cost.opportunity_cost_score, 4),
                net_score=round(net_score, 4),
                capital_efficiency=round(eff, 4),
                efficiency_rank=cost.efficiency_rank,
                is_actionable=inp.is_actionable,
                expected_reward_atr=inp.expected_reward_atr,
                expected_duration_seconds=inp.expected_duration_seconds,
                expected_mae_atr=inp.expected_mae_atr,
                conviction=inp.conviction,
                confidence=inp.confidence,
                uncertainty=inp.uncertainty,
                proj_position_size_modifier=inp.proj_position_size_modifier,
                proj_risk_appetite_modifier=inp.proj_risk_appetite_modifier,
            )
            scored.append((net_score, row))

        # ── Step 4: sort descending and assign ordinal ranks ───────────────
        scored.sort(key=lambda t: t[0], reverse=True)

        result: list[RankedOpportunity] = []
        for ordinal, (_, row) in enumerate(scored, start=1):
            # Frozen dataclass — reconstruct with the rank field set
            result.append(
                RankedOpportunity(
                    rank=ordinal,
                    **{f: getattr(row, f) for f in row.__dataclass_fields__ if f != "rank"},
                )
            )

        return result

    def top(
        self,
        candidates: list[OpportunityInput],
        n: int = 1,
        actionable_only: bool = True,
        now: float | None = None,
    ) -> list[RankedOpportunity]:
        """Return the top-N ranked candidates, optionally restricted to
        actionable setups (CONFIRMING / READY / EXECUTING).
        """
        ranked = self.rank(candidates, now=now)
        if actionable_only:
            ranked = [r for r in ranked if r.is_actionable]
        return ranked[:n]
