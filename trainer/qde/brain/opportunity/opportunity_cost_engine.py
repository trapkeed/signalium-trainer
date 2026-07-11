"""QDE Brain — opportunity: cost of capital and cross-symbol efficiency.

``OpportunityCostEngine`` answers one question the existing per-symbol
engines cannot: **if we commit capital and attention to opportunity X,
what do we forgo?**

Opportunity cost is meaningful only in a cross-symbol context — it
requires seeing *all* live candidates simultaneously to establish what
the best alternative is.  This module is therefore deliberately
stateless (no per-symbol memory of its own) and always operates on the
full current candidate set in a single pass.

What this module does
---------------------
1. Derives ``capital_efficiency`` for each candidate — expected reward
   (ATR multiples) earned per hour of capital deployment.  This makes
   duration a first-class citizen: a 2-ATR trade that closes in 30
   minutes beats a 2-ATR trade that locks capital for 8 hours.

2. Computes ``opportunity_cost_score`` (0-1): how much of the best
   available setup's efficiency does this candidate forgo?  0 = this IS
   the best candidate; 1 = taking this means missing the best entirely.

3. Ranks all candidates by capital_efficiency so the ranker can use the
   efficiency rank as one input without re-deriving it.

What this module does NOT do
-----------------------------
* Does not recompute conviction, uncertainty, or projection estimates —
  those are read from ``OpportunityInput`` fields that callers populate
  from the existing per-symbol engines (``ConvictionScore.conviction``,
  ``UncertaintyScore.composite``, ``ExpectedProjection.*``).
* Does not import ``QDEDecision`` or any execution-layer type.
* Does not decide whether to enter a trade — that boundary stays with
  ``qde_inference``.

``OpportunityInput`` — the input bag
--------------------------------------
A flat dataclass that bundles everything the three cross-symbol engines
need from the existing per-symbol outputs.  Callers construct one per
active opportunity from signals already in hand:

    inp = OpportunityInput(
        instrument   = opp.instrument,
        direction    = 1,
        opportunity_id = opp.opportunity_id,
        setup_state  = opp.state.value,
        conviction   = conv_score.conviction,
        confidence   = conv_score.confidence,
        conviction_profile = conv_score.profile.value,
        uncertainty  = unc_score.composite,
        uncertainty_level  = unc_score.level.value,
        expected_reward_atr  = proj.expected_reward_atr,
        expected_mae_atr     = proj.expected_mae_atr,
        expected_duration_seconds = proj.expected_duration_seconds,
        reward_sufficiency_score  = proj.reward_sufficiency_score,
        reward_risk_ratio         = proj.reward_risk_ratio,
        proj_opportunity_rank_signal = proj.opportunity_rank_signal,
        proj_position_size_modifier  = proj.position_size_modifier,
        proj_risk_appetite_modifier  = proj.risk_appetite_modifier,
        attention_weight   = attn_score.weight,
        has_open_position  = inp.has_open_position,
        confirmation_ticks = opp.confirmation_ticks,
        state_dwell_seconds = opp.dwell_seconds(),
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

# Minimum expected duration used as the denominator when computing
# capital efficiency — avoids divide-by-zero on very short setups and
# ensures a sub-quarter-hour setup still gets a finite (high) efficiency.
_MIN_DURATION_HOURS = 0.25

# Seconds → hours conversion
_S_PER_HOUR = 3600.0


# ── OpportunityInput ──────────────────────────────────────────────────────────


@dataclass
class OpportunityInput:
    """Flat bag of per-symbol signals needed by the cross-symbol engines.

    All scalar fields are already-computed values from the existing
    per-symbol engines.  Callers never re-derive them here.

    Direction convention: +1 = long, −1 = short (same as throughout
    the rest of the QDE).  ``setup_state`` is the string value of
    ``SetupState`` so this module has no hard import dependency on
    ``setup_lifecycle``.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    instrument:             str   = ""
    direction:              int   = 1      # +1 long, −1 short
    opportunity_id:         str   = ""
    setup_state:            str   = ""    # SetupState.value

    # ── Conviction / confidence — from ConvictionScore ─────────────────
    conviction:             float = 0.0   # 0-1
    confidence:             float = 0.0   # 0-1 (P(hypothesis correct))
    conviction_profile:     str   = ""    # ConvictionProfile.value

    # ── Uncertainty — from UncertaintyScore ────────────────────────────
    uncertainty:            float = 0.5   # 0-1 (higher = more ambiguous)
    uncertainty_level:      str   = ""    # UncertaintyLevel.value

    # ── Projection — from ExpectedProjection ───────────────────────────
    expected_reward_atr:    float = 0.0   # signed, ATR multiples
    expected_mae_atr:       float = 0.0   # non-negative
    expected_duration_seconds: float = 0.0
    reward_sufficiency_score:  float = 0.5  # 0-1, learned percentile
    reward_risk_ratio:         float = 0.0  # reward / mae
    proj_opportunity_rank_signal: float = 0.0
    proj_position_size_modifier:  float = 1.0
    proj_risk_appetite_modifier:  float = 0.0

    # ── Attention — from AttentionScoreAllocator ────────────────────────
    attention_weight: float = 0.0   # normalized cross-symbol weight

    # ── Execution friction — C11 additions ─────────────────────────────
    # These four fields are optional (all default to neutral values) so
    # existing callers that do not yet populate them continue to work
    # unchanged.  Callers that have execution-engine context should
    # populate them from ``SlippagePredictor`` / ``FillProbabilityEngine``
    # and the MIE liquidity signal.
    spread_cost_atr:       float = 0.0   # gross spread expressed in ATR units
    slippage_estimate_atr: float = 0.0   # estimated execution slippage in ATR
    liquidity_quality:     float = 0.5   # 0-1, from MIE/liquidity layer
    portfolio_interaction: float = 0.0   # −1 = concentrating, +1 = diversifying

    # ── Position context ────────────────────────────────────────────────
    has_open_position:  bool  = False
    confirmation_ticks: int   = 0
    state_dwell_seconds: float = 0.0

    @property
    def expected_duration_hours(self) -> float:
        return self.expected_duration_seconds / _S_PER_HOUR

    @property
    def is_actionable(self) -> bool:
        """True when the setup state indicates the opportunity is ready
        for execution consideration (READY or EXECUTING).  WATCHING /
        ARMED / CONFIRMING are not yet actionable for capital allocation.
        """
        return self.setup_state in ("READY", "EXECUTING", "CONFIRMING")

    def to_dict(self) -> dict:
        return {
            "instrument":                 self.instrument,
            "direction":                  self.direction,
            "opportunity_id":             self.opportunity_id,
            "setup_state":                self.setup_state,
            "conviction":                 round(self.conviction, 4),
            "confidence":                 round(self.confidence, 4),
            "conviction_profile":         self.conviction_profile,
            "uncertainty":                round(self.uncertainty, 4),
            "uncertainty_level":          self.uncertainty_level,
            "expected_reward_atr":        round(self.expected_reward_atr, 4),
            "expected_mae_atr":           round(self.expected_mae_atr, 4),
            "expected_duration_seconds":  round(self.expected_duration_seconds, 1),
            "reward_sufficiency_score":   round(self.reward_sufficiency_score, 4),
            "reward_risk_ratio":          round(self.reward_risk_ratio, 4),
            "proj_opportunity_rank_signal": round(self.proj_opportunity_rank_signal, 4),
            "proj_position_size_modifier":  round(self.proj_position_size_modifier, 4),
            "proj_risk_appetite_modifier":  round(self.proj_risk_appetite_modifier, 4),
            "attention_weight":           round(self.attention_weight, 4),
            # C11 execution-friction fields
            "spread_cost_atr":            round(self.spread_cost_atr, 4),
            "slippage_estimate_atr":      round(self.slippage_estimate_atr, 4),
            "liquidity_quality":          round(self.liquidity_quality, 4),
            "portfolio_interaction":      round(self.portfolio_interaction, 4),
            "has_open_position":          self.has_open_position,
            "confirmation_ticks":         self.confirmation_ticks,
            "state_dwell_seconds":        round(self.state_dwell_seconds, 1),
        }


# ── OpportunityCost ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpportunityCost:
    """Capital-efficiency and opportunity-cost metrics for one candidate.

    ``capital_efficiency``
        Expected reward (ATR multiples) earned per hour of deployment.
        Instrument-agnostic because the numerator is already in ATR units.

    ``opportunity_cost_score`` (0-1)
        What fraction of the *best* available candidate's efficiency this
        candidate forgoes.  0 = this candidate IS the best; 1 = taking
        this means missing the entire value of the best alternative.
        Callers should interpret this as "the price paid, in efficiency
        units, for choosing this over the best".

    ``forgone_efficiency``
        ``best_capital_efficiency - this_capital_efficiency``, in the
        same ATR/hour units as ``capital_efficiency``.  Negative when
        this candidate IS the best.

    ``efficiency_rank``
        Ordinal rank by ``capital_efficiency`` descending (1 = most
        efficient).  Ties broken by insertion order.
    """

    instrument:              str   = ""
    opportunity_id:          str   = ""
    capital_efficiency:      float = 0.0   # ATR/hour
    opportunity_cost_score:  float = 0.0   # 0-1
    forgone_efficiency:      float = 0.0   # ATR/hour (best - this)
    efficiency_rank:         int   = 1

    def to_dict(self) -> dict:
        return {
            "instrument":             self.instrument,
            "opportunity_id":         self.opportunity_id,
            "capital_efficiency":     round(self.capital_efficiency, 4),
            "opportunity_cost_score": round(self.opportunity_cost_score, 4),
            "forgone_efficiency":     round(self.forgone_efficiency, 4),
            "efficiency_rank":        self.efficiency_rank,
        }


# ── OpportunityCostEngine ─────────────────────────────────────────────────────


def _capital_efficiency(inp: OpportunityInput) -> float:
    """Expected reward per hour of deployment, floored at 0.

    Negative expected_reward_atr (the trade is historically a loser)
    yields a non-positive efficiency — such candidates will naturally
    rank last.
    """
    duration_h = max(inp.expected_duration_hours, _MIN_DURATION_HOURS)
    return inp.expected_reward_atr / duration_h


class OpportunityCostEngine:
    """Stateless cross-symbol capital-efficiency and opportunity-cost engine.

    ``compute()`` accepts the full set of current candidates and returns
    one ``OpportunityCost`` per candidate in a single O(N) pass.

    The engine is intentionally stateless so it can be called on every
    brain tick with a fresh candidate set without accumulating stale
    cost estimates from previous ticks.
    """

    def compute(
        self,
        candidates: list[OpportunityInput],
    ) -> dict[str, OpportunityCost]:
        """Compute opportunity cost for every candidate.

        Parameters
        ----------
        candidates:
            All currently active ``OpportunityInput`` records across all
            monitored symbols.  A single-element list still works
            correctly (opportunity cost is zero when there is no
            alternative).

        Returns
        -------
        ``dict[instrument → OpportunityCost]``
            Keyed by ``instrument``.  If two candidates share the same
            instrument (e.g. both a long and a short are queued), the
            better-ranked one is kept.
        """
        if not candidates:
            return {}

        # ── Compute raw capital efficiency for every candidate ─────────
        efficiencies: list[tuple[OpportunityInput, float]] = [
            (inp, _capital_efficiency(inp)) for inp in candidates
        ]

        best_eff = max(eff for _, eff in efficiencies) if efficiencies else 0.0
        best_eff = max(best_eff, 1e-9)   # avoid /0 when all candidates have 0 reward

        # ── Sort by efficiency descending to assign ranks ─────────────
        ranked = sorted(efficiencies, key=lambda t: t[1], reverse=True)

        result: dict[str, OpportunityCost] = {}
        for rank_idx, (inp, eff) in enumerate(ranked, start=1):
            forgone       = best_eff - eff
            cost_score    = max(0.0, min(1.0, forgone / best_eff))
            cost = OpportunityCost(
                instrument=inp.instrument,
                opportunity_id=inp.opportunity_id,
                capital_efficiency=round(eff, 6),
                opportunity_cost_score=round(cost_score, 4),
                forgone_efficiency=round(forgone, 6),
                efficiency_rank=rank_idx,
            )
            # Keep the higher-efficiency entry when instrument is duplicated
            if inp.instrument not in result or eff > result[inp.instrument].capital_efficiency:
                result[inp.instrument] = cost

        return result
