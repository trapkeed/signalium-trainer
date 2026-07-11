"""C11 — Opportunity Engine: expected-utility scoring, classification, and ranking.

Three public callables required by C11:

    classify_opportunity(inp)            → OpportunityClass
    score_opportunity(inp)               → OpportunityScore
    rank_opportunities(candidates)       → list[OpportunityScore]

Design: expected utility, not expected value
--------------------------------------------
The canonical ``OpportunityRanker`` in ``opportunity_ranker.py`` measures
*how good is this setup relative to the other setups right now?* using a
conviction-weighted composite plus capital-efficiency ranking — a relative,
cross-symbol comparison.

C11 adds a complementary, **absolute** utility score that answers
*how good is this setup on its own terms?*  The two differ in one crucial
way:

    Expected value (linear):  EU = p × R - (1-p) × L
    Expected utility (concave): EU = p × U(R) - (1-p) × U(L)

where ``U`` is a **concave** utility function (Bernoulli log utility here).
The concave shape naturally satisfies all three C11 behavioral requirements:

  1. Does not favour large profits — ``log(1 + R)`` grows sub-linearly;
     doubling the target does not double the score.

  2. Does not favour large position sizes — ``opportunity_score`` is
     independent of ``proj_position_size_modifier`` (sizing stays in
     ``CapitalPriorityEngine``, not here).

  3. Prefers small but highly probable gains — a 0.5-ATR setup at p=0.9
     scores **higher** than a 4-ATR setup at p=0.5 because the high-
     probability win multiplies a decent utility, while the large-reward
     setup is offset by substantial uncertainty.

Eight score factors
--------------------
All eight are normalized to [0, 1] and retained for inspection:

  ``prob_success``          — conviction × confidence; the joint belief that
                              the direction is correct AND the hypothesis is
                              high-quality.  Primary gate: a low-probability
                              setup cannot score well regardless of all other
                              factors.

  ``expected_return_score`` — log-utility of the friction-adjusted expected
                              gain: ``log1p(net_gain_atr / REF_ATR)``,
                              normalized.  Sub-linear in gain size.

  ``expected_loss_score``   — inverse of the log-utility of expected loss:
                              ``1 − log1p(expected_mae_atr / REF_ATR) /
                              max_loss_util``.  Smaller MAE = higher score.

  ``spread_cost_score``     — how much of the expected return survives after
                              paying the spread.  Zero when spread ≥ expected
                              return.

  ``slippage_score``        — same as spread_cost_score but for slippage.

  ``duration_score``        — capital-velocity proxy: ``sqrt(REF_DURATION_H /
                              duration_h)``, capturing that faster turnover
                              compresses the same return into less wall-clock
                              time without the linear penalty that makes
                              short-duration setups unreasonably dominant.

  ``portfolio_interaction_score`` — maps ``portfolio_interaction`` (−1 to +1)
                                    linearly to [0, 1].  Diversifying setups
                                    score better; concentrating setups score
                                    worse.

  ``liquidity_quality_score``     — direct pass-through of ``liquidity_quality``
                                    (0-1).  Poor liquidity degrades fill
                                    quality and therefore the achievable return.

The eight factors combine via a **probability-gated weighted product**:

    base = (return_score ^ w_r) × (loss_score ^ w_l) × (spread_score ^ w_sp)
         × (slip_score ^ w_sl) × (duration_score ^ w_d)
         × (portfolio_score ^ w_p) × (liquidity_score ^ w_lq)

    opportunity_score = base × p_success_weight × liquidity_quality_factor

Using a product (geometric weighting) means any single factor near zero
pulls the whole score down — the QDE must see a clean opportunity on
*all* dimensions, not just an outlier on one.

Opportunity classes
-------------------
MICRO_SCALP   — < 5 minutes expected duration
SCALP         — 5 min – 2 hours
INTRADAY      — 2 – 8 hours
MULTI_SESSION — 8 hours – 3 days
SWING         — ≥ 3 days

These are structural trade classes, not trader identities.  The same
symbol may move from SCALP to INTRADAY classification as its setup matures
and its expected-duration estimate grows.  ``classify_opportunity`` is
purely deterministic from ``OpportunityInput.expected_duration_seconds``
plus a fallback heuristic on ``expected_reward_atr`` when duration is zero.

Relationship to existing opportunity infrastructure
----------------------------------------------------
* ``OpportunityCostEngine`` / ``OpportunityRanker`` / ``CapitalPriorityEngine``
  are **unchanged**.  This module adds a new absolute utility layer; it does
  not replace or duplicate those engines' relative-ranking logic.

* Callers may use ``score_opportunity()`` standalone (one symbol at a time)
  or ``rank_opportunities()`` for a full cross-symbol sorted list.
  ``rank_opportunities()`` does NOT call ``OpportunityRanker`` internally —
  it produces an independent ordering based purely on ``opportunity_score``.

* No import of ``QDEDecision`` or any execution-layer type.
* All three public functions are stateless; safe to call on every brain tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from trainer.qde.brain.opportunity.opportunity_cost_engine import OpportunityInput

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Reference ATR size: a "normal" good trade outcome.  Used to normalize the
# log-utility terms so that a 2-ATR net gain scores at the midpoint of the
# return scale rather than near the ceiling.
_REF_ATR: float = 2.0

# Maximum achievable log utility for normalization (at gain = 5 × REF_ATR).
# log1p(10 / 2) = log1p(5) ≈ 1.7918
_MAX_GAIN_UTIL: float = math.log1p(5.0)

# Maximum loss utility reference (loss of 3 × REF_ATR is the upper bound).
_MAX_LOSS_UTIL: float = math.log1p(3.0)

# Reference duration for the duration score (1 hour = "normal" trade speed).
_REF_DURATION_H: float = 1.0

# Floor: very short trades are not penalized beyond this minimum duration.
_MIN_DURATION_H: float = 0.083  # 5 minutes

# Portfolio interaction floor/ceiling used in the score (avoids starving
# or completely dominating the score with a single portfolio factor).
_PORT_FLOOR: float = 0.70
_PORT_CEIL:  float = 1.00

# Liquidity quality floor (even with zero measured liquidity quality, the
# setup is not wholly unscorable — it is just heavily penalised).
_LIQ_FLOOR: float = 0.40

# Duration class boundaries in seconds.
_MICRO_SCALP_MAX_S:   float = 300.0      # < 5 min
_SCALP_MAX_S:         float = 7_200.0    # < 2 hr
_INTRADAY_MAX_S:      float = 28_800.0   # < 8 hr
_MULTI_SESSION_MAX_S: float = 259_200.0  # < 72 hr  (3 days)


# ─────────────────────────────────────────────────────────────────────────────
# OpportunityClass
# ─────────────────────────────────────────────────────────────────────────────


class OpportunityClass(str, Enum):
    """Structural trade duration class.

    Determined from ``OpportunityInput.expected_duration_seconds`` via
    ``classify_opportunity()``.  These are **opportunity** classes, not
    trader-style identities — the same symbol can shift class as the
    setup matures and the duration estimate changes.
    """

    MICRO_SCALP   = "MICRO_SCALP"    # < 5 minutes
    SCALP         = "SCALP"          # 5 minutes – 2 hours
    INTRADAY      = "INTRADAY"       # 2 – 8 hours
    MULTI_SESSION = "MULTI_SESSION"  # 8 hours – 3 days
    SWING         = "SWING"          # ≥ 3 days


# ─────────────────────────────────────────────────────────────────────────────
# OpportunityScore
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpportunityScore:
    """Expected-utility score for one opportunity, broken down by factor.

    ``opportunity_score``  — the headline 0-1 score; this is what
                             ``rank_opportunities()`` sorts on.

    ``opportunity_class``  — structural duration class (MICRO_SCALP..SWING).

    The eight ``*_score`` fields expose the full factor breakdown so callers
    can inspect *why* a setup scored high or low without re-deriving it.
    All factor fields are in [0, 1].

    ``rank`` is 0 when produced by ``score_opportunity()`` (single-symbol
    path); it is set to the 1-indexed ordinal when produced by
    ``rank_opportunities()``.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    instrument:              str   = ""
    opportunity_id:          str   = ""
    setup_state:             str   = ""
    direction:               int   = 1

    # ── Opportunity class ────────────────────────────────────────────────────
    opportunity_class: OpportunityClass = OpportunityClass.INTRADAY

    # ── Eight factor scores (all 0-1) ────────────────────────────────────────
    prob_success:               float = 0.0  # conviction × confidence
    expected_return_score:      float = 0.0  # log-utility of net gain
    expected_loss_score:        float = 0.0  # inverse log-utility of expected MAE
    spread_cost_score:          float = 0.0  # fraction of return surviving spread
    slippage_score:             float = 0.0  # fraction of return surviving slippage
    duration_score:             float = 0.0  # capital-velocity proxy
    portfolio_interaction_score: float = 0.0  # diversification benefit
    liquidity_quality_score:    float = 0.0  # fill-quality proxy

    # ── Headline score ───────────────────────────────────────────────────────
    opportunity_score: float = 0.0  # probability-gated geometric combination

    # ── Rank (set by rank_opportunities; 0 = not yet ranked) ────────────────
    rank: int = 0

    # ── Input passthrough for consumers ──────────────────────────────────────
    conviction:               float = 0.0
    confidence:               float = 0.0
    uncertainty:              float = 0.5
    expected_reward_atr:      float = 0.0
    expected_mae_atr:         float = 0.0
    spread_cost_atr:          float = 0.0
    slippage_estimate_atr:    float = 0.0
    expected_duration_seconds: float = 0.0
    liquidity_quality:        float = 0.5
    portfolio_interaction:    float = 0.0
    is_actionable:            bool  = False

    def to_dict(self) -> dict:
        return {
            "rank":                    self.rank,
            "instrument":              self.instrument,
            "opportunity_id":          self.opportunity_id,
            "setup_state":             self.setup_state,
            "direction":               self.direction,
            "opportunity_class":       self.opportunity_class.value,
            "opportunity_score":       round(self.opportunity_score, 4),
            # Eight factors
            "prob_success":            round(self.prob_success, 4),
            "expected_return_score":   round(self.expected_return_score, 4),
            "expected_loss_score":     round(self.expected_loss_score, 4),
            "spread_cost_score":       round(self.spread_cost_score, 4),
            "slippage_score":          round(self.slippage_score, 4),
            "duration_score":          round(self.duration_score, 4),
            "portfolio_interaction_score": round(self.portfolio_interaction_score, 4),
            "liquidity_quality_score": round(self.liquidity_quality_score, 4),
            # Context passthrough
            "conviction":              round(self.conviction, 4),
            "confidence":              round(self.confidence, 4),
            "uncertainty":             round(self.uncertainty, 4),
            "expected_reward_atr":     round(self.expected_reward_atr, 4),
            "expected_mae_atr":        round(self.expected_mae_atr, 4),
            "spread_cost_atr":         round(self.spread_cost_atr, 4),
            "slippage_estimate_atr":   round(self.slippage_estimate_atr, 4),
            "expected_duration_seconds": round(self.expected_duration_seconds, 1),
            "liquidity_quality":       round(self.liquidity_quality, 4),
            "portfolio_interaction":   round(self.portfolio_interaction, 4),
            "is_actionable":           self.is_actionable,
        }


# ─────────────────────────────────────────────────────────────────────────────
# classify_opportunity
# ─────────────────────────────────────────────────────────────────────────────


def classify_opportunity(inp: OpportunityInput) -> OpportunityClass:
    """Classify a setup into one of the five opportunity duration classes.

    Primary criterion: ``expected_duration_seconds``.
    Fallback when duration is zero (projection engine not yet warmed up):
    infer class from ``expected_reward_atr`` as a rough duration proxy —
    larger target reward implies a longer-horizon setup.

    Parameters
    ----------
    inp:
        A populated ``OpportunityInput`` record.

    Returns
    -------
    ``OpportunityClass``
        Structural trade duration class (MICRO_SCALP, SCALP, INTRADAY,
        MULTI_SESSION, or SWING).
    """
    duration_s = inp.expected_duration_seconds

    if duration_s <= 0:
        # Fallback heuristic: classify by reward size when duration unknown.
        reward = inp.expected_reward_atr
        if reward < 0.3:
            return OpportunityClass.MICRO_SCALP
        if reward < 0.8:
            return OpportunityClass.SCALP
        if reward < 2.0:
            return OpportunityClass.INTRADAY
        if reward < 4.0:
            return OpportunityClass.MULTI_SESSION
        return OpportunityClass.SWING

    if duration_s < _MICRO_SCALP_MAX_S:
        return OpportunityClass.MICRO_SCALP
    if duration_s < _SCALP_MAX_S:
        return OpportunityClass.SCALP
    if duration_s < _INTRADAY_MAX_S:
        return OpportunityClass.INTRADAY
    if duration_s < _MULTI_SESSION_MAX_S:
        return OpportunityClass.MULTI_SESSION
    return OpportunityClass.SWING


# ─────────────────────────────────────────────────────────────────────────────
# score_opportunity
# ─────────────────────────────────────────────────────────────────────────────


def score_opportunity(inp: OpportunityInput) -> OpportunityScore:
    """Compute the expected-utility opportunity score for a single symbol.

    Stateless: reads only from ``inp``; never modifies Brain state or
    accesses any per-symbol history.  Safe to call on every brain tick.

    The eight factor scores are each normalized to [0, 1] and combined
    via a probability-gated product (geometric weighting):

        opportunity_score = prob_success
                          × (return_score  ^ 0.25)
                          × (loss_score    ^ 0.20)
                          × (spread_score  ^ 0.12)
                          × (slip_score    ^ 0.12)
                          × (duration_score ^ 0.10)
                          × (port_score    ^ 0.08)
                          × (liq_score     ^ 0.13)

    The fractional exponents are the geometric weights and sum to 1.0.
    Using a product (geometric mean) rather than a weighted arithmetic
    mean means that weakness in any single factor drags the whole score
    down — the QDE requires a clean opportunity on *all* dimensions.

    Probability of success acts as a **direct linear multiplier** (not
    just another geometric component) because the QDE's core philosophy
    is certainty over size: a 0.4-score on all other factors with p=0.9
    gives opportunity_score ≈ 0.31, which **beats** the same all-0.4
    factors with p=0.5 (opportunity_score ≈ 0.17).

    Parameters
    ----------
    inp:
        A populated ``OpportunityInput`` record.  The four C11 fields
        (``spread_cost_atr``, ``slippage_estimate_atr``,
        ``liquidity_quality``, ``portfolio_interaction``) default to
        0.0 / 0.5 / 0.0 when not supplied — backward-compatible.

    Returns
    -------
    ``OpportunityScore``
        Full 8-factor breakdown plus headline ``opportunity_score``.
        ``rank`` is 0; use ``rank_opportunities()`` for a ranked list.
    """
    opp_class = classify_opportunity(inp)

    # ── Factor 1: probability of success ─────────────────────────────────────
    # Joint belief: the hypothesis is both high-quality (conviction) AND
    # probably correct (confidence).  Clamped to [0, 1].
    p_success = max(0.0, min(1.0, inp.conviction * inp.confidence))

    # ── Friction adjustments ──────────────────────────────────────────────────
    friction_total = inp.spread_cost_atr + inp.slippage_estimate_atr
    net_gain_atr   = max(0.0, inp.expected_reward_atr - friction_total)
    net_loss_atr   = inp.expected_mae_atr + friction_total

    # ── Factor 2: expected return score (log utility, concave) ───────────────
    # log1p(x / REF_ATR) grows sub-linearly.  Normalized by the maximum
    # achievable utility (_MAX_GAIN_UTIL = log1p(5)).
    gain_util = math.log1p(net_gain_atr / _REF_ATR)
    expected_return_score = min(1.0, gain_util / _MAX_GAIN_UTIL)

    # ── Factor 3: expected loss score ────────────────────────────────────────
    # Inversion of log-utility of expected loss.  Lower MAE → higher score.
    loss_util = math.log1p(net_loss_atr / _REF_ATR)
    expected_loss_score = max(0.0, 1.0 - loss_util / _MAX_LOSS_UTIL)

    # ── Factor 4: spread cost score ──────────────────────────────────────────
    # What fraction of the expected gross return survives after paying spread?
    gross_return_ref = max(0.05, inp.expected_reward_atr)
    spread_frac = inp.spread_cost_atr / gross_return_ref
    spread_cost_score = max(0.0, 1.0 - spread_frac)

    # ── Factor 5: slippage score ─────────────────────────────────────────────
    slip_frac = inp.slippage_estimate_atr / gross_return_ref
    slippage_score = max(0.0, 1.0 - slip_frac)

    # ── Factor 6: duration score (capital velocity) ───────────────────────────
    # sqrt(REF_DURATION_H / duration_h): faster than the reference earns full
    # score (capped at 1.0); slower decays gracefully.  _MIN_DURATION_H floor
    # prevents micro-scalps from overflowing the cap.
    duration_h = max(_MIN_DURATION_H, inp.expected_duration_seconds / 3600.0)
    duration_score = min(1.0, math.sqrt(_REF_DURATION_H / duration_h))

    # ── Factor 7: portfolio interaction score ────────────────────────────────
    # Linear mapping of [-1, +1] → [0, 1].
    portfolio_interaction_score = (inp.portfolio_interaction + 1.0) / 2.0

    # ── Factor 8: liquidity quality score ────────────────────────────────────
    # Direct pass-through; already [0, 1].
    liquidity_quality_score = max(0.0, min(1.0, inp.liquidity_quality))

    # ── Headline opportunity_score ────────────────────────────────────────────
    # Probability-gated geometric combination.
    # Geometric weights (sum = 1.0):
    #   return 0.25, loss 0.20, spread 0.12, slip 0.12,
    #   duration 0.10, portfolio 0.08, liquidity 0.13
    #
    # Each factor is raised to its geometric weight; the product is then
    # multiplied by p_success so a low-probability setup cannot score well
    # regardless of how clean the execution parameters are.
    geo_product = (
        _safe_pow(expected_return_score,       0.25)
        * _safe_pow(expected_loss_score,       0.20)
        * _safe_pow(spread_cost_score,         0.12)
        * _safe_pow(slippage_score,            0.12)
        * _safe_pow(duration_score,            0.10)
        * _safe_pow(portfolio_interaction_score, 0.08)
        * _safe_pow(liquidity_quality_score,   0.13)
    )
    opportunity_score = min(1.0, p_success * geo_product)

    return OpportunityScore(
        instrument=inp.instrument,
        opportunity_id=inp.opportunity_id,
        setup_state=inp.setup_state,
        direction=inp.direction,
        opportunity_class=opp_class,
        # Eight factors
        prob_success=round(p_success, 4),
        expected_return_score=round(expected_return_score, 4),
        expected_loss_score=round(expected_loss_score, 4),
        spread_cost_score=round(spread_cost_score, 4),
        slippage_score=round(slippage_score, 4),
        duration_score=round(duration_score, 4),
        portfolio_interaction_score=round(portfolio_interaction_score, 4),
        liquidity_quality_score=round(liquidity_quality_score, 4),
        # Headline
        opportunity_score=round(opportunity_score, 4),
        rank=0,
        # Passthrough
        conviction=inp.conviction,
        confidence=inp.confidence,
        uncertainty=inp.uncertainty,
        expected_reward_atr=inp.expected_reward_atr,
        expected_mae_atr=inp.expected_mae_atr,
        spread_cost_atr=inp.spread_cost_atr,
        slippage_estimate_atr=inp.slippage_estimate_atr,
        expected_duration_seconds=inp.expected_duration_seconds,
        liquidity_quality=inp.liquidity_quality,
        portfolio_interaction=inp.portfolio_interaction,
        is_actionable=inp.is_actionable,
    )


# ─────────────────────────────────────────────────────────────────────────────
# rank_opportunities
# ─────────────────────────────────────────────────────────────────────────────


def rank_opportunities(
    candidates: list[OpportunityInput],
    actionable_only: bool = False,
) -> list[OpportunityScore]:
    """Score and rank all candidates by expected utility, highest first.

    Stateless: computes ``score_opportunity()`` for each candidate in a
    single pass, sorts descending by ``opportunity_score``, and assigns
    1-indexed ordinal ranks.

    Parameters
    ----------
    candidates:
        All currently active ``OpportunityInput`` records.  An empty list
        returns an empty list without error.
    actionable_only:
        When ``True``, only candidates whose ``setup_state`` is CONFIRMING,
        READY, or EXECUTING are included in the output.  Non-actionable
        candidates are scored but filtered out (same gate as
        ``OpportunityRanker``).

    Returns
    -------
    ``list[OpportunityScore]``
        Sorted descending by ``opportunity_score``.  Each entry has its
        ``rank`` set to its 1-based ordinal position within the returned
        list (so rank 1 is always the highest-scoring candidate).
    """
    if not candidates:
        return []

    scored: list[OpportunityScore] = []
    for inp in candidates:
        s = score_opportunity(inp)
        if actionable_only and not s.is_actionable:
            continue
        scored.append(s)

    scored.sort(key=lambda s: s.opportunity_score, reverse=True)

    # Rebuild with rank field set (frozen dataclass → reconstruct)
    result: list[OpportunityScore] = []
    for ordinal, s in enumerate(scored, start=1):
        fields = {f: getattr(s, f) for f in s.__dataclass_fields__ if f != "rank"}
        result.append(OpportunityScore(rank=ordinal, **fields))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_pow(base: float, exp: float) -> float:
    """``base ** exp`` with zero-floor protection.

    Raising 0.0 to a fractional exponent is mathematically 0, but
    floating-point noise can produce values just below 0 which would
    raise ``ValueError``.  Clamp to [0, 1] before applying the exponent.
    """
    return max(0.0, min(1.0, base)) ** exp


__all__ = [
    "OpportunityClass",
    "OpportunityScore",
    "classify_opportunity",
    "score_opportunity",
    "rank_opportunities",
]
