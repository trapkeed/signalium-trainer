"""C13 — Patience Engine: scoring functions.

Two module-level public callables:

    patience_score(state)           → PatienceMetrics
    minimum_quality_threshold(state)→ float

Plus the reward math used internally by evaluate_wait_decision() and
resolve_skipped_opportunity():

    compute_no_action_reward(...)   → float   (prospective — at skip time)
    compute_realized_reward(...)    → float   (retrospective — at resolution)

NO_ACTION Reward design (C13 requirement #2)
--------------------------------------------
The NO_ACTION reward is a signal that "doing nothing was the right choice."

**Prospective reward** (computed at skip time from the wait_reasons alone):

    Base reward for waiting:              +0.20
    Each of these multiplies/adds:
      COGNITIVE_PASSIVE (IGNORE):         +0.20  (market incomprehensible)
      UNCERTAINTY_HIGH:                   +0.15  (reward proportional to excess)
      OVERTRADING:                        +0.15  (penalise frequency, reward restraint)
      CONFIRMATION_PENDING (MONITOR):     +0.10  (setup forming, good discipline)
      LOW_QUALITY:                        +0.05 × quality_gap (proportional to distance)
      IMPULSIVE_GUARD:                    +0.12  (correctly identified recklessness)
    If opportunity_score was very high (≥ 0.65): reward reduced by -0.15
      (the system might be over-cautious; don't over-reward)

    Clamped to [0.0, 1.0].

**Realized reward** (updated at resolution):

    CORRECT_SKIP:        +0.60 + 0.40 × tanh(pnl_avoided / 0.5) → [+0.60, +1.0]
    MISSED_OPPORTUNITY:  -0.40 - 0.40 × tanh(pnl_missed  / 0.5) → [-0.40, -0.80]
    NEUTRAL:             0.0

The tanh scaling means small pnl differences give modest rewards;
large differences push toward the ceiling/floor.

Adaptive threshold design (C13 requirement #6)
-----------------------------------------------
``minimum_quality_threshold(state)`` is a **pure read** of ``state`` — it
does not mutate anything.  ``PatienceEngine._update_threshold()`` is the
one place that mutates ``state.current_threshold``, called after each
resolved skip.

Adjustment rules (applied cumulatively then clamped):
  raise  +THRESHOLD_STEP_UP   if impulsive_entry_rate > 0.30
  raise  +THRESHOLD_STEP_UP   if discipline_score < 0.40
         (being wrong often → raise the bar)
  lower  -THRESHOLD_STEP_DOWN if miss_rate > 0.50
         (missing too many → lower the bar slightly)
  lower  -THRESHOLD_STEP_DOWN if drought: no trades or skips in the
         last DEFAULT_ROLLING_WINDOW worth of signals
         (don't lock the system out of all entries forever)

Result is clamped to [MIN_QUALITY_THRESHOLD, MAX_QUALITY_THRESHOLD].
"""

from __future__ import annotations

import math

from trainer.qde.brain.patience.patience_types import (
    DEFAULT_QUALITY_THRESHOLD,
    MAX_QUALITY_THRESHOLD,
    MIN_QUALITY_THRESHOLD,
    THRESHOLD_STEP_DOWN,
    THRESHOLD_STEP_UP,
    PatienceMetrics,
    PatienceState,
    SkipOutcome,
    WaitReason,
)


# ─────────────────────────────────────────────────────────────────────────────
# Prospective NO_ACTION reward
# ─────────────────────────────────────────────────────────────────────────────

_BASE_WAIT_REWARD       = 0.20
_HIGH_QUALITY_PENALTY   = 0.15   # deducted when opp_score ≥ HIGH_QUALITY_FLOOR
_HIGH_QUALITY_FLOOR     = 0.65


def compute_no_action_reward(
    wait_reasons:       list[WaitReason],
    opportunity_score:  float,
    uncertainty_score:  float = 0.0,
    quality_gap:        float = 0.0,
) -> float:
    """Compute the prospective NO_ACTION reward at skip time.

    A positive value tells the training layer "waiting here was a
    reasonable decision based on the information available."  The signal
    is imperfect (it doesn't know yet whether the skip was correct) but
    it provides an immediate learning gradient rather than waiting until
    the outcome is resolved hours/days later.

    Parameters
    ----------
    wait_reasons:
        The reasons the system chose to wait (subset of ``WaitReason``).
    opportunity_score:
        The opportunity's headline score at skip time.
    uncertainty_score:
        The brain's composite uncertainty score [0, 1].
    quality_gap:
        ``quality_threshold − opportunity_score``.  Positive = below bar.
    """
    reward = _BASE_WAIT_REWARD

    reason_set = set(wait_reasons)

    if WaitReason.COGNITIVE_PASSIVE in reason_set:
        reward += 0.20   # market was incomprehensible → strong basis for waiting

    if WaitReason.UNCERTAINTY_HIGH in reason_set:
        # Extra reward proportional to excess uncertainty above the threshold
        excess = max(0.0, uncertainty_score - 0.55)
        reward += 0.15 + 0.20 * excess

    if WaitReason.OVERTRADING in reason_set:
        reward += 0.15   # actively restraining frequency

    if WaitReason.CONFIRMATION_PENDING in reason_set:
        reward += 0.10   # discipline of waiting for setup maturity

    if WaitReason.IMPULSIVE_GUARD in reason_set:
        reward += 0.12   # correctly identified impulse pattern

    if WaitReason.LOW_QUALITY in reason_set:
        # Proportional to how far below the quality floor the opportunity is
        reward += 0.05 * max(0.0, quality_gap)

    if WaitReason.CAPITAL_UNAVAILABLE in reason_set:
        reward += 0.05   # minor; budget constraint is mechanical not discretionary

    # Penalty for being over-cautious on a high-quality opportunity
    if opportunity_score >= _HIGH_QUALITY_FLOOR:
        reward -= _HIGH_QUALITY_PENALTY

    return max(0.0, min(1.0, reward))


# ─────────────────────────────────────────────────────────────────────────────
# Realized NO_ACTION reward (set on resolution)
# ─────────────────────────────────────────────────────────────────────────────

_REALIZED_CORRECT_BASE  = +0.60
_REALIZED_CORRECT_RANGE = +0.40
_REALIZED_MISSED_BASE   = -0.40
_REALIZED_MISSED_RANGE  = -0.40


def compute_realized_reward(
    outcome:        SkipOutcome,
    pnl_estimate:   float = 0.0,
) -> float:
    """Compute the realized NO_ACTION reward once the outcome is known.

    Replaces the prospective reward stored on the ``SkippedOpportunity``.

    ``pnl_estimate`` semantics:

    * For ``CORRECT_SKIP``: positive value = magnitude of loss avoided
      (e.g. 0.5 = avoided ½ ATR adverse move).
    * For ``MISSED_OPPORTUNITY``: positive value = magnitude of profit missed.
    * For ``NEUTRAL``: unused.
    """
    if outcome == SkipOutcome.CORRECT_SKIP:
        scaling = math.tanh(abs(pnl_estimate) / 0.5)
        return _REALIZED_CORRECT_BASE + _REALIZED_CORRECT_RANGE * scaling

    if outcome == SkipOutcome.MISSED_OPPORTUNITY:
        scaling = math.tanh(abs(pnl_estimate) / 0.5)
        return _REALIZED_MISSED_BASE + _REALIZED_MISSED_RANGE * scaling

    # NEUTRAL or UNRESOLVED
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# patience_score  (public callable)
# ─────────────────────────────────────────────────────────────────────────────


def patience_score(state: PatienceState) -> PatienceMetrics:
    """Compute the patience discipline metrics for one ``PatienceState``.

    Returns an immutable ``PatienceMetrics`` snapshot.  Does not mutate
    ``state``.

    The ``composite_patience_score`` is the headline learning target:

        0.50 × discipline_score
      + 0.30 × (1 − impulsive_entry_rate)
      + 0.20 × (1 − miss_rate)

    Parameters
    ----------
    state:
        The ``PatienceState`` to score.  May belong to a single instrument
        or be an aggregated global state.
    """
    resolved = state.resolved_skips()
    resolved_count = max(1, len(resolved))

    discipline = state.correct_skips / resolved_count
    miss_rate  = state.missed_opportunities / resolved_count

    impulsive_rate = (
        state.impulsive_entries / max(1, state.total_trades)
        if state.total_trades > 0 else 0.0
    )

    composite = (
        0.50 * discipline
        + 0.30 * (1.0 - impulsive_rate)
        + 0.20 * (1.0 - miss_rate)
    )
    composite = max(0.0, min(1.0, composite))

    return PatienceMetrics(
        discipline_score=          round(discipline,     4),
        impulsive_entry_rate=      round(impulsive_rate, 4),
        miss_rate=                 round(miss_rate,      4),
        composite_patience_score=  round(composite,      4),
        total_skips=               state.total_skips,
        total_trades=              state.total_trades,
        correct_skips=             state.correct_skips,
        missed_opportunities=      state.missed_opportunities,
        resolved_skips=            len(resolved),
        overtrading_events=        state.overtrading_events,
    )


# ─────────────────────────────────────────────────────────────────────────────
# minimum_quality_threshold  (public callable)
# ─────────────────────────────────────────────────────────────────────────────


def minimum_quality_threshold(state: PatienceState) -> float:
    """Return the current adaptive minimum opportunity quality threshold.

    Pure read of ``state`` — does not mutate anything.  The actual
    adaptation (adjusting ``state.current_threshold``) is performed by
    ``PatienceEngine._update_threshold()`` and is called automatically
    each time a skipped opportunity is resolved.

    When ``state`` has no history (new instrument, no resolved skips),
    returns ``state.current_threshold`` unchanged (which starts at
    ``DEFAULT_QUALITY_THRESHOLD``).

    Adjustment logic (applied to ``state.current_threshold``):

    +THRESHOLD_STEP_UP   if impulsive_entry_rate > 0.30
    +THRESHOLD_STEP_UP   if discipline_score < 0.40  (wrong skips dominate)
    -THRESHOLD_STEP_DOWN if miss_rate > 0.50          (too many missed entries)
    -THRESHOLD_STEP_DOWN if drought                   (no recent activity)

    Clamped to [MIN_QUALITY_THRESHOLD, MAX_QUALITY_THRESHOLD].
    """
    resolved = state.resolved_skips()
    resolved_count = max(1, len(resolved))

    # If no history, return the stored current value as-is.
    if len(resolved) == 0 and state.total_trades == 0:
        return max(
            MIN_QUALITY_THRESHOLD,
            min(MAX_QUALITY_THRESHOLD, state.current_threshold),
        )

    discipline   = state.correct_skips   / resolved_count
    miss_rate_v  = state.missed_opportunities / resolved_count
    impulsive_r  = state.impulsive_entries / max(1, state.total_trades)

    adjustment = 0.0

    if impulsive_r > 0.30:
        adjustment += THRESHOLD_STEP_UP    # impulsive entries → raise bar

    if discipline < 0.40 and len(resolved) >= 5:
        adjustment += THRESHOLD_STEP_UP    # poor skip accuracy → raise bar

    if miss_rate_v > 0.50 and len(resolved) >= 5:
        adjustment -= THRESHOLD_STEP_DOWN  # missing too many → lower bar slightly

    # Drought detection: if all recent wait_flags are True (only waiting),
    # the threshold may be too high — give a small downward nudge.
    if len(state.recent_wait_flags) >= 20:
        recent_wait_rate = sum(state.recent_wait_flags) / len(state.recent_wait_flags)
        if recent_wait_rate > 0.90:
            adjustment -= THRESHOLD_STEP_DOWN

    proposed = state.current_threshold + adjustment
    return max(MIN_QUALITY_THRESHOLD, min(MAX_QUALITY_THRESHOLD, proposed))


__all__ = [
    "compute_no_action_reward",
    "compute_realized_reward",
    "patience_score",
    "minimum_quality_threshold",
]
