"""C13 — Patience Engine: maximise opportunity quality, not trade quantity.

The Patience Engine teaches the QDE that **doing nothing is a valid decision**.
It rewards correct inaction (avoiding bad trades, waiting for confirmation,
preserving capital) and penalises impulsive entries and overtrading.

Three public callables (C13 requirement)
-----------------------------------------
``evaluate_wait_decision(cognitive_decision, opp_score, ...) → WaitDecision``
    Stateless.  Evaluates whether the system should wait given the current
    cognitive verdict and opportunity score.  Collects all applicable
    ``WaitReason`` values (COGNITIVE_PASSIVE, UNCERTAINTY_HIGH, OVERTRADING,
    CONFIRMATION_PENDING, IMPULSIVE_GUARD, LOW_QUALITY, CAPITAL_UNAVAILABLE).
    Returns a ``WaitDecision`` with a prospective ``NO_ACTION`` reward signal
    and a pre-built ``SkippedOpportunity`` record ready for persistence.

``patience_score(state) → PatienceMetrics``
    Stateless.  Computes the patience discipline metrics from a
    ``PatienceState``.  The headline ``composite_patience_score`` (0–1) is
    the learning target:
        0.50 × discipline_score (fraction of skips that were correct)
      + 0.30 × (1 − impulsive_entry_rate)
      + 0.20 × (1 − miss_rate)

``minimum_quality_threshold(state) → float``
    Stateless.  Returns the current adaptive minimum opportunity score
    required before the system should consider committing to an entry.
    Starts at 0.20 (= _TIER_3_MIN from capital_priority_engine) and adapts:
    • Raises when impulsive entries or poor skip accuracy dominate.
    • Lowers when the system has been correctly identifying good setups
      or the miss rate is too high (over-cautious).
    • Bounded to [0.10, 0.55].

NO_ACTION reward (C13 requirement #2)
--------------------------------------
``WaitDecision.prospective_no_action_reward`` is a positive signal that can
be fed directly into the training pipeline as a reward for correct inaction.
It is replaced by ``SkippedOpportunity.realized_no_action_reward`` once the
market shows what would have happened (via
``PatienceEngine.resolve_skipped_opportunity()``).

    Prospective: +0.20 base + per-reason increments (see patience_scoring.py)
    Realized CORRECT_SKIP:       [+0.60, +1.00]  (avoided a loss)
    Realized MISSED_OPPORTUNITY: [−0.40, −0.80]  (missed a profit)
    Realized NEUTRAL:              0.00

Skipped opportunity tracking (C13 requirement #7)
--------------------------------------------------
Every ``WaitDecision`` with ``should_wait=True`` contains a
``SkippedOpportunity`` record.  ``PatienceEngine.record_skip()`` persists it
to the bounded history (max 500 per instrument).  Later, when price data is
available, ``PatienceEngine.resolve_skipped_opportunity(skip_id, outcome)``
updates the record with the actual market outcome so the model can learn
"I was correct not to trade" from every resolved skip.

Modules
-------
``patience_types``
    All enums (``WaitReason``, ``SkipOutcome``) and dataclasses
    (``SkippedOpportunity``, ``WaitDecision``, ``PatienceState``,
    ``PatienceMetrics``) plus constants.

``patience_scoring``
    ``patience_score()``, ``minimum_quality_threshold()``,
    ``compute_no_action_reward()``, ``compute_realized_reward()``.

``patience_engine``
    ``evaluate_wait_decision()``, ``PatienceEngine``.
"""

from __future__ import annotations

# ── Engine ────────────────────────────────────────────────────────────────────
from trainer.qde.brain.patience.patience_engine import (
    PatienceEngine,
    evaluate_wait_decision,
)

# ── Scoring ───────────────────────────────────────────────────────────────────
from trainer.qde.brain.patience.patience_scoring import (
    compute_no_action_reward,
    compute_realized_reward,
    minimum_quality_threshold,
    patience_score,
)

# ── Types ─────────────────────────────────────────────────────────────────────
from trainer.qde.brain.patience.patience_types import (
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_RESOLUTION_WINDOW_SECONDS,
    DEFAULT_ROLLING_WINDOW,
    IMPULSIVE_CONVICTION_FLOOR,
    IMPULSIVE_URGENCY_THRESHOLD,
    MAX_QUALITY_THRESHOLD,
    MAX_SKIPPED_RECORDS,
    MIN_QUALITY_THRESHOLD,
    OVERTRADING_RATIO_THRESHOLD,
    THRESHOLD_STEP_DOWN,
    THRESHOLD_STEP_UP,
    UNCERTAINTY_NO_TRADE_THRESHOLD,
    PatienceMetrics,
    PatienceState,
    SkipOutcome,
    SkippedOpportunity,
    WaitDecision,
    WaitReason,
)

__all__ = [
    # ── Three public callables (C13 interface) ────────────────────────────────
    "evaluate_wait_decision",
    "patience_score",
    "minimum_quality_threshold",
    # ── Engine ───────────────────────────────────────────────────────────────
    "PatienceEngine",
    # ── Types ────────────────────────────────────────────────────────────────
    "WaitReason",
    "SkipOutcome",
    "SkippedOpportunity",
    "WaitDecision",
    "PatienceState",
    "PatienceMetrics",
    # ── Reward math ──────────────────────────────────────────────────────────
    "compute_no_action_reward",
    "compute_realized_reward",
    # ── Constants ────────────────────────────────────────────────────────────
    "DEFAULT_QUALITY_THRESHOLD",
    "MIN_QUALITY_THRESHOLD",
    "MAX_QUALITY_THRESHOLD",
    "UNCERTAINTY_NO_TRADE_THRESHOLD",
    "IMPULSIVE_URGENCY_THRESHOLD",
    "IMPULSIVE_CONVICTION_FLOOR",
    "OVERTRADING_RATIO_THRESHOLD",
    "DEFAULT_ROLLING_WINDOW",
    "MAX_SKIPPED_RECORDS",
    "THRESHOLD_STEP_UP",
    "THRESHOLD_STEP_DOWN",
    "DEFAULT_RESOLUTION_WINDOW_SECONDS",
]
