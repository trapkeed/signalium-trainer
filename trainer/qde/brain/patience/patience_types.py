"""C13 — Patience Engine: data types, enums, and state containers.

Design principles
-----------------
``WaitReason``
    Discriminator for why the system chose inaction.  Multiple reasons
    can apply simultaneously (e.g. LOW_QUALITY + UNCERTAINTY_HIGH);
    they are collected as a list so downstream training signals can
    attribute the skip to its real drivers.

``SkippedOpportunity``
    Mutable record of one skip event.  Created when
    ``evaluate_wait_decision()`` returns ``should_wait=True`` and the
    caller persists it via ``PatienceEngine.record_skip()``.  Later
    resolved by calling ``PatienceEngine.resolve_skipped_opportunity()``
    once the market shows what would have happened.  Mutable so that
    the resolution fields (``outcome``, ``resolved_at``,
    ``no_action_reward``) can be updated in place without rebuilding the
    list.

``WaitDecision``
    Immutable result of ``evaluate_wait_decision()``.  The
    ``prospective_no_action_reward`` field is the immediate reward
    signal for correct inaction — it is positive when the system has
    good reasons to wait and negative only when the system is being
    overly conservative (opportunity_score very high and all other
    checks pass).

``PatienceState``
    Per-instrument mutable state owned by ``PatienceEngine``.  Holds
    rolling counts, the bounded history of skipped opportunities, and
    the current adaptive quality threshold.

``PatienceMetrics``
    Immutable snapshot produced by ``patience_score()``.  The headline
    ``composite_patience_score`` is the learning target — it should
    increase over time as the model learns to wait for confirmation and
    avoid impulsive entries.

Relationship to existing infrastructure
----------------------------------------
* ``CognitiveAction.WAIT / MONITOR / IGNORE`` (decision_types.py) are
  already first-class passive decisions.  C13 does not introduce a new
  action — it layers patience scoring and adaptive thresholds on top.
* ``prefers_no_trade()`` in uncertainty_engine.py is a binary gate at
  composite uncertainty ≥ 0.55.  C13 is complementary: it tracks the
  *quality of patience decisions over time* and adjusts the minimum
  opportunity score the system requires before committing.
* ``PriorityTier.PASS`` (capital_priority_engine.py) uses a static
  floor of ``DEFAULT_MIN_SCORE = 0.20``.  ``minimum_quality_threshold``
  produces the same floor but makes it adaptive: repeated impulsive
  entries raise it, a pattern of correct skips allows it to relax.
* No import of ``QDEDecision`` or any execution-layer type.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Quality floor: same as _TIER_3_MIN in capital_priority_engine so the two
# systems agree on the absolute minimum for consideration.
DEFAULT_QUALITY_THRESHOLD: float = 0.20

# Adaptive threshold bounds.
MIN_QUALITY_THRESHOLD: float = 0.10   # never drop below this (market is valid)
MAX_QUALITY_THRESHOLD: float = 0.55   # never exceed this (would block all entries)

# Uncertainty threshold above which NO_TRADE is preferred.
# Mirrors prefers_no_trade(score, threshold=0.55) in uncertainty_engine.py.
UNCERTAINTY_NO_TRADE_THRESHOLD: float = 0.55

# Entry urgency ceiling without matching conviction.
IMPULSIVE_URGENCY_THRESHOLD: float = 0.70  # urgency above this is "impulsive"
IMPULSIVE_CONVICTION_FLOOR: float  = 0.40  # conviction must be at least this

# Overtrading frequency ratio (today / baseline) above which we guard.
OVERTRADING_RATIO_THRESHOLD: float = 1.5

# Rolling window for patience metrics computation.
DEFAULT_ROLLING_WINDOW: int = 50

# Maximum skipped-opportunity records to retain per instrument.
MAX_SKIPPED_RECORDS: int = 500

# Threshold adjustment step applied by the adaptive engine each update.
THRESHOLD_STEP_UP: float   = 0.025   # raise quality bar
THRESHOLD_STEP_DOWN: float = 0.015   # lower quality bar (gentler)

# Resolution window: how long after a skip we wait before calling it NEUTRAL.
DEFAULT_RESOLUTION_WINDOW_SECONDS: float = 4 * 3600.0   # 4 hours default


# ─────────────────────────────────────────────────────────────────────────────
# WaitReason
# ─────────────────────────────────────────────────────────────────────────────


class WaitReason(str, Enum):
    """Why the patience engine recommended inaction.

    Ordered from hardest (structural/risk) to softest (discretionary)
    so callers can compare priority via ``list.index()``.
    """

    COGNITIVE_PASSIVE     = "COGNITIVE_PASSIVE"       # brain said WAIT/IGNORE
    UNCERTAINTY_HIGH      = "UNCERTAINTY_HIGH"         # uncertainty ≥ 0.55
    OVERTRADING           = "OVERTRADING"              # trade frequency too high
    CONFIRMATION_PENDING  = "CONFIRMATION_PENDING"     # brain said MONITOR — setup forming
    LOW_QUALITY           = "LOW_QUALITY"              # opportunity_score < threshold
    IMPULSIVE_GUARD       = "IMPULSIVE_GUARD"          # high urgency but low conviction
    CAPITAL_UNAVAILABLE   = "CAPITAL_UNAVAILABLE"      # PriorityTier.PASS (budget)


# ─────────────────────────────────────────────────────────────────────────────
# SkipOutcome
# ─────────────────────────────────────────────────────────────────────────────


class SkipOutcome(str, Enum):
    """Post-hoc outcome of a skipped opportunity.

    Resolved by ``PatienceEngine.resolve_skipped_opportunity()`` once
    sufficient price data is available.

    ``CORRECT_SKIP``
        Price moved against the direction that was skipped — the system
        correctly avoided a loss.

    ``MISSED_OPPORTUNITY``
        Price moved significantly with the skipped direction — the system
        was over-cautious and missed a profit.

    ``NEUTRAL``
        No significant price movement within the resolution window; the
        skip had negligible cost or benefit.

    ``UNRESOLVED``
        Outcome not yet known (default at creation time).
    """

    CORRECT_SKIP        = "CORRECT_SKIP"
    MISSED_OPPORTUNITY  = "MISSED_OPPORTUNITY"
    NEUTRAL             = "NEUTRAL"
    UNRESOLVED          = "UNRESOLVED"


# ─────────────────────────────────────────────────────────────────────────────
# SkippedOpportunity
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SkippedOpportunity:
    """Mutable record of one skip event.

    Created when ``evaluate_wait_decision()`` returns ``should_wait=True``
    and the caller (or ``PatienceEngine``) calls ``record_skip()``.
    Resolved by ``PatienceEngine.resolve_skipped_opportunity()`` which
    updates the ``outcome``, ``outcome_pnl_estimate``, ``resolved_at``,
    and ``no_action_reward`` fields.

    The ``no_action_reward`` at creation time is **prospective** (based
    on the reasons for skipping).  At resolution time it is replaced by
    the **realized** reward which reflects what actually happened in the
    market.
    """

    skip_id:                    str         = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument:                 str         = ""
    direction:                  str         = ""     # BeliefDirection.value

    # ── Snapshot at skip time ─────────────────────────────────────────────────
    opportunity_score_at_skip:  float       = 0.0
    quality_threshold_at_skip:  float       = DEFAULT_QUALITY_THRESHOLD
    wait_reasons:               list[WaitReason] = field(default_factory=list)
    primary_reason:             WaitReason | None = None
    cognitive_action:           str         = ""     # CognitiveAction.value
    uncertainty_score:          float       = 0.0
    conviction_score:           float       = 0.0
    entry_urgency:              float       = 0.0
    prospective_no_action_reward: float     = 0.0    # reward signal at skip time

    # ── Resolution fields (updated later) ────────────────────────────────────
    outcome:                    SkipOutcome = SkipOutcome.UNRESOLVED
    outcome_pnl_estimate:       float       = 0.0    # +ve = avoided loss, -ve = missed profit
    realized_no_action_reward:  float       = 0.0    # updated on resolution
    skipped_at:                 float       = field(default_factory=time.time)
    resolved_at:                float | None = None

    @property
    def is_resolved(self) -> bool:
        return self.outcome != SkipOutcome.UNRESOLVED

    @property
    def was_correct(self) -> bool:
        return self.outcome == SkipOutcome.CORRECT_SKIP

    @property
    def was_mistake(self) -> bool:
        return self.outcome == SkipOutcome.MISSED_OPPORTUNITY

    def resolve(
        self,
        outcome: SkipOutcome,
        pnl_estimate: float = 0.0,
        realized_reward: float = 0.0,
        now: float | None = None,
    ) -> None:
        """Update outcome fields in place.  Idempotent if called twice."""
        if self.is_resolved:
            return
        self.outcome = outcome
        self.outcome_pnl_estimate = pnl_estimate
        self.realized_no_action_reward = realized_reward
        self.resolved_at = now if now is not None else time.time()

    def to_dict(self) -> dict:
        return {
            "skip_id":                     self.skip_id,
            "instrument":                  self.instrument,
            "direction":                   self.direction,
            "opportunity_score_at_skip":   round(self.opportunity_score_at_skip, 4),
            "quality_threshold_at_skip":   round(self.quality_threshold_at_skip, 4),
            "wait_reasons":                [r.value for r in self.wait_reasons],
            "primary_reason":              self.primary_reason.value if self.primary_reason else None,
            "cognitive_action":            self.cognitive_action,
            "uncertainty_score":           round(self.uncertainty_score, 4),
            "conviction_score":            round(self.conviction_score, 4),
            "entry_urgency":               round(self.entry_urgency, 4),
            "prospective_no_action_reward": round(self.prospective_no_action_reward, 4),
            "outcome":                     self.outcome.value,
            "outcome_pnl_estimate":        round(self.outcome_pnl_estimate, 4),
            "realized_no_action_reward":   round(self.realized_no_action_reward, 4),
            "skipped_at":                  round(self.skipped_at, 3),
            "resolved_at":                 round(self.resolved_at, 3) if self.resolved_at else None,
            "is_resolved":                 self.is_resolved,
            "was_correct":                 self.was_correct,
        }


# ─────────────────────────────────────────────────────────────────────────────
# WaitDecision
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WaitDecision:
    """Immutable result of one ``evaluate_wait_decision()`` call.

    ``should_wait``
        Primary recommendation.  When ``True`` the caller should NOT
        commit to an entry.  When ``False`` the patience engine has no
        objection and the entry can proceed through the normal execution
        pipeline.

    ``prospective_no_action_reward``
        Immediate reward signal for choosing inaction.  Range [-1, +1].
        Positive: the system has well-grounded reasons to wait.
        Negative (small): the system is being overly conservative (e.g.
        opportunity_score is very high and all structural checks pass).
        Zero when ``should_wait=False`` (no reward for acting).

    ``quality_gap``
        ``quality_threshold − opportunity_score``.  Positive means the
        opportunity is below the current bar; negative means it exceeds
        it.  Useful for continuous reward shaping (larger gap → stronger
        avoidance signal).

    ``skip_record``
        Pre-built ``SkippedOpportunity`` instance, ready for the caller
        to persist via ``PatienceEngine.record_skip()``.  ``None`` when
        ``should_wait=False`` (no skip to record).
    """

    should_wait:                   bool
    wait_reasons:                  tuple[WaitReason, ...]
    primary_reason:                WaitReason | None
    prospective_no_action_reward:  float
    quality_gap:                   float   # threshold - opportunity_score (+ve = below bar)
    reasoning:                     str
    skip_record:                   SkippedOpportunity | None  = None

    def to_dict(self) -> dict:
        return {
            "should_wait":                  self.should_wait,
            "wait_reasons":                 [r.value for r in self.wait_reasons],
            "primary_reason":               self.primary_reason.value if self.primary_reason else None,
            "prospective_no_action_reward": round(self.prospective_no_action_reward, 4),
            "quality_gap":                  round(self.quality_gap, 4),
            "reasoning":                    self.reasoning,
            "skip_id":                      self.skip_record.skip_id if self.skip_record else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PatienceState
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PatienceState:
    """Per-instrument mutable state for the ``PatienceEngine``.

    Holds rolling counters, the bounded skip history, and the current
    adaptive quality threshold.  One instance per instrument (or one
    global instance if the engine covers all symbols).

    ``impulsive_entries``
        Entries where ``entry_urgency > IMPULSIVE_URGENCY_THRESHOLD``
        and ``conviction < IMPULSIVE_CONVICTION_FLOOR`` — the clearest
        indicator of overtrading behaviour.

    ``current_threshold``
        The live adaptive quality floor.  Raised by impulsive entries and
        overtrading events; lowered when repeated correct skips suggest
        the system is being too conservative.  ``minimum_quality_threshold``
        returns this value after bounding it to [MIN, MAX].

    ``recent_wait_flags``
        Fixed-length deque of booleans (True=waited, False=acted).
        Used to detect sustained avoidance patterns without scanning the
        full skip history on every tick.
    """

    instrument:             str   = ""

    # ── Rolling counters ──────────────────────────────────────────────────────
    total_trades:           int   = 0
    total_skips:            int   = 0
    correct_skips:          int   = 0       # resolved as CORRECT_SKIP
    missed_opportunities:   int   = 0       # resolved as MISSED_OPPORTUNITY
    impulsive_entries:      int   = 0       # entries flagged as impulsive
    overtrading_events:     int   = 0       # is_overtrading=True detections

    # ── Adaptive threshold ────────────────────────────────────────────────────
    current_threshold:      float = DEFAULT_QUALITY_THRESHOLD
    last_threshold_update:  float = 0.0

    # ── Bounded skip history ──────────────────────────────────────────────────
    skipped_opportunities:  list[SkippedOpportunity] = field(default_factory=list)

    # ── Recent-decision ring buffer (True=waited, False=acted) ───────────────
    recent_wait_flags:      Deque[bool] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_ROLLING_WINDOW)
    )

    def _append_skip(self, skip: SkippedOpportunity) -> None:
        self.skipped_opportunities.append(skip)
        if len(self.skipped_opportunities) > MAX_SKIPPED_RECORDS:
            del self.skipped_opportunities[0]

    def pending_skips(self) -> list[SkippedOpportunity]:
        """Skipped opportunities whose outcome has not yet been resolved."""
        return [s for s in self.skipped_opportunities if not s.is_resolved]

    def resolved_skips(self) -> list[SkippedOpportunity]:
        """Skipped opportunities that have been resolved."""
        return [s for s in self.skipped_opportunities if s.is_resolved]

    def find_skip(self, skip_id: str) -> SkippedOpportunity | None:
        for s in reversed(self.skipped_opportunities):
            if s.skip_id == skip_id:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "instrument":            self.instrument,
            "total_trades":          self.total_trades,
            "total_skips":           self.total_skips,
            "correct_skips":         self.correct_skips,
            "missed_opportunities":  self.missed_opportunities,
            "impulsive_entries":     self.impulsive_entries,
            "overtrading_events":    self.overtrading_events,
            "current_threshold":     round(self.current_threshold, 4),
            "pending_skips":         len(self.pending_skips()),
            "total_stored_skips":    len(self.skipped_opportunities),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PatienceMetrics
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatienceMetrics:
    """Immutable snapshot produced by ``patience_score()``.

    All scores are in [0, 1]; higher is always better (more patient /
    more disciplined).

    ``discipline_score``
        Fraction of resolved skips that were correct (CORRECT_SKIP).
        ``correct_skips / max(1, resolved_count)``.

    ``impulsive_entry_rate``
        Fraction of total entries flagged as impulsive.
        ``impulsive_entries / max(1, total_trades)``.
        Lower is better; inverted to ``1 − rate`` in composite scoring.

    ``miss_rate``
        Fraction of resolved skips that were MISSED_OPPORTUNITY.
        ``missed_opportunities / max(1, resolved_count)``.
        Lower is better; inverted in composite scoring.

    ``composite_patience_score``
        Weighted composite (the headline learning target):
        ``0.5 × discipline_score + 0.3 × (1 - impulsive_rate) + 0.2 × (1 - miss_rate)``
    """

    discipline_score:          float
    impulsive_entry_rate:      float
    miss_rate:                 float
    composite_patience_score:  float
    total_skips:               int
    total_trades:              int
    correct_skips:             int
    missed_opportunities:      int
    resolved_skips:            int
    overtrading_events:        int

    def to_dict(self) -> dict:
        return {
            "discipline_score":         round(self.discipline_score,         4),
            "impulsive_entry_rate":     round(self.impulsive_entry_rate,     4),
            "miss_rate":                round(self.miss_rate,                4),
            "composite_patience_score": round(self.composite_patience_score, 4),
            "total_skips":              self.total_skips,
            "total_trades":             self.total_trades,
            "correct_skips":            self.correct_skips,
            "missed_opportunities":     self.missed_opportunities,
            "resolved_skips":           self.resolved_skips,
            "overtrading_events":       self.overtrading_events,
        }


__all__ = [
    # Enums
    "WaitReason",
    "SkipOutcome",
    # Dataclasses
    "SkippedOpportunity",
    "WaitDecision",
    "PatienceState",
    "PatienceMetrics",
    # Constants
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
