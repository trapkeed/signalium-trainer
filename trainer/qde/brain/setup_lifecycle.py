"""QDE Brain — setup lifecycle (opportunity-queue state machine).

Defines the execution-facing progression a *developing opportunity*
moves through before QDE will ever consider entering a trade from it:

    WATCHING -> ARMED -> CONFIRMING -> READY -> EXECUTING -> COOLDOWN

This is a third, still-distinct vocabulary from the two that already
exist in this codebase:

  - `qde_observer_types.ObserverState` (OBSERVING/PREPARING/ARMED/...)
    is a *per-symbol*, single-slot lifecycle owned by
    `QDEMarketObserver` that also covers the in-trade side
    (POSITION_ACTIVE/MANAGING/EXITING) — one state per symbol at a
    time, tied to whatever position (if any) is currently open.
  - `hypothesis_lifecycle.HypothesisMaturity` (OBSERVING/BUILDING/
    HIGH_CONVICTION/EXECUTABLE/...) is a *pure projection* of one
    hypothesis's own fields — it has no memory of its own and answers
    "how mature is this belief right now", not "has this specific
    opportunity been watched, armed, and confirmed over time".

`SetupState` is deliberately different: a symbol may have several
opportunities queued at once (e.g. two different hypotheses maturing
on different timeframes), each tracked independently, each requiring
*sustained* evidence (multiple observed ticks in CONFIRMING, not a
single strong reading) before reaching READY. It never re-implements
maturity/conviction/uncertainty scoring — `next_state()` only
*consumes* those readings (via `SetupEvidence`) to decide whether an
opportunity has earned its next stage. It also never places or
confirms an order itself: reaching EXECUTING is a signal for an
external decision layer to act on, and moving out of EXECUTING again
is caller-driven (`opportunity_tracker.OpportunityTracker.mark_executing`
/ `.complete`), not something this pure function infers on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from trainer.qde.brain.brain_types import NarrativeTone
from trainer.qde.brain.conviction_engine import ConvictionScore, is_trade_worthy
from trainer.qde.brain.hypothesis_lifecycle import HypothesisMaturity
from trainer.qde.brain.uncertainty_engine import UncertaintyScore, prefers_no_trade


class SetupState(str, Enum):
    """Where one developing opportunity sits in the observe-before-enter
    progression.
    """

    WATCHING = "WATCHING"  # registered, thin evidence, just observing
    ARMED = "ARMED"  # setup quality rising, warrants closer attention
    CONFIRMING = "CONFIRMING"  # accumulating sustained confirming evidence
    READY = "READY"  # sufficient evidence has accumulated to justify entry
    EXECUTING = "EXECUTING"  # handed off to the decision layer to act on
    COOLDOWN = "COOLDOWN"  # post-execution or post-invalidation quiet period


# Valid transition map, mirroring the `qde_observer_types.VALID_TRANSITIONS`
# convention so a malformed caller cannot jump the queue (e.g. straight
# from WATCHING to READY without ever being ARMED/CONFIRMING).
VALID_TRANSITIONS: dict[SetupState, frozenset[SetupState]] = {
    SetupState.WATCHING: frozenset({SetupState.WATCHING, SetupState.ARMED, SetupState.COOLDOWN}),
    SetupState.ARMED: frozenset(
        {SetupState.ARMED, SetupState.CONFIRMING, SetupState.WATCHING, SetupState.COOLDOWN}
    ),
    SetupState.CONFIRMING: frozenset(
        {SetupState.CONFIRMING, SetupState.READY, SetupState.ARMED, SetupState.COOLDOWN}
    ),
    SetupState.READY: frozenset({SetupState.READY, SetupState.EXECUTING, SetupState.COOLDOWN}),
    SetupState.EXECUTING: frozenset({SetupState.EXECUTING, SetupState.COOLDOWN}),
    SetupState.COOLDOWN: frozenset({SetupState.COOLDOWN, SetupState.WATCHING}),
}


def is_valid_transition(from_state: SetupState, to_state: SetupState) -> bool:
    return to_state in VALID_TRANSITIONS.get(from_state, frozenset())


@dataclass(frozen=True)
class SetupThresholds:
    """Tunable gates for the WATCHING -> ... -> READY progression."""

    min_attention_weight_to_arm: float = 0.10
    arm_min_maturity_rank: int = 1  # HypothesisMaturity.BUILDING or higher
    confirm_min_maturity_rank: int = 2  # HypothesisMaturity.HIGH_CONVICTION or higher
    min_confirmation_ticks: int = 2  # sustained evidence, not one strong tick
    ready_min_conviction: float = 0.6
    ready_min_confidence: float = 0.4
    uncertainty_threshold: float = 0.55  # prefer NO_TRADE at/above this
    max_ready_dwell_seconds: float = 900.0  # missed entry window if unconsumed this long
    max_executing_dwell_seconds: float = 60.0  # orphaned handoff safety net
    cooldown_seconds: float = 300.0


DEFAULT_SETUP_THRESHOLDS = SetupThresholds()

_MATURITY_RANK = {
    HypothesisMaturity.OBSERVING: 0,
    HypothesisMaturity.BUILDING: 1,
    HypothesisMaturity.HIGH_CONVICTION: 2,
    HypothesisMaturity.EXECUTABLE: 3,
    HypothesisMaturity.INVALIDATED: -1,
    HypothesisMaturity.EXPIRED: -1,
}


@dataclass(frozen=True)
class SetupEvidence:
    """Everything `next_state()` needs to judge one opportunity's
    progress this tick — all of it read from engines that already own
    this data (`hypothesis_engine`, `attention_engine`,
    `narratives.NarrativeEngine`); this module never re-derives any of
    it.
    """

    maturity: HypothesisMaturity = HypothesisMaturity.OBSERVING
    conviction: ConvictionScore | None = None
    uncertainty: UncertaintyScore | None = None
    attention_weight: float = 0.0
    narrative_tone: NarrativeTone = NarrativeTone.NEUTRAL
    narrative_snapshot_text: str = ""
    dwell_seconds: float = 0.0


def _maturity_terminal(maturity: HypothesisMaturity) -> bool:
    return maturity in (HypothesisMaturity.INVALIDATED, HypothesisMaturity.EXPIRED)


def _is_trade_worthy(evidence: SetupEvidence, thresholds: SetupThresholds) -> bool:
    if evidence.conviction is None:
        return False
    return is_trade_worthy(
        evidence.conviction,
        min_conviction=thresholds.ready_min_conviction,
        min_confidence=thresholds.ready_min_confidence,
    )


def _understands_market(evidence: SetupEvidence, thresholds: SetupThresholds) -> bool:
    if evidence.uncertainty is None:
        return True
    return not prefers_no_trade(evidence.uncertainty, threshold=thresholds.uncertainty_threshold)


def next_state(
    current: SetupState,
    evidence: SetupEvidence,
    confirmation_ticks: int = 0,
    thresholds: SetupThresholds = DEFAULT_SETUP_THRESHOLDS,
) -> SetupState:
    """Decide the next `SetupState` given the current state plus this
    tick's evidence. Pure function — never mutates `evidence`, never
    touches storage, and never places an order; it only ever proposes
    the *next* queue position, which the caller (`OpportunityTracker`/
    `SetupQueue`) is responsible for actually applying and recording.

    QDE's explicit "I do not understand this market" recognition is
    enforced here uniformly: regardless of maturity or conviction, high
    market ambiguity can only ever push an opportunity *backward*
    (toward COOLDOWN) or hold it in place — it can never advance a
    setup toward READY/EXECUTING.
    """
    # Terminal hypothesis outcomes (invalidated/expired) always retire
    # the opportunity, from any state, including an in-flight execution
    # handoff (an execution already underway is left for the caller to
    # resolve via `complete()`, not silently reclassified here).
    if _maturity_terminal(evidence.maturity) and current != SetupState.EXECUTING:
        return SetupState.COOLDOWN

    understands_market = _understands_market(evidence, thresholds)
    maturity_rank = _MATURITY_RANK.get(evidence.maturity, 0)

    if current == SetupState.WATCHING:
        if (
            understands_market
            and maturity_rank >= thresholds.arm_min_maturity_rank
            and evidence.attention_weight >= thresholds.min_attention_weight_to_arm
        ):
            return SetupState.ARMED
        return SetupState.WATCHING

    if current == SetupState.ARMED:
        if not understands_market:
            return SetupState.WATCHING
        if maturity_rank < thresholds.arm_min_maturity_rank:
            return SetupState.WATCHING
        if maturity_rank >= thresholds.confirm_min_maturity_rank:
            return SetupState.CONFIRMING
        return SetupState.ARMED

    if current == SetupState.CONFIRMING:
        if not understands_market:
            return SetupState.ARMED
        if maturity_rank < thresholds.confirm_min_maturity_rank:
            return SetupState.ARMED
        # A breakout-pending narrative corroborates the setup, letting
        # sustained evidence count a little faster — it never bypasses
        # the sustained-evidence requirement outright.
        required_ticks = thresholds.min_confirmation_ticks
        if evidence.narrative_tone == NarrativeTone.BREAKOUT_PENDING:
            required_ticks = max(1, required_ticks - 1)
        if confirmation_ticks >= required_ticks and _is_trade_worthy(evidence, thresholds):
            return SetupState.READY
        return SetupState.CONFIRMING

    if current == SetupState.READY:
        if not understands_market or maturity_rank < thresholds.confirm_min_maturity_rank:
            return SetupState.COOLDOWN
        if evidence.dwell_seconds >= thresholds.max_ready_dwell_seconds:
            return SetupState.COOLDOWN  # missed the entry window
        return SetupState.READY

    if current == SetupState.EXECUTING:
        if evidence.dwell_seconds >= thresholds.max_executing_dwell_seconds:
            return SetupState.COOLDOWN  # orphaned handoff safety net
        return SetupState.EXECUTING

    # COOLDOWN
    if evidence.dwell_seconds >= thresholds.cooldown_seconds and not _maturity_terminal(
        evidence.maturity
    ):
        return SetupState.WATCHING
    return SetupState.COOLDOWN


__all__ = [
    "SetupState",
    "VALID_TRANSITIONS",
    "is_valid_transition",
    "SetupThresholds",
    "DEFAULT_SETUP_THRESHOLDS",
    "SetupEvidence",
    "next_state",
]
