"""C13 — Patience Engine: evaluate_wait_decision() + PatienceEngine.

Three module-level public callables (C13 requirement):

    evaluate_wait_decision(cognitive_decision, opp_score, ...) → WaitDecision
    patience_score(state)                                       → PatienceMetrics
    minimum_quality_threshold(state)                            → float

The module-level ``patience_score`` and ``minimum_quality_threshold`` are
re-exported from ``patience_scoring`` for a single unified import surface
via the package ``__init__``.

``PatienceEngine``
    Stateful manager.  One instance per symbol (mirroring ``BrainState``'s
    per-symbol design) or one global instance.  Wraps the three stateless
    functions and adds:

    * Automatic threshold adaptation on each resolved skip.
    * Skip history management (bounded, searchable by skip_id).
    * Impulsive-entry and overtrading tracking.
    * ``age_pending_skips()`` — resolve very old skips as NEUTRAL when
      no explicit outcome arrives within the resolution window.

evaluate_wait_decision logic
-----------------------------
Checks are applied in descending priority order; multiple reasons can
be collected simultaneously (the list is the full set of reasons that
apply, the primary_reason is the highest-priority one):

  1. COGNITIVE_PASSIVE   — action is WAIT or IGNORE
  2. UNCERTAINTY_HIGH    — uncertainty_score ≥ UNCERTAINTY_NO_TRADE_THRESHOLD (0.55)
  3. OVERTRADING         — account_features.is_overtrading
                           OR trade_frequency_ratio ≥ OVERTRADING_RATIO_THRESHOLD
  4. CONFIRMATION_PENDING— action is MONITOR (setup forming)
  5. IMPULSIVE_GUARD     — entry_urgency > 0.70 and conviction_score < 0.40
  6. LOW_QUALITY         — opp_score.opportunity_score < quality_threshold
  7. CAPITAL_UNAVAILABLE — explicitly passed by caller (e.g. PriorityTier.PASS)

If ANY reason fires → ``should_wait = True``.
If no reason fires  → ``should_wait = False``; skip_record is None;
                       prospective_no_action_reward is 0.

Relationship to existing infrastructure
----------------------------------------
* Takes ``CognitiveDecision`` (decision/decision_types.py) — reads .action,
  .uncertainty_score, .conviction_score, .entry_urgency, .direction.
* Takes ``OpportunityScore`` (opportunity/opportunity_score.py) — reads
  .opportunity_score, .opportunity_class.
* Optionally takes ``AccountFeatures`` (qde_account_models.py) — reads
  .is_overtrading, .trade_frequency_ratio.
* Does NOT import ``QDEDecision`` or any execution-layer type.
* Does NOT modify any input object.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from trainer.qde.brain.patience.patience_scoring import (
    compute_no_action_reward,
    compute_realized_reward,
    minimum_quality_threshold,   # re-export
    patience_score,               # re-export
)
from trainer.qde.brain.patience.patience_types import (
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_RESOLUTION_WINDOW_SECONDS,
    IMPULSIVE_CONVICTION_FLOOR,
    IMPULSIVE_URGENCY_THRESHOLD,
    OVERTRADING_RATIO_THRESHOLD,
    THRESHOLD_STEP_DOWN,
    THRESHOLD_STEP_UP,
    MAX_QUALITY_THRESHOLD,
    MIN_QUALITY_THRESHOLD,
    UNCERTAINTY_NO_TRADE_THRESHOLD,
    PatienceMetrics,
    PatienceState,
    SkipOutcome,
    SkippedOpportunity,
    WaitDecision,
    WaitReason,
)

if TYPE_CHECKING:
    from trainer.qde.brain.decision.decision_types import CognitiveDecision
    from trainer.qde.brain.opportunity.opportunity_score import OpportunityScore
    from trainer.qde.qde_account_models import AccountFeatures


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_wait_decision   (public callable)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_wait_decision(
    cognitive_decision:     "CognitiveDecision",
    opp_score:              "OpportunityScore",
    quality_threshold:      float | None = None,
    account_features:       "AccountFeatures | None" = None,
    capital_unavailable:    bool = False,
    now:                    float | None = None,
) -> WaitDecision:
    """Evaluate whether the system should wait rather than act.

    Pure stateless function — does not mutate any argument.  Returns a
    ``WaitDecision`` whose ``skip_record`` field contains a pre-built
    ``SkippedOpportunity`` ready for the caller to persist if they track
    skips.

    Parameters
    ----------
    cognitive_decision:
        The Brain's current cognitive verdict (``CognitiveDecision``).
        Read fields: ``action``, ``uncertainty_score``, ``conviction_score``,
        ``entry_urgency``, ``direction``, ``instrument``.
    opp_score:
        The opportunity score for the candidate entry (``OpportunityScore``
        from C11).  Read fields: ``opportunity_score``, ``opportunity_class``.
    quality_threshold:
        Minimum opportunity score required to proceed.  When ``None``,
        defaults to ``DEFAULT_QUALITY_THRESHOLD (0.20)``.  Pass the output
        of ``minimum_quality_threshold(state)`` for adaptive behaviour.
    account_features:
        Optional ``AccountFeatures`` from the account pipeline.  When
        provided, ``is_overtrading`` and ``trade_frequency_ratio`` are
        checked.
    capital_unavailable:
        Explicitly signals that the capital priority engine assigned
        ``PriorityTier.PASS`` (budget exhausted).  When ``True``,
        ``CAPITAL_UNAVAILABLE`` is added to the wait reasons.
    now:
        Wall-clock timestamp (seconds since epoch).  Defaults to
        ``time.time()``.

    Returns
    -------
    ``WaitDecision``
        ``should_wait=False`` when no reason fires (system can proceed).
        ``should_wait=True`` with a populated ``skip_record`` otherwise.
    """
    from trainer.qde.brain.decision.decision_types import CognitiveAction

    now = now if now is not None else time.time()
    threshold = quality_threshold if quality_threshold is not None else DEFAULT_QUALITY_THRESHOLD

    reasons: list[WaitReason] = []

    # ── 1. Cognitive action check (highest priority) ──────────────────────────
    action = cognitive_decision.action
    if action == CognitiveAction.IGNORE or action == CognitiveAction.WAIT:
        reasons.append(WaitReason.COGNITIVE_PASSIVE)

    # ── 2. Uncertainty ────────────────────────────────────────────────────────
    if cognitive_decision.uncertainty_score >= UNCERTAINTY_NO_TRADE_THRESHOLD:
        reasons.append(WaitReason.UNCERTAINTY_HIGH)

    # ── 3. Overtrading ────────────────────────────────────────────────────────
    if account_features is not None:
        if account_features.is_overtrading:
            reasons.append(WaitReason.OVERTRADING)
        elif (
            hasattr(account_features, "trade_frequency_ratio")
            and account_features.trade_frequency_ratio >= OVERTRADING_RATIO_THRESHOLD
        ):
            reasons.append(WaitReason.OVERTRADING)

    # ── 4. MONITOR → confirmation pending ────────────────────────────────────
    if action == CognitiveAction.MONITOR:
        reasons.append(WaitReason.CONFIRMATION_PENDING)

    # ── 5. Impulsive entry guard ──────────────────────────────────────────────
    if (
        cognitive_decision.entry_urgency > IMPULSIVE_URGENCY_THRESHOLD
        and cognitive_decision.conviction_score < IMPULSIVE_CONVICTION_FLOOR
    ):
        reasons.append(WaitReason.IMPULSIVE_GUARD)

    # ── 6. Quality floor ──────────────────────────────────────────────────────
    quality_gap = threshold - opp_score.opportunity_score
    if quality_gap > 0.0:
        reasons.append(WaitReason.LOW_QUALITY)

    # ── 7. Capital unavailable ────────────────────────────────────────────────
    if capital_unavailable:
        reasons.append(WaitReason.CAPITAL_UNAVAILABLE)

    should_wait = len(reasons) > 0
    primary_reason = reasons[0] if reasons else None

    if not should_wait:
        return WaitDecision(
            should_wait=False,
            wait_reasons=(),
            primary_reason=None,
            prospective_no_action_reward=0.0,
            quality_gap=quality_gap,
            reasoning="No wait conditions met — system may proceed.",
            skip_record=None,
        )

    # ── Prospective NO_ACTION reward ──────────────────────────────────────────
    reward = compute_no_action_reward(
        wait_reasons=reasons,
        opportunity_score=opp_score.opportunity_score,
        uncertainty_score=cognitive_decision.uncertainty_score,
        quality_gap=quality_gap,
    )

    # ── Build reasoning string ────────────────────────────────────────────────
    reason_parts = [
        f"{r.value}({_reason_detail(r, cognitive_decision, opp_score, threshold, account_features)})"
        for r in reasons
    ]
    reasoning = "Wait: " + "; ".join(reason_parts) + f". NO_ACTION reward={reward:.3f}"

    # ── Build skip record ─────────────────────────────────────────────────────
    skip_record = SkippedOpportunity(
        instrument=cognitive_decision.instrument,
        direction=cognitive_decision.direction.value,
        opportunity_score_at_skip=opp_score.opportunity_score,
        quality_threshold_at_skip=threshold,
        wait_reasons=list(reasons),
        primary_reason=primary_reason,
        cognitive_action=cognitive_decision.action.value,
        uncertainty_score=cognitive_decision.uncertainty_score,
        conviction_score=cognitive_decision.conviction_score,
        entry_urgency=cognitive_decision.entry_urgency,
        prospective_no_action_reward=reward,
        skipped_at=now,
    )

    return WaitDecision(
        should_wait=True,
        wait_reasons=tuple(reasons),
        primary_reason=primary_reason,
        prospective_no_action_reward=reward,
        quality_gap=quality_gap,
        reasoning=reasoning,
        skip_record=skip_record,
    )


def _reason_detail(
    reason: WaitReason,
    cd: "CognitiveDecision",
    os: "OpportunityScore",
    threshold: float,
    af: "AccountFeatures | None",
) -> str:
    """Brief detail string embedded in the reasoning narrative."""
    if reason == WaitReason.COGNITIVE_PASSIVE:
        return cd.action.value
    if reason == WaitReason.UNCERTAINTY_HIGH:
        return f"unc={cd.uncertainty_score:.2f}"
    if reason == WaitReason.OVERTRADING:
        freq = getattr(af, "trade_frequency_ratio", "?") if af else "?"
        return f"freq_ratio={freq}"
    if reason == WaitReason.CONFIRMATION_PENDING:
        return f"MONITOR conviction={cd.conviction_score:.2f}"
    if reason == WaitReason.IMPULSIVE_GUARD:
        return f"urgency={cd.entry_urgency:.2f} conv={cd.conviction_score:.2f}"
    if reason == WaitReason.LOW_QUALITY:
        return f"score={os.opportunity_score:.3f}<{threshold:.3f}"
    if reason == WaitReason.CAPITAL_UNAVAILABLE:
        return "PriorityTier.PASS"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# PatienceEngine
# ─────────────────────────────────────────────────────────────────────────────


class PatienceEngine:
    """Stateful manager for one instrument's patience discipline.

    Wraps the three stateless public callables and maintains a
    ``PatienceState`` that accumulates skip history, trade counts, and
    the current adaptive quality threshold.

    Typical usage::

        engine = PatienceEngine(instrument="EURUSD")

        # On each brain tick:
        decision = engine.evaluate_wait_decision(cog_dec, opp_score)
        if decision.should_wait:
            skip = engine.record_skip(decision)   # persist the skip
        else:
            engine.record_trade(was_impulsive=False)

        # Later, when market data shows what would have happened:
        engine.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.CORRECT_SKIP, pnl_estimate=0.3)

        # Query current patience metrics:
        metrics   = engine.patience_score()
        threshold = engine.minimum_quality_threshold()
    """

    def __init__(
        self,
        instrument:         str   = "",
        initial_threshold:  float = DEFAULT_QUALITY_THRESHOLD,
    ) -> None:
        self._state = PatienceState(
            instrument=instrument,
            current_threshold=initial_threshold,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> PatienceState:
        return self._state

    @property
    def instrument(self) -> str:
        return self._state.instrument

    # ── Three public callables (engine-method form) ────────────────────────────

    def evaluate_wait_decision(
        self,
        cognitive_decision:     "CognitiveDecision",
        opp_score:              "OpportunityScore",
        account_features:       "AccountFeatures | None" = None,
        capital_unavailable:    bool = False,
        now:                    float | None = None,
    ) -> WaitDecision:
        """Evaluate whether to wait.  Uses the current adaptive threshold.

        Delegates to the module-level ``evaluate_wait_decision()``
        with the engine's current adaptive threshold from
        ``self.minimum_quality_threshold()``.
        """
        threshold = self.minimum_quality_threshold()
        return evaluate_wait_decision(
            cognitive_decision=cognitive_decision,
            opp_score=opp_score,
            quality_threshold=threshold,
            account_features=account_features,
            capital_unavailable=capital_unavailable,
            now=now,
        )

    def patience_score(self) -> PatienceMetrics:
        """Current patience discipline metrics."""
        return patience_score(self._state)

    def minimum_quality_threshold(self) -> float:
        """Current adaptive minimum opportunity quality threshold."""
        return minimum_quality_threshold(self._state)

    # ── Skip lifecycle ─────────────────────────────────────────────────────────

    def record_skip(
        self,
        decision: WaitDecision,
    ) -> SkippedOpportunity | None:
        """Persist a skip from a ``WaitDecision`` result.

        Returns the stored ``SkippedOpportunity`` or ``None`` if
        ``decision.skip_record`` is absent (i.e. ``should_wait=False``).
        """
        if decision.skip_record is None:
            return None
        skip = decision.skip_record
        self._state._append_skip(skip)
        self._state.total_skips += 1
        self._state.recent_wait_flags.append(True)
        return skip

    def record_trade(self, was_impulsive: bool = False) -> None:
        """Record an entry decision (trade was placed, not skipped).

        Parameters
        ----------
        was_impulsive:
            ``True`` when the entry had ``entry_urgency > IMPULSIVE_URGENCY_THRESHOLD``
            and ``conviction_score < IMPULSIVE_CONVICTION_FLOOR`` — i.e.
            the caller determined the entry was reckless.  Increments
            ``state.impulsive_entries`` which raises the adaptive threshold.
        """
        self._state.total_trades += 1
        self._state.recent_wait_flags.append(False)
        if was_impulsive:
            self._state.impulsive_entries += 1

    def record_overtrading_event(self) -> None:
        """Record one overtrading detection event.

        Called externally when the account pipeline flags ``is_overtrading``
        so the patience engine can track frequency pressure independently
        of individual skip decisions.
        """
        self._state.overtrading_events += 1

    def resolve_skipped_opportunity(
        self,
        skip_id:        str,
        outcome:        SkipOutcome,
        pnl_estimate:   float = 0.0,
        now:            float | None = None,
    ) -> SkippedOpportunity | None:
        """Resolve the outcome of a previously skipped opportunity.

        Finds the ``SkippedOpportunity`` by ``skip_id``, calls
        ``.resolve()`` with the computed realized reward, updates the
        patience-state counters, and adapts the quality threshold.

        Parameters
        ----------
        skip_id:
            The ``skip_id`` returned by ``record_skip()``.
        outcome:
            ``CORRECT_SKIP``, ``MISSED_OPPORTUNITY``, or ``NEUTRAL``.
        pnl_estimate:
            Magnitude of the pnl avoided (CORRECT_SKIP) or missed
            (MISSED_OPPORTUNITY).  Unsigned; sign is inferred from outcome.
        now:
            Resolution timestamp.

        Returns
        -------
        The resolved ``SkippedOpportunity``, or ``None`` if ``skip_id``
        not found.
        """
        now = now if now is not None else time.time()
        skip = self._state.find_skip(skip_id)
        if skip is None or skip.is_resolved:
            return skip

        realized_reward = compute_realized_reward(outcome, pnl_estimate)
        skip.resolve(outcome, pnl_estimate, realized_reward, now)

        # Update counters
        if outcome == SkipOutcome.CORRECT_SKIP:
            self._state.correct_skips += 1
        elif outcome == SkipOutcome.MISSED_OPPORTUNITY:
            self._state.missed_opportunities += 1

        # Adapt the threshold based on the new resolved data
        self._update_threshold(now)
        return skip

    def age_pending_skips(
        self,
        resolution_window_seconds: float = DEFAULT_RESOLUTION_WINDOW_SECONDS,
        now: float | None = None,
    ) -> int:
        """Auto-resolve pending skips that have exceeded the resolution window.

        When no explicit outcome is provided within ``resolution_window_seconds``
        after the skip, resolves as ``NEUTRAL`` (no significant movement).

        Returns the number of skips auto-resolved.
        """
        now = now if now is not None else time.time()
        resolved_count = 0
        for skip in self._state.pending_skips():
            if now - skip.skipped_at >= resolution_window_seconds:
                skip.resolve(SkipOutcome.NEUTRAL, 0.0, 0.0, now)
                resolved_count += 1
        return resolved_count

    # ── Queries ────────────────────────────────────────────────────────────────

    def pending_skips(self) -> list[SkippedOpportunity]:
        """All skipped opportunities whose outcome is not yet resolved."""
        return self._state.pending_skips()

    def recent_skips(self, n: int = 10) -> list[SkippedOpportunity]:
        """Most recent ``n`` skipped opportunities (resolved or pending)."""
        return self._state.skipped_opportunities[-n:]

    def summary(self) -> dict:
        """Diagnostic snapshot of current patience state and metrics."""
        metrics = self.patience_score()
        return {
            "state":            self._state.to_dict(),
            "metrics":          metrics.to_dict(),
            "adaptive_threshold": round(self.minimum_quality_threshold(), 4),
            "pending_skips":    len(self.pending_skips()),
        }

    # ── Internal threshold adaptation ─────────────────────────────────────────

    def _update_threshold(self, now: float) -> None:
        """Mutate ``state.current_threshold`` based on latest resolved data.

        Called automatically by ``resolve_skipped_opportunity()``.
        Pure mutation — does not return a value.
        """
        proposed = minimum_quality_threshold(self._state)
        self._state.current_threshold = proposed
        self._state.last_threshold_update = now


# ─────────────────────────────────────────────────────────────────────────────
# Module-level re-exports (convenience — all three callables from one import)
# ─────────────────────────────────────────────────────────────────────────────

# patience_score and minimum_quality_threshold imported from patience_scoring
# so they don't need to be redefined here.

__all__ = [
    # ── Three public callables (C13 interface) ────────────────────────────────
    "evaluate_wait_decision",
    "patience_score",
    "minimum_quality_threshold",
    # ── Engine ───────────────────────────────────────────────────────────────
    "PatienceEngine",
]
