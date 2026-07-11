"""C12 — Thesis Aging: three public callables + ThesisAgingEngine.

Three stateless module-level functions (the C12 public interface):

    age_thesis(hyp, state, now=None)              → ThesisAgingResult
    expire_thesis(hyp, state, reason="", now=None)→ None
    update_confidence(hyp, state, delta, ...)     → None

``ThesisAgingEngine`` wraps a collection of ``ThesisAgingState`` objects
and provides the same three operations plus bulk ``age_all()`` and
lifecycle helpers (``register``, ``state_for``, ``drop``).

Decay model
-----------
Confidence follows **exponential decay** from the last confirmation event:

    new_confidence = old_confidence × exp(−λ × elapsed)

where ``λ = ln(2) / half_life_seconds`` and ``elapsed`` is the time since
the most recent CONFIRMATION event (or, if none yet, since the hypothesis
was formed).

This means:
* A freshly confirmed hypothesis retains its confidence for one full
  half-life before losing 50 % — the decay clock is reset on each
  confirmation, rewarding hypotheses that continue to receive market
  support.
* A hypothesis that receives no confirmation decays at a rate determined
  by its opportunity class (MICRO_SCALP decays in minutes; SWING in weeks).

Expiry conditions
-----------------
A thesis is expired (``HypothesisStatus.EXPIRED``) when *any* of:

1. Decayed confidence falls below ``profile.expiry_threshold``.
2. Thesis age (``now − created_at``) exceeds ``profile.max_age_seconds``.
3. ``expire_thesis()`` is called explicitly.
4. ``update_confidence()`` applies an INVALIDATION delta that drives
   confidence below the expiry threshold.

Relationship to existing infrastructure
----------------------------------------
* ``Hypothesis`` and ``HypothesisStatus`` in ``brain_models.py`` are
  **untouched** — this module mutates ``hyp.confidence`` and ``hyp.status``
  (EXPIRED) through the existing public interface; it never bypasses it.
* ``HypothesisTracker.age()`` (in ``hypotheses.py``) does binary staleness
  expiry based on ``is_stale()`` (``last_updated_at`` timeout).  C12 is
  complementary: it applies *gradual* decay driven by time-since-confirmation
  with per-class half-lives.  Both can coexist; ``HypothesisTracker.age()``
  remains unchanged.
* No imports of ``QDEDecision`` or any execution-layer type.
* All three stateless functions are safe to call on every brain tick.
"""

from __future__ import annotations

import time
from typing import Callable

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.brain_types import HypothesisStatus
from trainer.qde.brain.thesis.thesis_types import (
    DEFAULT_DECAY_PROFILES as DEFAULT_DECAY_PROFILES,  # re-export
)
from trainer.qde.brain.thesis.thesis_types import (
    AgingRecord,
    ThesisAgingResult,
    ThesisAgingState,
    ThesisEvent,
    ThesisEventType,
    profile_for_class,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _is_terminal(hyp: Hypothesis) -> bool:
    return hyp.status in (
        HypothesisStatus.EXPIRED,
        HypothesisStatus.INVALIDATED,
        HypothesisStatus.CONFIRMED,
    )


def _do_expire(
    hyp: Hypothesis,
    state: ThesisAgingState,
    reason: str,
    trigger: ThesisEventType,
    conf_before: float,
    now: float,
) -> None:
    """Shared expiry finalisation used by age_thesis and expire_thesis."""
    hyp.status = HypothesisStatus.EXPIRED
    hyp.confidence = 0.0
    hyp.last_updated_at = now

    event = ThesisEvent(
        event_type=trigger,
        evidence=reason,
        confidence_before=conf_before,
        confidence_after=0.0,
        occurred_at=now,
    )
    state._append_event(event, is_confirmation=False)
    state.is_expired = True
    state.expired_at = now

    record = AgingRecord(
        hypothesis_id=state.hypothesis_id,
        confidence=0.0,
        age_seconds=state.age(now),
        elapsed_since_confirm=state.elapsed_since_confirmation(now),
        confirmation_count=state.confirmation_count,
        invalidation_count=state.invalidation_count,
        trigger=trigger,
        occurred_at=now,
    )
    state._append_history(record)


# ─────────────────────────────────────────────────────────────────────────────
# age_thesis
# ─────────────────────────────────────────────────────────────────────────────


def age_thesis(
    hyp: Hypothesis,
    state: ThesisAgingState,
    now: float | None = None,
) -> ThesisAgingResult:
    """Apply time-based confidence decay to one hypothesis.

    Reads the decay parameters from ``state.decay_profile`` (set at
    registration time via ``ThesisAgingEngine.register()`` or constructed
    directly from ``profile_for_class()``).

    Returns a ``ThesisAgingResult`` describing what happened.  The result
    has ``decay_applied=False`` when the tick was skipped (hypothesis
    already terminal, or too soon since the last decay tick).

    The function mutates:
      * ``hyp.confidence`` — reduced by the decay factor.
      * ``hyp.status``    — set to EXPIRED if confidence or age threshold
                            is crossed.
      * ``hyp.last_updated_at`` — touched when decay is applied.
      * ``state.*``       — event log and aging history updated.

    Parameters
    ----------
    hyp:
        The live ``Hypothesis`` object to age.  Mutated in place.
    state:
        The companion ``ThesisAgingState`` for this hypothesis.  Mutated
        in place.
    now:
        Wall-clock timestamp (seconds since epoch).  Defaults to
        ``time.time()`` when not supplied.
    """
    now = now if now is not None else time.time()
    profile = state.decay_profile
    conf_before = hyp.confidence

    # ── Guard: already terminal ───────────────────────────────────────────────
    if _is_terminal(hyp):
        return ThesisAgingResult(
            hypothesis_id=hyp.hypothesis_id,
            instrument=hyp.instrument,
            confidence_before=conf_before,
            confidence_after=conf_before,
            age_seconds=state.age(now),
            elapsed_since_confirm=state.elapsed_since_confirmation(now),
            decay_factor=1.0,
            expired=hyp.status == HypothesisStatus.EXPIRED,
            decay_applied=False,
            reason="already terminal",
        )

    # ── Guard: too soon since last decay tick ─────────────────────────────────
    since_last = now - state.last_decayed_at
    if since_last < profile.min_decay_interval_seconds:
        return ThesisAgingResult(
            hypothesis_id=hyp.hypothesis_id,
            instrument=hyp.instrument,
            confidence_before=conf_before,
            confidence_after=conf_before,
            age_seconds=state.age(now),
            elapsed_since_confirm=state.elapsed_since_confirmation(now),
            decay_factor=1.0,
            expired=False,
            decay_applied=False,
            reason=f"too soon ({since_last:.1f}s < {profile.min_decay_interval_seconds}s interval)",
        )

    # ── Hard age expiry check ─────────────────────────────────────────────────
    age_s = state.age(now)
    if age_s >= profile.max_age_seconds:
        reason = f"max_age exceeded: {age_s:.0f}s >= {profile.max_age_seconds:.0f}s"
        _do_expire(hyp, state, reason, ThesisEventType.EXPIRY, conf_before, now)
        return ThesisAgingResult(
            hypothesis_id=hyp.hypothesis_id,
            instrument=hyp.instrument,
            confidence_before=conf_before,
            confidence_after=0.0,
            age_seconds=age_s,
            elapsed_since_confirm=state.elapsed_since_confirmation(now),
            decay_factor=0.0,
            expired=True,
            decay_applied=True,
            reason=reason,
        )

    # ── Apply exponential decay ───────────────────────────────────────────────
    elapsed = state.elapsed_since_confirmation(now)
    factor = profile.decay_factor(elapsed)
    new_conf = max(0.0, conf_before * factor)

    # Write state
    hyp.confidence = new_conf
    hyp.last_updated_at = now
    state.last_decayed_at = now

    # Record event
    decay_event = ThesisEvent(
        event_type=ThesisEventType.DECAY_TICK,
        evidence=f"elapsed_since_confirm={elapsed:.0f}s factor={factor:.4f}",
        confidence_before=conf_before,
        confidence_after=new_conf,
        occurred_at=now,
    )
    # Decay ticks are stored in invalidation_events to avoid inflating
    # the confirmation count — they are "negative" pressure on the thesis.
    state._append_event(decay_event, is_confirmation=False)

    # Record aging history snapshot
    record = AgingRecord(
        hypothesis_id=state.hypothesis_id,
        confidence=new_conf,
        age_seconds=age_s,
        elapsed_since_confirm=elapsed,
        confirmation_count=state.confirmation_count,
        invalidation_count=state.invalidation_count,
        trigger=ThesisEventType.DECAY_TICK,
        occurred_at=now,
    )
    state._append_history(record)

    # ── Check expiry threshold after decay ────────────────────────────────────
    if new_conf < profile.expiry_threshold:
        reason = f"confidence {new_conf:.4f} below expiry threshold " f"{profile.expiry_threshold}"
        _do_expire(hyp, state, reason, ThesisEventType.EXPIRY, new_conf, now)
        return ThesisAgingResult(
            hypothesis_id=hyp.hypothesis_id,
            instrument=hyp.instrument,
            confidence_before=conf_before,
            confidence_after=0.0,
            age_seconds=age_s,
            elapsed_since_confirm=elapsed,
            decay_factor=factor,
            expired=True,
            decay_applied=True,
            reason=reason,
        )

    return ThesisAgingResult(
        hypothesis_id=hyp.hypothesis_id,
        instrument=hyp.instrument,
        confidence_before=conf_before,
        confidence_after=new_conf,
        age_seconds=age_s,
        elapsed_since_confirm=elapsed,
        decay_factor=factor,
        expired=False,
        decay_applied=True,
        reason=f"decayed by factor {factor:.4f} over {elapsed:.0f}s",
    )


# ─────────────────────────────────────────────────────────────────────────────
# expire_thesis
# ─────────────────────────────────────────────────────────────────────────────


def expire_thesis(
    hyp: Hypothesis,
    state: ThesisAgingState,
    reason: str = "",
    now: float | None = None,
) -> None:
    """Force-expire a thesis immediately, regardless of its current age
    or confidence.

    Idempotent: calling on an already-expired or otherwise terminal
    hypothesis is a no-op.

    Mutates ``hyp.status``, ``hyp.confidence``, ``hyp.last_updated_at``,
    and the corresponding ``state`` fields.

    Parameters
    ----------
    hyp:
        The ``Hypothesis`` to expire.  Mutated in place.
    state:
        The companion ``ThesisAgingState``.  Mutated in place.
    reason:
        Human-readable explanation stored in both the event log and
        ``hyp.invalidation_reason`` for diagnostics.
    now:
        Wall-clock timestamp.  Defaults to ``time.time()``.
    """
    if _is_terminal(hyp):
        return

    now = now if now is not None else time.time()
    conf_before = hyp.confidence
    full_reason = reason or "explicit expire_thesis() call"
    hyp.invalidation_reason = full_reason
    _do_expire(hyp, state, full_reason, ThesisEventType.FORCE_EXPIRE, conf_before, now)


# ─────────────────────────────────────────────────────────────────────────────
# update_confidence
# ─────────────────────────────────────────────────────────────────────────────


def update_confidence(
    hyp: Hypothesis,
    state: ThesisAgingState,
    delta: float,
    event_type: ThesisEventType = ThesisEventType.CONFIRMATION,
    evidence: str = "",
    now: float | None = None,
) -> None:
    """Record a confidence-altering event and apply the delta to the
    hypothesis, then reset (on confirmation) or preserve (on invalidation)
    the decay clock accordingly.

    CONFIRMATION events reset ``state.last_confirmation_at`` to ``now``,
    restarting the decay clock.  INVALIDATION events accumulate without
    resetting the clock — the thesis continues to age from its last
    confirmation.

    When the post-update confidence falls below
    ``state.decay_profile.expiry_threshold``, the thesis is expired in the
    same call.

    Parameters
    ----------
    hyp:
        The ``Hypothesis`` to update.  ``hyp.confidence`` is mutated.
    state:
        The companion ``ThesisAgingState``.  Event log and history updated.
    delta:
        Signed confidence change.  Positive for confirmations, negative for
        invalidations.  Applied as ``hyp.confidence = clamp(old + delta, 0, 1)``.
    event_type:
        ``CONFIRMATION`` or ``INVALIDATION``.  All other values are treated
        as CONFIRMATION for decay-clock purposes.
    evidence:
        Human-readable description of the market event.
    now:
        Wall-clock timestamp.  Defaults to ``time.time()``.
    """
    if _is_terminal(hyp):
        return

    now = now if now is not None else time.time()
    profile = state.decay_profile
    conf_before = hyp.confidence
    new_conf = max(0.0, min(1.0, conf_before + delta))

    hyp.confidence = new_conf
    hyp.last_updated_at = now

    is_confirmation = event_type == ThesisEventType.CONFIRMATION

    event = ThesisEvent(
        event_type=event_type,
        evidence=evidence,
        confidence_before=conf_before,
        confidence_after=new_conf,
        occurred_at=now,
    )
    state._append_event(event, is_confirmation=is_confirmation)

    if is_confirmation:
        # Reset the decay clock on every confirmation.
        state.last_confirmation_at = now

    # Record aging history snapshot
    record = AgingRecord(
        hypothesis_id=state.hypothesis_id,
        confidence=new_conf,
        age_seconds=state.age(now),
        elapsed_since_confirm=state.elapsed_since_confirmation(now),
        confirmation_count=state.confirmation_count,
        invalidation_count=state.invalidation_count,
        trigger=event_type,
        occurred_at=now,
    )
    state._append_history(record)

    # Auto-expire if confidence has crossed the floor.
    if new_conf < profile.expiry_threshold:
        reason = (
            f"confidence {new_conf:.4f} below expiry threshold "
            f"{profile.expiry_threshold} after {event_type.value}"
        )
        hyp.invalidation_reason = reason
        _do_expire(hyp, state, reason, ThesisEventType.EXPIRY, new_conf, now)


# ─────────────────────────────────────────────────────────────────────────────
# ThesisAgingEngine
# ─────────────────────────────────────────────────────────────────────────────


class ThesisAgingEngine:
    """Stateful manager for a collection of per-hypothesis aging states.

    ``ThesisAgingEngine`` is the primary high-level interface when a caller
    manages multiple hypotheses (e.g. ``QDEBrain`` at the per-symbol level).
    It wraps the three stateless functions and adds:

    * **Registration** — ``register(hyp, opp_class_value)`` creates and
      stores a ``ThesisAgingState`` for each new hypothesis.
    * **Bulk aging** — ``age_all(hypotheses)`` calls ``age_thesis()`` for
      every live hypothesis in one pass.
    * **Lifecycle cleanup** — ``drop(hypothesis_id)`` removes the state
      when a hypothesis is pruned from ``BrainState.hypotheses``.

    The engine is **not** stateless — it holds per-hypothesis aging state.
    It is safe to construct one per symbol (mirroring the per-symbol
    ``BrainState`` pattern) or one global engine keyed by hypothesis_id.
    """

    def __init__(self) -> None:
        self._states: dict[str, ThesisAgingState] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def register(
        self,
        hyp: Hypothesis,
        opp_class_value: str = "INTRADAY",
        now: float | None = None,
    ) -> ThesisAgingState:
        """Create and store a ``ThesisAgingState`` for ``hyp``.

        Idempotent: if the hypothesis is already registered, the existing
        state is returned unchanged (re-registration does not reset events
        or history).

        Parameters
        ----------
        hyp:
            The ``Hypothesis`` to register.
        opp_class_value:
            String value of the ``OpportunityClass`` enum
            (``"MICRO_SCALP"``, ``"SCALP"``, ``"INTRADAY"``,
            ``"MULTI_SESSION"``, ``"SWING"``).  Defaults to ``"INTRADAY"``.
        now:
            Registration timestamp.  Defaults to ``time.time()``.
        """
        if hyp.hypothesis_id in self._states:
            return self._states[hyp.hypothesis_id]

        now = now if now is not None else time.time()
        profile = profile_for_class(opp_class_value)
        state = ThesisAgingState(
            hypothesis_id=hyp.hypothesis_id,
            instrument=hyp.instrument,
            created_at=hyp.created_at,
            decay_profile=profile,
            last_confirmation_at=hyp.created_at,
            last_decayed_at=hyp.created_at,
        )
        self._states[hyp.hypothesis_id] = state
        return state

    def drop(self, hypothesis_id: str) -> None:
        """Remove the aging state for ``hypothesis_id``.

        Called when the hypothesis is pruned from ``BrainState.hypotheses``
        to prevent memory leaks.  Safe to call on unregistered IDs.
        """
        self._states.pop(hypothesis_id, None)

    def state_for(self, hypothesis_id: str) -> ThesisAgingState | None:
        """Return the aging state for ``hypothesis_id``, or ``None``."""
        return self._states.get(hypothesis_id)

    # ── Per-hypothesis operations ─────────────────────────────────────────────

    def age_thesis(
        self,
        hyp: Hypothesis,
        now: float | None = None,
    ) -> ThesisAgingResult | None:
        """Apply time-based decay to ``hyp``.

        Returns ``None`` if ``hyp`` is not registered with this engine.
        Auto-registers with the INTRADAY profile when called on an
        unregistered hypothesis so callers do not need to register before
        ageing — though explicit registration with the correct class is
        preferred.
        """
        state = self._states.get(hyp.hypothesis_id)
        if state is None:
            state = self.register(hyp, "INTRADAY", now=now)
        return age_thesis(hyp, state, now=now)

    def expire_thesis(
        self,
        hyp: Hypothesis,
        reason: str = "",
        now: float | None = None,
    ) -> None:
        """Force-expire ``hyp``.  Auto-registers if needed."""
        state = self._states.get(hyp.hypothesis_id)
        if state is None:
            state = self.register(hyp, "INTRADAY", now=now)
        expire_thesis(hyp, state, reason=reason, now=now)

    def update_confidence(
        self,
        hyp: Hypothesis,
        delta: float,
        event_type: ThesisEventType = ThesisEventType.CONFIRMATION,
        evidence: str = "",
        now: float | None = None,
    ) -> None:
        """Record an event and apply ``delta`` to ``hyp.confidence``.
        Auto-registers if needed.
        """
        state = self._states.get(hyp.hypothesis_id)
        if state is None:
            state = self.register(hyp, "INTRADAY", now=now)
        update_confidence(hyp, state, delta, event_type, evidence, now=now)

    # ── Bulk operations ───────────────────────────────────────────────────────

    def age_all(
        self,
        hypotheses: dict[str, Hypothesis],
        opp_class_fn: Callable[[Hypothesis], str] | None = None,
        now: float | None = None,
    ) -> list[ThesisAgingResult]:
        """Apply ``age_thesis()`` to every non-terminal hypothesis.

        Parameters
        ----------
        hypotheses:
            The per-symbol hypothesis dict from ``BrainState.hypotheses``.
        opp_class_fn:
            Optional callable that maps a ``Hypothesis`` to its
            ``OpportunityClass`` string value, used to auto-register
            hypotheses that have not been explicitly registered.  When
            ``None``, unregistered hypotheses are registered as INTRADAY.
        now:
            Shared timestamp for the batch.

        Returns
        -------
        ``list[ThesisAgingResult]``
            One result per hypothesis where decay was attempted
            (terminals are skipped).
        """
        now = now if now is not None else time.time()
        results: list[ThesisAgingResult] = []
        for hyp in hypotheses.values():
            if _is_terminal(hyp):
                continue
            state = self._states.get(hyp.hypothesis_id)
            if state is None:
                cls_value = opp_class_fn(hyp) if opp_class_fn else "INTRADAY"
                state = self.register(hyp, cls_value, now=now)
            result = age_thesis(hyp, state, now=now)
            results.append(result)
        return results

    def summary(self) -> dict:
        """Diagnostic snapshot of all tracked aging states."""
        return {
            "tracked_count": len(self._states),
            "expired_count": sum(1 for s in self._states.values() if s.is_expired),
            "states": {hid: s.to_dict() for hid, s in self._states.items()},
        }


__all__ = [
    # Stateless functions (C12 public interface)
    "age_thesis",
    "expire_thesis",
    "update_confidence",
    # Engine
    "ThesisAgingEngine",
    # Re-exported for convenience
    "DEFAULT_DECAY_PROFILES",
]
