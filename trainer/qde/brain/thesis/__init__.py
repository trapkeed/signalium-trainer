"""C12 — Thesis Aging: per-hypothesis confidence decay, expiry, and history.

This package implements time-based confidence decay for QDE hypotheses.
When the market fails to confirm a thesis, confidence decays exponentially
at a rate calibrated to the opportunity class:

    MICRO_SCALP   — half-life ≈ 10 minutes,  hard expiry 15 minutes
    SCALP         — half-life ≈ 30 minutes,  hard expiry 60 minutes
    INTRADAY      — half-life ≈ 4 hours,     hard expiry 8 hours
    MULTI_SESSION — half-life ≈ 2 days,      hard expiry 3 days
    SWING         — half-life ≈ 2 weeks,     hard expiry 3 weeks

Three public callables (C12 requirement)
-----------------------------------------
``age_thesis(hyp, state, now=None) → ThesisAgingResult``
    Apply time-based confidence decay.  The decay clock runs from the last
    CONFIRMATION event; each confirmation resets the clock.  Expires the
    thesis automatically when confidence falls below the class threshold
    or the hypothesis exceeds its maximum age.

``expire_thesis(hyp, state, reason="", now=None) → None``
    Force-expire a thesis immediately, regardless of age or confidence.
    Idempotent on already-terminal hypotheses.

``update_confidence(hyp, state, delta, event_type=CONFIRMATION, ...) → None``
    Record a structured confirmation or invalidation event and apply the
    confidence delta.  CONFIRMATION events reset the decay clock so a
    thesis that keeps receiving market support does not decay.

Modules
-------
``thesis_types``
    All data classes: ``ThesisDecayProfile``, ``ThesisAgingState``,
    ``ThesisEvent``, ``AgingRecord``, ``ThesisAgingResult``,
    ``ThesisEventType``, ``DEFAULT_DECAY_PROFILES``, ``profile_for_class``.

``thesis_aging``
    The three stateless public functions and ``ThesisAgingEngine`` (the
    stateful wrapper for managing a collection of aging states).

Design principles
-----------------
* ``Hypothesis`` and ``HypothesisTracker`` in the parent package are
  **not modified** — this package only mutates the existing public fields
  (``hyp.confidence``, ``hyp.status``, ``hyp.last_updated_at``) through
  the existing ``Hypothesis`` interface.
* No import of ``QDEDecision`` or any execution-layer type.
* Stateless functions are safe to call on every brain tick.
"""

from __future__ import annotations

# ── Functions + Engine ────────────────────────────────────────────────────────
from trainer.qde.brain.thesis.thesis_aging import (
    ThesisAgingEngine,
    age_thesis,
    expire_thesis,
    update_confidence,
)

# ── Types ─────────────────────────────────────────────────────────────────────
from trainer.qde.brain.thesis.thesis_types import (
    DEFAULT_DECAY_PROFILES,
    DEFAULT_EXPIRY_THRESHOLD,
    MAX_EVENTS,
    MAX_HISTORY,
    MIN_DECAY_INTERVAL_SECONDS,
    AgingRecord,
    ThesisAgingResult,
    ThesisAgingState,
    ThesisDecayProfile,
    ThesisEvent,
    ThesisEventType,
    profile_for_class,
)

__all__ = [
    # ── Types ─────────────────────────────────────────────────────────────────
    "ThesisEventType",
    "ThesisEvent",
    "ThesisDecayProfile",
    "DEFAULT_DECAY_PROFILES",
    "profile_for_class",
    "AgingRecord",
    "ThesisAgingState",
    "ThesisAgingResult",
    # ── Public callables (C12 interface) ──────────────────────────────────────
    "age_thesis",
    "expire_thesis",
    "update_confidence",
    # ── Engine ────────────────────────────────────────────────────────────────
    "ThesisAgingEngine",
    # ── Constants ─────────────────────────────────────────────────────────────
    "DEFAULT_EXPIRY_THRESHOLD",
    "MAX_EVENTS",
    "MAX_HISTORY",
    "MIN_DECAY_INTERVAL_SECONDS",
]
