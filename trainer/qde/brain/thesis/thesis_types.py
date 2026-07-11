"""C12 — Thesis Aging: types, decay profiles, and event records.

All data structures used by the thesis aging system.  No mutation logic
lives here — see ``thesis_aging.py`` for the three public callables.

Key design choices
------------------
``ThesisDecayProfile``
    Frozen dataclass that bundles the three parameters that govern how a
    thesis's confidence evolves over time without confirmation:

    * ``half_life_seconds`` — time until confidence halves in the absence
      of any confirmation event (exponential decay clock starts from the
      last confirmation, or from hypothesis formation if none yet).

    * ``max_age_seconds`` — hard upper bound on thesis lifetime; a thesis
      that reaches this age is expired regardless of its confidence level.
      Defaults to 4× the half-life so a thesis can survive a full two
      half-life cycles before being unconditionally retired.

    * ``expiry_threshold`` — confidence floor; when decayed confidence
      falls below this value the thesis is auto-expired.  Keeps the
      system from tracking near-zero-conviction theses that would never
      drive a decision anyway.

``ThesisAgingState``
    Mutable per-hypothesis companion that holds the three pieces of state
    that the existing ``Hypothesis`` dataclass does not capture:

    * A structured list of **confirmation events** (timestamped) so the
      aging engine knows when the decay clock was last reset.
    * A structured list of **invalidation events** (timestamped) for
      post-hoc analysis and pattern recognition.
    * An **aging history** — a bounded time-series of ``AgingRecord``
      snapshots, one written each time decay is applied, so callers can
      reconstruct how confidence evolved and why.

    This deliberately wraps the ``Hypothesis`` rather than modifying it,
    keeping ``brain_models.Hypothesis`` unchanged.

``AgingRecord``
    Immutable snapshot of a thesis's confidence and event counts at a
    specific moment in time.  Written on every ``age_thesis()`` call and
    on every ``update_confidence()`` call so the full decay trajectory
    is preserved for inspection and learning.

``ThesisAgingResult``
    Immutable result returned by ``age_thesis()``.  Summarises what
    happened in one decay tick so callers can log, alert, or react
    (e.g. demote an opportunity if its thesis confidence crossed a
    tier boundary).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Avoid circular imports at runtime

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maximum number of events retained per hypothesis (rolling window).
MAX_EVENTS = 50

# Maximum number of AgingRecord snapshots retained per hypothesis.
MAX_HISTORY = 200

# Confidence below this value triggers auto-expiry on the next decay tick.
DEFAULT_EXPIRY_THRESHOLD = 0.08

# Minimum interval between consecutive decay ticks (seconds).  Prevents
# accumulation of many zero-delta records when the system is called on
# every market tick.
MIN_DECAY_INTERVAL_SECONDS = 10.0

# Per-class half-life and max-age defaults (in seconds).
# Values derived from C12 spec (midpoint of given range for half-life;
# upper end of range for max-age).
#
# Opportunity class │ Half-life range │ Half-life used │ Max age (upper)
# ──────────────────┼─────────────────┼────────────────┼────────────────
# MICRO_SCALP       │ 5-15 min        │ 10 min (600s)  │ 15 min (900s)
# SCALP             │ 15-60 min       │ 30 min (1800s) │ 60 min (3600s)
# INTRADAY          │ 1-8 hr          │ 4 hr (14400s)  │ 8 hr (28800s)
# MULTI_SESSION     │ 1-3 days        │ 2 days(172800) │ 3 days(259200)
# SWING             │ 1-3 weeks       │ 2 wk (1209600) │ 3 wk (1814400)

_PROFILE_PARAMS: dict[str, tuple[float, float]] = {
    "MICRO_SCALP": (600.0, 900.0),
    "SCALP": (1800.0, 3600.0),
    "INTRADAY": (14400.0, 28800.0),
    "MULTI_SESSION": (172800.0, 259200.0),
    "SWING": (1209600.0, 1814400.0),
}

# Fall-through default for callers that have not classified the opportunity.
_DEFAULT_CLASS = "INTRADAY"


# ─────────────────────────────────────────────────────────────────────────────
# ThesisEventType
# ─────────────────────────────────────────────────────────────────────────────


class ThesisEventType(str, Enum):
    """Discriminator for entries in the per-hypothesis event log."""

    CONFIRMATION = "CONFIRMATION"  # market confirms the thesis direction
    INVALIDATION = "INVALIDATION"  # market contradicts the thesis direction
    DECAY_TICK = "DECAY_TICK"  # scheduled confidence decay applied
    EXPIRY = "EXPIRY"  # thesis expired (age or threshold)
    FORCE_EXPIRE = "FORCE_EXPIRE"  # explicit ``expire_thesis()`` call


# ─────────────────────────────────────────────────────────────────────────────
# ThesisEvent
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThesisEvent:
    """One timestamped event in a thesis's lifecycle.

    ``confidence_before`` and ``confidence_after`` allow the full
    confidence trajectory to be reconstructed from the event log alone,
    independent of ``AgingRecord`` snapshots.
    """

    event_type: ThesisEventType
    evidence: str = ""
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    occurred_at: float = field(default_factory=time.time)

    @property
    def delta(self) -> float:
        return self.confidence_after - self.confidence_before

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "evidence": self.evidence,
            "confidence_before": round(self.confidence_before, 4),
            "confidence_after": round(self.confidence_after, 4),
            "delta": round(self.delta, 4),
            "occurred_at": round(self.occurred_at, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ThesisDecayProfile
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThesisDecayProfile:
    """Decay parameters for one opportunity class.

    All three fields are in seconds.

    ``half_life_seconds``
        Time for confidence to halve with no confirmation.
        Derived from ``λ = ln(2) / half_life_seconds``; applied as
        ``new_confidence = old × exp(−λ × elapsed)``.

    ``max_age_seconds``
        Hard expiry ceiling.  A thesis older than this is force-expired
        on the next ``age_thesis()`` call regardless of its confidence.

    ``expiry_threshold``
        Minimum confidence before auto-expiry.  When decayed confidence
        falls below this floor the thesis is expired in the same tick.

    ``min_decay_interval_seconds``
        Guard against calling decay too frequently (e.g. on every tick
        for a SWING hypothesis).  If less than this has elapsed since the
        last decay tick, ``age_thesis()`` returns a no-op result.
    """

    half_life_seconds: float
    max_age_seconds: float
    expiry_threshold: float = DEFAULT_EXPIRY_THRESHOLD
    min_decay_interval_seconds: float = MIN_DECAY_INTERVAL_SECONDS

    @property
    def decay_rate(self) -> float:
        """λ in the exponential decay formula ``exp(−λ × t)``."""
        return math.log(2.0) / max(1.0, self.half_life_seconds)

    def decay_factor(self, elapsed_seconds: float) -> float:
        """Fraction of confidence retained after ``elapsed_seconds``
        with no confirmation.  Always in (0, 1].
        """
        return math.exp(-self.decay_rate * max(0.0, elapsed_seconds))

    def to_dict(self) -> dict:
        return {
            "half_life_seconds": self.half_life_seconds,
            "max_age_seconds": self.max_age_seconds,
            "expiry_threshold": self.expiry_threshold,
            "min_decay_interval_seconds": self.min_decay_interval_seconds,
            "decay_rate": round(self.decay_rate, 8),
        }


def _build_profile(class_name: str) -> ThesisDecayProfile:
    half_life, max_age = _PROFILE_PARAMS.get(class_name, _PROFILE_PARAMS[_DEFAULT_CLASS])
    return ThesisDecayProfile(half_life_seconds=half_life, max_age_seconds=max_age)


# Default profiles, one per OpportunityClass.  Callers import
# ``DEFAULT_DECAY_PROFILES[opp_class.value]`` or use
# ``profile_for_class(opp_class_value)``.
DEFAULT_DECAY_PROFILES: dict[str, ThesisDecayProfile] = {
    name: _build_profile(name) for name in _PROFILE_PARAMS
}


def profile_for_class(opp_class_value: str) -> ThesisDecayProfile:
    """Return the ``ThesisDecayProfile`` for ``opp_class_value``.

    Falls back to the INTRADAY profile for unrecognised values so callers
    never need to guard against ``KeyError``.
    """
    return DEFAULT_DECAY_PROFILES.get(opp_class_value, DEFAULT_DECAY_PROFILES[_DEFAULT_CLASS])


# ─────────────────────────────────────────────────────────────────────────────
# AgingRecord
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgingRecord:
    """Immutable snapshot of a thesis's state at a moment in time.

    One record is written per ``age_thesis()`` call (decay tick) and
    per ``update_confidence()`` call (event) so the full trajectory is
    available for inspection and training signal derivation.
    """

    hypothesis_id: str
    confidence: float
    age_seconds: float  # now − created_at
    elapsed_since_confirm: float  # now − last_confirmation_at
    confirmation_count: int
    invalidation_count: int
    trigger: ThesisEventType
    occurred_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "confidence": round(self.confidence, 4),
            "age_seconds": round(self.age_seconds, 1),
            "elapsed_since_confirm": round(self.elapsed_since_confirm, 1),
            "confirmation_count": self.confirmation_count,
            "invalidation_count": self.invalidation_count,
            "trigger": self.trigger.value,
            "occurred_at": round(self.occurred_at, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ThesisAgingState
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ThesisAgingState:
    """Per-hypothesis mutable companion that holds the aging metadata the
    ``Hypothesis`` dataclass does not capture.

    ``confirmation_events`` and ``invalidation_events``
        Rolling bounded lists of ``ThesisEvent`` records so the full
        event history (up to ``MAX_EVENTS`` entries each) is available
        for inspection without modifying the core ``Hypothesis`` dataclass.

    ``aging_history``
        Bounded time-series of ``AgingRecord`` snapshots — one written
        per decay tick and per confidence-update call.

    ``last_confirmation_at``
        Timestamp of the most recent ``CONFIRMATION`` event.  The decay
        clock runs from this point (not from ``hypothesis.last_updated_at``
        which is updated by all reinforcements, not only confirmations).
        Initialised to ``created_at`` so a freshly-formed hypothesis starts
        decaying immediately.

    ``last_decayed_at``
        Timestamp of the most recent decay tick.  Used to enforce
        ``min_decay_interval_seconds`` so the system is not saturated with
        near-zero-delta decay records on high-frequency ticks.

    ``decay_profile``
        The ``ThesisDecayProfile`` for the opportunity class this thesis
        belongs to.  Set at registration time; immutable thereafter.

    ``is_expired`` / ``expired_at``
        Terminal flags set by ``expire_thesis()``.  Redundant with
        ``hypothesis.status == EXPIRED`` but maintained here so callers
        can check the aging state independently of the hypothesis object.
    """

    hypothesis_id: str
    instrument: str
    created_at: float
    decay_profile: ThesisDecayProfile

    # ── Event logs (rolling, bounded) ────────────────────────────────────────
    confirmation_events: list[ThesisEvent] = field(default_factory=list)
    invalidation_events: list[ThesisEvent] = field(default_factory=list)

    # ── Aging history (rolling, bounded) ─────────────────────────────────────
    aging_history: list[AgingRecord] = field(default_factory=list)

    # ── Decay clock ──────────────────────────────────────────────────────────
    last_confirmation_at: float = 0.0  # set to created_at in __post_init__
    last_decayed_at: float = 0.0  # set to created_at in __post_init__

    # ── Terminal flags ────────────────────────────────────────────────────────
    is_expired: bool = False
    expired_at: float | None = None

    def __post_init__(self) -> None:
        # Start both clocks from the hypothesis formation time so the very
        # first decay tick measures elapsed correctly.
        if self.last_confirmation_at == 0.0:
            self.last_confirmation_at = self.created_at
        if self.last_decayed_at == 0.0:
            self.last_decayed_at = self.created_at

    # ── Computed properties ──────────────────────────────────────────────────

    def age(self, now: float | None = None) -> float:
        """Seconds elapsed since hypothesis formation (``created_at``)."""
        now = now if now is not None else time.time()
        return max(0.0, now - self.created_at)

    def elapsed_since_confirmation(self, now: float | None = None) -> float:
        """Seconds since the last confirmation event (or formation)."""
        now = now if now is not None else time.time()
        return max(0.0, now - self.last_confirmation_at)

    @property
    def confirmation_count(self) -> int:
        return len(self.confirmation_events)

    @property
    def invalidation_count(self) -> int:
        return len(self.invalidation_events)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _append_event(self, event: ThesisEvent, is_confirmation: bool) -> None:
        target = self.confirmation_events if is_confirmation else self.invalidation_events
        target.append(event)
        if len(target) > MAX_EVENTS:
            del target[0]

    def _append_history(self, record: AgingRecord) -> None:
        self.aging_history.append(record)
        if len(self.aging_history) > MAX_HISTORY:
            del self.aging_history[0]

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "instrument": self.instrument,
            "created_at": round(self.created_at, 3),
            "decay_profile": self.decay_profile.to_dict(),
            "confirmation_count": self.confirmation_count,
            "invalidation_count": self.invalidation_count,
            "last_confirmation_at": round(self.last_confirmation_at, 3),
            "last_decayed_at": round(self.last_decayed_at, 3),
            "is_expired": self.is_expired,
            "expired_at": round(self.expired_at, 3) if self.expired_at else None,
            "aging_history_len": len(self.aging_history),
            "confirmation_events": [e.to_dict() for e in self.confirmation_events[-5:]],
            "invalidation_events": [e.to_dict() for e in self.invalidation_events[-5:]],
        }


# ─────────────────────────────────────────────────────────────────────────────
# ThesisAgingResult
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThesisAgingResult:
    """Immutable result of one ``age_thesis()`` call.

    ``expired``
        True when this tick triggered expiry (either confidence crossed
        ``expiry_threshold`` or the thesis exceeded ``max_age_seconds``).

    ``confidence_before`` / ``confidence_after``
        Values before and after the decay tick.

    ``decay_applied``
        False when the tick was skipped (elapsed < min_decay_interval or
        hypothesis was already terminal).

    ``reason``
        Human-readable explanation of the outcome, useful for logging.
    """

    hypothesis_id: str
    instrument: str
    confidence_before: float
    confidence_after: float
    age_seconds: float
    elapsed_since_confirm: float
    decay_factor: float  # multiplier applied (1.0 = no decay)
    expired: bool
    decay_applied: bool
    reason: str = ""

    @property
    def confidence_delta(self) -> float:
        return self.confidence_after - self.confidence_before

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "instrument": self.instrument,
            "confidence_before": round(self.confidence_before, 4),
            "confidence_after": round(self.confidence_after, 4),
            "confidence_delta": round(self.confidence_delta, 4),
            "age_seconds": round(self.age_seconds, 1),
            "elapsed_since_confirm": round(self.elapsed_since_confirm, 1),
            "decay_factor": round(self.decay_factor, 6),
            "expired": self.expired,
            "decay_applied": self.decay_applied,
            "reason": self.reason,
        }


__all__ = [
    "ThesisEventType",
    "ThesisEvent",
    "ThesisDecayProfile",
    "DEFAULT_DECAY_PROFILES",
    "profile_for_class",
    "AgingRecord",
    "ThesisAgingState",
    "ThesisAgingResult",
    # Constants
    "MAX_EVENTS",
    "MAX_HISTORY",
    "DEFAULT_EXPIRY_THRESHOLD",
    "MIN_DECAY_INTERVAL_SECONDS",
]
