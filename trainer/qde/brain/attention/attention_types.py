"""C10 — Attention tier classification, resource budgets, and evolution records.

Defines the four named attention tiers (WATCH → OBSERVE → HIGH_ATTENTION →
CRITICAL_ATTENTION), the resource cadence each tier implies, and the
lightweight data structures used to persist attention evolution in memory.

Design constraints
------------------
* No imports from ``qde_inference``, ``execution/``, or ``protocol/``.
* No imports from ``attention_engine`` or ``attention_allocator`` — this
  module is a pure types layer; circular-import-free.
* Tier thresholds are tunable constants, not hard-wired inside functions,
  so an operator can widen/narrow CRITICAL_ATTENTION without touching logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

# ─────────────────────────────────────────────────────────────────────────────
# Tier enum
# ─────────────────────────────────────────────────────────────────────────────


class AttentionTier(str, Enum):
    """Named priority tier for a symbol's current attention level.

    Maps linearly from a low composite score (WATCH) to maximum urgency
    (CRITICAL_ATTENTION).  The tier determines the ``ResourceBudget``
    allocated to the symbol — observation frequency, memory-write frequency,
    inference frequency, and reasoning-cycle rate.

    WATCH            — score < 0.25: baseline, minimal cadence.  Symbol is
                       tracked but unlikely to produce an actionable setup
                       in the near term.
    OBSERVE          — 0.25 ≤ score < 0.50: standard cadence.  Some signals
                       present; worth watching at a normal rate.
    HIGH_ATTENTION   — 0.50 ≤ score < 0.75: accelerated cadence.  Multiple
                       signals converging; deploy faster observation and
                       inference cycles.
    CRITICAL_ATTENTION — score ≥ 0.75: maximum resources.  Strong multi-
                       factor signal; run observation and inference every
                       tick, reasoning every minute.
    """

    WATCH = "WATCH"
    OBSERVE = "OBSERVE"
    HIGH_ATTENTION = "HIGH_ATTENTION"
    CRITICAL_ATTENTION = "CRITICAL_ATTENTION"


# Lower-bound composite score (inclusive) for each tier.
TIER_THRESHOLDS: dict[AttentionTier, float] = {
    AttentionTier.WATCH: 0.0,
    AttentionTier.OBSERVE: 0.25,
    AttentionTier.HIGH_ATTENTION: 0.50,
    AttentionTier.CRITICAL_ATTENTION: 0.75,
}


def tier_from_composite(composite: float) -> AttentionTier:
    """Map a 0-1 composite attention score to the corresponding tier.

    Uses the ``TIER_THRESHOLDS`` table — the highest tier whose lower
    bound does not exceed ``composite``.
    """
    if composite >= TIER_THRESHOLDS[AttentionTier.CRITICAL_ATTENTION]:
        return AttentionTier.CRITICAL_ATTENTION
    if composite >= TIER_THRESHOLDS[AttentionTier.HIGH_ATTENTION]:
        return AttentionTier.HIGH_ATTENTION
    if composite >= TIER_THRESHOLDS[AttentionTier.OBSERVE]:
        return AttentionTier.OBSERVE
    return AttentionTier.WATCH


# ─────────────────────────────────────────────────────────────────────────────
# Resource budgets
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResourceBudget:
    """Per-tier resource cadence specification.

    All interval fields are in seconds.  A value of ``0.0`` means "run on
    every market-data tick" (i.e. no rate-limiting at this tier).

    ``observation_interval_s``   — Minimum seconds between observation ticks
                                   for this symbol.
    ``memory_update_interval_s`` — Minimum seconds between memory writes.
    ``inference_interval_s``     — Minimum seconds between model-inference
                                   calls.
    ``reasoning_cycles_per_hour``— Target number of expensive Brain.reflect()
                                   calls per hour; 0 = disabled.
    """

    observation_interval_s: float
    memory_update_interval_s: float
    inference_interval_s: float
    reasoning_cycles_per_hour: float

    @property
    def reasoning_interval_s(self) -> float:
        """Seconds between reasoning cycles (inverse of ``reasoning_cycles_per_hour``)."""
        if self.reasoning_cycles_per_hour <= 0:
            return float("inf")
        return 3600.0 / self.reasoning_cycles_per_hour

    def to_dict(self) -> dict:
        return {
            "observation_interval_s": self.observation_interval_s,
            "memory_update_interval_s": self.memory_update_interval_s,
            "inference_interval_s": self.inference_interval_s,
            "reasoning_cycles_per_hour": self.reasoning_cycles_per_hour,
            "reasoning_interval_s": round(self.reasoning_interval_s, 1),
        }


#: Default per-tier resource budgets.
#:
#: Production defaults tuned for a live forex/index trading context.
#: Callers may pass a custom ``dict[AttentionTier, ResourceBudget]`` to
#: ``ResourceAllocator`` to override any or all tiers.
TIER_RESOURCES: dict[AttentionTier, ResourceBudget] = {
    AttentionTier.WATCH: ResourceBudget(
        observation_interval_s=300.0,  # every 5 minutes
        memory_update_interval_s=600.0,  # every 10 minutes
        inference_interval_s=300.0,  # every 5 minutes
        reasoning_cycles_per_hour=1.0,  # once per hour
    ),
    AttentionTier.OBSERVE: ResourceBudget(
        observation_interval_s=60.0,  # every minute
        memory_update_interval_s=120.0,  # every 2 minutes
        inference_interval_s=60.0,  # every minute
        reasoning_cycles_per_hour=4.0,  # every 15 minutes
    ),
    AttentionTier.HIGH_ATTENTION: ResourceBudget(
        observation_interval_s=15.0,  # every 15 seconds
        memory_update_interval_s=30.0,  # every 30 seconds
        inference_interval_s=15.0,  # every 15 seconds
        reasoning_cycles_per_hour=12.0,  # every 5 minutes
    ),
    AttentionTier.CRITICAL_ATTENTION: ResourceBudget(
        observation_interval_s=0.0,  # every tick
        memory_update_interval_s=5.0,  # every 5 seconds
        inference_interval_s=0.0,  # every tick
        reasoning_cycles_per_hour=60.0,  # every minute
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Attention snapshot (point-in-time state for memory persistence)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AttentionSnapshot:
    """Point-in-time attention state for a symbol.

    Produced by ``AttentionAPI.calculate_attention()`` and stored by
    ``AttentionMemory.record()`` so the QDE can observe how attention
    has evolved over time and query "when did HIGH_ATTENTION start?".
    """

    instrument: str
    tier: AttentionTier
    composite: float  # 0-1 raw composite score
    weight: float  # normalized weight across all tracked symbols
    signals: dict  # factor breakdown from OpportunitySignals.to_dict()
    resources: ResourceBudget
    captured_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "tier": self.tier.value,
            "composite": round(self.composite, 4),
            "weight": round(self.weight, 4),
            "signals": self.signals,
            "resources": self.resources.to_dict(),
            "captured_at": round(self.captured_at, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Attention evolution record (tier-transition history)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AttentionEvolutionRecord:
    """A tier-transition event stored in attention memory.

    Written only when the tier changes (or on initial observation), so the
    evolution log is compact and efficient to query for patterns like
    "when did CRITICAL_ATTENTION start?" or "how long was this symbol
    in HIGH_ATTENTION?".
    """

    instrument: str
    from_tier: AttentionTier | None  # ``None`` on the first recorded event
    to_tier: AttentionTier
    composite_at_transition: float
    transitioned_at: float = field(default_factory=time.time)
    dominant_signal: str = ""  # which factor drove the transition

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "from_tier": self.from_tier.value if self.from_tier else None,
            "to_tier": self.to_tier.value,
            "composite_at_transition": round(self.composite_at_transition, 4),
            "transitioned_at": round(self.transitioned_at, 3),
            "dominant_signal": self.dominant_signal,
        }


__all__ = [
    "AttentionTier",
    "TIER_THRESHOLDS",
    "tier_from_composite",
    "ResourceBudget",
    "TIER_RESOURCES",
    "AttentionSnapshot",
    "AttentionEvolutionRecord",
]
