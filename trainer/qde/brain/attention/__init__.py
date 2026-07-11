"""trainer.qde.brain.attention — C10 Attention Engine.

This subpackage extends the existing two-layer attention system with:

  * Four new attention factors: ``volume_anomaly``, ``key_level_proximity``,
    ``spread_change``, ``failed_move`` (via ``signal_enricher``).

  * Four named attention tiers: ``WATCH``, ``OBSERVE``, ``HIGH_ATTENTION``,
    ``CRITICAL_ATTENTION`` (via ``attention_types``).

  * Per-tier resource budgets controlling observation frequency, memory-update
    frequency, inference frequency, and reasoning-cycle rate (via
    ``TIER_RESOURCES``).

  * Attention evolution memory tracking tier transitions and snapshots
    (via ``AttentionMemory``).

  * A high-level ``AttentionAPI`` wrapping the existing ``AttentionEngine``
    and exposing the three primary callables the C10 spec requires:

      ``calculate_attention(observation, context=None, ...) → AttentionSnapshot``
      ``rank_symbols()                                      → list[RankedSymbol]``
      ``allocate_resources(instrument)                      → ResourceBudget``

What this package does NOT do
------------------------------
* It does **not** replace ``attention.AttentionAllocator`` (lightweight
  normalization), ``attention_allocator.AttentionScoreAllocator`` (6-signal
  decaying composite), or ``attention_engine.AttentionEngine`` (orchestrator).
  Those remain the canonical implementations; this package adds a layer on
  top of them.

* It does **not** own scheduling, network I/O, or Brain state management.
  All of that stays in its current owners.
"""

from trainer.qde.brain.attention.attention_api import (
    AttentionAPI,
    RankedSymbol,
    allocate_resources,
    calculate_attention,
    rank_symbols,
)
from trainer.qde.brain.attention.attention_memory import (
    DEFAULT_SNAPSHOT_DEPTH,
    DEFAULT_TRANSITION_DEPTH,
    AttentionMemory,
)
from trainer.qde.brain.attention.attention_types import (
    TIER_RESOURCES,
    TIER_THRESHOLDS,
    AttentionEvolutionRecord,
    AttentionSnapshot,
    AttentionTier,
    ResourceBudget,
    tier_from_composite,
)
from trainer.qde.brain.attention.signal_enricher import (
    ExtendedSignalContext,
    SignalEnricher,
    enrich_signals,
)

__all__ = [
    # Tiers and thresholds
    "AttentionTier",
    "TIER_THRESHOLDS",
    "tier_from_composite",
    # Resource budgets
    "ResourceBudget",
    "TIER_RESOURCES",
    # Evolution memory types
    "AttentionSnapshot",
    "AttentionEvolutionRecord",
    # Memory store
    "AttentionMemory",
    "DEFAULT_SNAPSHOT_DEPTH",
    "DEFAULT_TRANSITION_DEPTH",
    # Signal enrichment (4 new C10 factors)
    "ExtendedSignalContext",
    "enrich_signals",
    "SignalEnricher",
    # Primary API
    "AttentionAPI",
    "RankedSymbol",
    # Convenience wrappers
    "calculate_attention",
    "rank_symbols",
    "allocate_resources",
]
