"""QDE Brain — multi-timeframe context: top-level engine.

`MultiTimeframeContextEngine` is the single integration point for the rest
of the Brain.  It owns one `TimeframeMemory` and one `ContextAlignmentEngine`
for all symbols and exposes four integration points:

1. **Ingestion** — `on_bar()` for individual timeframe updates (including
   the Monthly/Weekly bars that don't exist in `MultiTimeframeSnapshot`),
   and `on_snapshot()` to ingest all five overlapping timeframes at once
   from the existing `TimeframeSynchronizer` pipeline.

2. **Inference context** — `context_for_inference()` returns a compact dict
   namespaced under ``mtf_*`` for merging into
   `QDEInferenceInput.brain_context`.

3. **Hypothesis seeds** — `suggest_hypothesis_seeds()` returns typed
   `ContextHypothesisSeed` objects for the `HypothesisEngine` to consume.

4. **Conviction modifier** — `conviction_modifier()` returns a signed
   ±0.3 adjustment for `compute_conviction` callers.

No hardcoded directional rules
-------------------------------
All direction and pattern information flows from `CrossTimeframeContext`,
which is computed purely from observational momentum data.  Neither this
module nor its helpers ever branches on "if H4 is bullish" or assigns a
direction from a fixed rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.context.context_alignment_engine import (
    AlignmentPattern,
    ContextAlignmentEngine,
    CrossTimeframeContext,
)
from trainer.qde.brain.context.timeframe_memory import (
    TIMEFRAME_TO_CONTEXT,
    ContextTimeframe,
    TimeframeMemory,
    coarse_timeframes,
    fine_timeframes,
)
from trainer.qde.qde_timeframe_models import MultiTimeframeSnapshot, TimeframeFeatureVector

# ── ContextHypothesisSeed ─────────────────────────────────────────────────────

# Minimum contributing timeframes before emitting seeds.
_MIN_SEED_TFS = 3

# Minimum pattern confidence before emitting seeds.
_MIN_SEED_CONF = 0.48


@dataclass
class ContextHypothesisSeed:
    """A context-derived hypothesis seed for the HypothesisEngine.

    `direction` is derived from `dominant_direction` in
    `CrossTimeframeContext` — the sign of the coarse-group weighted-average
    momentum.  For PULLBACK patterns, the seed direction reflects the coarser-
    TF momentum sign (the direction the fine-TF counter-move is expected to
    resolve toward), not a hardcoded assertion.

    Callers pass this to `HypothesisEngine.create()` or equivalent; this
    class never places or influences a trade itself.
    """

    instrument: str
    thesis: str
    direction: BeliefDirection
    confidence: float             # 0–1
    source_pattern: AlignmentPattern
    evidence: str                 # the CrossTimeframeContext.summary

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "thesis": self.thesis,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "source_pattern": self.source_pattern.value,
            "evidence": self.evidence,
        }


# ── MultiTimeframeContextEngine ───────────────────────────────────────────────


class MultiTimeframeContextEngine:
    """Orchestrates per-symbol multi-timeframe context for the entire Brain.

    One instance is intended to be long-lived for the process.  It holds
    state (a `TimeframeMemory`) but exposes only pure read methods for all
    integrations; ingestion is explicit (`on_bar` / `on_snapshot`).
    """

    def __init__(self, history_depth: int = 12) -> None:
        self._memory = TimeframeMemory(history_depth=history_depth)
        self._engine = ContextAlignmentEngine()
        self._contexts: dict[str, CrossTimeframeContext | None] = {}

    # ── Ingestion ─────────────────────────────────────────────────────────

    def on_bar(
        self,
        instrument: str,
        timeframe: ContextTimeframe,
        feature_vector: TimeframeFeatureVector,
        now: float | None = None,
    ) -> CrossTimeframeContext:
        """Ingest a single-timeframe feature vector and recompute context.

        The primary path for Monthly and Weekly bars, which are not present
        in `MultiTimeframeSnapshot`.  Also usable for any source that feeds
        individual timeframes independently (e.g. a streaming provider that
        emits H4 bars as they close).
        """
        ts = now if now is not None else time.time()
        self._memory.record(instrument, timeframe, feature_vector, now=ts)
        return self._recompute(instrument, ts)

    def on_snapshot(
        self,
        instrument: str,
        snapshot: MultiTimeframeSnapshot,
        now: float | None = None,
    ) -> CrossTimeframeContext:
        """Ingest all timeframes from a `MultiTimeframeSnapshot` at once.

        Bridges `qde_timeframe_types.Timeframe` → `ContextTimeframe` for the
        five overlapping members (D1/H4/H1/M15/M5).  M1 (execution) is
        silently skipped.  Monthly and Weekly bars must come through
        `on_bar()` separately when available.
        """
        ts = now if now is not None else time.time()
        for tf, ctx_tf in TIMEFRAME_TO_CONTEXT.items():
            fv = snapshot.features.get(tf)
            if fv is not None:
                self._memory.record(instrument, ctx_tf, fv, now=ts)
        return self._recompute(instrument, ts)

    def _recompute(self, instrument: str, now: float) -> CrossTimeframeContext:
        obs = self._memory.current_observations(instrument, now=now)
        ctx = self._engine.compute(instrument, obs, self._memory, now=now)
        self._contexts[instrument] = ctx
        return ctx

    # ── Read ──────────────────────────────────────────────────────────────

    def context_for(self, instrument: str) -> CrossTimeframeContext | None:
        """Most recent cross-timeframe context, or None if not yet computed."""
        return self._contexts.get(instrument)

    def context_for_inference(self, instrument: str) -> dict:
        """Compact dict for merging into `QDEInferenceInput.brain_context`.

        All keys are namespaced with ``mtf_`` to avoid collisions with other
        context providers (narrative engine, conviction engine, etc.).

        A caller merges this with `brain_context` like::

            brain_context.update(engine.context_for_inference(instrument))
        """
        ctx = self._contexts.get(instrument)
        if ctx is None:
            return {
                "mtf_pattern": AlignmentPattern.MIXED.value,
                "mtf_pattern_confidence": 0.0,
                "mtf_alignment_score": 0.0,
                "mtf_dominant_direction": 0,
                "mtf_coarse_momentum": 0.0,
                "mtf_fine_momentum": 0.0,
                "mtf_pullback_depth": 0.0,
                "mtf_timeframes_contributing": 0,
                "mtf_summary": "no context",
            }
        return {
            "mtf_pattern": ctx.pattern.value,
            "mtf_pattern_confidence": ctx.pattern_confidence,
            "mtf_alignment_score": ctx.alignment_score,
            "mtf_dominant_direction": ctx.dominant_direction,
            "mtf_coarse_momentum": ctx.coarse_momentum,
            "mtf_fine_momentum": ctx.fine_momentum,
            "mtf_pullback_depth": ctx.pullback_depth,
            "mtf_timeframes_contributing": ctx.timeframes_contributing,
            "mtf_summary": ctx.summary,
        }

    def suggest_hypothesis_seeds(
        self,
        instrument: str,
    ) -> list[ContextHypothesisSeed]:
        """Context-derived hypothesis seeds for the HypothesisEngine.

        Seeds are emitted only when pattern confidence ≥ `_MIN_SEED_CONF`,
        enough timeframes contributed, and the coarse group has a clear
        direction.  `direction` is always derived from `dominant_direction`
        in the context — never assigned by a hardcoded rule.

        Callers are responsible for passing these to `HypothesisEngine` (or
        equivalent); this method never forms a hypothesis itself.
        """
        ctx = self._contexts.get(instrument)
        if ctx is None:
            return []
        if ctx.timeframes_contributing < _MIN_SEED_TFS:
            return []
        if ctx.pattern_confidence < _MIN_SEED_CONF:
            return []
        if ctx.dominant_direction == 0:
            return []

        dom = ctx.dominant_direction
        direction = BeliefDirection.BULLISH if dom > 0 else BeliefDirection.BEARISH

        seeds: list[ContextHypothesisSeed] = []

        if ctx.pattern == AlignmentPattern.PULLBACK:
            # The fine-TF counter-move may resolve in the coarse-TF direction.
            seeds.append(
                ContextHypothesisSeed(
                    instrument=instrument,
                    thesis=(
                        "Fine-TF counter-move may resolve in coarse-TF direction"
                    ),
                    direction=direction,
                    confidence=ctx.pattern_confidence * 0.80,
                    source_pattern=ctx.pattern,
                    evidence=ctx.summary,
                )
            )

        elif ctx.pattern == AlignmentPattern.CONTINUATION:
            seeds.append(
                ContextHypothesisSeed(
                    instrument=instrument,
                    thesis=(
                        "Cross-timeframe momentum aligned and fine-TF strengthening"
                    ),
                    direction=direction,
                    confidence=ctx.pattern_confidence * 0.90,
                    source_pattern=ctx.pattern,
                    evidence=ctx.summary,
                )
            )

        elif ctx.pattern == AlignmentPattern.ALIGNED:
            seeds.append(
                ContextHypothesisSeed(
                    instrument=instrument,
                    thesis="Broad cross-timeframe momentum alignment",
                    direction=direction,
                    confidence=ctx.pattern_confidence * 0.72,
                    source_pattern=ctx.pattern,
                    evidence=ctx.summary,
                )
            )

        elif ctx.pattern == AlignmentPattern.EXHAUSTION:
            # Exhaustion seeds lean against the dominant direction.
            opp = (
                BeliefDirection.BEARISH
                if direction == BeliefDirection.BULLISH
                else BeliefDirection.BULLISH
            )
            seeds.append(
                ContextHypothesisSeed(
                    instrument=instrument,
                    thesis=(
                        "Momentum weakening across timeframes — possible exhaustion"
                    ),
                    direction=opp,
                    confidence=ctx.pattern_confidence * 0.58,
                    source_pattern=ctx.pattern,
                    evidence=ctx.summary,
                )
            )

        return seeds

    def conviction_modifier(
        self,
        instrument: str,
        direction: BeliefDirection,
    ) -> float:
        """Signed conviction modifier in [−0.3, +0.3].

        Positive means context supports `direction`; negative means context
        undermines it.  Derived from `dominant_direction`, `alignment_score`,
        and `pattern_confidence` — no hardcoded per-pattern numbers.

        Callers add this to their base conviction score::

            final = compute_conviction(inp) + engine.conviction_modifier(sym, dir)
        """
        ctx = self._contexts.get(instrument)
        if ctx is None or ctx.timeframes_contributing < _MIN_SEED_TFS:
            return 0.0
        if ctx.dominant_direction == 0:
            return 0.0

        dir_int = 1 if direction == BeliefDirection.BULLISH else -1
        # +1 if direction agrees with dominant, −1 if it opposes
        agreement = 1.0 if dir_int == ctx.dominant_direction else -1.0

        strength = ctx.alignment_score * ctx.pattern_confidence
        return round(agreement * 0.30 * strength, 4)

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        """Serialise current memory + context for optional external storage.

        Compatible with `load_snapshot()` for round-trips.  Callers may
        store this dict in any JSON column (e.g. the QDEBrainDatabase) — no
        schema change is required.
        """
        ctx = self._contexts.get(instrument)
        return {
            "memory": self._memory.to_snapshot(instrument),
            "context": ctx.to_dict() if ctx is not None else None,
        }

    def load_snapshot(self, instrument: str, data: dict) -> None:
        """Restore memory + context from a previously serialised snapshot."""
        if "memory" in data and isinstance(data["memory"], list):
            self._memory.load_snapshot(instrument, data["memory"])
        if data.get("context"):
            self._contexts[instrument] = CrossTimeframeContext.from_dict(data["context"])
