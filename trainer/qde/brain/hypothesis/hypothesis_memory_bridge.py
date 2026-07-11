"""QDE Brain — hypothesis memory bridge.

Connects the new hypothesis set / competing-scenario layer (C4) to the
cognitive memory layer built in C2 (episodic, semantic, working memory)
so that hypothesis generation is informed by historical experience rather
than operating purely on live-tick narrative seeds.

What this module provides
--------------------------

1. **Recall context** — "Have we been in this kind of market before?"
   Retrieve similar past episodes and extract per-direction historical
   win rates.  Use these to bias the initial confidence of new hypothesis
   seeds so, e.g., BULLISH hypotheses in a regime that historically
   produced mostly failing BULLISH setups start with lower prior confidence.

2. **Seed enrichment** — After recall, adjust the ``initial_confidence``
   of each ``HypothesisSeed`` based on the historical win rate for its
   direction.  A direction that has historically resolved correctly 70%
   of the time gets a +0.08 boost; one that resolved correctly only 30%
   of the time gets a −0.08 penalty.

3. **Episode storage** — When a hypothesis reaches a significant
   resolution (CONFIRMED, INVALIDATED, PROMOTED), store it as an episode
   in episodic memory so it becomes part of the recall pool for future
   hypothesis generation.

4. **Failure-bias query** — Read per-direction failure counts from
   ``HypothesisFailureTracker`` to further adjust seed confidence
   (separately from the episodic-memory recall).

Design constraints
-------------------
* All C2 imports are soft (``try/except ImportError``) so the hypothesis
  package works even without the full C2 layer.
* No ``QDEDecision`` imports.
* No SQLite / disk I/O here; persistence handled by callers.
* The bridge does NOT call ``HypothesisEngine`` or mutate ``Hypothesis``
  objects.  It returns adjusted seeds and context; callers decide what to
  do with them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.narrative.market_narrative_engine import HypothesisSeed

# C2 soft imports
try:
    from trainer.qde.brain.episodic.episodic_memory import (
        Episode,
        EpisodeFeatures,
        EpisodeOutcome,
        EpisodePortfolioState,
        EpisodicMemory,
    )
    _HAS_EPISODIC = True
except ImportError:
    _HAS_EPISODIC = False

try:
    from trainer.qde.brain.semantic.semantic_memory import MarketConcept, SemanticMemory
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False

# ── Configuration ─────────────────────────────────────────────────────────────

_RECALL_K           = 20      # episodes to retrieve
_RECENCY_LAMBDA     = 1.0 / 14.0
_MAX_CONFIDENCE_ADJ = 0.12    # cap on confidence adjustment (±)
_MIN_EPISODES_FOR_ADJ = 4     # minimum resolved episodes before adjusting


# ── Output types ──────────────────────────────────────────────────────────────


@dataclass
class HypothesisMemoryContext:
    """Result of one memory-recall pass for hypothesis generation.

    ``similar_episode_count``
        Number of similar past episodes found.

    ``win_rate_by_direction``
        Historical win rate by direction (BULLISH/BEARISH/NEUTRAL → 0-1).
        None = insufficient history for that direction.

    ``confidence_adj_by_direction``
        Confidence adjustment to apply per direction: positive = history
        supports this direction, negative = history contradicts it.
        Range: [−0.12, +0.12].

    ``relevant_concept``
        Best-matching semantic concept for the current feature vector (or None).

    ``concept_similarity``
        Cosine similarity to ``relevant_concept`` (0 if no concept found).

    ``failure_bias``
        Per-direction recent failure count from ``HypothesisFailureTracker``
        (if provided).  Higher = more failures recorded for this direction.
    """

    similar_episode_count:  int                      = 0
    win_rate_by_direction:  dict[str, float | None]  = field(
        default_factory=lambda: {d.value: None for d in BeliefDirection}
    )
    confidence_adj_by_direction: dict[str, float]    = field(
        default_factory=lambda: {d.value: 0.0 for d in BeliefDirection}
    )
    relevant_concept:       object | None            = None
    concept_similarity:     float                    = 0.0
    failure_bias:           dict[str, float]         = field(
        default_factory=lambda: {d.value: 0.0 for d in BeliefDirection}
    )

    def to_dict(self) -> dict:
        return {
            "similar_episode_count":    self.similar_episode_count,
            "win_rate_by_direction":    {
                k: round(v, 4) if v is not None else None
                for k, v in self.win_rate_by_direction.items()
            },
            "confidence_adj_by_direction": {
                k: round(v, 4) for k, v in self.confidence_adj_by_direction.items()
            },
            "concept_similarity":       round(self.concept_similarity, 4),
            "concept_id": (
                getattr(self.relevant_concept, "concept_id", None)
                if self.relevant_concept else None
            ),
            "failure_bias":             {k: round(v, 2) for k, v in self.failure_bias.items()},
        }


# ── Bridge ────────────────────────────────────────────────────────────────────


class HypothesisMemoryBridge:
    """Connect the hypothesis layer (C4) to episodic/semantic memory (C2).

    All C2 backends are optional — pass ``None`` for any that are not
    available; the bridge gracefully skips that recall step.

    Usage::

        bridge = HypothesisMemoryBridge(episodic_memory=emem, semantic_memory=smem)

        ctx = bridge.recall_context(
            instrument="EURUSD",
            features=feat_dict,
            session="LONDON",
            market_regime="TREND_UP",
            now=now,
        )

        enriched_seeds = bridge.enrich_seeds(seeds, ctx)
        # Use enriched_seeds to create hypotheses with adjusted confidence.
    """

    def __init__(
        self,
        episodic_memory:  object | None = None,   # EpisodicMemory | None
        semantic_memory:  object | None = None,   # SemanticMemory | None
        failure_tracker:  object | None = None,   # HypothesisFailureTracker | None
        recall_k:         int           = _RECALL_K,
        recency_lambda:   float         = _RECENCY_LAMBDA,
        max_conf_adj:     float         = _MAX_CONFIDENCE_ADJ,
        min_episodes:     int           = _MIN_EPISODES_FOR_ADJ,
    ) -> None:
        self._episodic  = episodic_memory
        self._semantic  = semantic_memory
        self._failure   = failure_tracker
        self._k         = recall_k
        self._lambda    = recency_lambda
        self._max_adj   = max_conf_adj
        self._min_eps   = min_episodes

    # ── Primary recall entry point ────────────────────────────────────────

    def recall_context(
        self,
        instrument:         str,
        features:           dict,
        session:            str          = "",
        market_regime:      str          = "",
        now:                float | None = None,
    ) -> HypothesisMemoryContext:
        """Recall context from all available memory sources.

        Parameters
        ----------
        instrument:
            Symbol being analyzed.
        features:
            12-dim feature dict (keys matching ``EpisodeFeatures`` field names).
        session:
            Current session label (e.g. "LONDON").
        market_regime:
            Current market regime string (e.g. "TREND_UP").
        now:
            Override current time (for replay / testing).
        """
        now = now if now is not None else time.time()
        ctx = HypothesisMemoryContext()

        # 1. Episodic recall
        if _HAS_EPISODIC and self._episodic is not None:
            ctx = self._recall_episodic(ctx, instrument, features, session, now)

        # 2. Semantic recall
        if _HAS_SEMANTIC and self._semantic is not None:
            ctx = self._recall_semantic(ctx, features)

        # 3. Failure-tracker bias
        if self._failure is not None:
            ctx = self._recall_failure_bias(ctx, instrument)

        return ctx

    # ── Seed enrichment ───────────────────────────────────────────────────

    def enrich_seeds(
        self,
        seeds:   list[HypothesisSeed],
        ctx:     HypothesisMemoryContext,
    ) -> list[HypothesisSeed]:
        """Return a new list of seeds with confidence adjusted by memory context.

        The original seeds are not mutated.  Returns a new list where each
        seed's ``confidence`` has been shifted by the direction-specific
        confidence adjustment from ``ctx``.

        Concept-similarity boost: if the semantic memory found a relevant
        concept with high similarity, apply an additional small boost to all
        seeds whose direction matches the concept's dominant win_rate direction.
        """
        import copy
        result: list[HypothesisSeed] = []
        for seed in seeds:
            adj = ctx.confidence_adj_by_direction.get(seed.direction.value, 0.0)

            # Concept similarity boost (small — concept confirms general direction)
            if ctx.concept_similarity > 0.70 and ctx.relevant_concept is not None:
                wr = getattr(ctx.relevant_concept, "win_rate", 0.5)
                if wr > 0.60:
                    adj += (ctx.concept_similarity - 0.70) * 0.15

            adj        = max(-self._max_adj, min(self._max_adj, adj))
            new_conf   = max(0.05, min(0.95, seed.confidence + adj))
            new_seed   = copy.copy(seed)
            new_seed.confidence = new_conf
            result.append(new_seed)

        return result

    # ── Episode storage ───────────────────────────────────────────────────

    def store_hypothesis_episode(
        self,
        hypothesis_id:     str,
        instrument:        str,
        thesis:            str,
        direction:         BeliefDirection,
        features:          dict,
        action:            str                = "NO_TRADE",
        confidence:        float              = 0.0,
        outcome_correct:   bool               = False,
        reward_atr:        float              = 0.0,
        resolved:          bool               = False,
        regime_context:    str                = "",
        session_context:   str                = "",
        now:               float | None       = None,
    ) -> str:
        """Write a hypothesis resolution as an episode to episodic memory.

        Returns the episode_id created, or "" if episodic memory is not
        available.

        Call this when a hypothesis reaches CONFIRMED, INVALIDATED, or
        PROMOTED so future recalls can find it as a historical precedent.
        """
        if not _HAS_EPISODIC or self._episodic is None:
            return ""

        now = now if now is not None else time.time()

        try:
            feat = EpisodeFeatures.from_dict(features)  # type: ignore[name-defined]
            outcome = EpisodeOutcome(  # type: ignore[name-defined]
                direction_correct=outcome_correct,
                reward_atr=reward_atr,
                resolved=resolved,
            )
            ep = Episode(  # type: ignore[name-defined]
                instrument=instrument,
                timestamp=now,
                features=feat,
                portfolio=EpisodePortfolioState(),  # type: ignore[name-defined]
                active_hypotheses=[hypothesis_id],
                action=action.upper(),
                confidence=confidence,
                outcome=outcome,
                trade_result_atr=reward_atr,
                regime_context=regime_context,
                session_context=session_context,
                reasoning=f"hypothesis:{direction.value}:{thesis[:60]}",
            )
            self._episodic.write_memory(ep, now=now)  # type: ignore[union-attr]
            logger.debug(
                f"[HypMemBridge] stored episode {ep.episode_id} for "
                f"hypothesis {hypothesis_id[:8]} on {instrument}"
            )
            return ep.episode_id
        except Exception as exc:
            logger.warning(f"[HypMemBridge] store_hypothesis_episode failed: {exc}")
            return ""

    # ── Internal helpers ──────────────────────────────────────────────────

    def _recall_episodic(
        self,
        ctx:        HypothesisMemoryContext,
        instrument: str,
        features:   dict,
        session:    str,
        now:        float,
    ) -> HypothesisMemoryContext:
        try:
            feat    = EpisodeFeatures.from_dict(features)  # type: ignore[name-defined]
            similar = self._episodic.retrieve_similar(  # type: ignore[union-attr]
                features=feat,
                instrument=instrument,
                k=self._k,
                session=session if session else None,
                resolved_only=True,
                now=now,
            )
        except Exception as exc:
            logger.debug(f"[HypMemBridge] episodic recall failed: {exc}")
            return ctx

        ctx.similar_episode_count = len(similar)
        if not similar:
            return ctx

        # Accumulate win rates by direction
        dir_corrects: dict[str, list[float]] = {d.value: [] for d in BeliefDirection}
        for se in similar:
            if not se.outcome.resolved:
                continue
            # Map action → direction
            dir_val = _action_to_direction(se.action).value
            dir_corrects[dir_val].append(1.0 if se.outcome.direction_correct else 0.0)

        for d in BeliefDirection:
            corrects = dir_corrects[d.value]
            if len(corrects) >= self._min_eps:
                wr  = sum(corrects) / len(corrects)
                ctx.win_rate_by_direction[d.value] = round(wr, 4)
                # Adjustment: centre at 0.5 → ±max_adj
                raw_adj = (wr - 0.5) * self._max_adj * 2.0
                ctx.confidence_adj_by_direction[d.value] = round(
                    max(-self._max_adj, min(self._max_adj, raw_adj)), 4
                )
            else:
                ctx.win_rate_by_direction[d.value] = None

        return ctx

    def _recall_semantic(
        self,
        ctx:      HypothesisMemoryContext,
        features: dict,
    ) -> HypothesisMemoryContext:
        try:
            feat_vec = [
                features.get("volatility",       0.5),
                (features.get("momentum", 0.0) + 1.0) / 2.0,
                features.get("trend_strength",   0.0),
                features.get("range_compression", 0.0),
                features.get("absorption",       0.0),
                features.get("volume_balance",   0.5),
                features.get("session_quality",  0.8),
                features.get("microstructure",   0.8),
                features.get("regime_numeric",   0.5),
                features.get("hypothesis_conf",  0.0),
                features.get("attention_weight", 0.5),
                features.get("spread_norm",      0.5),
            ]
            results = self._semantic.recall(  # type: ignore[union-attr]
                feature_vec=feat_vec,
                n=1,
                published_only=True,
            )
        except Exception as exc:
            logger.debug(f"[HypMemBridge] semantic recall failed: {exc}")
            return ctx

        if results:
            best = results[0]
            ctx.relevant_concept    = best.concept
            ctx.concept_similarity  = best.similarity

        return ctx

    def _recall_failure_bias(
        self,
        ctx:        HypothesisMemoryContext,
        instrument: str,
    ) -> HypothesisMemoryContext:
        """Add failure-count bias from the failure tracker."""
        try:
            bias = self._failure.direction_bias_summary(instrument)  # type: ignore[union-attr]
            ctx.failure_bias = {k: float(v) for k, v in bias.items()}
        except Exception:
            pass
        return ctx


def _action_to_direction(action: str) -> BeliefDirection:
    """Map episode action to BeliefDirection for win-rate bucketing."""
    a = action.upper()
    if a in ("ENTER", "SCALE_IN", "HOLD"):
        return BeliefDirection.BULLISH    # proxy: entering = bullish conviction
    if a in ("EXIT", "SCALE_OUT"):
        return BeliefDirection.BEARISH
    return BeliefDirection.NEUTRAL


__all__ = [
    "HypothesisMemoryBridge",
    "HypothesisMemoryContext",
]
