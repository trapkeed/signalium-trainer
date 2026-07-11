"""QDE Brain — narrative memory bridge.

Connects the typed narrative layer (Phase C3) to the cognitive memory
layer built in Phase C2 (``trainer.qde.brain.episodic``,
``trainer.qde.brain.semantic``, ``trainer.qde.brain.memory``).

What this module provides
--------------------------
``NarrativeMemoryBridge`` answers three questions on each narrative tick:

1. **Memory recall** — "Have we been here before?"
   Retrieve similar past episodes from episodic memory.  Their historical
   narrative outcomes (which NarrativeKind turned out to be correct in
   similar market states) provide evidence that the probability engine can
   use to boost or dampen individual narrative kinds.

2. **Portfolio influence** — "Does our current book position change which
   narrative matters?"
   A heavy long position makes a DISTRIBUTION or REVERSAL narrative more
   consequential than an ACCUMULATION one.  Portfolio exposure and session
   P&L produce a ``portfolio_influence`` dict that callers can apply as
   adjustments to narrative confidences.

3. **Episode storage** — "Record this narrative for future recall."
   When a narrative reaches ESTABLISHED, store it as an episode in episodic
   memory so it becomes part of the recall pool for future ticks.

Design constraints
-------------------
* This module imports from C2 (``episodic``, ``semantic``, ``memory``)
  but never the reverse.
* All imports are soft (wrapped in ``try/except ImportError``) so the
  narrative layer continues working even without the C2 layer installed
  (e.g. in test environments that only import the narrative subpackage).
* No ``QDEDecision`` imports anywhere.
* No SQLite / disk I/O; persistence is handled by callers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.narrative.narrative_store import NarrativeKind, SymbolNarrative

# C2 imports — soft (optional)
try:
    from trainer.qde.brain.episodic.episodic_memory import (
        Episode,
        EpisodeFeatures,
        EpisodeOutcome,
        EpisodePortfolioState,
    )

    _HAS_EPISODIC = True
except ImportError:
    _HAS_EPISODIC = False

try:
    from trainer.qde.brain.semantic.semantic_memory import (  # noqa: F401
        SemanticMemory,
    )

    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False

try:
    from trainer.qde.brain.memory.working_memory_store import (  # noqa: F401
        WorkingMemoryStore,
    )

    _HAS_WORKING = True
except ImportError:
    _HAS_WORKING = False


# ── Output types ──────────────────────────────────────────────────────────────


@dataclass
class NarrativeMemoryContext:
    """Result of a memory-recall pass for one narrative generation call.

    ``similar_episodes``
        Up to K episodes from episodic memory whose market features are
        most similar to the current state, weighted by recency.

    ``historical_narrative_outcomes``
        Per-kind dictionary of historical win rates derived from similar
        past episodes that had resolved narrative outcomes.
        ``None`` entries mean the kind has never been seen in similar states.

    ``avg_reward_by_kind``
        Per-kind average realized reward from similar episodes.

    ``kind_confidence_adjustments``
        Derived adjustments: +/- delta to apply to narrative confidence
        for each kind.  Positive means history supports this narrative;
        negative means history contradicts it.  Range: [−0.15, +0.15].

    ``top_concept``
        The best-matching semantic concept for the current market state,
        or None if no concept matches.

    ``concept_similarity``
        Cosine similarity to ``top_concept`` (0 if no concept).

    ``portfolio_influence``
        Dict describing how the current portfolio state modifies which
        narratives are most relevant.  Keys:
          ``heavy_long_bias``   — 0-1 (biases toward DISTRIBUTION/REVERSAL)
          ``heavy_short_bias``  — 0-1 (biases toward ACCUMULATION/BREAKOUT)
          ``drawdown_penalty``  — 0-1 (dampens all directional narratives)
          ``session_pnl_boost`` — +/- adjustment from session P&L
    """

    similar_episodes: list = field(default_factory=list)
    historical_narrative_outcomes: dict[str, float | None] = field(
        default_factory=lambda: {k.value: None for k in NarrativeKind}
    )
    avg_reward_by_kind: dict[str, float] = field(
        default_factory=lambda: {k.value: 0.0 for k in NarrativeKind}
    )
    kind_confidence_adjustments: dict[str, float] = field(
        default_factory=lambda: {k.value: 0.0 for k in NarrativeKind}
    )
    top_concept: object | None = None  # MarketConcept | None
    concept_similarity: float = 0.0
    portfolio_influence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "similar_episode_count": len(self.similar_episodes),
            "historical_narrative_outcomes": {
                k: round(v, 4) if v is not None else None
                for k, v in self.historical_narrative_outcomes.items()
            },
            "avg_reward_by_kind": {k: round(v, 4) for k, v in self.avg_reward_by_kind.items()},
            "kind_confidence_adjustments": {
                k: round(v, 4) for k, v in self.kind_confidence_adjustments.items()
            },
            "concept_similarity": round(self.concept_similarity, 4),
            "top_concept_id": (
                self.top_concept.concept_id  # type: ignore[union-attr]
                if self.top_concept is not None
                else None
            ),
            "portfolio_influence": self.portfolio_influence,
        }


@dataclass
class NarrativePortfolioContext:
    """Portfolio state passed into the memory bridge for portfolio influence.

    Mirrors ``EpisodePortfolioState`` but is kept as a plain dataclass so
    callers do not need to import from the episodic package.

    All fields have neutral defaults so the bridge degrades gracefully when
    no portfolio context is supplied.
    """

    open_positions: int = 0
    total_exposure: float = 0.0  # 0-1 fraction of capital
    unrealized_pnl: float = 0.0  # ATR units (signed)
    drawdown: float = 0.0  # 0-1 current drawdown
    session_pnl: float = 0.0  # ATR units in current session
    equity_normalized: float = 1.0

    def to_dict(self) -> dict:
        return {
            "open_positions": self.open_positions,
            "total_exposure": round(self.total_exposure, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "drawdown": round(self.drawdown, 4),
            "session_pnl": round(self.session_pnl, 4),
            "equity_normalized": round(self.equity_normalized, 4),
        }


# ── Bridge ────────────────────────────────────────────────────────────────────


class NarrativeMemoryBridge:
    """Connect the narrative layer to episodic and semantic memory (C2).

    All C2 objects (``EpisodicMemory``, ``SemanticMemory``,
    ``WorkingMemoryStore``) are optional — pass ``None`` for any that are
    not available, and the bridge will silently skip that recall step.

    Usage::

        bridge = NarrativeMemoryBridge(
            episodic_memory=emem,
            semantic_memory=smem,
        )

        # Before generating narratives, recall context:
        ctx = bridge.recall_context(
            instrument="EURUSD",
            features=feat_dict,
            session="LONDON",
            portfolio=portfolio_ctx,
            active_narratives=current_narratives,
        )

        # After storing a narrative episode:
        bridge.store_narrative_episode(narrative, features, portfolio, "ENTER", 0.72)
    """

    def __init__(
        self,
        episodic_memory: object | None = None,  # EpisodicMemory | None
        semantic_memory: object | None = None,  # SemanticMemory | None
        working_memory: object | None = None,  # WorkingMemoryStore | None
        recall_k: int = 20,
        recency_lambda: float = 1.0 / 14.0,
        adj_scale: float = 0.15,  # max confidence adjustment
    ) -> None:
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._working = working_memory
        self._k = recall_k
        self._lambda = recency_lambda
        self._adj_scale = adj_scale

    # ── Primary recall entry point ────────────────────────────────────────

    def recall_context(
        self,
        instrument: str,
        features: dict,
        session: str = "",
        portfolio: NarrativePortfolioContext | None = None,
        active_narratives: list[SymbolNarrative] = (),
        now: float | None = None,
    ) -> NarrativeMemoryContext:
        """Recall relevant past episodes and compute portfolio influence.

        Parameters
        ----------
        instrument:
            Symbol being analyzed.
        features:
            Dict matching ``EpisodeFeatures.to_dict()`` format (the 12-dim
            feature vector fields).  Derived from the current tick's
            observation input.
        session:
            Current session label (e.g. "LONDON").
        portfolio:
            Current portfolio state, or None for no portfolio influence.
        active_narratives:
            The current active narratives (used to cross-reference concept
            similarity boosts).
        """
        now = now if now is not None else time.time()
        ctx = NarrativeMemoryContext()

        # 1. Episodic memory recall
        if _HAS_EPISODIC and self._episodic is not None:
            ctx = self._recall_from_episodic(ctx, instrument, features, session, now)

        # 2. Semantic memory recall
        if _HAS_SEMANTIC and self._semantic is not None:
            ctx = self._recall_from_semantic(ctx, features)

        # 3. Portfolio influence
        if portfolio is not None:
            ctx.portfolio_influence = self._compute_portfolio_influence(
                portfolio, active_narratives
            )

        return ctx

    # ── Episodic store / update ───────────────────────────────────────────

    def store_narrative_episode(
        self,
        narrative: SymbolNarrative,
        features: dict,
        portfolio: NarrativePortfolioContext | None = None,
        action: str = "NO_TRADE",
        confidence: float = 0.0,
        reasoning: str = "",
        now: float | None = None,
    ) -> str:
        """Write an established narrative as an episode to episodic memory.

        Returns the ``episode_id`` of the created episode, or "" if episodic
        memory is not available.

        The episode captures the market state at the moment the narrative
        reached ESTABLISHED, so future calls to ``recall_context`` can find
        it as a relevant historical precedent.
        """
        if not _HAS_EPISODIC or self._episodic is None:
            return ""

        now = now if now is not None else time.time()

        port_state = None
        if portfolio is not None:
            port_state = EpisodePortfolioState(  # type: ignore[name-defined]
                open_positions=portfolio.open_positions,
                total_exposure=portfolio.total_exposure,
                unrealized_pnl=portfolio.unrealized_pnl,
                drawdown=portfolio.drawdown,
                session_pnl=portfolio.session_pnl,
                equity_normalized=portfolio.equity_normalized,
            )
        else:
            port_state = EpisodePortfolioState()  # type: ignore[name-defined]

        feat = EpisodeFeatures.from_dict(features) if _HAS_EPISODIC else None  # type: ignore[name-defined]
        if feat is None:
            return ""

        ep = Episode(  # type: ignore[name-defined]
            instrument=narrative.instrument,
            timestamp=now,
            features=feat,
            portfolio=port_state,
            active_hypotheses=[],
            action=action.upper(),
            confidence=confidence,
            reasoning=reasoning or f"narrative:{narrative.kind.value} established",
            regime_context=narrative.regime_context,
            session_context=narrative.session_context,
        )
        try:
            self._episodic.write_memory(ep, now=now)  # type: ignore[union-attr]
            logger.debug(
                f"[MemoryBridge] stored narrative episode {ep.episode_id} "
                f"{narrative.instrument}/{narrative.kind.value}"
            )
            return ep.episode_id
        except Exception as exc:
            logger.warning(f"[MemoryBridge] store_narrative_episode failed: {exc}")
            return ""

    def update_episode_outcome(
        self,
        instrument: str,
        episode_id: str,
        direction_correct: bool,
        reward_atr: float,
        exit_reason: str = "",
        bars_to_close: int = 0,
        now: float | None = None,
    ) -> bool:
        """Resolve an episode's outcome in episodic memory."""
        if not _HAS_EPISODIC or self._episodic is None:
            return False
        now = now if now is not None else time.time()
        outcome = EpisodeOutcome(  # type: ignore[name-defined]
            direction_correct=direction_correct,
            reward_atr=reward_atr,
            bars_to_close=bars_to_close,
            exit_reason=exit_reason,
            resolved=True,
        )
        try:
            return self._episodic.update_outcome(instrument, episode_id, outcome)  # type: ignore[union-attr]
        except Exception:
            return False

    # ── Apply context to narratives ───────────────────────────────────────

    def apply_memory_context(
        self,
        narratives: list[SymbolNarrative],
        ctx: NarrativeMemoryContext,
        max_adj: float = 0.12,
    ) -> None:
        """Adjust active narrative confidences using memory context.

        Applies ``ctx.kind_confidence_adjustments`` to each narrative's
        confidence, bounded by ``max_adj``.  Mutates in place so the effect
        is visible in the next probability computation.

        This should be called AFTER ``NarrativeEvolution.evolve`` and BEFORE
        ``NarrativeProbabilityEngine.compute``.
        """
        for n in narratives:
            adj = ctx.kind_confidence_adjustments.get(n.kind.value, 0.0)
            if abs(adj) < 1e-9:
                continue
            clamped = max(-max_adj, min(max_adj, adj))
            n.confidence = max(0.0, min(1.0, n.confidence + clamped))

    # ── Internal helpers ──────────────────────────────────────────────────

    def _recall_from_episodic(
        self,
        ctx: NarrativeMemoryContext,
        instrument: str,
        features: dict,
        session: str,
        now: float,
    ) -> NarrativeMemoryContext:
        """Populate ctx from episodic memory recall."""
        try:
            feat = EpisodeFeatures.from_dict(features)  # type: ignore[name-defined]
            similar = self._episodic.retrieve_similar(  # type: ignore[union-attr]
                features=feat,
                instrument=instrument,
                k=self._k,
                session=session if session else None,
                resolved_only=True,
                now=now,
            )
        except Exception as exc:
            logger.debug(f"[MemoryBridge] episodic recall failed: {exc}")
            return ctx

        ctx.similar_episodes = similar  # list[SimilarEpisode]

        if not similar:
            return ctx

        # Compute per-kind historical outcome statistics from similar episodes
        kind_correct: dict[str, list[float]] = {k.value: [] for k in NarrativeKind}
        kind_reward: dict[str, list[float]] = {k.value: [] for k in NarrativeKind}

        for se in similar:
            if not se.outcome.resolved:
                continue
            # The episode's action is a proxy for the dominant narrative kind
            # at that time.  We map it to the most probable narrative kind
            # using regime context and session.
            kind_value = _action_to_narrative_kind(se.action, se.regime_context)
            if kind_value and kind_value in kind_correct:
                kind_correct[kind_value].append(1.0 if se.outcome.direction_correct else 0.0)
                kind_reward[kind_value].append(se.reward_atr)

        for kind in NarrativeKind:
            corrects = kind_correct[kind.value]
            rewards = kind_reward[kind.value]
            if corrects:
                win_rate = sum(corrects) / len(corrects)
                ctx.historical_narrative_outcomes[kind.value] = round(win_rate, 4)
                ctx.avg_reward_by_kind[kind.value] = (
                    round(sum(rewards) / len(rewards), 4) if rewards else 0.0
                )
            else:
                ctx.historical_narrative_outcomes[kind.value] = None

        # Compute confidence adjustments: positive when history supports,
        # negative when history contradicts.
        for kind in NarrativeKind:
            win = ctx.historical_narrative_outcomes.get(kind.value)
            if win is None:
                ctx.kind_confidence_adjustments[kind.value] = 0.0
                continue
            # Centre at 0.5 (neutral): win>0.5 → positive, win<0.5 → negative
            raw_adj = (win - 0.5) * self._adj_scale * 2.0
            ctx.kind_confidence_adjustments[kind.value] = round(
                max(-self._adj_scale, min(self._adj_scale, raw_adj)), 4
            )

        return ctx

    def _recall_from_semantic(
        self,
        ctx: NarrativeMemoryContext,
        features: dict,
    ) -> NarrativeMemoryContext:
        """Populate ctx from semantic memory recall."""
        try:
            feat_vec = [
                features.get("volatility", 0.5),
                (features.get("momentum", 0.0) + 1.0) / 2.0,
                features.get("trend_strength", 0.0),
                features.get("range_compression", 0.0),
                features.get("absorption", 0.0),
                features.get("volume_balance", 0.5),
                features.get("session_quality", 0.8),
                features.get("microstructure", 0.8),
                features.get("regime_numeric", 0.5),
                features.get("hypothesis_conf", 0.0),
                features.get("attention_weight", 0.5),
                features.get("spread_norm", 0.5),
            ]
            results = self._semantic.recall(  # type: ignore[union-attr]
                feature_vec=feat_vec,
                n=3,
                published_only=True,
            )
        except Exception as exc:
            logger.debug(f"[MemoryBridge] semantic recall failed: {exc}")
            return ctx

        if not results:
            return ctx

        top = results[0]
        ctx.top_concept = top.concept
        ctx.concept_similarity = top.similarity

        # If a concept has a strong win_rate, boost the narrative kind that
        # most closely corresponds to it.
        boost_kind = _concept_to_narrative_kind(top.concept)
        if boost_kind is not None and top.similarity > 0.70:
            boost = round(top.similarity * top.concept.win_rate * 0.10, 4)  # type: ignore[union-attr]
            existing = ctx.kind_confidence_adjustments.get(boost_kind.value, 0.0)
            ctx.kind_confidence_adjustments[boost_kind.value] = round(
                max(-self._adj_scale, min(self._adj_scale, existing + boost)), 4
            )

        return ctx

    @staticmethod
    def _compute_portfolio_influence(
        portfolio: NarrativePortfolioContext,
        active_narratives: list[SymbolNarrative],
    ) -> dict:
        """Compute how portfolio state modifies narrative relevance.

        Heavy long exposure → DISTRIBUTION and REVERSAL are more consequential
        (we need to be alert to narrative invalidation).

        Heavy short exposure → ACCUMULATION and BREAKOUT are more consequential.

        Drawdown → dampen all directional narratives (protect capital).
        """
        exposure = portfolio.total_exposure
        drawdown = portfolio.drawdown
        spnl = portfolio.session_pnl  # signed ATR units
        unrealized = portfolio.unrealized_pnl  # signed ATR units

        # Long/short bias from open positions + unrealized PnL sign
        long_bias = 0.0
        short_bias = 0.0
        if exposure > 0.0:
            if unrealized >= 0:
                long_bias = exposure * 0.80  # net long
            else:
                short_bias = exposure * 0.60  # losing long → treat as short-leaning

        long_bias = round(min(1.0, long_bias), 4)
        short_bias = round(min(1.0, short_bias), 4)

        # Drawdown penalty: at max_drawdown=0.20 the penalty is 0.40 (large)
        dd_penalty = round(min(1.0, drawdown * 2.0), 4)

        # Session P&L boost (positive session → slight confidence boost to
        # continuation narratives; negative → slight boost to reversal narratives)
        spnl_boost = round(max(-0.10, min(0.10, spnl * 0.05)), 4)

        return {
            "heavy_long_bias": long_bias,
            "heavy_short_bias": short_bias,
            "drawdown_penalty": dd_penalty,
            "session_pnl_boost": spnl_boost,
            "exposure": round(exposure, 4),
        }


# ── Kind mapping helpers ──────────────────────────────────────────────────────


def _action_to_narrative_kind(action: str, regime: str) -> str | None:
    """Heuristic: map episode action + regime → the most likely narrative kind.

    This is a best-effort mapping.  Episodes don't directly record the
    narrative kind — we infer it from the action and market regime.
    """
    regime = regime.upper()
    if action == "ENTER":
        if "TREND" in regime:
            return NarrativeKind.TREND_CONTINUATION.value
        if "RANGE" in regime or "CHOP" in regime:
            return NarrativeKind.BREAKOUT.value
        return NarrativeKind.TREND_CONTINUATION.value
    if action == "EXIT":
        if "TREND" in regime:
            return NarrativeKind.EXHAUSTION.value
        return NarrativeKind.REVERSAL.value
    return None


def _concept_to_narrative_kind(concept: object) -> NarrativeKind | None:
    """Map a semantic concept's kind to the nearest NarrativeKind."""
    if concept is None:
        return None
    kind_str = getattr(getattr(concept, "kind", None), "value", "")

    # ConceptKind → NarrativeKind mapping (best approximation)
    mapping: dict[str, NarrativeKind] = {
        "BREAKOUT_AFTER_COMPRESSION": NarrativeKind.BREAKOUT,
        "FAILED_LIQUIDITY_SWEEP": NarrativeKind.LIQUIDITY_GRAB,
        "TREND_CONTINUATION_PULLBACK": NarrativeKind.TREND_CONTINUATION,
        "MEAN_REVERSION_AFTER_EXPANSION": NarrativeKind.REVERSAL,
        "ACCUMULATION_BREAKOUT": NarrativeKind.ACCUMULATION,
        "DISTRIBUTION_REVERSAL": NarrativeKind.DISTRIBUTION,
        "STOP_HUNT_REVERSAL": NarrativeKind.LIQUIDITY_GRAB,
        "RANGE_FADE": NarrativeKind.DISTRIBUTION,
        "VOLATILITY_CONTRACTION": NarrativeKind.ACCUMULATION,
        "VOLATILITY_EXPANSION": NarrativeKind.BREAKOUT,
        "SESSION_OPEN_MOMENTUM": NarrativeKind.TREND_CONTINUATION,
        "NEWS_SPIKE_FADE": NarrativeKind.EXHAUSTION,
    }
    return mapping.get(kind_str)


__all__ = [
    "NarrativeMemoryBridge",
    "NarrativeMemoryContext",
    "NarrativePortfolioContext",
]
