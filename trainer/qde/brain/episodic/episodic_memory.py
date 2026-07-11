"""QDE Brain — episodic memory: complete market experience records.

``EpisodicMemory`` stores, indexes, and retrieves complete historical market
experiences — one ``Episode`` per trading event that the QDE observed and
acted on (or chose not to act on).  Each episode is a rich record capturing
the full context at the moment of the decision and the outcome that followed.

What is an episode?
-------------------
An episode is the QDE's subjective memory of one market moment:

  * The raw market features at that moment (as a float vector).
  * The portfolio state (exposure, open P&L, drawdown, …).
  * Which hypotheses were active and at what confidence.
  * What action was chosen (ENTER / EXIT / SCALE / NO_TRADE).
  * The confidence and reasoning behind that choice.
  * The outcome — what the market actually did afterwards.
  * The trade result (realized reward in ATR units).
  * The regime and session context.

All twelve fields from the spec are captured.

How recall works
----------------
``retrieve_similar`` uses cosine similarity over the episode's
``EpisodeFeatures.as_vector()`` to find historically similar market
conditions, weighted by recency decay ``e^(-λ × age_days)``.  This is the
same pattern used by ``InstitutionalIntentEngine``, ``RewardRiskEngine``,
and ``RegimeTransitionPredictor`` — consistent with the whole brain package.

``recall`` supports structured filtering by symbol, session, volatility
state, regime (market structure), and portfolio state bucket.

Indexing
--------
``EpisodicIndex`` maintains per-symbol inverted indexes by session,
volatility state (as a string key), and market structure (regime label).
This allows O(n) → O(k) filtered search when there are many episodes.

Persistence
-----------
``to_snapshot(instrument)`` / ``load_snapshot(instrument, data)`` for each
instrument, consistent with the rest of the brain package.  Callers are
responsible for writing snapshots to durable storage (e.g. via the
cognitive persistence layer).
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

# ── Hyper-parameters ──────────────────────────────────────────────────────────

_DEFAULT_MAX_EPISODES   = 5000    # per symbol
_DEFAULT_K              = 50      # top-K for similarity search
_RECENCY_LAMBDA_DEFAULT = 1.0 / 14.0  # decay halves ~every 10 days (e^-1 at 14)
_MIN_SIM_THRESHOLD      = 0.0     # minimum cosine similarity to include in results


# ── EpisodeFeatures ───────────────────────────────────────────────────────────


@dataclass
class EpisodeFeatures:
    """Raw numerical feature snapshot at the moment of the episode.

    This is the vector used for cosine-similarity search in
    ``retrieve_similar``.  All features are normalized to [0, 1] or
    signed −1..+1 to keep the cosine well-defined.

    The 12-dimensional default vector covers the signals most predictive
    of whether historical similar conditions resolved similarly:

      volatility       — normalized 0-1 volatility intensity
      momentum         — signed −1..+1 directional momentum
      trend_strength   — 0-1 trend strength
      range_compression — 0-1 price range compression
      absorption       — 0-1 level absorption signal
      volume_balance   — 0-1 buy/sell volume balance (0.5=neutral)
      session_quality  — 0-1 session liquidity quality
      microstructure   — 0-1 microstructure quality
      regime_numeric   — 0-1 encoded current regime (see ``regime_encode``)
      hypothesis_conf  — 0-1 mean confidence of active hypotheses
      attention_weight — 0-1 current attention allocation
      spread_norm      — 0-1 normalized bid/ask spread

    Additional raw features can be stored in ``extra`` for future use.
    """

    volatility:        float = 0.5
    momentum:          float = 0.0   # −1..+1
    trend_strength:    float = 0.0
    range_compression: float = 0.0
    absorption:        float = 0.0
    volume_balance:    float = 0.5   # 0.5=neutral
    session_quality:   float = 0.8
    microstructure:    float = 0.8
    regime_numeric:    float = 0.5   # 0-1 regime encoding
    hypothesis_conf:   float = 0.0
    attention_weight:  float = 0.5
    spread_norm:       float = 0.5
    extra:             dict  = field(default_factory=dict)

    def as_vector(self) -> list[float]:
        """12-dimensional cosine-similarity feature vector."""
        return [
            self.volatility,
            (self.momentum + 1.0) / 2.0,   # normalize −1..+1 → 0..1
            self.trend_strength,
            self.range_compression,
            self.absorption,
            self.volume_balance,
            self.session_quality,
            self.microstructure,
            self.regime_numeric,
            self.hypothesis_conf,
            self.attention_weight,
            self.spread_norm,
        ]

    def to_dict(self) -> dict:
        return {
            "volatility":        round(self.volatility, 4),
            "momentum":          round(self.momentum, 4),
            "trend_strength":    round(self.trend_strength, 4),
            "range_compression": round(self.range_compression, 4),
            "absorption":        round(self.absorption, 4),
            "volume_balance":    round(self.volume_balance, 4),
            "session_quality":   round(self.session_quality, 4),
            "microstructure":    round(self.microstructure, 4),
            "regime_numeric":    round(self.regime_numeric, 4),
            "hypothesis_conf":   round(self.hypothesis_conf, 4),
            "attention_weight":  round(self.attention_weight, 4),
            "spread_norm":       round(self.spread_norm, 4),
            "extra":             self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EpisodeFeatures:
        return cls(
            volatility=d.get("volatility", 0.5),
            momentum=d.get("momentum", 0.0),
            trend_strength=d.get("trend_strength", 0.0),
            range_compression=d.get("range_compression", 0.0),
            absorption=d.get("absorption", 0.0),
            volume_balance=d.get("volume_balance", 0.5),
            session_quality=d.get("session_quality", 0.8),
            microstructure=d.get("microstructure", 0.8),
            regime_numeric=d.get("regime_numeric", 0.5),
            hypothesis_conf=d.get("hypothesis_conf", 0.0),
            attention_weight=d.get("attention_weight", 0.5),
            spread_norm=d.get("spread_norm", 0.5),
            extra=d.get("extra", {}),
        )


# ── EpisodePortfolioState ─────────────────────────────────────────────────────


@dataclass
class EpisodePortfolioState:
    """Portfolio state at the moment of the episode.

    These fields capture how exposed the QDE was at the time, which
    influences which historical comparisons are most relevant (e.g. an
    entry in a fully-exposed portfolio is different from one in a fresh
    account).
    """

    open_positions:    int   = 0
    total_exposure:    float = 0.0   # 0-1 fraction of capital deployed
    unrealized_pnl:    float = 0.0   # ATR units
    drawdown:          float = 0.0   # 0-1 current drawdown
    session_pnl:       float = 0.0   # ATR units in the current session
    equity_normalized: float = 1.0   # current equity / starting equity

    def exposure_bucket(self) -> str:
        """Coarse bucket for index partitioning."""
        if self.total_exposure < 0.20:
            return "light"
        if self.total_exposure < 0.60:
            return "moderate"
        return "heavy"

    def to_dict(self) -> dict:
        return {
            "open_positions":    self.open_positions,
            "total_exposure":    round(self.total_exposure, 4),
            "unrealized_pnl":    round(self.unrealized_pnl, 4),
            "drawdown":          round(self.drawdown, 4),
            "session_pnl":       round(self.session_pnl, 4),
            "equity_normalized": round(self.equity_normalized, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> EpisodePortfolioState:
        return cls(
            open_positions=d.get("open_positions", 0),
            total_exposure=d.get("total_exposure", 0.0),
            unrealized_pnl=d.get("unrealized_pnl", 0.0),
            drawdown=d.get("drawdown", 0.0),
            session_pnl=d.get("session_pnl", 0.0),
            equity_normalized=d.get("equity_normalized", 1.0),
        )


# ── EpisodeOutcome ────────────────────────────────────────────────────────────


@dataclass
class EpisodeOutcome:
    """What actually happened after the episode's decision.

    ``direction_correct`` — did the price move in the predicted direction?
    ``reward_atr``        — realized reward in ATR units (signed; positive=profit).
    ``mae_atr``           — maximum adverse excursion in ATR units.
    ``mfe_atr``           — maximum favorable excursion in ATR units.
    ``bars_to_close``     — bars until the position was closed (or 0 if not yet).
    ``exit_reason``       — e.g. "TP_HIT", "SL_HIT", "MANUAL", "TIMEOUT".
    ``market_moved``      — how much price moved (ATR units) over the next N bars.
    ``resolved``          — True once the outcome is known.
    """

    direction_correct: bool  = False
    reward_atr:        float = 0.0
    mae_atr:           float = 0.0
    mfe_atr:           float = 0.0
    bars_to_close:     int   = 0
    exit_reason:       str   = ""
    market_moved:      float = 0.0
    resolved:          bool  = False

    def to_dict(self) -> dict:
        return {
            "direction_correct": self.direction_correct,
            "reward_atr":        round(self.reward_atr, 6),
            "mae_atr":           round(self.mae_atr, 6),
            "mfe_atr":           round(self.mfe_atr, 6),
            "bars_to_close":     self.bars_to_close,
            "exit_reason":       self.exit_reason,
            "market_moved":      round(self.market_moved, 6),
            "resolved":          self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EpisodeOutcome:
        return cls(
            direction_correct=d.get("direction_correct", False),
            reward_atr=d.get("reward_atr", 0.0),
            mae_atr=d.get("mae_atr", 0.0),
            mfe_atr=d.get("mfe_atr", 0.0),
            bars_to_close=d.get("bars_to_close", 0),
            exit_reason=d.get("exit_reason", ""),
            market_moved=d.get("market_moved", 0.0),
            resolved=d.get("resolved", False),
        )


# ── Episode ───────────────────────────────────────────────────────────────────


@dataclass
class Episode:
    """Complete historical market experience record.

    All twelve required fields from the Phase C2 spec:
      symbol, timestamp, raw features, portfolio state,
      active hypotheses, chosen action, confidence, reasoning,
      outcome, trade result, regime context, session context.
    """

    episode_id:          str                  = field(
        default_factory=lambda: str(uuid.uuid4())[:14]
    )
    # 1. symbol
    instrument:          str                  = ""
    # 2. timestamp
    timestamp:           float                = field(default_factory=time.time)
    # 3. raw features (float vector + metadata)
    features:            EpisodeFeatures      = field(default_factory=EpisodeFeatures)
    # 4. portfolio state
    portfolio:           EpisodePortfolioState = field(
        default_factory=EpisodePortfolioState
    )
    # 5. active hypotheses (snapshot of live hypothesis dicts at the time)
    active_hypotheses:   list[dict]           = field(default_factory=list)
    # 6. chosen action
    action:              str                  = "NO_TRADE"  # ENTER|EXIT|SCALE|NO_TRADE
    # 7. confidence
    confidence:          float                = 0.0
    # 8. reasoning (narrative/thesis summary)
    reasoning:           str                  = ""
    # 9. outcome
    outcome:             EpisodeOutcome       = field(default_factory=EpisodeOutcome)
    # 10. trade result (alias: outcome.reward_atr — kept as a top-level
    #     convenience field so callers don't need to unwrap outcome)
    trade_result_atr:    float                = 0.0
    # 11. regime context
    regime_context:      str                  = ""   # e.g. "TRENDING", "RANGE"
    # 12. session context
    session_context:     str                  = ""   # e.g. "LONDON", "NY"

    # Optional: which concept (if any) this episode was assigned to by
    # SemanticMemory.  Populated externally after semantic clustering.
    concept_id:          str                  = ""

    def resolve(self, outcome: EpisodeOutcome) -> None:
        """Populate outcome fields after the trade closes."""
        self.outcome           = outcome
        self.trade_result_atr  = outcome.reward_atr

    def to_dict(self) -> dict:
        return {
            "episode_id":         self.episode_id,
            "instrument":         self.instrument,
            "timestamp":          round(self.timestamp, 3),
            "features":           self.features.to_dict(),
            "portfolio":          self.portfolio.to_dict(),
            "active_hypotheses":  self.active_hypotheses,
            "action":             self.action,
            "confidence":         round(self.confidence, 4),
            "reasoning":          self.reasoning,
            "outcome":            self.outcome.to_dict(),
            "trade_result_atr":   round(self.trade_result_atr, 6),
            "regime_context":     self.regime_context,
            "session_context":    self.session_context,
            "concept_id":         self.concept_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        return cls(
            episode_id=d.get("episode_id", str(uuid.uuid4())[:14]),
            instrument=d.get("instrument", ""),
            timestamp=d.get("timestamp", 0.0),
            features=EpisodeFeatures.from_dict(d.get("features", {})),
            portfolio=EpisodePortfolioState.from_dict(d.get("portfolio", {})),
            active_hypotheses=d.get("active_hypotheses", []),
            action=d.get("action", "NO_TRADE"),
            confidence=d.get("confidence", 0.0),
            reasoning=d.get("reasoning", ""),
            outcome=EpisodeOutcome.from_dict(d.get("outcome", {})),
            trade_result_atr=d.get("trade_result_atr", 0.0),
            regime_context=d.get("regime_context", ""),
            session_context=d.get("session_context", ""),
            concept_id=d.get("concept_id", ""),
        )


# ── SimilarEpisode ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SimilarEpisode:
    """One result from ``EpisodicMemory.retrieve_similar``.

    ``similarity``          — cosine similarity between query and episode.
    ``recency_weight``      — e^(−λ × age_days) decay factor.
    ``combined_weight``     — ``similarity × recency_weight``, the ranking key.
    ``outcome``             — the episode's outcome snapshot.
    ``historical_confidence`` — the confidence with which the action was taken.
    ``reward_atr``          — realized reward in ATR units.
    ``episode_id``          — the source episode's ID.
    ``regime_context``      — regime label at the time.
    ``session_context``     — session label at the time.
    """

    similarity:           float
    recency_weight:       float
    combined_weight:      float
    outcome:              EpisodeOutcome
    historical_confidence: float
    reward_atr:           float
    episode_id:           str
    regime_context:       str
    session_context:      str
    action:               str

    def to_dict(self) -> dict:
        return {
            "episode_id":            self.episode_id,
            "similarity":            round(self.similarity, 4),
            "recency_weight":        round(self.recency_weight, 4),
            "combined_weight":       round(self.combined_weight, 4),
            "historical_confidence": round(self.historical_confidence, 4),
            "reward_atr":            round(self.reward_atr, 6),
            "regime_context":        self.regime_context,
            "session_context":       self.session_context,
            "action":                self.action,
            "outcome":               self.outcome.to_dict(),
        }


# ── EpisodicIndex ─────────────────────────────────────────────────────────────


class EpisodicIndex:
    """Inverted index for fast filtered retrieval of episodes.

    Indexes by:
      * session       — ``episode.session_context``
      * volatility    — coarse vol bucket derived from
                        ``episode.features.volatility``
      * structure     — ``episode.regime_context`` (market structure label)
      * portfolio     — ``episode.portfolio.exposure_bucket()``

    Each value in the index is a list of ``episode_id`` strings.  The
    main ``EpisodicMemory`` store is the source of truth; this index only
    provides fast lookups.
    """

    def __init__(self) -> None:
        # Each key maps to list[episode_id]
        self._by_session:   dict[str, list[str]] = defaultdict(list)
        self._by_vol:       dict[str, list[str]] = defaultdict(list)
        self._by_structure: dict[str, list[str]] = defaultdict(list)
        self._by_portfolio: dict[str, list[str]] = defaultdict(list)

    def add(self, ep: Episode) -> None:
        eid = ep.episode_id
        if ep.session_context:
            self._by_session[ep.session_context].append(eid)
        vol_key = _vol_bucket(ep.features.volatility)
        self._by_vol[vol_key].append(eid)
        if ep.regime_context:
            self._by_structure[ep.regime_context.upper()].append(eid)
        self._by_portfolio[ep.portfolio.exposure_bucket()].append(eid)

    def remove(self, ep: Episode) -> None:
        """Remove one episode from all indexes.  O(n) — only called during
        pruning when the whole deque is pruned from the front."""
        eid = ep.episode_id
        _remove_from(self._by_session, ep.session_context, eid)
        _remove_from(self._by_vol, _vol_bucket(ep.features.volatility), eid)
        _remove_from(self._by_structure, ep.regime_context.upper(), eid)
        _remove_from(self._by_portfolio, ep.portfolio.exposure_bucket(), eid)

    def ids_for_session(self, session: str) -> list[str]:
        return list(self._by_session.get(session, []))

    def ids_for_vol(self, volatility: float) -> list[str]:
        return list(self._by_vol.get(_vol_bucket(volatility), []))

    def ids_for_structure(self, regime: str) -> list[str]:
        return list(self._by_structure.get(regime.upper(), []))

    def ids_for_portfolio(self, exposure_bucket: str) -> list[str]:
        return list(self._by_portfolio.get(exposure_bucket, []))

    def clear(self) -> None:
        self._by_session.clear()
        self._by_vol.clear()
        self._by_structure.clear()
        self._by_portfolio.clear()


# ── EpisodicMemory ────────────────────────────────────────────────────────────


class EpisodicMemory:
    """Multi-symbol store of complete market experience records.

    One instance is shared across all instruments.  Per-symbol stores are
    bounded FIFOs (oldest episodes are evicted when the limit is reached).

    Key capabilities
    ----------------
    *Append* — ``write_memory(episode)``
    *Filtered recall* — ``recall(instrument, session=..., vol_state=..., regime=...)``
    *Similarity recall* — ``retrieve_similar(features, instrument, k=...)``
    *Persistence* — ``to_snapshot(instrument)`` / ``load_snapshot(instrument, data)``
    """

    def __init__(
        self,
        max_episodes:     int   = _DEFAULT_MAX_EPISODES,
        k:                int   = _DEFAULT_K,
        recency_lambda:   float = _RECENCY_LAMBDA_DEFAULT,
        min_sim:          float = _MIN_SIM_THRESHOLD,
    ) -> None:
        self._max         = max(1, max_episodes)
        self._k           = max(1, k)
        self._lambda      = max(1e-9, recency_lambda)
        self._min_sim     = min_sim

        # instrument → deque[Episode]  (oldest first)
        self._store: dict[str, deque[Episode]] = {}
        # instrument → dict[episode_id, Episode]  (fast lookup)
        self._by_id: dict[str, dict[str, Episode]] = {}
        # instrument → EpisodicIndex
        self._index: dict[str, EpisodicIndex] = {}

    # ── Write ─────────────────────────────────────────────────────────────

    def write_memory(self, episode: Episode, now: float | None = None) -> None:
        """Append a new episode to the store.

        If the per-symbol buffer is at capacity the oldest episode is evicted
        and removed from the index.  ``episode.timestamp`` is set to ``now``
        if it is zero.
        """
        now = now if now is not None else time.time()
        if episode.timestamp == 0.0:
            episode.timestamp = now

        sym   = episode.instrument
        store = self._store.setdefault(sym, deque())
        by_id = self._by_id.setdefault(sym, {})
        idx   = self._index.setdefault(sym, EpisodicIndex())

        # Evict oldest if at capacity
        if len(store) >= self._max:
            evicted = store.popleft()
            by_id.pop(evicted.episode_id, None)
            idx.remove(evicted)

        store.append(episode)
        by_id[episode.episode_id] = episode
        idx.add(episode)

        logger.debug(
            f"[Episodic] {sym} episode {episode.episode_id} "
            f"action={episode.action} regime={episode.regime_context} "
            f"session={episode.session_context}"
        )

    def update_outcome(self, instrument: str, episode_id: str, outcome: EpisodeOutcome) -> bool:
        """Populate the outcome of an already-stored episode after trade close.

        Returns True if the episode was found and updated.
        """
        ep = self._by_id.get(instrument, {}).get(episode_id)
        if ep is None:
            return False
        ep.resolve(outcome)
        return True

    # ── Recall ────────────────────────────────────────────────────────────

    def recall(
        self,
        instrument:   str,
        n:            int   = 50,
        session:      str | None = None,
        vol_state:    str | None = None,   # "low" | "medium" | "high"
        regime:       str | None = None,
        portfolio:    str | None = None,   # "light" | "moderate" | "heavy"
        action:       str | None = None,
        resolved_only: bool = False,
        since:        float | None = None,
    ) -> list[Episode]:
        """Retrieve recent episodes filtered by one or more dimensions.

        Filters are AND-combined: all specified conditions must be met.
        Returns episodes oldest-first.

        Parameters
        ----------
        instrument:     Symbol to query.
        n:              Maximum number of results.
        session:        Filter to episodes in this session (e.g. "LONDON").
        vol_state:      Filter by volatility bucket ("low", "medium", "high").
        regime:         Filter by regime label (case-insensitive).
        portfolio:      Filter by portfolio exposure bucket.
        action:         Filter by chosen action (e.g. "ENTER").
        resolved_only:  Only return episodes with a resolved outcome.
        since:          Only return episodes with timestamp ≥ since.
        """
        store = self._store.get(instrument)
        if store is None:
            return []

        idx   = self._index.get(instrument, EpisodicIndex())
        by_id = self._by_id.get(instrument, {})

        # Collect candidate IDs from indexes (AND of non-None filters)
        candidate_ids: set[str] | None = None

        def _intersect(ids: list[str]) -> None:
            nonlocal candidate_ids
            s = set(ids)
            candidate_ids = s if candidate_ids is None else candidate_ids & s

        if session is not None:
            _intersect(idx.ids_for_session(session))
        if vol_state is not None:
            _intersect(idx.ids_for_vol(_vol_from_bucket(vol_state)))
        if regime is not None:
            _intersect(idx.ids_for_structure(regime))
        if portfolio is not None:
            _intersect(idx.ids_for_portfolio(portfolio))

        if candidate_ids is not None:
            candidates = [
                by_id[eid] for eid in candidate_ids if eid in by_id
            ]
        else:
            candidates = list(store)

        # Apply remaining filters that aren't indexed
        if action is not None:
            candidates = [ep for ep in candidates if ep.action == action.upper()]
        if resolved_only:
            candidates = [ep for ep in candidates if ep.outcome.resolved]
        if since is not None:
            candidates = [ep for ep in candidates if ep.timestamp >= since]

        # Sort oldest→newest, return last n
        candidates.sort(key=lambda ep: ep.timestamp)
        return candidates[-n:]

    def retrieve_similar(
        self,
        features:      EpisodeFeatures,
        instrument:    str,
        k:             int   | None = None,
        session:       str | None = None,
        regime:        str | None = None,
        resolved_only: bool = True,
        now:           float | None = None,
    ) -> list[SimilarEpisode]:
        """Find the K most similar historical episodes to ``features``.

        Uses cosine similarity over ``EpisodeFeatures.as_vector()`` combined
        with recency decay ``e^(−λ × age_days)`` — identical weighting
        strategy to ``InstitutionalIntentEngine``.

        Returns a list of ``SimilarEpisode`` sorted by ``combined_weight``
        descending (most relevant first).

        Also returns:
          * ``outcomes`` — the resolved ``EpisodeOutcome`` objects.
          * ``historical_confidence`` — confidence at decision time.
          * ``average_reward`` — mean realized reward across the K results.
            (Call ``average_reward_of(results)`` to get it.)

        Parameters
        ----------
        features:       Query feature vector.
        instrument:     Symbol to search within.
        k:              Number of results.  Defaults to engine default.
        session:        If given, pre-filter to this session.
        regime:         If given, pre-filter to this regime.
        resolved_only:  Only compare against episodes with known outcomes.
        now:            Override current time.
        """
        now = now if now is not None else time.time()
        k   = k if k is not None else self._k

        query = features.as_vector()

        candidates = self.recall(
            instrument,
            n=self._max,
            session=session,
            regime=regime,
            resolved_only=resolved_only,
        )

        scored: list[tuple[float, float, Episode]] = []
        for ep in candidates:
            sim = _cosine_sim(query, ep.features.as_vector())
            if sim < self._min_sim:
                continue
            age_days = max(0.0, now - ep.timestamp) / 86400.0
            rw       = math.exp(-self._lambda * age_days)
            scored.append((sim, rw, ep))

        # Sort by combined weight descending
        scored.sort(key=lambda t: t[0] * t[1], reverse=True)
        top = scored[:k]

        return [
            SimilarEpisode(
                similarity=sim,
                recency_weight=rw,
                combined_weight=sim * rw,
                outcome=ep.outcome,
                historical_confidence=ep.confidence,
                reward_atr=ep.trade_result_atr,
                episode_id=ep.episode_id,
                regime_context=ep.regime_context,
                session_context=ep.session_context,
                action=ep.action,
            )
            for sim, rw, ep in top
        ]

    # ── Statistics ────────────────────────────────────────────────────────

    @staticmethod
    def average_reward(results: list[SimilarEpisode]) -> float:
        """Weighted average reward across similar episodes.

        Uses ``combined_weight`` as the per-episode weight so that
        very similar AND very recent episodes contribute more.
        """
        if not results:
            return 0.0
        total_w  = sum(r.combined_weight for r in results)
        if total_w < 1e-12:
            return 0.0
        return sum(r.reward_atr * r.combined_weight for r in results) / total_w

    @staticmethod
    def average_confidence(results: list[SimilarEpisode]) -> float:
        """Weighted average historical confidence across similar episodes."""
        if not results:
            return 0.0
        total_w = sum(r.combined_weight for r in results)
        if total_w < 1e-12:
            return 0.0
        return sum(r.historical_confidence * r.combined_weight for r in results) / total_w

    @staticmethod
    def win_rate(results: list[SimilarEpisode]) -> float:
        """Fraction of similar episodes (uniform weight) with positive reward."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.reward_atr > 0) / len(results)

    def episode_count(self, instrument: str) -> int:
        return len(self._store.get(instrument, deque()))

    def get_by_id(self, instrument: str, episode_id: str) -> Episode | None:
        return self._by_id.get(instrument, {}).get(episode_id)

    def instruments(self) -> list[str]:
        return sorted(self._store.keys())

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        return {
            "max_episodes": self._max,
            "episodes":     [ep.to_dict() for ep in self._store.get(instrument, [])],
        }

    def load_snapshot(self, instrument: str, data: dict) -> None:
        q: deque[Episode] = deque()
        by_id: dict[str, Episode] = {}
        idx   = EpisodicIndex()

        for d in data.get("episodes", []):
            try:
                ep = Episode.from_dict(d)
                q.append(ep)
                by_id[ep.episode_id] = ep
                idx.add(ep)
            except Exception:
                pass

        self._store[instrument] = q
        self._by_id[instrument] = by_id
        self._index[instrument] = idx


# ── Internal helpers ──────────────────────────────────────────────────────────


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity ∈ [−1, 1].  Returns 0.0 for zero vectors."""
    if len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _vol_bucket(volatility: float) -> str:
    """Map 0-1 volatility to a coarse bucket label."""
    if volatility < 0.33:
        return "low"
    if volatility < 0.67:
        return "medium"
    return "high"


def _vol_from_bucket(label: str) -> float:
    """Representative float for a vol bucket (used for index lookup)."""
    return {"low": 0.15, "medium": 0.50, "high": 0.85}.get(label.lower(), 0.50)


def _remove_from(d: dict[str, list[str]], key: str, eid: str) -> None:
    """Remove ``eid`` from ``d[key]`` if present."""
    lst = d.get(key)
    if lst and eid in lst:
        lst.remove(eid)
