"""QDE Brain — semantic memory: learned market concepts.

``SemanticMemory`` turns a stream of ``Episode`` records (from episodic
memory) into a set of named, self-organizing ``MarketConcept`` objects that
represent recurring market patterns the QDE has encountered.

Concepts emerge automatically
-------------------------------
There are no hardcoded rules like "breakout after compression = ATR slope > X".
Instead, when an episode is ingested, the system computes the cosine
similarity between the episode's feature vector and the centroid of every
existing concept.  If the best match exceeds ``creation_threshold`` the
episode is assigned to that concept and the centroid is updated via an EMA.
If no concept is similar enough, a new concept is proposed when there are
enough supporting episodes clustered nearby (or immediately if the store is
empty).

Named concept kinds
--------------------
``ConceptKind`` provides a vocabulary of labels that the system tries to
infer from outcome patterns after enough examples accumulate.  The kind
label is *advisory* — it is attached after the fact by
``_infer_concept_kind``, not before.  Callers can also supply a
``ConceptSeed`` to pre-label a cluster direction.

Concept update (online centroid learning)
------------------------------------------
A concept's centroid is the exponential moving average of all member
feature vectors::

    centroid = (1 − α) × centroid + α × new_vector

This is equivalent to how ``SymbolPersonality.update_trait`` works — a
gradually-adapting running average with no large single-step jumps.

Persistence
-----------
``to_snapshot()`` / ``load_snapshot()`` serialize the full concept store
in JSON-safe dicts.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.episodic.episodic_memory import Episode

# ── Hyper-parameters ──────────────────────────────────────────────────────────

_DEFAULT_MAX_CONCEPTS     = 200    # total across all instruments
_DEFAULT_CREATION_SIM     = 0.80   # below this → candidate for new concept
_DEFAULT_MIN_MEMBERS      = 3      # minimum before a concept is published
_DEFAULT_CENTROID_ALPHA   = 0.10   # EMA alpha for centroid update
_DEFAULT_CONCEPT_DEPTH    = 10_000 # maximum member episode IDs tracked per concept
_OUTCOME_WIN_ALPHA        = 0.15   # EMA for win_rate / avg_reward update


# ── ConceptKind ───────────────────────────────────────────────────────────────


class ConceptKind(str, Enum):
    """Labels assigned to market concepts after pattern discovery.

    These are the *interpretations* the system learns to associate with
    feature clusters — they are never hardcoded; a concept starts as
    ``UNKNOWN`` and is reclassified as its outcome statistics accumulate.

    The classification heuristics in ``_infer_concept_kind`` look at:
    * ``avg_reward_atr``  — positive = profitable pattern, negative = trap
    * ``win_rate``        — fraction of member episodes with direction_correct
    * Dominant feature values from the centroid (volatility, trend_strength,
      range_compression, absorption)
    """

    BREAKOUT_AFTER_COMPRESSION    = "BREAKOUT_AFTER_COMPRESSION"
    FAILED_LIQUIDITY_SWEEP        = "FAILED_LIQUIDITY_SWEEP"
    TREND_CONTINUATION_PULLBACK   = "TREND_CONTINUATION_PULLBACK"
    MEAN_REVERSION_AFTER_EXPANSION = "MEAN_REVERSION_AFTER_EXPANSION"
    ACCUMULATION_BREAKOUT         = "ACCUMULATION_BREAKOUT"
    DISTRIBUTION_REVERSAL         = "DISTRIBUTION_REVERSAL"
    STOP_HUNT_REVERSAL            = "STOP_HUNT_REVERSAL"
    RANGE_FADE                    = "RANGE_FADE"
    VOLATILITY_CONTRACTION        = "VOLATILITY_CONTRACTION"
    VOLATILITY_EXPANSION          = "VOLATILITY_EXPANSION"
    SESSION_OPEN_MOMENTUM         = "SESSION_OPEN_MOMENTUM"
    NEWS_SPIKE_FADE               = "NEWS_SPIKE_FADE"
    UNKNOWN                       = "UNKNOWN"


# ── ConceptSeed ───────────────────────────────────────────────────────────────


@dataclass
class ConceptSeed:
    """Optional caller-supplied label direction for a new concept.

    When passed to ``write_memory`` alongside an episode, the concept
    created for this episode (or the best-matching existing one) will
    be initialized with ``kind`` if it is currently ``UNKNOWN``.

    This lets callers provide a hint without forcing any label — the
    system will override with a data-derived kind once enough members
    accumulate.
    """

    kind:        ConceptKind = ConceptKind.UNKNOWN
    description: str         = ""


# ── MarketConcept ─────────────────────────────────────────────────────────────


@dataclass
class MarketConcept:
    """One learned market pattern.

    ``centroid``         — 12-dimensional average feature vector.  Updated
                           via EMA as new member episodes arrive.
    ``member_ids``       — episode IDs that belong to this concept (bounded).
    ``member_count``     — total number of episodes ever assigned (including
                           those pruned from ``member_ids``).
    ``win_rate``         — EMA-updated fraction of members where
                           ``direction_correct=True``.
    ``avg_reward_atr``   — EMA-updated average realized reward.
    ``occurrence_count`` — number of times this concept was matched (a
                           superset of ``member_count`` because matching
                           can happen without full ingestion).
    ``published``        — True once ``member_count >= min_members``.
    ``kind``             — inferred or caller-supplied concept label.
    """

    concept_id:       str          = field(
        default_factory=lambda: str(uuid.uuid4())[:12]
    )
    instrument:       str          = ""   # "" = cross-instrument
    kind:             ConceptKind  = ConceptKind.UNKNOWN
    description:      str          = ""
    centroid:         list[float]  = field(default_factory=lambda: [0.5] * 12)
    member_ids:       list[str]    = field(default_factory=list)
    member_count:     int          = 0
    occurrence_count: int          = 0
    win_rate:         float        = 0.5
    avg_reward_atr:   float        = 0.0
    published:        bool         = False
    created_at:       float        = field(default_factory=time.time)
    last_updated_at:  float        = field(default_factory=time.time)

    def update_centroid(
        self, vec: list[float], alpha: float = _DEFAULT_CENTROID_ALPHA
    ) -> None:
        """EMA-update the centroid toward ``vec``."""
        if len(vec) != len(self.centroid):
            return
        self.centroid = [
            (1.0 - alpha) * c + alpha * v
            for c, v in zip(self.centroid, vec)
        ]
        self.last_updated_at = time.time()

    def update_outcome(self, direction_correct: bool, reward_atr: float) -> None:
        """EMA-update win rate and average reward with one new outcome."""
        self.win_rate       = (1 - _OUTCOME_WIN_ALPHA) * self.win_rate + \
                               _OUTCOME_WIN_ALPHA * (1.0 if direction_correct else 0.0)
        self.avg_reward_atr = (1 - _OUTCOME_WIN_ALPHA) * self.avg_reward_atr + \
                               _OUTCOME_WIN_ALPHA * reward_atr

    def to_dict(self) -> dict:
        return {
            "concept_id":       self.concept_id,
            "instrument":       self.instrument,
            "kind":             self.kind.value,
            "description":      self.description,
            "centroid":         [round(v, 6) for v in self.centroid],
            "member_ids":       self.member_ids[-50:],   # keep last 50 in snapshot
            "member_count":     self.member_count,
            "occurrence_count": self.occurrence_count,
            "win_rate":         round(self.win_rate, 4),
            "avg_reward_atr":   round(self.avg_reward_atr, 6),
            "published":        self.published,
            "created_at":       round(self.created_at, 3),
            "last_updated_at":  round(self.last_updated_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> MarketConcept:
        return cls(
            concept_id=d.get("concept_id", str(uuid.uuid4())[:12]),
            instrument=d.get("instrument", ""),
            kind=ConceptKind(d.get("kind", ConceptKind.UNKNOWN.value)),
            description=d.get("description", ""),
            centroid=d.get("centroid", [0.5] * 12),
            member_ids=d.get("member_ids", []),
            member_count=d.get("member_count", 0),
            occurrence_count=d.get("occurrence_count", 0),
            win_rate=d.get("win_rate", 0.5),
            avg_reward_atr=d.get("avg_reward_atr", 0.0),
            published=d.get("published", False),
            created_at=d.get("created_at", 0.0),
            last_updated_at=d.get("last_updated_at", 0.0),
        )


# ── SemanticRecallResult ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SemanticRecallResult:
    """One result from ``SemanticMemory.recall``.

    ``similarity``      — cosine similarity to the query feature vector.
    ``concept``         — the matched concept.
    ``confidence``      — adjusted confidence: ``win_rate × similarity``.
    """

    similarity:  float
    concept:     MarketConcept
    confidence:  float

    def to_dict(self) -> dict:
        return {
            "concept_id":     self.concept.concept_id,
            "kind":           self.concept.kind.value,
            "similarity":     round(self.similarity, 4),
            "confidence":     round(self.confidence, 4),
            "win_rate":       round(self.concept.win_rate, 4),
            "avg_reward_atr": round(self.concept.avg_reward_atr, 6),
            "member_count":   self.concept.member_count,
            "published":      self.concept.published,
        }


# ── SemanticMemory ────────────────────────────────────────────────────────────


class SemanticMemory:
    """Online concept-clustering semantic memory store.

    Concepts emerge from episode patterns without manual labelling.  The
    only control knobs are:

    ``creation_threshold``
        Cosine similarity below which an episode is too dissimilar to
        all existing centroids to join any of them — triggering a new
        concept (once a pending-episode count threshold is met).

    ``min_members``
        Minimum member count before a concept is ``published`` (visible
        in ``recall``).  Unpublished concepts act as candidate clusters
        that may merge with later episodes.

    ``centroid_alpha``
        EMA alpha for centroid updates.  Small values (0.05–0.15) give
        stable centroids; larger values react faster to distribution
        shifts.
    """

    def __init__(
        self,
        max_concepts:       int   = _DEFAULT_MAX_CONCEPTS,
        creation_threshold: float = _DEFAULT_CREATION_SIM,
        min_members:        int   = _DEFAULT_MIN_MEMBERS,
        centroid_alpha:     float = _DEFAULT_CENTROID_ALPHA,
        concept_depth:      int   = _DEFAULT_CONCEPT_DEPTH,
    ) -> None:
        self._max_concepts      = max_concepts
        self._creation_threshold = creation_threshold
        self._min_members       = min_members
        self._alpha             = centroid_alpha
        self._depth             = concept_depth

        # concept_id → MarketConcept
        self._concepts: dict[str, MarketConcept] = {}
        # Pending episodes that haven't found a home yet:
        # feature_vec + optional seed + episode_id
        self._pending: list[tuple[list[float], str, ConceptSeed | None]] = []

    # ── Write ─────────────────────────────────────────────────────────────

    def write_memory(
        self,
        episode: Episode,
        seed:    ConceptSeed | None = None,
        now:     float | None = None,
    ) -> MarketConcept | None:
        """Ingest one episode into the semantic store.

        Attempts to assign the episode to the most similar existing
        concept.  If no concept is similar enough, the episode goes into
        a pending pool; when enough pending episodes are close to each
        other, a new concept is formed.

        Returns the concept the episode was assigned to, or ``None`` if
        the episode is pending.
        """
        now = now if now is not None else time.time()
        vec = episode.features.as_vector()

        best_concept, best_sim = self._find_best(vec, episode.instrument)

        if best_concept is not None and best_sim >= self._creation_threshold:
            # Assign to existing concept
            self._assign(best_concept, episode, vec, seed, now)
            return best_concept

        # No good match — add to pending
        self._pending.append((vec, episode.episode_id, seed))

        # Try to form a new concept from pending pool
        new_concept = self._try_form_concept(vec, episode.instrument, seed, now)
        if new_concept is not None:
            # Assign this episode to the new concept too
            self._assign(new_concept, episode, vec, seed, now)
            return new_concept

        return None

    # ── Recall ────────────────────────────────────────────────────────────

    def recall(
        self,
        feature_vec:     list[float],
        instrument:      str  = "",
        n:               int  = 10,
        published_only:  bool = True,
        min_win_rate:    float = 0.0,
    ) -> list[SemanticRecallResult]:
        """Retrieve the N most similar published concepts.

        Parameters
        ----------
        feature_vec:    Current market feature vector (same 12-dim as Episode).
        instrument:     If non-empty, only return concepts for this instrument
                        or cross-instrument concepts (``instrument=""``).
        n:              Maximum results.
        published_only: Skip concepts that haven't reached ``min_members``.
        min_win_rate:   Filter by minimum historical win rate.
        """
        candidates = list(self._concepts.values())

        if published_only:
            candidates = [c for c in candidates if c.published]
        if instrument:
            candidates = [
                c for c in candidates
                if not c.instrument or c.instrument == instrument
            ]
        if min_win_rate > 0.0:
            candidates = [c for c in candidates if c.win_rate >= min_win_rate]

        scored: list[tuple[float, MarketConcept]] = []
        for concept in candidates:
            sim = _cosine_sim(feature_vec, concept.centroid)
            scored.append((sim, concept))

        scored.sort(key=lambda t: t[0], reverse=True)

        results: list[SemanticRecallResult] = []
        for sim, concept in scored[:n]:
            concept.occurrence_count += 1
            results.append(SemanticRecallResult(
                similarity=round(sim, 4),
                concept=concept,
                confidence=round(sim * concept.win_rate, 4),
            ))
        return results

    def concepts(
        self,
        published_only: bool = True,
        instrument:     str  = "",
    ) -> list[MarketConcept]:
        """Return all concepts, optionally filtered."""
        cs = list(self._concepts.values())
        if published_only:
            cs = [c for c in cs if c.published]
        if instrument:
            cs = [c for c in cs if not c.instrument or c.instrument == instrument]
        return sorted(cs, key=lambda c: c.member_count, reverse=True)

    def concept_count(self, published_only: bool = True) -> int:
        if published_only:
            return sum(1 for c in self._concepts.values() if c.published)
        return len(self._concepts)

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        return {
            "concepts":  [c.to_dict() for c in self._concepts.values()],
            "pending_count": len(self._pending),  # pending vecs are ephemeral
        }

    def load_snapshot(self, data: dict) -> None:
        self._concepts.clear()
        self._pending.clear()
        for d in data.get("concepts", []):
            try:
                c = MarketConcept.from_dict(d)
                self._concepts[c.concept_id] = c
            except Exception:
                pass

    # ── Internals ─────────────────────────────────────────────────────────

    def _find_best(
        self,
        vec:        list[float],
        instrument: str,
    ) -> tuple[MarketConcept | None, float]:
        """Find the most similar existing concept for ``instrument``.

        Cross-instrument concepts (``concept.instrument == ""``) are always
        considered; per-instrument concepts only match their own instrument.
        """
        best_sim    = -1.0
        best_c: MarketConcept | None = None

        for c in self._concepts.values():
            if c.instrument and c.instrument != instrument:
                continue
            sim = _cosine_sim(vec, c.centroid)
            if sim > best_sim:
                best_sim = sim
                best_c   = c

        return best_c, best_sim

    def _assign(
        self,
        concept:  MarketConcept,
        episode:  Episode,
        vec:      list[float],
        seed:     ConceptSeed | None,
        now:      float,
    ) -> None:
        """Assign ``episode`` to ``concept`` and update its state."""
        concept.update_centroid(vec, self._alpha)
        concept.member_count     += 1
        concept.occurrence_count += 1
        concept.last_updated_at   = now

        # Track member IDs (bounded)
        concept.member_ids.append(episode.episode_id)
        if len(concept.member_ids) > self._depth:
            concept.member_ids = concept.member_ids[-self._depth:]

        # Update outcome statistics if available
        if episode.outcome.resolved:
            concept.update_outcome(
                episode.outcome.direction_correct,
                episode.outcome.reward_atr,
            )

        # Publish once min_members reached
        if not concept.published and concept.member_count >= self._min_members:
            concept.published = True
            # Re-infer kind from accumulated statistics
            concept.kind = _infer_concept_kind(concept)
            logger.debug(
                f"[Semantic] concept {concept.concept_id} published "
                f"kind={concept.kind.value} members={concept.member_count}"
            )

        # Apply seed label if concept is still UNKNOWN
        if seed and concept.kind == ConceptKind.UNKNOWN:
            concept.kind        = seed.kind
            concept.description = seed.description

        # Propagate concept_id to episode
        episode.concept_id = concept.concept_id

    def _try_form_concept(
        self,
        vec:        list[float],
        instrument: str,
        seed:       ConceptSeed | None,
        now:        float,
    ) -> MarketConcept | None:
        """Try to form a new concept from pending episodes near ``vec``.

        A new concept is created when:
          1. There are at least ``min_members − 1`` pending vecs with
             cosine similarity ≥ ``creation_threshold`` to ``vec``.
          2. The total number of concepts is below ``max_concepts``.

        The new concept's centroid is the mean of the matching pending vecs
        (including the current vec).

        If created, matching pending vecs are removed from the pool.
        """
        if len(self._concepts) >= self._max_concepts:
            return None

        # Check pending pool for similar vecs
        similar_pending: list[list[float]] = [vec]
        remaining: list[tuple[list[float], str, ConceptSeed | None]] = []

        for pv, pid, pseed in self._pending:
            sim = _cosine_sim(vec, pv)
            if sim >= self._creation_threshold:
                similar_pending.append(pv)
            else:
                remaining.append((pv, pid, pseed))

        # Need min_members similar vecs to form a concept
        if len(similar_pending) < self._min_members:
            # Not enough yet — leave pending pool unchanged
            return None

        # Compute mean centroid
        n      = len(similar_pending)
        centroid = [
            sum(v[i] for v in similar_pending) / n
            for i in range(len(vec))
        ]

        concept = MarketConcept(
            instrument=instrument,
            kind=seed.kind if seed else ConceptKind.UNKNOWN,
            description=seed.description if seed else "",
            centroid=centroid,
            created_at=now,
            last_updated_at=now,
        )
        self._concepts[concept.concept_id] = concept
        self._pending = remaining   # remove consumed pending vecs

        logger.debug(
            f"[Semantic] new concept {concept.concept_id} "
            f"from {len(similar_pending)} pending vecs "
            f"instrument={instrument or 'cross'}"
        )
        return concept


# ── Kind inference ────────────────────────────────────────────────────────────


def _infer_concept_kind(concept: MarketConcept) -> ConceptKind:
    """Infer a concept kind from its centroid + outcome statistics.

    All logic is based on the centroid's feature values and the
    accumulated win_rate / avg_reward — never on hardcoded market rules.

    Feature vector positions:
      [0] volatility        [1] momentum (normalized)  [2] trend_strength
      [3] range_compression [4] absorption             [5] volume_balance
      [6] session_quality   [7] microstructure         [8] regime_numeric
      [9] hypothesis_conf   [10] attention_weight      [11] spread_norm
    """
    if len(concept.centroid) < 12:
        return ConceptKind.UNKNOWN

    c = concept.centroid
    volatility        = c[0]
    momentum_norm     = c[1]   # 0=bearish, 0.5=neutral, 1=bullish
    trend_strength    = c[2]
    range_compression = c[3]
    absorption        = c[4]
    volume_balance    = c[5]   # >0.5=buy dominant
    win               = concept.win_rate
    avg_rw            = concept.avg_reward_atr

    # Signals for key patterns — threshold-free (uses relative magnitude)
    high_compression  = range_compression > 0.55
    high_absorption   = absorption > 0.50
    high_trend        = trend_strength > 0.55
    high_volatility   = volatility > 0.65
    low_volatility    = volatility < 0.35
    bullish_momentum  = momentum_norm > 0.60
    bearish_momentum  = momentum_norm < 0.40
    buy_dominant      = volume_balance > 0.60
    sell_dominant     = volume_balance < 0.40
    profitable        = avg_rw > 0.0 and win > 0.5

    if high_compression and profitable:
        if high_absorption:
            return ConceptKind.BREAKOUT_AFTER_COMPRESSION
        if buy_dominant:
            return ConceptKind.ACCUMULATION_BREAKOUT
        if sell_dominant:
            return ConceptKind.DISTRIBUTION_REVERSAL

    if high_trend and profitable:
        if bullish_momentum or bearish_momentum:
            return ConceptKind.TREND_CONTINUATION_PULLBACK

    if high_volatility and not profitable:
        return ConceptKind.FAILED_LIQUIDITY_SWEEP

    if high_volatility and profitable and abs(momentum_norm - 0.5) > 0.2:
        return ConceptKind.STOP_HUNT_REVERSAL

    if low_volatility and high_compression:
        return ConceptKind.VOLATILITY_CONTRACTION

    if high_volatility and not high_trend:
        return ConceptKind.VOLATILITY_EXPANSION

    if high_trend and not profitable:
        return ConceptKind.MEAN_REVERSION_AFTER_EXPANSION

    if high_absorption and not high_trend:
        return ConceptKind.RANGE_FADE

    return ConceptKind.UNKNOWN


# ── Cosine similarity helper ──────────────────────────────────────────────────


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
