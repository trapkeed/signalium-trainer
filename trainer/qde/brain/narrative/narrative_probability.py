"""QDE Brain — narrative probability distribution.

Converts the current set of active `SymbolNarrative` objects into a
normalized probability distribution across all seven `NarrativeKind` values.

Why a probability distribution?
---------------------------------
Individual narrative confidence values are not normalized — they represent
the engine's certainty *within* a single narrative, not the relative weight
of each story *compared with others*.  Two simultaneous narratives can both
have 0.80 confidence and neither "wins" in isolation.  The probability
distribution answers a different question:

    "Given everything active right now, how likely is each narrative kind
     to be the correct interpretation of this market?"

The answer looks like:
    TREND_CONTINUATION  : 0.64
    LIQUIDITY_GRAB      : 0.23
    DISTRIBUTION        : 0.13
    (remaining 4 kinds) : ~0.00  (epsilon prior)

Algorithm
---------
1. For each of the 7 `NarrativeKind` values, collect the narrative's
   confidence if it is currently active, or a small epsilon prior otherwise.
2. Apply a maturity weight: ESTABLISHED narratives contribute fully,
   DEVELOPING at 0.75, FORMING at 0.40, FADING at 0.20.
3. Normalize the weighted confidences to sum to 1.0.
4. Compute Shannon entropy over the distribution as a measure of certainty:
   low entropy = one dominant story; high entropy = ambiguous.

The distribution is purely a function of the inputs; it does not mutate any
narrative.  ``NarrativeProbabilityEngine`` is stateless and safe to reuse
across symbols and ticks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from trainer.qde.brain.narrative.narrative_store import (
    NarrativeKind,
    NarrativeMaturity,
    SymbolNarrative,
)

# ── Maturity weights ──────────────────────────────────────────────────────────

_MATURITY_WEIGHT: dict[NarrativeMaturity, float] = {
    NarrativeMaturity.ESTABLISHED: 1.00,
    NarrativeMaturity.DEVELOPING:  0.75,
    NarrativeMaturity.FORMING:     0.40,
    NarrativeMaturity.FADING:      0.20,
    NarrativeMaturity.EXPIRED:     0.00,
}

# Small prior for kinds that have no active narrative.
# Prevents zero-probability surprise and allows entropy to be well-defined.
_EPSILON_PRIOR = 0.01

_ALL_KINDS: list[NarrativeKind] = list(NarrativeKind)


# ── Output types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NarrativeProbability:
    """One kind's probability in the current distribution.

    ``raw_weight``   — unnormalized weight before softmax-style normalization.
    ``probability``  — normalized 0-1 probability; all 7 sum to 1.0.
    ``confidence``   — the underlying narrative's confidence (0 if inactive).
    ``maturity``     — the underlying narrative's maturity (EXPIRED if inactive).
    ``active``       — True if this kind has a living, non-EXPIRED narrative.
    """

    kind:        NarrativeKind
    raw_weight:  float
    probability: float
    confidence:  float
    maturity:    NarrativeMaturity
    active:      bool

    def to_dict(self) -> dict:
        return {
            "kind":        self.kind.value,
            "probability": round(self.probability, 4),
            "confidence":  round(self.confidence, 4),
            "maturity":    self.maturity.value,
            "active":      self.active,
        }


@dataclass
class NarrativeProbabilityDistribution:
    """Normalized probability distribution across all seven narrative kinds.

    Attributes
    ----------
    probs:
        Full list of ``NarrativeProbability`` objects, one per kind, sorted
        by probability descending.
    dominant:
        The ``NarrativeKind`` with the highest probability, or ``None`` when
        all kinds have equal probability (completely ambiguous).
    dominant_probability:
        Probability of the dominant kind (0-1).
    entropy:
        Shannon entropy of the distribution in nats (not bits).
        Range: 0 (one kind dominates) → ln(7) ≈ 1.95 (maximum uncertainty).
    certainty:
        1 - (entropy / max_entropy).  0 = maximum uncertainty, 1 = certainty.
    """

    probs:                list[NarrativeProbability] = field(default_factory=list)
    dominant:             NarrativeKind | None       = None
    dominant_probability: float                      = 0.0
    entropy:              float                      = 0.0
    certainty:            float                      = 0.0

    # ── Query helpers ─────────────────────────────────────────────────────

    def probability_of(self, kind: NarrativeKind) -> float:
        """Probability for a specific narrative kind."""
        for p in self.probs:
            if p.kind == kind:
                return p.probability
        return 0.0

    def top_k(self, k: int = 3) -> list[NarrativeProbability]:
        """Return the top-k kinds by probability (already sorted descending)."""
        return self.probs[:max(1, k)]

    def active_probs(self) -> list[NarrativeProbability]:
        """Only the kinds that have an active (non-epsilon) narrative."""
        return [p for p in self.probs if p.active]

    def as_dict_probs(self) -> dict[str, float]:
        """Simple kind.value → probability mapping for serialization."""
        return {p.kind.value: round(p.probability, 4) for p in self.probs}

    def to_dict(self) -> dict:
        return {
            "probabilities":         [p.to_dict() for p in self.probs],
            "dominant":              self.dominant.value if self.dominant else None,
            "dominant_probability":  round(self.dominant_probability, 4),
            "entropy":               round(self.entropy, 4),
            "certainty":             round(self.certainty, 4),
            "active_count":          sum(1 for p in self.probs if p.active),
        }


# ── Engine ────────────────────────────────────────────────────────────────────

_MAX_ENTROPY = math.log(len(_ALL_KINDS))   # ln(7) ≈ 1.946


class NarrativeProbabilityEngine:
    """Stateless converter: active narratives → probability distribution.

    Usage::

        engine = NarrativeProbabilityEngine()
        active = narrative_store.active(store)
        dist   = engine.compute(active)

        print(dist.dominant)               # NarrativeKind.TREND_CONTINUATION
        print(dist.dominant_probability)   # 0.64
        print(dist.top_k(3))               # top 3 kinds with probabilities
        print(dist.certainty)              # 0 = uniform, 1 = one-dominant

    Parameters
    ----------
    epsilon_prior:
        Probability floor for inactive kinds.  Default 0.01 keeps zero-
        probability states out of the distribution.  Set to 0 to get hard
        zero for inactive kinds (not recommended — breaks entropy).
    """

    def __init__(self, epsilon_prior: float = _EPSILON_PRIOR) -> None:
        self._eps = max(0.0, epsilon_prior)

    def compute(
        self,
        narratives:    list[SymbolNarrative],
        boost_kind:    NarrativeKind | None = None,
        boost_amount:  float = 0.0,
    ) -> NarrativeProbabilityDistribution:
        """Compute the probability distribution for the given narrative list.

        Parameters
        ----------
        narratives:
            All active (non-EXPIRED) narratives for one symbol.  Expired
            narratives are silently ignored.
        boost_kind:
            Optional kind to apply an additive weight boost to before
            normalization.  Useful when memory recall or portfolio context
            provides additional evidence for a particular kind.
        boost_amount:
            Additive boost for ``boost_kind`` (0–0.5 typical).
        """
        # Build a lookup for quick access
        by_kind: dict[NarrativeKind, SymbolNarrative] = {}
        for n in narratives:
            if n.maturity != NarrativeMaturity.EXPIRED:
                by_kind[n.kind] = n

        # Raw weight for each kind
        raw: dict[NarrativeKind, float] = {}
        for kind in _ALL_KINDS:
            n = by_kind.get(kind)
            if n is None:
                raw[kind] = self._eps
            else:
                mw = _MATURITY_WEIGHT.get(n.maturity, 0.0)
                raw[kind] = max(self._eps, n.confidence * mw)

        # Apply optional boost
        if boost_kind is not None and boost_amount > 0.0:
            raw[boost_kind] = raw.get(boost_kind, self._eps) + boost_amount

        # Normalize
        total = sum(raw.values())
        if total < 1e-12:
            # Fallback: uniform distribution
            probs_norm = {k: 1.0 / len(_ALL_KINDS) for k in _ALL_KINDS}
        else:
            probs_norm = {k: v / total for k, v in raw.items()}

        # Build result objects — store full-precision probability to keep
        # sum(p.probability) == 1.0 exactly; round only in to_dict().
        result_probs: list[NarrativeProbability] = [
            NarrativeProbability(
                kind=kind,
                raw_weight=raw[kind],
                probability=probs_norm[kind],
                confidence=by_kind[kind].confidence if kind in by_kind else 0.0,
                maturity=by_kind[kind].maturity if kind in by_kind else NarrativeMaturity.EXPIRED,
                active=kind in by_kind,
            )
            for kind in _ALL_KINDS
        ]
        # Sort by probability descending
        result_probs.sort(key=lambda p: p.probability, reverse=True)

        # Entropy
        entropy = -sum(
            p.probability * math.log(max(p.probability, 1e-15))
            for p in result_probs
        )
        certainty = 1.0 - (entropy / _MAX_ENTROPY) if _MAX_ENTROPY > 0 else 0.0

        # Dominant
        dominant_p = result_probs[0]
        # Consider it dominant only if it is genuinely above a flat distribution
        is_dominant = dominant_p.active and dominant_p.probability > (1.0 / len(_ALL_KINDS) + 0.05)

        return NarrativeProbabilityDistribution(
            probs=result_probs,
            dominant=dominant_p.kind if is_dominant else None,
            dominant_probability=dominant_p.probability,
            entropy=round(entropy, 4),
            certainty=round(max(0.0, min(1.0, certainty)), 4),
        )

    @staticmethod
    def from_confidences(
        kind_confidences: dict[NarrativeKind, float],
        epsilon_prior:    float = _EPSILON_PRIOR,
    ) -> NarrativeProbabilityDistribution:
        """Convenience constructor from a raw {kind: confidence} dict.

        Useful for tests and for callers that hold narrative confidences
        in a different form than a list of ``SymbolNarrative`` objects.
        """
        engine = NarrativeProbabilityEngine(epsilon_prior=epsilon_prior)
        # Build fake narratives with DEVELOPING maturity for each kind
        from trainer.qde.brain.narrative.narrative_store import NarrativeMaturity
        narratives: list[SymbolNarrative] = []
        for kind, conf in kind_confidences.items():
            n = SymbolNarrative(
                kind=kind,
                confidence=conf,
                maturity=NarrativeMaturity.DEVELOPING,
            )
            narratives.append(n)
        return engine.compute(narratives)


__all__ = [
    "NarrativeProbability",
    "NarrativeProbabilityDistribution",
    "NarrativeProbabilityEngine",
]
