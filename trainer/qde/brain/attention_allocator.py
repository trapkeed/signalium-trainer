"""QDE Brain — multi-signal attention scoring and allocation.

`attention.AttentionAllocator` already exists and is used internally by
`brain.py`'s per-tick pipeline for a lightweight salience score
(readiness + volatility quality + hypothesis activity + experience
magnitude), normalized across symbols. That allocator's normalization
step (`reallocate`) is correct and reused here as-is — this module does
not reimplement it.

What's missing is the richer scoring this feature asks for: a
per-symbol score built from six named, independently-decaying signal
components (`opportunity_scanner.OpportunitySignals`), which
continuously rebalances as new opportunities arrive and decays when
they go quiet — rather than being recomputed from scratch each tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from trainer.qde.brain.attention_base import AttentionAllocator
from trainer.qde.brain.brain_models import AttentionAllocation
from trainer.qde.brain.opportunity_scanner import OpportunitySignals

# Half-life-style decay: an untouched attention score loses this fraction
# of its remaining value per second of silence, so attention "cools off"
# smoothly rather than snapping to zero or lingering forever.
DEFAULT_DECAY_PER_SECOND = 0.0015


@dataclass(frozen=True)
class AttentionWeights:
    """Relative importance of each opportunity signal in the composite
    score.

    The first six weights (defaults sum to 1.0) cover the original signals
    produced by ``opportunity_scanner.scan()``.  The four new C10 weights
    cover the additional signals produced by
    ``signal_enricher.enrich_signals()``; they default to 0.0 so that the
    composite score is unchanged for callers that never supply an
    ``ExtendedSignalContext`` (backward-compatible).

    When the C10 weights are non-zero, they add on top of the original six
    and may push the raw composite above 1.0 for a tick where many factors
    converge — the composite is clamped to 1.0 in ``AttentionScoreAllocator
    .update()`` via ``min(1.0, ...)``.
    """

    # Original six weights
    hypothesis_activity: float = 0.25
    volatility_expansion: float = 0.20
    liquidity_event: float = 0.15
    news_impact: float = 0.15
    unusual_behavior: float = 0.15
    session_transition: float = 0.10

    # C10: four new weights (default 0.0 = backward-compatible)
    volume_anomaly: float = 0.0
    key_level_proximity: float = 0.0
    spread_change: float = 0.0
    failed_move: float = 0.0

    def score(self, signals: OpportunitySignals) -> float:
        return max(
            0.0,
            self.hypothesis_activity * signals.hypothesis_activity
            + self.volatility_expansion * signals.volatility_expansion
            + self.liquidity_event * signals.liquidity_event
            + self.news_impact * signals.news_impact
            + self.unusual_behavior * signals.unusual_behavior
            + self.session_transition * signals.session_transition
            + self.volume_anomaly * signals.volume_anomaly
            + self.key_level_proximity * signals.key_level_proximity
            + self.spread_change * signals.spread_change
            + self.failed_move * signals.failed_move,
        )


#: C10-tuned default weights that activate the four new signals.
#: These shift weight slightly from the original six to keep the
#: total budget reasonable while giving meaningful signal contribution
#: from the new factors when context is available.
C10_WEIGHTS = AttentionWeights(
    hypothesis_activity=0.20,
    volatility_expansion=0.15,
    liquidity_event=0.12,
    news_impact=0.10,
    unusual_behavior=0.10,
    session_transition=0.08,
    # C10 additions
    volume_anomaly=0.10,
    key_level_proximity=0.08,
    spread_change=0.05,
    failed_move=0.02,
)


DEFAULT_WEIGHTS = AttentionWeights()


@dataclass
class AttentionScore:
    """Persistent per-symbol attention state maintained across ticks."""

    instrument: str
    composite: float = 0.0
    weight: float = 0.0
    last_signals: OpportunitySignals = field(default_factory=OpportunitySignals)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "composite": round(self.composite, 4),
            "weight": round(self.weight, 4),
            "last_signals": self.last_signals.to_dict(),
            "updated_at": self.updated_at,
        }


class AttentionScoreAllocator:
    """Maintains a continuously-rebalancing attention score per symbol.

    Two responsibilities kept separate:
      - `update()` folds a new opportunity reading into a symbol's
        running composite score (decaying the prior value first, so a
        quiet symbol's score fades even without a fresh signal).
      - `reallocate()` normalizes all current composite scores into
        cross-symbol weights, delegating to the existing
        `attention.AttentionAllocator` rather than re-deriving
        normalization math.
    """

    def __init__(
        self,
        weights: AttentionWeights = DEFAULT_WEIGHTS,
        decay_per_second: float = DEFAULT_DECAY_PER_SECOND,
    ) -> None:
        self._weights = weights
        self._decay_per_second = max(0.0, decay_per_second)
        self._scores: dict[str, AttentionScore] = {}
        self._normalizer = AttentionAllocator()

    def _decay(self, score: AttentionScore, now: float) -> float:
        elapsed = max(0.0, now - score.updated_at)
        if elapsed <= 0 or self._decay_per_second <= 0:
            return score.composite
        retained = max(0.0, 1.0 - self._decay_per_second) ** elapsed
        return score.composite * retained

    def update(
        self, instrument: str, signals: OpportunitySignals, now: float | None = None
    ) -> AttentionScore:
        now = now if now is not None else time.time()
        prior = self._scores.get(instrument)
        decayed_prior = self._decay(prior, now) if prior is not None else 0.0
        fresh = self._weights.score(signals)
        # New opportunities boost on top of whatever residual attention
        # remains, rather than fully overwriting it — a symbol that was
        # already interesting stays elevated a little longer.
        composite = min(1.0, max(decayed_prior, fresh) + 0.25 * min(decayed_prior, fresh))

        score = self._scores.setdefault(instrument, AttentionScore(instrument=instrument))
        score.composite = composite
        score.last_signals = signals
        score.updated_at = now
        return score

    def decay_all(self, now: float | None = None) -> None:
        """Apply pure time-decay to every tracked symbol without a fresh
        signal this cycle — this is what keeps attention continuously
        rebalancing even for symbols that haven't ticked recently.
        """
        now = now if now is not None else time.time()
        for score in self._scores.values():
            score.composite = self._decay(score, now)
            score.updated_at = now

    def reallocate(self, now: float | None = None) -> dict[str, AttentionAllocation]:
        salience_by_instrument = {
            instrument: score.composite for instrument, score in self._scores.items()
        }
        allocations = self._normalizer.reallocate(salience_by_instrument, now=now)
        for instrument, allocation in allocations.items():
            self._scores[instrument].weight = allocation.weight
        return allocations

    def high_attention(self, threshold: float = 0.15) -> list[str]:
        """Symbols whose current normalized weight meets `threshold`,
        ranked highest-first. Callers use this to gate expensive
        reasoning cycles to only the symbols that deserve them.
        """
        ranked = sorted(self._scores.values(), key=lambda s: s.weight, reverse=True)
        return [s.instrument for s in ranked if s.weight >= threshold]

    def get(self, instrument: str) -> AttentionScore | None:
        return self._scores.get(instrument)

    def all(self) -> dict[str, AttentionScore]:
        return dict(self._scores)

    def remove(self, instrument: str) -> None:
        self._scores.pop(instrument, None)
