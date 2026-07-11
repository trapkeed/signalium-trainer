"""QDE Brain — hypothesis set (competing scenarios).

A ``HypothesisSet`` is the core data structure for "thinking in possibilities
rather than predictions".  It groups a set of competing ``Hypothesis`` objects
for one instrument at one point in time into named scenarios whose posterior
probabilities always sum to 1.0:

    Scenario A (0.58)  trend continuation — price finds support on M15 EMA
    Scenario B (0.27)  fake breakout      — sweep above resistance, reversal
    Scenario C (0.15)  deep retracement   — correction to H1 50% Fib

The ``HypothesisSet`` never re-implements hypothesis storage or mutation —
those stay in ``HypothesisTracker`` / ``HypothesisEngine`` exactly as before.
This module only adds the scenario-grouping and probability-distribution layer:

  * Assigning a human-readable label and rank to each competing thesis.
  * Maintaining a normalized posterior probability per scenario.
  * Storing a timestamped ``ProbabilitySnapshot`` on every update so the
    evolution of the distribution can be inspected.
  * Tracking which scenario was eventually promoted (the "winning" thesis).

Design rules
------------
* ``HypothesisSet`` is pure data — no I/O, no calls to ``HypothesisEngine``.
  Mutation is done by ``HypothesisAPI``.
* ``HypothesisScenario`` holds a reference to the underlying ``Hypothesis``
  object; callers must not mutate the hypothesis through this reference —
  use ``HypothesisEngine`` for that.
* Probability values are full-precision floats; rounding happens only in
  ``to_dict()``.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.narrative.narrative_store import NarrativeKind

# Scenario label prefixes (A, B, C, … up to H for 8 competing scenarios)
_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# ── Scenario ──────────────────────────────────────────────────────────────────


@dataclass
class HypothesisScenario:
    """One named competing scenario within a ``HypothesisSet``.

    ``probability`` is this scenario's posterior probability in the current
    distribution — **not** the underlying ``hypothesis.confidence``.  The two
    can diverge: a scenario's confidence can be high while its probability is
    low if the competing scenarios are collectively much stronger.

    ``raw_score`` is the unnormalized score before normalization (used during
    Bayesian updates — callers should read ``probability`` for the posterior).

    ``is_active`` is True while the scenario is still live (not invalidated).
    An invalidated scenario retains its record in the set for lineage purposes
    but its probability is forced to 0 and is excluded from renormalization.
    """

    scenario_id: str  # short key, e.g. "A"
    label: str  # full label, e.g. "Scenario A: trend continuation"
    hypothesis_id: str
    hypothesis: Hypothesis
    probability: float = 0.0
    raw_score: float = 0.0
    probability_delta: float = 0.0  # last update change
    rank: int = 0  # 1 = most probable
    is_active: bool = True
    source_narrative_kind: NarrativeKind | None = None
    created_at: float = field(default_factory=time.time)

    def deactivate(self) -> None:
        self.is_active = False
        self.probability = 0.0
        self.rank = 0

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "label": self.label,
            "hypothesis_id": self.hypothesis_id,
            "probability": round(self.probability, 4),
            "probability_delta": round(self.probability_delta, 4),
            "rank": self.rank,
            "is_active": self.is_active,
            "direction": self.hypothesis.direction.value,
            "confidence": round(self.hypothesis.confidence, 4),
            "status": self.hypothesis.status.value,
            "thesis": self.hypothesis.thesis,
            "source_narrative_kind": (
                self.source_narrative_kind.value if self.source_narrative_kind else None
            ),
        }


# ── Probability snapshot ──────────────────────────────────────────────────────


@dataclass
class ProbabilitySnapshot:
    """Timestamped record of the probability distribution at one moment.

    Stored in ``HypothesisSet.probability_history`` to allow inspection of
    how the competing scenarios evolved over the life of the set.
    """

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    probabilities: dict[str, float] = field(default_factory=dict)  # scenario_id → prob
    entropy: float = 0.0
    certainty: float = 0.0
    trigger: str = "INIT"  # INIT | EVIDENCE | PROMOTION | INVALIDATION | AGE

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": round(self.timestamp, 3),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "entropy": round(self.entropy, 4),
            "certainty": round(self.certainty, 4),
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProbabilitySnapshot:
        return cls(
            snapshot_id=d.get("snapshot_id", str(uuid.uuid4())[:8]),
            timestamp=d.get("timestamp", 0.0),
            probabilities=d.get("probabilities", {}),
            entropy=d.get("entropy", 0.0),
            certainty=d.get("certainty", 0.0),
            trigger=d.get("trigger", "INIT"),
        )


# ── HypothesisSet ─────────────────────────────────────────────────────────────

_MAX_HISTORY = 50  # keep at most this many probability snapshots


@dataclass
class HypothesisSet:
    """A set of named, competing hypotheses for one instrument.

    ``scenarios``
        The competing scenarios, ordered by their current probability rank
        (most probable first after the first normalization).

    ``invalidated_scenario_ids``
        Scenarios that have been invalidated — retained for lineage but
        excluded from probability normalization.

    ``promoted_scenario_id``
        The scenario (if any) that has been explicitly promoted to dominant —
        i.e. the one the system currently believes is most likely to play out.

    ``probability_history``
        Bounded ring of ``ProbabilitySnapshot`` objects, one per
        ``update_probabilities`` call.
    """

    set_id: str
    instrument: str
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    scenarios: list[HypothesisScenario] = field(default_factory=list)
    invalidated_scenario_ids: set[str] = field(default_factory=set)
    promoted_scenario_id: str | None = None
    generation_count: int = 0  # update_probabilities calls
    probability_history: list[ProbabilitySnapshot] = field(default_factory=list)

    # ── Query helpers ─────────────────────────────────────────────────────

    @property
    def active_scenarios(self) -> list[HypothesisScenario]:
        """Scenarios that have not been invalidated, sorted by rank."""
        active = [s for s in self.scenarios if s.is_active]
        return sorted(active, key=lambda s: s.rank)

    @property
    def dominant(self) -> HypothesisScenario | None:
        """The scenario with rank=1 among active scenarios, or None."""
        active = self.active_scenarios
        return active[0] if active else None

    @property
    def dominant_direction(self) -> BeliefDirection:
        """Direction of the dominant scenario, or NEUTRAL if none."""
        dom = self.dominant
        return dom.hypothesis.direction if dom else BeliefDirection.NEUTRAL

    def scenario_by_id(self, scenario_id: str) -> HypothesisScenario | None:
        for s in self.scenarios:
            if s.scenario_id == scenario_id:
                return s
        return None

    def scenario_for_hypothesis(self, hypothesis_id: str) -> HypothesisScenario | None:
        for s in self.scenarios:
            if s.hypothesis_id == hypothesis_id:
                return s
        return None

    def probability_of(self, scenario_id: str) -> float:
        s = self.scenario_by_id(scenario_id)
        return s.probability if s else 0.0

    def entropy(self) -> float:
        """Current Shannon entropy of the probability distribution."""
        probs = [s.probability for s in self.scenarios if s.is_active and s.probability > 0]
        if not probs:
            return 0.0
        return -sum(p * math.log(max(p, 1e-15)) for p in probs)

    def certainty(self) -> float:
        """1 - normalized entropy.  0 = maximally uncertain, 1 = fully certain."""
        n = max(2, sum(1 for s in self.scenarios if s.is_active))
        max_ent = math.log(n)
        if max_ent < 1e-12:
            return 1.0
        return max(0.0, min(1.0, 1.0 - self.entropy() / max_ent))

    def latest_snapshot(self) -> ProbabilitySnapshot | None:
        return self.probability_history[-1] if self.probability_history else None

    # ── Mutation helpers (called by HypothesisAPI, not public) ───────────

    def _add_scenario(
        self,
        hypothesis: Hypothesis,
        narrative_kind: NarrativeKind | None = None,
    ) -> HypothesisScenario:
        """Add a new competing scenario to the set."""
        idx = len(self.scenarios)
        sid = _LABELS[idx] if idx < len(_LABELS) else str(idx + 1)
        label = f"Scenario {sid}: {hypothesis.thesis}"
        s = HypothesisScenario(
            scenario_id=sid,
            label=label,
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis=hypothesis,
            source_narrative_kind=narrative_kind,
        )
        self.scenarios.append(s)
        return s

    def _apply_probabilities(
        self,
        probs: dict[str, float],
        trigger: str,
        now: float | None = None,
    ) -> None:
        """Apply a new probability distribution and record a snapshot.

        ``probs`` maps scenario_id → posterior probability (should sum to 1).
        """
        now = now if now is not None else time.time()
        for s in self.scenarios:
            old = s.probability
            new = probs.get(s.scenario_id, 0.0)
            s.probability_delta = new - old
            s.probability = new

        # Re-rank
        active = sorted(
            [s for s in self.scenarios if s.is_active],
            key=lambda s: s.probability,
            reverse=True,
        )
        for rank, s in enumerate(active, start=1):
            s.rank = rank
        for s in self.scenarios:
            if not s.is_active:
                s.rank = 0

        self.generation_count += 1
        self.last_updated_at = now

        snap = ProbabilitySnapshot(
            timestamp=now,
            probabilities={s.scenario_id: s.probability for s in self.scenarios},
            entropy=round(self.entropy(), 4),
            certainty=round(self.certainty(), 4),
            trigger=trigger,
        )
        self.probability_history.append(snap)
        if len(self.probability_history) > _MAX_HISTORY:
            self.probability_history = self.probability_history[-_MAX_HISTORY:]

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "set_id": self.set_id,
            "instrument": self.instrument,
            "created_at": round(self.created_at, 3),
            "last_updated_at": round(self.last_updated_at, 3),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "invalidated_scenario_ids": list(self.invalidated_scenario_ids),
            "promoted_scenario_id": self.promoted_scenario_id,
            "generation_count": self.generation_count,
            "entropy": round(self.entropy(), 4),
            "certainty": round(self.certainty(), 4),
            "dominant_scenario_id": self.dominant.scenario_id if self.dominant else None,
            "dominant_direction": self.dominant_direction.value,
            "probability_history": [h.to_dict() for h in self.probability_history[-10:]],
        }


def make_hypothesis_set(instrument: str, set_id: str | None = None) -> HypothesisSet:
    """Factory: create an empty HypothesisSet for ``instrument``."""
    return HypothesisSet(
        set_id=set_id or str(uuid.uuid4())[:12],
        instrument=instrument,
    )


__all__ = [
    "HypothesisScenario",
    "HypothesisSet",
    "ProbabilitySnapshot",
    "make_hypothesis_set",
]
