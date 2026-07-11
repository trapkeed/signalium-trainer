"""QDE Brain — hypothesis subpackage (Phase C4).

This package adds the "think in possibilities rather than predictions" layer
on top of the existing hypothesis infrastructure (``hypotheses.py``,
``hypothesis_engine.py``, ``hypothesis_lifecycle.py``,
``hypothesis_registry.py``).

What was already there (untouched)
-------------------------------------
  hypotheses.py             ``HypothesisTracker`` — form/reinforce/weaken/age
  hypothesis_engine.py      ``HypothesisEngine``  — orchestration façade
  hypothesis_lifecycle.py   ``HypothesisMaturity`` / ``compute_maturity``
  hypothesis_registry.py    ``HypothesisRegistry`` / ``HypothesisView``
  brain_models.py           ``Hypothesis`` dataclass

What this package adds (Phase C4)
-------------------------------------
  hypothesis_set.py          ``HypothesisSet`` / ``HypothesisScenario``
                             Competing scenarios with normalized probabilities
                             that sum to 1.0.  Per-tick probability snapshots.

  hypothesis_probability.py  ``HypothesisProbabilityEngine``
                             Bayesian-style probability updates over a set.

  hypothesis_failure_tracker.py ``HypothesisFailureTracker``
                             Structured failure records: failure mode taxonomy,
                             lesson extraction, per-direction win-rate tracking.

  hypothesis_lineage.py      ``HypothesisLineage`` / ``LineageNode``
                             Parent/child tree: root hypothesis → competing
                             sub-scenarios → resolution.

  hypothesis_memory_bridge.py ``HypothesisMemoryBridge``
                             Connects C4 to C2 episodic/semantic memory for
                             seed confidence biasing and episode storage.

  hypothesis_api.py          ``HypothesisAPI``
                             Four spec-required methods:
                               generate_hypotheses()
                               update_probabilities()
                               invalidate_hypothesis()
                               promote_hypothesis()

Integration example
-------------------
::

    # At initialization:
    api = HypothesisAPI(
        hypothesis_engine=brain.hypothesis_engine,
        memory_bridge=HypothesisMemoryBridge(episodic_memory=emem),
    )

    # On each tick where narrative seeds are available:
    seeds = narrative_engine.suggest_hypothesis_seeds(instrument)
    result = api.generate_hypotheses(instrument, inp, seeds, features=feat_dict)

    # Display competing scenarios:
    for s in result.hypothesis_set.active_scenarios:
        print(f"  {s.label}: {s.probability:.0%}")
    # Scenario A: trend continuation: 64%
    # Scenario B: fake breakout:      24%
    # Scenario C: deep retracement:   12%

    # When new evidence arrives:
    api.update_probabilities(result.hypothesis_set, {"A": +0.4, "B": -0.2})

    # When a scenario is disproven:
    api.invalidate_hypothesis("hyp-abc123", reason="price broke support")

    # When a scenario plays out:
    api.promote_hypothesis("hyp-def456", reason="breakout confirmed on volume")
"""

from trainer.qde.brain.hypothesis.hypothesis_api import (
    HypothesisAPI,
    HypothesisGenerationResult,
)
from trainer.qde.brain.hypothesis.hypothesis_failure_tracker import (
    FailureMode,
    HypothesisFailureRecord,
    HypothesisFailureTracker,
)
from trainer.qde.brain.hypothesis.hypothesis_lineage import (
    HypothesisLineage,
    LineageNode,
    ResolutionState,
)
from trainer.qde.brain.hypothesis.hypothesis_memory_bridge import (
    HypothesisMemoryBridge,
    HypothesisMemoryContext,
)
from trainer.qde.brain.hypothesis.hypothesis_probability import (
    HypothesisProbabilityEngine,
    ProbabilityUpdateResult,
)
from trainer.qde.brain.hypothesis.hypothesis_set import (
    HypothesisScenario,
    HypothesisSet,
    ProbabilitySnapshot,
    make_hypothesis_set,
)

__all__ = [
    # ── Primary API (Phase C4) ────────────────────────────────────────────
    "HypothesisAPI",
    "HypothesisGenerationResult",
    # ── Scenario set ──────────────────────────────────────────────────────
    "HypothesisSet",
    "HypothesisScenario",
    "ProbabilitySnapshot",
    "make_hypothesis_set",
    # ── Probability engine ────────────────────────────────────────────────
    "HypothesisProbabilityEngine",
    "ProbabilityUpdateResult",
    # ── Failure tracking ──────────────────────────────────────────────────
    "HypothesisFailureTracker",
    "HypothesisFailureRecord",
    "FailureMode",
    # ── Lineage ───────────────────────────────────────────────────────────
    "HypothesisLineage",
    "LineageNode",
    "ResolutionState",
    # ── Memory bridge ─────────────────────────────────────────────────────
    "HypothesisMemoryBridge",
    "HypothesisMemoryContext",
]
