"""QDE Brain — hypothesis maturity lifecycle.

This module adds a second, *execution-facing* status vocabulary on top of
the existing narrative-tracking `Hypothesis` (`brain_models.py`) /
`HypothesisTracker` (`hypotheses.py`). It does not duplicate their
create/reinforce/weaken/invalidate machinery or introduce a parallel
storage model — it purely *projects* a `HypothesisMaturity` from the
existing `Hypothesis` fields (confidence, reinforcement count,
contradicting-evidence count, age, underlying `HypothesisStatus`).

Why a second vocabulary
------------------------
`HypothesisStatus` (FORMING/ACTIVE/REINFORCED/WEAKENING/INVALIDATED/
CONFIRMED/EXPIRED) describes the *narrative* lifecycle of a belief — how
QDE's internal story about a symbol is evolving. `HypothesisMaturity`
(OBSERVING/BUILDING/HIGH_CONVICTION/EXECUTABLE/INVALIDATED/EXPIRED)
answers a different, execution-facing question: *is this thesis mature
enough that a trade should emerge from it, rather than from an isolated
per-tick prediction?* Mixing the two into one enum would force every
narrative status change to also mean something about execution
readiness, which is not true (e.g. a `REINFORCED` hypothesis with thin
evidence should not automatically be tradeable).

Trade execution should emerge from mature hypotheses
------------------------------------------------------
Only `HypothesisMaturity.EXECUTABLE` denotes a hypothesis mature enough
to justify a trade. Reaching it requires *sustained* conviction — high
confidence, multiple independent reinforcements, evidence not currently
dominated by contradictions, and a minimum dwell time — not a single
strong tick. This mirrors `qde_observer.py`'s readiness-score gating
(never gate off one instant) and keeps the *only* place execution
decisions are actually made at `qde_inference.derive_decision_fields`;
this module supplies one more piece of context that decision layer may
consult (via `hypothesis_engine.HypothesisEngine.executable_hypotheses`),
it never places orders itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.brain_types import HypothesisStatus


class HypothesisMaturity(str, Enum):
    """Execution-facing maturity of a hypothesis."""

    OBSERVING = "OBSERVING"  # just formed, thin/no evidence yet
    BUILDING = "BUILDING"  # gathering evidence, conviction not yet high
    HIGH_CONVICTION = "HIGH_CONVICTION"  # strong, repeatedly-reinforced belief
    EXECUTABLE = "EXECUTABLE"  # mature enough that a trade may emerge from it
    INVALIDATED = "INVALIDATED"  # disproven by evidence
    EXPIRED = "EXPIRED"  # aged out or explicitly retired without resolving


_TERMINAL_MATURITIES = frozenset({HypothesisMaturity.INVALIDATED, HypothesisMaturity.EXPIRED})


def is_terminal_maturity(maturity: HypothesisMaturity) -> bool:
    return maturity in _TERMINAL_MATURITIES


@dataclass(frozen=True)
class MaturityThresholds:
    """Tunable gates for the OBSERVING → ... → EXECUTABLE progression."""

    building_confidence: float = 0.35
    high_conviction_confidence: float = 0.6
    executable_confidence: float = 0.8
    high_conviction_min_reinforcements: int = 2
    executable_min_reinforcements: int = 3
    executable_min_age_seconds: float = 300.0  # 5 minutes of sustained conviction


DEFAULT_MATURITY_THRESHOLDS = MaturityThresholds()


def hypothesis_age_seconds(hypothesis: Hypothesis, now: float | None = None) -> float:
    now = now if now is not None else time.time()
    return max(0.0, now - hypothesis.created_at)


def compute_maturity(
    hypothesis: Hypothesis,
    now: float | None = None,
    thresholds: MaturityThresholds = DEFAULT_MATURITY_THRESHOLDS,
) -> HypothesisMaturity:
    """Project the execution-facing maturity of a narrative `Hypothesis`.

    Pure function of the hypothesis's current fields — never mutates the
    hypothesis and never touches storage, so it is safe to call as often
    as needed (e.g. on every registry query) without side effects.
    """
    now = now if now is not None else time.time()

    if hypothesis.status == HypothesisStatus.INVALIDATED:
        return HypothesisMaturity.INVALIDATED
    if hypothesis.status == HypothesisStatus.EXPIRED:
        return HypothesisMaturity.EXPIRED
    if hypothesis.status == HypothesisStatus.CONFIRMED:
        # Already fully proven out — the strongest possible validation.
        return HypothesisMaturity.EXECUTABLE

    contradicting = len(hypothesis.contradicting_evidence)
    supporting = len(hypothesis.supporting_evidence)
    age = hypothesis_age_seconds(hypothesis, now)
    currently_weakening = hypothesis.status == HypothesisStatus.WEAKENING

    if hypothesis.confidence < thresholds.building_confidence:
        return HypothesisMaturity.OBSERVING

    if currently_weakening or contradicting > supporting:
        # Actively degrading conviction can never be judged "mature"
        # enough to execute on, regardless of raw confidence level.
        return HypothesisMaturity.BUILDING

    if (
        hypothesis.confidence >= thresholds.executable_confidence
        and hypothesis.reinforcement_count >= thresholds.executable_min_reinforcements
        and age >= thresholds.executable_min_age_seconds
    ):
        return HypothesisMaturity.EXECUTABLE

    if (
        hypothesis.confidence >= thresholds.high_conviction_confidence
        and hypothesis.reinforcement_count >= thresholds.high_conviction_min_reinforcements
    ):
        return HypothesisMaturity.HIGH_CONVICTION

    return HypothesisMaturity.BUILDING


def maturity_rank(maturity: HypothesisMaturity) -> int:
    """Ordinal rank for comparing maturity progress (higher = more mature
    toward execution). Terminal states are ranked separately from the
    living progression since they are not "further along", just resolved.
    """
    order = {
        HypothesisMaturity.OBSERVING: 0,
        HypothesisMaturity.BUILDING: 1,
        HypothesisMaturity.HIGH_CONVICTION: 2,
        HypothesisMaturity.EXECUTABLE: 3,
        HypothesisMaturity.INVALIDATED: -1,
        HypothesisMaturity.EXPIRED: -1,
    }
    return order[maturity]
