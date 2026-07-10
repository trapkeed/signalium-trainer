"""QDE Brain — hypothesis engine.

`HypothesisEngine` is the single orchestration facade for the five
required hypothesis operations (create / reinforce / weaken / invalidate
/ retire). It does not reimplement hypothesis storage or narrative
mutation logic — those already exist and are reused as-is:

  * storage         -> `BrainState.hypotheses` (via `QDEBrain.get_state`)
  * create/reinforce/
    weaken/invalidate -> `hypotheses.HypothesisTracker` (unchanged)
  * maturity         -> `hypothesis_lifecycle.compute_maturity` (pure projection)
  * cross-symbol lookup -> `hypothesis_registry.HypothesisRegistry`

What this module *adds* that did not exist before:
  * `retire()` — an explicit, human/engine-initiated close-out of a
    hypothesis that never resolved, distinct from `age()`'s time-based
    expiry (both land on `HypothesisStatus.EXPIRED`, but retirement
    records *why* someone chose to stop tracking it).
  * Maturity-milestone → long-term-memory bridging: crossing into
    HIGH_CONVICTION, EXECUTABLE, INVALIDATED, or EXPIRED for the first
    time is recorded as an `ExperienceRecord` via the existing
    `QDEBrain.record_experience`, so the personality/learning loop
    (`long_term_memory.py`) accumulates trade-maturity outcomes the same
    way it already accumulates any other experience — no parallel
    learning path.
  * `executable_hypotheses()` / `mature_direction()` — the concrete
    integration point for "trade execution should emerge from mature
    hypotheses rather than isolated predictions": a decision layer
    (e.g. `qde_inference.py`) can consult these before treating a raw
    per-tick signal as tradeable, without this module ever placing an
    order itself (that boundary stays exactly where it already is, at
    `qde_inference.derive_decision_fields`).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.brain_models import BrainState, Hypothesis
from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.hypotheses import HypothesisTracker
from trainer.qde.brain.hypothesis_lifecycle import (
    DEFAULT_MATURITY_THRESHOLDS,
    HypothesisMaturity,
    MaturityThresholds,
    compute_maturity,
)
from trainer.qde.brain.hypothesis_registry import HypothesisRegistry, HypothesisView
from trainer.qde.brain.working_memory import WorkingMemory

if TYPE_CHECKING:
    from trainer.qde.brain.brain import QDEBrain

# Milestones worth surfacing into long-term memory — reaching (or first
# crossing into) any of these is a meaningful, learnable event; the living
# progression through OBSERVING/BUILDING is too noisy to log every tick.
_MILESTONE_MATURITIES = frozenset(
    {
        HypothesisMaturity.HIGH_CONVICTION,
        HypothesisMaturity.EXECUTABLE,
        HypothesisMaturity.INVALIDATED,
        HypothesisMaturity.EXPIRED,
    }
)


class HypothesisEngine:
    """Create, reinforce, weaken, invalidate, and retire hypotheses; track
    which ones have matured enough for execution to emerge from.
    """

    def __init__(
        self,
        brain: "QDEBrain",
        thresholds: MaturityThresholds = DEFAULT_MATURITY_THRESHOLDS,
    ) -> None:
        self._brain = brain
        self._tracker = HypothesisTracker()
        self._thresholds = thresholds
        self.registry = HypothesisRegistry(brain, thresholds=thresholds)
        self._working_memory = WorkingMemory()
        # Last-known maturity per hypothesis, so milestone crossings can be
        # detected (and logged to long-term memory) exactly once each.
        self._last_maturity: dict[str, HypothesisMaturity] = {}

    # ── Core operations ──────────────────────────────────────────────────

    def create(
        self,
        symbol: str,
        thesis: str,
        direction: BeliefDirection = BeliefDirection.NEUTRAL,
        confidence: float = 0.3,
        evidence: str | None = None,
        now: float | None = None,
    ) -> Hypothesis:
        """Create a hypothesis for `symbol` (or reinforce an equivalent
        already-active one — see `HypothesisTracker.form`, dedup by
        thesis text is intentional and not duplicated here).
        """
        now = now if now is not None else time.time()
        state = self._brain.get_state(symbol)
        hyp = self._tracker.form(state.hypotheses, symbol, thesis, direction, confidence, now=now)
        if evidence:
            self._tracker.reinforce(state.hypotheses, hyp.hypothesis_id, evidence)
        self._log_memory(symbol, hyp, "HYPOTHESIS_CREATED", now)
        self._check_milestone(symbol, hyp, now)
        self._persist(symbol, state)
        return hyp

    def reinforce(self, hypothesis_id: str, evidence: str) -> Hypothesis | None:
        located = self._locate(hypothesis_id)
        if located is None:
            return None
        symbol, state = located
        now = time.time()
        hyp = self._tracker.reinforce(state.hypotheses, hypothesis_id, evidence)
        if hyp is None:
            return None
        self._log_memory(symbol, hyp, "HYPOTHESIS_REINFORCED", now)
        self._check_milestone(symbol, hyp, now)
        self._persist(symbol, state)
        return hyp

    def weaken(self, hypothesis_id: str, evidence: str) -> Hypothesis | None:
        located = self._locate(hypothesis_id)
        if located is None:
            return None
        symbol, state = located
        now = time.time()
        hyp = self._tracker.weaken(state.hypotheses, hypothesis_id, evidence)
        if hyp is None:
            return None
        self._log_memory(symbol, hyp, "HYPOTHESIS_WEAKENED", now)
        self._check_milestone(symbol, hyp, now)
        self._persist(symbol, state)
        return hyp

    def invalidate(self, hypothesis_id: str, reason: str) -> Hypothesis | None:
        located = self._locate(hypothesis_id)
        if located is None:
            return None
        symbol, state = located
        now = time.time()
        hyp = self._tracker.invalidate(state.hypotheses, hypothesis_id, reason)
        if hyp is None:
            return None
        self._log_memory(symbol, hyp, "HYPOTHESIS_INVALIDATED", now)
        self._check_milestone(symbol, hyp, now)
        self._persist(symbol, state)
        return hyp

    def retire(self, hypothesis_id: str, reason: str) -> Hypothesis | None:
        """Explicitly stop tracking a hypothesis that never resolved
        either way (not disproven, just no longer relevant — e.g. thesis
        superseded, symbol delisted, regime context no longer applies).

        Distinct from `age()`'s automatic staleness expiry: this is an
        intentional close-out, recorded as such in `invalidation_reason`
        (reusing that field rather than adding a parallel one — a retired
        hypothesis is, mechanically, an expired one with a known cause).
        """
        located = self._locate(hypothesis_id)
        if located is None:
            return None
        symbol, state = located
        hyp = state.hypotheses.get(hypothesis_id)
        if hyp is None or hyp.is_terminal():
            return hyp
        now = time.time()
        from trainer.qde.brain.brain_types import HypothesisStatus

        hyp.status = HypothesisStatus.EXPIRED
        hyp.invalidation_reason = f"RETIRED: {reason}"
        hyp.last_updated_at = now
        self._log_memory(symbol, hyp, "HYPOTHESIS_RETIRED", now)
        self._check_milestone(symbol, hyp, now)
        self._persist(symbol, state)
        return hyp

    def age_all(self, now: float | None = None) -> int:
        """Expire stale hypotheses across every tracked symbol. Delegates
        entirely to `HypothesisTracker.age` per symbol — no reimplemented
        staleness logic.
        """
        now = now if now is not None else time.time()
        expired = 0
        for symbol, state in self._brain.all_states().items():
            n = self._tracker.age(state.hypotheses, now=now)
            if n:
                expired += n
                for hyp in state.hypotheses.values():
                    self._check_milestone(symbol, hyp, now)
                self._persist(symbol, state)
        return expired

    # ── Execution integration ────────────────────────────────────────────

    def maturity_of(self, hypothesis_id: str) -> HypothesisMaturity | None:
        located = self._locate(hypothesis_id)
        if located is None:
            return None
        _, state = located
        hyp = state.hypotheses.get(hypothesis_id)
        if hyp is None:
            return None
        return compute_maturity(hyp, thresholds=self._thresholds)

    def executable_hypotheses(self, symbol: str | None = None) -> list[HypothesisView]:
        """Hypotheses mature enough that a trade may emerge from them.
        Intended to be consulted by the decision/inference layer instead
        of acting on an isolated per-tick prediction alone.
        """
        return self.registry.executable(symbol)

    def mature_direction(self, symbol: str) -> BeliefDirection | None:
        """The direction implied by `symbol`'s most confident executable
        hypothesis, or `None` if nothing has matured to EXECUTABLE yet —
        i.e. there is currently no hypothesis-backed basis to trade from.
        """
        views = self.executable_hypotheses(symbol)
        if not views:
            return None
        return views[0].hypothesis.direction

    # ── Internals ─────────────────────────────────────────────────────────

    def _locate(self, hypothesis_id: str) -> tuple[str, BrainState] | None:
        for symbol, state in self._brain.all_states().items():
            if hypothesis_id in state.hypotheses:
                return symbol, state
        return None

    def _check_milestone(self, symbol: str, hyp: Hypothesis, now: float) -> None:
        maturity = compute_maturity(hyp, now=now, thresholds=self._thresholds)
        prior = self._last_maturity.get(hyp.hypothesis_id)
        self._last_maturity[hyp.hypothesis_id] = maturity
        if maturity == prior or maturity not in _MILESTONE_MATURITIES:
            return

        outcome_quality = {
            HypothesisMaturity.HIGH_CONVICTION: 0.3,
            HypothesisMaturity.EXECUTABLE: 0.6,
            HypothesisMaturity.INVALIDATED: -0.5,
            HypothesisMaturity.EXPIRED: -0.1,
        }[maturity]

        self._brain.record_experience(
            instrument=symbol,
            kind=f"HYPOTHESIS_{maturity.value}",
            description=(
                f"Hypothesis '{hyp.thesis}' reached {maturity.value} "
                f"(confidence {hyp.confidence:.0%}, "
                f"{hyp.reinforcement_count} reinforcements)."
            ),
            outcome_quality=outcome_quality,
            regime="",
            tags=["hypothesis_maturity", maturity.value.lower()],
        )

    def _log_memory(self, symbol: str, hyp: Hypothesis, kind: str, now: float) -> None:
        state = self._brain.get_state(symbol)
        self._working_memory.add(
            state.working_memory,
            instrument=symbol,
            kind="HYPOTHESIS_UPDATE",
            content={
                "hypothesis_id": hyp.hypothesis_id,
                "event": kind,
                "thesis": hyp.thesis,
                "status": hyp.status.value,
                "confidence": round(hyp.confidence, 4),
            },
            salience=min(1.0, 0.4 + hyp.confidence * 0.4),
            now=now,
        )

    def _persist(self, symbol: str, state: BrainState) -> None:
        db = getattr(self._brain, "_db", None)
        if db is not None:
            db.upsert_state(state.to_dict())
