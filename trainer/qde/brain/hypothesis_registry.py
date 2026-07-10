"""QDE Brain — cross-symbol hypothesis registry.

`HypothesisRegistry` is a thin, read-oriented query facade over the
hypotheses `QDEBrain` already stores per symbol (`BrainState.hypotheses`,
owned/mutated by `hypotheses.HypothesisTracker`). It introduces no new
storage — hypotheses continue to live exactly where they always have —
it only adds the cross-symbol lookups and maturity classification needed
to answer questions like "which hypotheses across all tracked symbols
are executable right now?", which `BrainState` alone cannot answer
without iterating every symbol by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.hypothesis_lifecycle import (
    DEFAULT_MATURITY_THRESHOLDS,
    HypothesisMaturity,
    MaturityThresholds,
    compute_maturity,
    hypothesis_age_seconds,
)

if TYPE_CHECKING:
    from trainer.qde.brain.brain import QDEBrain


@dataclass(frozen=True)
class HypothesisView:
    """A hypothesis paired with its projected execution maturity — the
    unit the registry hands back, so callers never need to re-derive
    maturity themselves.
    """

    instrument: str
    hypothesis: Hypothesis
    maturity: HypothesisMaturity
    age_seconds: float

    def to_dict(self) -> dict:
        d = self.hypothesis.to_dict()
        d["maturity"] = self.maturity.value
        d["age_seconds"] = round(self.age_seconds, 3)
        return d


class HypothesisRegistry:
    """Queries over every hypothesis `QDEBrain` currently knows about."""

    def __init__(
        self,
        brain: "QDEBrain",
        thresholds: MaturityThresholds = DEFAULT_MATURITY_THRESHOLDS,
    ) -> None:
        self._brain = brain
        self._thresholds = thresholds

    def _views(self, now: float | None = None) -> list[HypothesisView]:
        import time as _time

        now = now if now is not None else _time.time()
        views: list[HypothesisView] = []
        for instrument, state in self._brain.all_states().items():
            for hyp in state.hypotheses.values():
                views.append(
                    HypothesisView(
                        instrument=instrument,
                        hypothesis=hyp,
                        maturity=compute_maturity(hyp, now=now, thresholds=self._thresholds),
                        age_seconds=hypothesis_age_seconds(hyp, now=now),
                    )
                )
        return views

    def get(self, hypothesis_id: str) -> HypothesisView | None:
        for view in self._views():
            if view.hypothesis.hypothesis_id == hypothesis_id:
                return view
        return None

    def all(self) -> list[HypothesisView]:
        return self._views()

    def for_symbol(self, instrument: str) -> list[HypothesisView]:
        return [v for v in self._views() if v.instrument == instrument]

    def by_maturity(self, maturity: HypothesisMaturity) -> list[HypothesisView]:
        return [v for v in self._views() if v.maturity == maturity]

    def executable(self, instrument: str | None = None) -> list[HypothesisView]:
        """Hypotheses mature enough that a trade may emerge from them,
        most-confident first. This is the intended integration point for
        an execution/decision layer that wants trade signals to come from
        sustained conviction rather than an isolated per-tick prediction.
        """
        views = self.by_maturity(HypothesisMaturity.EXECUTABLE)
        if instrument is not None:
            views = [v for v in views if v.instrument == instrument]
        return sorted(views, key=lambda v: v.hypothesis.confidence, reverse=True)

    def active(self, instrument: str | None = None) -> list[HypothesisView]:
        from trainer.qde.brain.hypothesis_lifecycle import is_terminal_maturity

        views = [v for v in self._views() if not is_terminal_maturity(v.maturity)]
        if instrument is not None:
            views = [v for v in views if v.instrument == instrument]
        return views
