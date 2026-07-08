"""Phase S4 — hindsight-optimal lifecycle state-transition labels.

Explicitly NOT the confidence/news-risk threshold rules used by
`trainer.lifecycle.transition_engine`/`trainer.lifecycle.position_monitor`
for live advisory recommendations. Per the S4 spec: "Train using future
outcomes." Given a realized `TradeOutcomePath`, this derives the sequence of
lifecycle-state transitions that, in hindsight, would have captured the
trade's actual best outcome — reusing `LifecycleState` (`trainer.lifecycle.
lifecycle_states`) as the label vocabulary rather than inventing a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trainer.lifecycle.lifecycle_states import LifecycleState
from trainer.training_signals.outcome_path import TradeOutcomePath

# Fraction of the eventual MFE at which it would have been safe, in
# hindsight, to move the position to a protected (breakeven-or-better)
# state. This is a labeling-shape constant, not a live trading rule.
_PROTECT_AT_MFE_FRACTION = 0.5
# Fraction of the eventual MFE that marks entry into the "let it run/trail"
# phase of the hindsight-optimal path.
_RUNNER_AT_MFE_FRACTION = 0.85


@dataclass
class LifecycleTransitionPoint:
    bar_index: int
    state: LifecycleState
    reason: str

    def to_dict(self) -> dict:
        return {"bar_index": self.bar_index, "state": self.state.value, "reason": self.reason}


@dataclass
class HindsightLifecycleLabel:
    instrument: str
    transitions: list[LifecycleTransitionPoint] = field(default_factory=list)
    realized_return_pct: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "transitions": [t.to_dict() for t in self.transitions],
            "realized_return_pct": round(self.realized_return_pct, 5),
            "rationale": self.rationale,
        }


class HindsightLifecycleLabeler:
    """Derives the hindsight-optimal lifecycle-state transition sequence for
    one realized trade outcome.

    The path always starts `ACTIVE` at bar 0. From there:

    - Once favorable excursion first reaches `_PROTECT_AT_MFE_FRACTION` of
      the eventual MFE, the label transitions to `PROTECTED` (in hindsight,
      the position was safe to move to breakeven-or-better from this point).
    - Once favorable excursion first reaches `_RUNNER_AT_MFE_FRACTION` of the
      eventual MFE, the label transitions to `RUNNER` then `TRAILING`
      (in hindsight, this was the "let it run" phase).
    - The path always ends at `CLOSED` on the final bar — `EXIT_READY`
      immediately precedes it, marking the bar before close.
    - If the trade never moved favorably at all (MFE == 0), the label skips
      straight from `ACTIVE` to `STOPPED` at the final bar, reflecting that
      in hindsight there was never a safe point to protect or run the trade.
    """

    def label(self, path: TradeOutcomePath) -> HindsightLifecycleLabel:
        realized_return_pct = path.realized_return_pct()

        if not path.future_bars:
            return HindsightLifecycleLabel(
                instrument=path.instrument,
                transitions=[
                    LifecycleTransitionPoint(0, LifecycleState.ACTIVE, "Trade opened."),
                ],
                realized_return_pct=realized_return_pct,
                rationale="No future bars available — only the entry transition can be labeled.",
            )

        transitions = [LifecycleTransitionPoint(0, LifecycleState.ACTIVE, "Trade opened.")]

        mfe, mfe_idx = path.max_favorable_excursion()
        last_idx = len(path.future_bars) - 1

        if mfe_idx < 0 or mfe <= 0:
            transitions.append(
                LifecycleTransitionPoint(
                    last_idx,
                    LifecycleState.STOPPED,
                    "Trade never moved favorably — no protect/runner phase was reachable.",
                )
            )
            rationale = (
                "Path never produced a favorable excursion; hindsight-optimal "
                "sequence is ACTIVE -> STOPPED with no intermediate states."
            )
            return HindsightLifecycleLabel(
                instrument=path.instrument,
                transitions=transitions,
                realized_return_pct=realized_return_pct,
                rationale=rationale,
            )

        protect_threshold = mfe * _PROTECT_AT_MFE_FRACTION
        runner_threshold = mfe * _RUNNER_AT_MFE_FRACTION

        protect_idx = self._first_index_reaching(path, protect_threshold)
        runner_idx = self._first_index_reaching(path, runner_threshold)

        if protect_idx is not None:
            transitions.append(
                LifecycleTransitionPoint(
                    protect_idx,
                    LifecycleState.PROTECTED,
                    f"Favorable excursion reached {_PROTECT_AT_MFE_FRACTION:.0%} of "
                    "eventual MFE — safe to protect from this bar onward.",
                )
            )

        if runner_idx is not None and runner_idx != protect_idx:
            transitions.append(
                LifecycleTransitionPoint(
                    runner_idx,
                    LifecycleState.RUNNER,
                    f"Favorable excursion reached {_RUNNER_AT_MFE_FRACTION:.0%} of "
                    "eventual MFE — hindsight-optimal 'let it run' phase begins.",
                )
            )
            if runner_idx != mfe_idx:
                transitions.append(
                    LifecycleTransitionPoint(
                        mfe_idx,
                        LifecycleState.TRAILING,
                        "Peak favorable excursion reached — trailing phase begins.",
                    )
                )

        if last_idx != transitions[-1].bar_index:
            transitions.append(
                LifecycleTransitionPoint(
                    last_idx,
                    LifecycleState.EXIT_READY,
                    "Final bar of the realized path — exit was due.",
                )
            )
        transitions.append(
            LifecycleTransitionPoint(last_idx, LifecycleState.CLOSED, "Trade closed.")
        )

        rationale = (
            f"MFE={mfe:.5f} at bar {mfe_idx}; hindsight-optimal sequence "
            f"protects at bar {protect_idx}, {'runs at bar ' + str(runner_idx) if runner_idx is not None else 'never reaches runner threshold'}, "
            f"closes at bar {last_idx}."
        )

        return HindsightLifecycleLabel(
            instrument=path.instrument,
            transitions=transitions,
            realized_return_pct=realized_return_pct,
            rationale=rationale,
        )

    @staticmethod
    def _first_index_reaching(path: TradeOutcomePath, threshold: float) -> int | None:
        for i, bar in enumerate(path.future_bars):
            if path.favorable_excursion(bar) >= threshold:
                return i
        return None
