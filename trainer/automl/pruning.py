"""Phase 16 — AutoML Early Pruning Strategies.

Terminates poor-performing experiments early to save compute resources.

Strategies:
  - MedianPruner           : prune if score < median of completed trials at same step
  - SuccessiveHalvingPruner: keep top 1/eta fraction at each rung
  - NoPruner               : never prune (baseline)

Usage::

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    decision = pruner.should_prune(trial_id="t-01", step=5, score=0.42,
                                   history=completed_trial_scores)
    if decision.prune:
        stop_trial()
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Pruning decision
# ---------------------------------------------------------------------------


@dataclass
class PruningDecision:
    """Result of a pruning evaluation.

    Attributes:
        prune: True if the trial should be stopped.
        trial_id: The trial being evaluated.
        step: Evaluation step at which decision was made.
        reason: Human-readable explanation.
        score: Current trial score.
        threshold: Score threshold that triggered the decision.
    """

    prune: bool
    trial_id: str
    step: int
    reason: str = ""
    score: float = 0.0
    threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prune": self.prune,
            "trial_id": self.trial_id,
            "step": self.step,
            "reason": self.reason,
            "score": self.score,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# Base pruner
# ---------------------------------------------------------------------------


class BasePruner:
    """Abstract pruner interface."""

    name: str = "base"

    def should_prune(
        self, trial_id: str, step: int, score: float, history: dict[str, list[tuple[int, float]]]
    ) -> PruningDecision:
        """Determine if trial should be pruned.

        Args:
            trial_id: ID of the trial being evaluated.
            step: Current training step (epoch, fold, etc.).
            score: Current intermediate score (higher = better).
            history: Mapping of trial_id → list of (step, score) tuples
                     for all *other* trials.

        Returns:
            PruningDecision with prune=True to stop, False to continue.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name}


# ---------------------------------------------------------------------------
# No-op pruner
# ---------------------------------------------------------------------------


class NoPruner(BasePruner):
    """Never prunes any trial."""

    name = "none"

    def should_prune(
        self, trial_id: str, step: int, score: float, history: dict[str, list[tuple[int, float]]]
    ) -> PruningDecision:
        return PruningDecision(
            prune=False, trial_id=trial_id, step=step, score=score, reason="pruning disabled"
        )


# ---------------------------------------------------------------------------
# Median pruner
# ---------------------------------------------------------------------------


class MedianPruner(BasePruner):
    """Prune if current score is below the median of completed trials at the same step.

    Args:
        n_startup_trials: Number of trials to complete before pruning begins.
        n_warmup_steps: Number of steps before pruning begins within a trial.
        interval_steps: Pruning only evaluated every N steps.
        percentile: Prune if below this percentile (50 = median).
    """

    name = "median"

    def __init__(
        self,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        percentile: float = 50.0,
    ) -> None:
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
        self.percentile = percentile

    def should_prune(
        self, trial_id: str, step: int, score: float, history: dict[str, list[tuple[int, float]]]
    ) -> PruningDecision:
        n_completed = len(history)

        if n_completed < self.n_startup_trials:
            return PruningDecision(
                prune=False,
                trial_id=trial_id,
                step=step,
                score=score,
                reason=f"startup: {n_completed}/{self.n_startup_trials} trials completed",
            )

        if step < self.n_warmup_steps:
            return PruningDecision(
                prune=False,
                trial_id=trial_id,
                step=step,
                score=score,
                reason=f"warmup: step {step}/{self.n_warmup_steps}",
            )

        if step % self.interval_steps != 0:
            return PruningDecision(
                prune=False,
                trial_id=trial_id,
                step=step,
                score=score,
                reason=f"not an evaluation step (interval={self.interval_steps})",
            )

        scores_at_step = []
        for tid, steps in history.items():
            if tid == trial_id:
                continue
            for s, v in steps:
                if s == step:
                    scores_at_step.append(v)
                    break
            else:
                closest = [v for s, v in steps if s <= step]
                if closest:
                    scores_at_step.append(closest[-1])

        if not scores_at_step:
            return PruningDecision(
                prune=False,
                trial_id=trial_id,
                step=step,
                score=score,
                reason="no reference scores at this step",
            )

        if self.percentile == 50.0:
            threshold = statistics.median(scores_at_step)
        else:
            sorted_scores = sorted(scores_at_step)
            idx = int(len(sorted_scores) * self.percentile / 100)
            threshold = sorted_scores[min(idx, len(sorted_scores) - 1)]

        should_stop = score < threshold
        return PruningDecision(
            prune=should_stop,
            trial_id=trial_id,
            step=step,
            score=score,
            threshold=threshold,
            reason=(
                f"score {score:.4f} below p{self.percentile:.0f}={threshold:.4f}"
                if should_stop
                else f"score {score:.4f} >= p{self.percentile:.0f}={threshold:.4f}"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "n_startup_trials": self.n_startup_trials,
            "n_warmup_steps": self.n_warmup_steps,
            "interval_steps": self.interval_steps,
            "percentile": self.percentile,
        }


# ---------------------------------------------------------------------------
# Successive halving pruner
# ---------------------------------------------------------------------------


class SuccessiveHalvingPruner(BasePruner):
    """Keep the top 1/eta fraction at each rung level.

    Implements Asynchronous Successive Halving (ASHA) pruning.

    Args:
        min_resource: Minimum resource before first rung evaluation.
        reduction_factor: Elimination factor eta (default 3 → keep top 1/3).
        min_early_stopping_rate: Number of rungs to skip before pruning starts.
    """

    name = "successive_halving"

    def __init__(
        self, min_resource: int = 1, reduction_factor: int = 3, min_early_stopping_rate: int = 0
    ) -> None:
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.min_early_stopping_rate = min_early_stopping_rate

    def _rung_levels(self, max_step: int) -> list[int]:
        """Return rung resource levels from min_resource to max_step."""
        rungs = []
        r = self.min_resource
        while r <= max_step:
            rungs.append(r)
            r = int(r * self.reduction_factor)
        return rungs

    def should_prune(
        self, trial_id: str, step: int, score: float, history: dict[str, list[tuple[int, float]]]
    ) -> PruningDecision:
        if step < self.min_resource:
            return PruningDecision(
                prune=False,
                trial_id=trial_id,
                step=step,
                score=score,
                reason=f"below min_resource ({self.min_resource})",
            )

        max_step = max((s for steps in history.values() for s, _ in steps), default=step)
        rungs = self._rung_levels(max(step, max_step))

        rung = None
        for r in sorted(rungs, reverse=True):
            if step >= r:
                rung = r
                break

        if rung is None:
            return PruningDecision(
                prune=False,
                trial_id=trial_id,
                step=step,
                score=score,
                reason="not at a rung level",
            )

        rung_idx = rungs.index(rung)
        if rung_idx < self.min_early_stopping_rate:
            return PruningDecision(
                prune=False,
                trial_id=trial_id,
                step=step,
                score=score,
                reason=f"rung {rung_idx} < min_early_stopping_rate {self.min_early_stopping_rate}",
            )

        scores_at_rung = []
        for tid, steps in history.items():
            if tid == trial_id:
                continue
            for s, v in steps:
                if s >= rung:
                    scores_at_rung.append(v)
                    break

        if not scores_at_rung:
            return PruningDecision(
                prune=False,
                trial_id=trial_id,
                step=step,
                score=score,
                reason="no peers at this rung",
            )

        n_keep = max(1, len(scores_at_rung) // self.reduction_factor)
        threshold = sorted(scores_at_rung, reverse=True)[n_keep - 1]
        should_stop = score < threshold

        return PruningDecision(
            prune=should_stop,
            trial_id=trial_id,
            step=step,
            score=score,
            threshold=threshold,
            reason=(
                f"rung {rung}: score {score:.4f} < threshold {threshold:.4f} "
                f"(keep top {n_keep}/{len(scores_at_rung)})"
                if should_stop
                else f"rung {rung}: score {score:.4f} >= threshold {threshold:.4f}"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "min_resource": self.min_resource,
            "reduction_factor": self.reduction_factor,
            "min_early_stopping_rate": self.min_early_stopping_rate,
        }


# ---------------------------------------------------------------------------
# Pruner registry
# ---------------------------------------------------------------------------


def build_pruner(config: dict[str, Any]) -> BasePruner:
    """Instantiate a pruner from a config dict.

    Config keys:
        type: "none" | "median" | "successive_halving"
        ...any additional constructor kwargs
    """
    ptype = config.get("type", "none")
    if ptype == "none":
        return NoPruner()
    elif ptype == "median":
        return MedianPruner(
            n_startup_trials=config.get("n_startup_trials", 5),
            n_warmup_steps=config.get("n_warmup_steps", 0),
            interval_steps=config.get("interval_steps", 1),
            percentile=config.get("percentile", 50.0),
        )
    elif ptype == "successive_halving":
        return SuccessiveHalvingPruner(
            min_resource=config.get("min_resource", 1),
            reduction_factor=config.get("reduction_factor", 3),
            min_early_stopping_rate=config.get("min_early_stopping_rate", 0),
        )
    raise ValueError(f"Unknown pruner type: {ptype!r}")
