"""QDE Brain — hypothesis probability engine.

Converts raw ``Hypothesis.confidence`` scores across a ``HypothesisSet``
into a normalized posterior probability distribution, and performs
Bayesian-style updates as new evidence arrives.

Why "Bayesian-style"?
----------------------
A full Bayesian update would require per-hypothesis likelihood functions,
which is impractical without a training signal.  Instead we use a
*log-space multiplicative update*:

    log_posterior(i) = log_prior(i) + dampening × evidence_strength(i)
    posterior(i)     = softmax(log_posterior)

This gives the key Bayesian properties we care about:

  * Strong positive evidence raises a scenario's probability while
    automatically lowering competing scenarios' probabilities.
  * Strong negative evidence does the reverse.
  * The distribution always sums to 1.
  * The prior (derived from confidence) is never ignored — small
    evidence updates move the distribution gradually.
  * A scenario with prior probability 0 (invalidated) stays at 0.

The ``dampening`` parameter (default 0.4) controls how aggressively
evidence moves the distribution.  Low dampening (0.1) = cautious;
high dampening (0.8) = aggressive.

Initial distribution
--------------------
``compute_initial`` converts ``Hypothesis.confidence`` values to initial
probabilities via epsilon-smoothed normalization:

    weight(i) = max(epsilon, confidence(i))
    prior(i)  = weight(i) / sum(weight)

An epsilon of 0.02 prevents any active scenario from having zero prior
probability before any evidence is seen (consistent with C3
NarrativeProbabilityEngine's approach).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trainer.qde.brain.hypothesis.hypothesis_set import HypothesisScenario, HypothesisSet

_EPSILON         = 0.02     # prior floor for active scenarios
_DEFAULT_DAMPEN  = 0.40     # log-space evidence dampening
_MAX_ENTROPY_2   = math.log(2)


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbabilityUpdateResult:
    """Output of one ``HypothesisProbabilityEngine`` update call.

    ``new_probs``    — new posterior: scenario_id → probability.
    ``prior_probs``  — the probabilities before this update.
    ``entropy``      — Shannon entropy of the new distribution (nats).
    ``certainty``    — 1 - normalized entropy.  0 = uniform, 1 = fully certain.
    ``dominant_id``  — scenario_id with the highest posterior (or None if all equal).
    ``shift``        — total probability mass moved = Σ|new - prior| / 2.
    """

    new_probs:    dict[str, float]
    prior_probs:  dict[str, float]
    entropy:      float
    certainty:    float
    dominant_id:  str | None
    shift:        float

    def to_dict(self) -> dict:
        return {
            "new_probs":   {k: round(v, 4) for k, v in self.new_probs.items()},
            "entropy":     round(self.entropy, 4),
            "certainty":   round(self.certainty, 4),
            "dominant_id": self.dominant_id,
            "shift":       round(self.shift, 4),
        }


# ── Engine ────────────────────────────────────────────────────────────────────


class HypothesisProbabilityEngine:
    """Stateless probability distribution engine over a ``HypothesisSet``.

    All methods are pure functions of their inputs; no state is held.

    Usage::

        engine = HypothesisProbabilityEngine()

        # On initial generation:
        probs = engine.compute_initial(hypothesis_set)
        hypothesis_set._apply_probabilities(probs, trigger="INIT")

        # On new evidence:
        evidence = {"A": +0.6, "B": -0.3}  # Scenario A confirmed, B weakened
        result = engine.bayesian_update(hypothesis_set, evidence)
        hypothesis_set._apply_probabilities(result.new_probs, trigger="EVIDENCE")
    """

    def __init__(
        self,
        epsilon:   float = _EPSILON,
        dampening: float = _DEFAULT_DAMPEN,
    ) -> None:
        self._eps    = epsilon
        self._dampen = dampening

    # ── Primary entry points ──────────────────────────────────────────────

    def compute_initial(self, hypothesis_set: HypothesisSet) -> dict[str, float]:
        """Compute the initial probability distribution from ``Hypothesis.confidence``.

        Only active scenarios receive probability mass; invalidated scenarios
        get 0.  Returns a normalized dict: scenario_id → probability.
        """
        active = [s for s in hypothesis_set.scenarios if s.is_active]
        if not active:
            return {}

        weights = {s.scenario_id: max(self._eps, s.hypothesis.confidence) for s in active}
        return self._normalize(weights)

    def bayesian_update(
        self,
        hypothesis_set:  HypothesisSet,
        evidence:        dict[str, float],
        dampening:       float | None = None,
    ) -> ProbabilityUpdateResult:
        """Apply a Bayesian-style update to the current distribution.

        Parameters
        ----------
        hypothesis_set:
            The set to update.  Invalidated scenarios are automatically
            excluded (probability forced to 0).
        evidence:
            Dict mapping scenario_id → evidence strength.
            Positive values (0–1) support the scenario.
            Negative values (−1–0) contradict it.
            Missing scenario_ids are treated as no-evidence (0).
        dampening:
            Override the engine's default dampening coefficient.
            Higher = more aggressive update.  Range: 0.05–1.0.

        Returns
        -------
        ProbabilityUpdateResult
            Contains new probabilities but does NOT apply them to the set.
            Call ``hypothesis_set._apply_probabilities(result.new_probs, ...)``
            to commit.
        """
        d = dampening if dampening is not None else self._dampen

        active = [s for s in hypothesis_set.scenarios if s.is_active]
        if not active:
            return ProbabilityUpdateResult(
                new_probs={}, prior_probs={}, entropy=0.0,
                certainty=1.0, dominant_id=None, shift=0.0,
            )

        # Current probabilities as the prior
        prior: dict[str, float] = {}
        for s in active:
            prior[s.scenario_id] = max(self._eps, s.probability if s.probability > 0 else self._eps)

        # Log-space update
        log_posterior: dict[str, float] = {}
        for s in active:
            evid = evidence.get(s.scenario_id, 0.0)
            clamped = max(-1.0, min(1.0, evid))
            log_posterior[s.scenario_id] = math.log(prior[s.scenario_id]) + d * clamped

        new_probs = self._softmax(log_posterior)

        # Add back invalidated scenarios with prob 0
        for s in hypothesis_set.scenarios:
            if not s.is_active:
                new_probs[s.scenario_id] = 0.0

        entropy  = self._entropy(new_probs)
        n_active = max(2, len(active))
        max_ent  = math.log(n_active)
        certainty = max(0.0, min(1.0, 1.0 - entropy / max_ent)) if max_ent > 0 else 1.0

        dominant_id = max(new_probs, key=lambda k: new_probs[k]) if new_probs else None
        shift = sum(abs(new_probs.get(k, 0.0) - prior.get(k, 0.0)) for k in new_probs) / 2.0

        return ProbabilityUpdateResult(
            new_probs=new_probs,
            prior_probs={s.scenario_id: s.probability for s in hypothesis_set.scenarios},
            entropy=round(entropy, 4),
            certainty=round(certainty, 4),
            dominant_id=dominant_id,
            shift=round(shift, 4),
        )

    def from_confidence_update(
        self,
        hypothesis_set: HypothesisSet,
    ) -> dict[str, float]:
        """Re-derive probabilities purely from current ``Hypothesis.confidence``
        values, ignoring the existing distribution.

        Use when a hypothesis's confidence has been externally mutated (e.g.
        after ``HypothesisEngine.reinforce`` or ``weaken``) and you want the
        probability distribution to reflect the new balance of confidences.
        """
        return self.compute_initial(hypothesis_set)

    # ── Helpers ───────────────────────────────────────────────────────────

    def dominant_scenario(self, hypothesis_set: HypothesisSet) -> HypothesisScenario | None:
        """Active scenario with the highest probability."""
        active = [s for s in hypothesis_set.scenarios if s.is_active]
        if not active:
            return None
        best = max(active, key=lambda s: s.probability)
        # Only return as dominant if clearly above a flat distribution
        threshold = 1.0 / max(1, len(active)) + 0.05
        return best if best.probability > threshold else None

    def entropy_of(self, hypothesis_set: HypothesisSet) -> float:
        probs = [s.probability for s in hypothesis_set.scenarios
                 if s.is_active and s.probability > 0]
        return self._entropy({str(i): p for i, p in enumerate(probs)})

    def certainty_of(self, hypothesis_set: HypothesisSet) -> float:
        n = max(2, sum(1 for s in hypothesis_set.scenarios if s.is_active))
        e = self.entropy_of(hypothesis_set)
        max_ent = math.log(n)
        return max(0.0, min(1.0, 1.0 - e / max_ent)) if max_ent > 0 else 1.0

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total < 1e-12:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}

    @staticmethod
    def _softmax(log_scores: dict[str, float]) -> dict[str, float]:
        """Numerically-stable softmax in log-space."""
        if not log_scores:
            return {}
        max_log = max(log_scores.values())
        exps = {k: math.exp(v - max_log) for k, v in log_scores.items()}
        total = sum(exps.values())
        if total < 1e-12:
            n = len(exps)
            return {k: 1.0 / n for k in exps}
        return {k: v / total for k, v in exps.items()}

    @staticmethod
    def _entropy(probs: dict[str, float]) -> float:
        return -sum(p * math.log(max(p, 1e-15)) for p in probs.values() if p > 0)


__all__ = [
    "HypothesisProbabilityEngine",
    "ProbabilityUpdateResult",
]
