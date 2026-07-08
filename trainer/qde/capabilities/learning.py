"""Phase 28.3 — Capability 11: Learning.

Consumes ReflectionInsights to update internal priors, capability trust
scores, and regime performance beliefs. Produces LearningUpdate objects
that the QDE can apply to its weight registry.

CONTRACT
--------
- Returns LearningUpdate (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trainer.qde.capabilities.cap_models import (
    LearningUpdate,
    ReflectionInsight,
)


@dataclass
class CapabilityPrior:
    """Bayesian-style prior for one capability's trust score."""

    capability: str
    trust_score: float = 0.70  # 0-1
    n_samples: int = 0
    cumulative_accuracy: float = 0.0
    learning_rate: float = 0.05

    def update(self, accuracy: float) -> float:
        """Online exponential moving average update."""
        self.n_samples += 1
        self.cumulative_accuracy += accuracy
        # EMA update: new = old * (1-lr) + accuracy * lr
        self.trust_score = (
            self.trust_score * (1.0 - self.learning_rate) + accuracy * self.learning_rate
        )
        self.trust_score = max(0.05, min(0.99, self.trust_score))
        return self.trust_score


class LearningCapability:
    """Maintains and updates priors from reflection insights.

    This capability is the final stage of the QDE improvement loop:
    Reflection → Learning → Updated priors → Next reasoning cycle.

    It operates post-trade. It does not participate in live reasoning.
    All outputs are weight recommendations — the QDE decides how to apply them.
    """

    CAPABILITY_NAME = "learning"

    # All 11 capabilities start with neutral trust
    DEFAULT_TRUST = 0.70
    LEARNING_RATE = 0.05

    def __init__(self) -> None:
        self._priors: dict[str, CapabilityPrior] = {}
        self._regime_priors: dict[str, dict[str, list[float]]] = {}
        self._insights_processed: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def learn(
        self,
        insight: ReflectionInsight,
        snapshot_id: str = "",
    ) -> LearningUpdate:
        """Apply a reflection insight and return updated weights.

        Args:
            insight: Output from the Reflection capability.
            snapshot_id: Snapshot reference for this learning cycle.

        Returns:
            LearningUpdate — advisory only, not a trade instruction.
        """
        old_scores = {k: v.trust_score for k, v in self._priors.items()}
        updated_weights: dict[str, float] = {}

        # Update each capability's trust score
        for cap, accuracy in insight.capability_accuracy.items():
            prior = self._get_or_create_prior(cap)
            adjustment = insight.suggested_weight_adjustments.get(cap, 0.0)
            effective_accuracy = max(0.0, min(1.0, accuracy + adjustment))
            new_score = prior.update(effective_accuracy)
            updated_weights[cap] = new_score

        # Track convergence delta
        new_scores = {k: v.trust_score for k, v in self._priors.items()}
        delta = self._convergence_delta(old_scores, new_scores)

        # Update regime priors
        regime_key = insight.metadata.get("regime", "") if hasattr(insight, "metadata") else ""
        regime_updates = self._update_regime_priors(regime_key, insight)

        self._insights_processed += 1
        current_trust = {k: round(v.trust_score, 4) for k, v in self._priors.items()}
        n_used = sum(p.n_samples for p in self._priors.values())

        return LearningUpdate(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot_id or f"learn-{self._insights_processed}",
            confidence=0.90,
            updated_weights=updated_weights,
            prior_updates=regime_updates,
            capability_trust_scores=current_trust,
            regime_performance_priors=self._export_regime_priors(),
            learning_rate_used=self.LEARNING_RATE,
            n_samples_used=n_used,
            convergence_delta=delta,
            notes=(
                f"Processed insight {self._insights_processed}. "
                f"Convergence delta={delta:.6f}. "
                f"{'Reasoning correct.' if insight.reasoning_was_correct else 'Reasoning incorrect — updating priors down.'}"
            ),
        )

    def get_trust_scores(self) -> dict[str, float]:
        """Return the current capability trust scores."""
        return {k: round(v.trust_score, 4) for k, v in self._priors.items()}

    def get_regime_priors(self) -> dict[str, dict[str, float]]:
        """Return regime performance summary."""
        return self._export_regime_priors()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_or_create_prior(self, capability: str) -> CapabilityPrior:
        if capability not in self._priors:
            self._priors[capability] = CapabilityPrior(
                capability=capability,
                trust_score=self.DEFAULT_TRUST,
                learning_rate=self.LEARNING_RATE,
            )
        return self._priors[capability]

    def _convergence_delta(
        self,
        old: dict[str, float],
        new: dict[str, float],
    ) -> float:
        all_keys = set(old) | set(new)
        if not all_keys:
            return 0.0
        sq_sum = sum((new.get(k, 0.0) - old.get(k, 0.0)) ** 2 for k in all_keys)
        return math.sqrt(sq_sum / len(all_keys))

    def _update_regime_priors(
        self,
        regime: str,
        insight: ReflectionInsight,
    ) -> dict[str, float]:
        if not regime:
            return {}
        if regime not in self._regime_priors:
            self._regime_priors[regime] = {"pnl_r": [], "accuracy": []}
        # Store recent samples
        samples = self._regime_priors[regime]
        actual_accuracy = 1.0 if insight.reasoning_was_correct else 0.0
        samples["accuracy"].append(actual_accuracy)
        if len(samples["accuracy"]) > 100:
            samples["accuracy"].pop(0)
        n = len(samples["accuracy"])
        return {
            "regime": regime,
            "n_samples": float(n),
            "avg_accuracy": sum(samples["accuracy"]) / n if n else 0.0,
        }

    def _export_regime_priors(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for regime, samples in self._regime_priors.items():
            acc = samples.get("accuracy", [])
            n = len(acc)
            result[regime] = {
                "n_samples": float(n),
                "avg_accuracy": round(sum(acc) / n, 4) if n else 0.0,
            }
        return result
