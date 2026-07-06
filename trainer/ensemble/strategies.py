"""Phase 17 — AEIP Ensemble Combination Strategies.

Implements all ensemble combination strategies:

  - SimpleAveragingStrategy     : unweighted mean of probabilities
  - WeightedAveragingStrategy   : weight-adjusted mean
  - PerformanceWeightedStrategy : weights from recent expert performance
  - StackingStrategy            : learned linear combiner (meta-features)
  - BlendingStrategy            : holdout-based blending
  - VotingStrategy              : hard or soft majority vote
  - MixtureOfExpertsStrategy    : dynamic expert gating
  - MetaLearningStrategy        : defers to the EnsembleMetaModel

All strategies share the same interface::

    strategy = WeightedAveragingStrategy()
    result = strategy.combine(predictions, weights)

Usage::

    from trainer.ensemble.strategies import build_strategy
    strategy = build_strategy("soft_voting")
    result = strategy.combine(predictions, weights)
    print(result.buy_prob, result.confidence)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from trainer.ensemble.expert import ExpertPrediction

# ---------------------------------------------------------------------------
# Ensemble result
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    """Combined output of the ensemble.

    Attributes:
        buy_prob: Final BUY probability [0, 1].
        sell_prob: Final SELL probability [0, 1].
        no_trade_prob: Final NO TRADE probability [0, 1].
        confidence: Ensemble-level confidence [0, 1].
        uncertainty: Ensemble-level uncertainty [0, 1].
        strategy: Name of the combination strategy used.
        n_experts: Number of experts in the ensemble.
        contributing_experts: List of expert_ids that contributed.
        weight_map: Dict of expert_id → applied weight.
        agreement_score: Inter-expert agreement [0, 1] (1 = unanimous).
        metadata: Strategy-specific metadata.
    """

    buy_prob: float = 0.33
    sell_prob: float = 0.33
    no_trade_prob: float = 0.34
    confidence: float = 0.5
    uncertainty: float = 0.5
    strategy: str = ""
    n_experts: int = 0
    contributing_experts: list[str] = field(default_factory=list)
    weight_map: dict[str, float] = field(default_factory=dict)
    agreement_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.buy_prob + self.sell_prob + self.no_trade_prob
        if total > 0 and abs(total - 1.0) > 1e-4:
            self.buy_prob /= total
            self.sell_prob /= total
            self.no_trade_prob /= total

    @property
    def predicted_class(self) -> str:
        probs = {"buy": self.buy_prob, "sell": self.sell_prob, "no_trade": self.no_trade_prob}
        return max(probs, key=probs.get)

    @property
    def institutional_forecast(self) -> str:
        """Human-readable institutional forecast string."""
        cls = self.predicted_class
        conf = self.confidence
        if conf >= 0.80:
            strength = "HIGH CONFIDENCE"
        elif conf >= 0.60:
            strength = "MODERATE CONFIDENCE"
        else:
            strength = "LOW CONFIDENCE"
        label = {"buy": "LONG", "sell": "SHORT", "no_trade": "NEUTRAL"}[cls]
        return f"{strength} {label} | conf={conf:.2f} agree={self.agreement_score:.2f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "buy_prob": self.buy_prob,
            "sell_prob": self.sell_prob,
            "no_trade_prob": self.no_trade_prob,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "predicted_class": self.predicted_class,
            "institutional_forecast": self.institutional_forecast,
            "strategy": self.strategy,
            "n_experts": self.n_experts,
            "contributing_experts": self.contributing_experts,
            "weight_map": self.weight_map,
            "agreement_score": self.agreement_score,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------


class BaseEnsembleStrategy:
    """Abstract base for ensemble strategies."""

    name: str = "base"

    def combine(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None = None
    ) -> EnsembleResult:
        raise NotImplementedError

    def _agreement(self, predictions: list[ExpertPrediction]) -> float:
        """Compute inter-expert agreement [0, 1]."""
        if not predictions:
            return 0.0
        classes = [p.predicted_class for p in predictions]
        majority = max(set(classes), key=classes.count)
        return classes.count(majority) / len(classes)

    def _normalize_weights(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None
    ) -> dict[str, float]:
        """Return normalized weights for the given predictions."""
        if not weights:
            w = {p.expert_id: 1.0 for p in predictions}
        else:
            w = {p.expert_id: weights.get(p.expert_id, 1.0) for p in predictions}
        total = sum(w.values())
        if total <= 0:
            return {k: 1.0 / len(w) for k in w}
        return {k: v / total for k, v in w.items()}


# ---------------------------------------------------------------------------
# Simple averaging
# ---------------------------------------------------------------------------


class SimpleAveragingStrategy(BaseEnsembleStrategy):
    """Unweighted mean of all expert probabilities."""

    name = "simple_averaging"

    def combine(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None = None
    ) -> EnsembleResult:
        if not predictions:
            return EnsembleResult(strategy=self.name)
        n = len(predictions)
        buy = sum(p.buy_prob for p in predictions) / n
        sell = sum(p.sell_prob for p in predictions) / n
        no_trade = sum(p.no_trade_prob for p in predictions) / n
        conf = sum(p.confidence for p in predictions) / n
        unc = sum(p.uncertainty for p in predictions) / n
        return EnsembleResult(
            buy_prob=buy,
            sell_prob=sell,
            no_trade_prob=no_trade,
            confidence=conf,
            uncertainty=unc,
            strategy=self.name,
            n_experts=n,
            contributing_experts=[p.expert_id for p in predictions],
            weight_map={p.expert_id: 1.0 / n for p in predictions},
            agreement_score=self._agreement(predictions),
        )


# ---------------------------------------------------------------------------
# Weighted averaging
# ---------------------------------------------------------------------------


class WeightedAveragingStrategy(BaseEnsembleStrategy):
    """Weight-adjusted mean of expert probabilities."""

    name = "weighted_averaging"

    def combine(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None = None
    ) -> EnsembleResult:
        if not predictions:
            return EnsembleResult(strategy=self.name)
        w = self._normalize_weights(predictions, weights)
        buy = sum(w[p.expert_id] * p.buy_prob for p in predictions)
        sell = sum(w[p.expert_id] * p.sell_prob for p in predictions)
        no_trade = sum(w[p.expert_id] * p.no_trade_prob for p in predictions)
        conf = sum(w[p.expert_id] * p.confidence for p in predictions)
        unc = sum(w[p.expert_id] * p.uncertainty for p in predictions)
        return EnsembleResult(
            buy_prob=buy,
            sell_prob=sell,
            no_trade_prob=no_trade,
            confidence=conf,
            uncertainty=unc,
            strategy=self.name,
            n_experts=len(predictions),
            contributing_experts=[p.expert_id for p in predictions],
            weight_map=w,
            agreement_score=self._agreement(predictions),
        )


# ---------------------------------------------------------------------------
# Performance-weighted averaging
# ---------------------------------------------------------------------------


class PerformanceWeightedStrategy(BaseEnsembleStrategy):
    """Weights derived from recent expert accuracy (higher accuracy = higher weight)."""

    name = "performance_weighted"

    def combine(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None = None
    ) -> EnsembleResult:
        if not predictions:
            return EnsembleResult(strategy=self.name)
        w = self._normalize_weights(predictions, weights)
        buy = sum(w[p.expert_id] * p.buy_prob for p in predictions)
        sell = sum(w[p.expert_id] * p.sell_prob for p in predictions)
        no_trade = sum(w[p.expert_id] * p.no_trade_prob for p in predictions)
        conf = sum(w[p.expert_id] * p.confidence for p in predictions)
        unc = sum(w[p.expert_id] * p.uncertainty for p in predictions)
        return EnsembleResult(
            buy_prob=buy,
            sell_prob=sell,
            no_trade_prob=no_trade,
            confidence=conf,
            uncertainty=unc,
            strategy=self.name,
            n_experts=len(predictions),
            contributing_experts=[p.expert_id for p in predictions],
            weight_map=w,
            agreement_score=self._agreement(predictions),
        )


# ---------------------------------------------------------------------------
# Hard voting
# ---------------------------------------------------------------------------


class HardVotingStrategy(BaseEnsembleStrategy):
    """Majority class vote — winning class gets probability 1.0."""

    name = "hard_voting"

    def combine(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None = None
    ) -> EnsembleResult:
        if not predictions:
            return EnsembleResult(strategy=self.name)
        w = self._normalize_weights(predictions, weights)
        votes: dict[str, float] = {"buy": 0.0, "sell": 0.0, "no_trade": 0.0}
        for p in predictions:
            votes[p.predicted_class] += w[p.expert_id]
        winner = max(votes, key=votes.get)
        buy = 1.0 if winner == "buy" else 0.0
        sell = 1.0 if winner == "sell" else 0.0
        no_trade = 1.0 if winner == "no_trade" else 0.0
        agreement = votes[winner]
        conf = agreement * sum(p.confidence * w[p.expert_id] for p in predictions)
        unc = 1.0 - agreement
        return EnsembleResult(
            buy_prob=buy,
            sell_prob=sell,
            no_trade_prob=no_trade,
            confidence=min(1.0, conf),
            uncertainty=unc,
            strategy=self.name,
            n_experts=len(predictions),
            contributing_experts=[p.expert_id for p in predictions],
            weight_map=w,
            agreement_score=agreement,
            metadata={"votes": votes},
        )


# ---------------------------------------------------------------------------
# Soft voting
# ---------------------------------------------------------------------------


class SoftVotingStrategy(BaseEnsembleStrategy):
    """Weighted sum of class probabilities across experts."""

    name = "soft_voting"

    def combine(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None = None
    ) -> EnsembleResult:
        if not predictions:
            return EnsembleResult(strategy=self.name)
        w = self._normalize_weights(predictions, weights)
        buy = sum(w[p.expert_id] * p.buy_prob for p in predictions)
        sell = sum(w[p.expert_id] * p.sell_prob for p in predictions)
        no_trade = sum(w[p.expert_id] * p.no_trade_prob for p in predictions)
        conf = sum(w[p.expert_id] * p.confidence for p in predictions)
        unc = sum(w[p.expert_id] * p.uncertainty for p in predictions)
        agreement = self._agreement(predictions)
        return EnsembleResult(
            buy_prob=buy,
            sell_prob=sell,
            no_trade_prob=no_trade,
            confidence=conf * (0.5 + 0.5 * agreement),
            uncertainty=unc,
            strategy=self.name,
            n_experts=len(predictions),
            contributing_experts=[p.expert_id for p in predictions],
            weight_map=w,
            agreement_score=agreement,
        )


# ---------------------------------------------------------------------------
# Mixture of Experts (gating network)
# ---------------------------------------------------------------------------


class MixtureOfExpertsStrategy(BaseEnsembleStrategy):
    """Dynamic expert gating using softmax-normalized weights.

    The gating function transforms raw weights to softmax probabilities,
    concentrating probability mass on the highest-weighted expert.

    Args:
        temperature: Softmax temperature (lower = sharper selection,
                     higher = more uniform). Default 1.0.
    """

    name = "mixture_of_experts"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(temperature, 1e-6)

    def _softmax(self, weights: dict[str, float]) -> dict[str, float]:
        vals = list(weights.values())
        keys = list(weights.keys())
        scaled = [v / self.temperature for v in vals]
        max_v = max(scaled)
        exps = [math.exp(v - max_v) for v in scaled]
        total = sum(exps)
        return {k: e / total for k, e in zip(keys, exps)}

    def combine(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None = None
    ) -> EnsembleResult:
        if not predictions:
            return EnsembleResult(strategy=self.name)
        raw_w = {
            p.expert_id: weights.get(p.expert_id, 1.0) if weights else 1.0 for p in predictions
        }
        gate_w = self._softmax(raw_w)
        buy = sum(gate_w[p.expert_id] * p.buy_prob for p in predictions)
        sell = sum(gate_w[p.expert_id] * p.sell_prob for p in predictions)
        no_trade = sum(gate_w[p.expert_id] * p.no_trade_prob for p in predictions)
        conf = sum(gate_w[p.expert_id] * p.confidence for p in predictions)
        unc = sum(gate_w[p.expert_id] * p.uncertainty for p in predictions)
        top_expert = max(gate_w, key=gate_w.get)
        return EnsembleResult(
            buy_prob=buy,
            sell_prob=sell,
            no_trade_prob=no_trade,
            confidence=conf,
            uncertainty=unc,
            strategy=self.name,
            n_experts=len(predictions),
            contributing_experts=[p.expert_id for p in predictions],
            weight_map=gate_w,
            agreement_score=self._agreement(predictions),
            metadata={
                "gate_weights": gate_w,
                "top_expert": top_expert,
                "temperature": self.temperature,
            },
        )


# ---------------------------------------------------------------------------
# Stacking (learned linear combiner)
# ---------------------------------------------------------------------------


class StackingStrategy(BaseEnsembleStrategy):
    """Linear stacking with a learned weight vector per class.

    The stacker learns one weight per expert per class from a small set of
    training examples using online gradient descent.

    Args:
        learning_rate: Stacker learning rate.
        regularization: L2 regularization coefficient.
    """

    name = "stacking"

    def __init__(self, learning_rate: float = 0.01, regularization: float = 1e-4) -> None:
        self.lr = learning_rate
        self.reg = regularization
        self._weights: dict[str, dict[str, float]] = {}

    def _get_weight(self, expert_id: str, cls: str) -> float:
        return self._weights.get(expert_id, {}).get(cls, 1.0)

    def update(self, predictions: list[ExpertPrediction], true_class: str) -> None:
        """Online gradient descent update on stacker weights."""
        class_map = {"buy": 0, "sell": 1, "no_trade": 2}
        for p in predictions:
            if p.expert_id not in self._weights:
                self._weights[p.expert_id] = {"buy": 1.0, "sell": 1.0, "no_trade": 1.0}
            pred_probs = {"buy": p.buy_prob, "sell": p.sell_prob, "no_trade": p.no_trade_prob}
            for cls in ("buy", "sell", "no_trade"):
                target = 1.0 if cls == true_class else 0.0
                grad = pred_probs[cls] - target
                w = self._weights[p.expert_id][cls]
                self._weights[p.expert_id][cls] = w - self.lr * (grad + self.reg * w)

    def combine(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None = None
    ) -> EnsembleResult:
        if not predictions:
            return EnsembleResult(strategy=self.name)
        w = self._normalize_weights(predictions, weights)
        buy = sell = no_trade = 0.0
        for p in predictions:
            wb = self._get_weight(p.expert_id, "buy")
            ws = self._get_weight(p.expert_id, "sell")
            wn = self._get_weight(p.expert_id, "no_trade")
            ew = w[p.expert_id]
            buy += ew * wb * p.buy_prob
            sell += ew * ws * p.sell_prob
            no_trade += ew * wn * p.no_trade_prob
        total = buy + sell + no_trade
        if total > 0:
            buy /= total
            sell /= total
            no_trade /= total
        else:
            buy = sell = 1 / 3
            no_trade = 1 / 3
        conf = sum(w[p.expert_id] * p.confidence for p in predictions)
        unc = sum(w[p.expert_id] * p.uncertainty for p in predictions)
        return EnsembleResult(
            buy_prob=buy,
            sell_prob=sell,
            no_trade_prob=no_trade,
            confidence=conf,
            uncertainty=unc,
            strategy=self.name,
            n_experts=len(predictions),
            contributing_experts=[p.expert_id for p in predictions],
            weight_map=w,
            agreement_score=self._agreement(predictions),
        )


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------


class BlendingStrategy(BaseEnsembleStrategy):
    """Holdout-trained weighted blending.

    Identical to WeightedAveragingStrategy but semantically distinct —
    weights are expected to be calibrated from a holdout validation set.
    """

    name = "blending"

    def combine(
        self, predictions: list[ExpertPrediction], weights: dict[str, float] | None = None
    ) -> EnsembleResult:
        if not predictions:
            return EnsembleResult(strategy=self.name)
        w = self._normalize_weights(predictions, weights)
        buy = sum(w[p.expert_id] * p.buy_prob for p in predictions)
        sell = sum(w[p.expert_id] * p.sell_prob for p in predictions)
        no_trade = sum(w[p.expert_id] * p.no_trade_prob for p in predictions)
        conf = sum(w[p.expert_id] * p.confidence for p in predictions)
        unc = sum(w[p.expert_id] * p.uncertainty for p in predictions)
        return EnsembleResult(
            buy_prob=buy,
            sell_prob=sell,
            no_trade_prob=no_trade,
            confidence=conf,
            uncertainty=unc,
            strategy=self.name,
            n_experts=len(predictions),
            contributing_experts=[p.expert_id for p in predictions],
            weight_map=w,
            agreement_score=self._agreement(predictions),
        )


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


_STRATEGY_MAP = {
    "simple_averaging": SimpleAveragingStrategy,
    "weighted_averaging": WeightedAveragingStrategy,
    "performance_weighted": PerformanceWeightedStrategy,
    "hard_voting": HardVotingStrategy,
    "soft_voting": SoftVotingStrategy,
    "mixture_of_experts": MixtureOfExpertsStrategy,
    "stacking": StackingStrategy,
    "blending": BlendingStrategy,
}


def build_strategy(name: str, **kwargs: Any) -> BaseEnsembleStrategy:
    """Instantiate an ensemble strategy by name.

    Args:
        name: Strategy name (see _STRATEGY_MAP keys).
        **kwargs: Constructor arguments (e.g. temperature=0.5).

    Raises:
        ValueError: If strategy name is unknown.
    """
    cls = _STRATEGY_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown ensemble strategy {name!r}. " f"Available: {sorted(_STRATEGY_MAP)}"
        )
    return cls(**kwargs)


AVAILABLE_STRATEGIES = sorted(_STRATEGY_MAP.keys())
