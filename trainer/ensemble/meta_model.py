"""Phase 17 — AEIP Ensemble Meta Model.

The EnsembleMetaModel learns to combine expert predictions optimally.

Inputs to the meta model:
  - Predictions from every expert (buy/sell/no_trade probabilities)
  - Per-expert confidence
  - Per-expert uncertainty
  - Current market regime
  - Instrument
  - Timeframe
  - Recent expert performance
  - Historical reliability

Outputs:
  - Final BUY probability
  - Final SELL probability
  - Final NO TRADE probability
  - Final Confidence
  - Final Institutional Forecast

Implementation: pure-Python logistic regression / linear model with
online gradient descent. No PyTorch dependency in the meta layer.

Usage::

    meta = EnsembleMetaModel()
    meta.fit(training_examples)
    result = meta.predict(predictions, context)
    print(result.institutional_forecast)
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from trainer.ensemble.expert import ExpertPrediction
from trainer.ensemble.strategies import WeightedAveragingStrategy

# ---------------------------------------------------------------------------
# Meta model prediction
# ---------------------------------------------------------------------------


@dataclass
class MetaPrediction:
    """Output of the Ensemble Meta Model.

    Extends EnsembleResult with meta-model-specific fields.
    """

    buy_prob: float = 0.33
    sell_prob: float = 0.33
    no_trade_prob: float = 0.34
    confidence: float = 0.5
    uncertainty: float = 0.5
    strategy: str = "meta_learning"
    n_experts: int = 0
    contributing_experts: list[str] = field(default_factory=list)
    weight_map: dict[str, float] = field(default_factory=dict)
    agreement_score: float = 0.0
    regime: str = "unknown"
    instrument: str = ""
    timeframe: str = ""
    meta_version: str = "1.0"
    n_training_samples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

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
        cls = self.predicted_class
        conf = self.confidence
        agree = self.agreement_score
        if conf >= 0.80 and agree >= 0.75:
            grade = "A"
        elif conf >= 0.65 and agree >= 0.60:
            grade = "B"
        elif conf >= 0.50:
            grade = "C"
        else:
            grade = "D"
        label = {"buy": "LONG", "sell": "SHORT", "no_trade": "NEUTRAL"}[cls]
        return (
            f"[{grade}] {label} | {self.regime} | "
            f"conf={conf:.2f} | agree={agree:.2f} | "
            f"n_experts={self.n_experts}"
        )

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
            "regime": self.regime,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "meta_version": self.meta_version,
            "n_training_samples": self.n_training_samples,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


class MetaFeatureBuilder:
    """Converts expert predictions + context into a meta-model feature vector."""

    REGIMES = [
        "trending_up",
        "trending_down",
        "ranging",
        "volatile",
        "high_volatility",
        "low_volatility",
        "breakout",
        "pullback",
        "crisis",
        "unknown",
    ]

    INSTRUMENTS = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "USDCHF",
        "AUDUSD",
        "USDCAD",
        "NZDUSD",
        "EURGBP",
        "EURJPY",
        "GBPJPY",
        "XAUUSD",
        "XAGUSD",
        "USOIL",
        "US30",
        "NAS100",
    ]

    TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

    def build(
        self,
        predictions: list[ExpertPrediction],
        weights: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> list[float]:
        """Build feature vector from predictions and context."""
        ctx = context or {}
        features: list[float] = []

        if predictions:
            avg_buy = sum(p.buy_prob for p in predictions) / len(predictions)
            avg_sell = sum(p.sell_prob for p in predictions) / len(predictions)
            avg_nt = sum(p.no_trade_prob for p in predictions) / len(predictions)
            avg_conf = sum(p.confidence for p in predictions) / len(predictions)
            avg_unc = sum(p.uncertainty for p in predictions) / len(predictions)
            std_buy = _std([p.buy_prob for p in predictions])
            std_sell = _std([p.sell_prob for p in predictions])
            features += [
                avg_buy,
                avg_sell,
                avg_nt,
                avg_conf,
                avg_unc,
                std_buy,
                std_sell,
                len(predictions) / 10.0,
            ]
        else:
            features += [0.33, 0.33, 0.34, 0.5, 0.5, 0.0, 0.0, 0.0]

        for p in predictions:
            w = weights.get(p.expert_id, 1.0 / max(len(predictions), 1))
            features += [w * p.buy_prob, w * p.sell_prob, w * p.no_trade_prob, w * p.confidence]

        regime = ctx.get("regime", "unknown")
        regime_ohe = [1.0 if r == regime else 0.0 for r in self.REGIMES]
        features += regime_ohe

        return features

    def n_base_features(self) -> int:
        return 8 + len(self.REGIMES)


# ---------------------------------------------------------------------------
# Linear meta model (pure Python)
# ---------------------------------------------------------------------------


class LinearMetaLayer:
    """A simple linear layer for meta learning.

    Maps feature vector → 3 logits (buy, sell, no_trade).

    Uses online SGD updates.
    """

    def __init__(
        self, n_features: int, learning_rate: float = 0.01, regularization: float = 1e-4
    ) -> None:
        self.n_features = n_features
        self.lr = learning_rate
        self.reg = regularization
        self._W: list[list[float]] = [[0.0] * n_features for _ in range(3)]
        self._b: list[float] = [0.33, 0.33, 0.34]
        self._n_updates = 0

    def forward(self, x: list[float]) -> list[float]:
        """Return logits for [buy, sell, no_trade]."""
        logits = []
        for i in range(3):
            val = self._b[i] + sum(
                self._W[i][j] * x[j] for j in range(min(len(x), self.n_features))
            )
            logits.append(val)
        return logits

    def softmax(self, logits: list[float]) -> list[float]:
        max_v = max(logits)
        exps = [math.exp(v - max_v) for v in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def predict(self, x: list[float]) -> list[float]:
        return self.softmax(self.forward(x))

    def update(self, x: list[float], target_idx: int) -> float:
        """Single SGD step. Returns cross-entropy loss."""
        logits = self.forward(x)
        probs = self.softmax(logits)
        loss = -math.log(max(probs[target_idx], 1e-9))
        for i in range(3):
            target = 1.0 if i == target_idx else 0.0
            grad = probs[i] - target
            self._b[i] -= self.lr * grad
            for j in range(min(len(x), self.n_features)):
                self._W[i][j] -= self.lr * (grad * x[j] + self.reg * self._W[i][j])
        self._n_updates += 1
        return loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "lr": self.lr,
            "reg": self.reg,
            "W": self._W,
            "b": self._b,
            "n_updates": self._n_updates,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LinearMetaLayer:
        layer = LinearMetaLayer(d["n_features"], d["lr"], d["reg"])
        layer._W = d["W"]
        layer._b = d["b"]
        layer._n_updates = d["n_updates"]
        return layer


# ---------------------------------------------------------------------------
# Ensemble meta model
# ---------------------------------------------------------------------------


class EnsembleMetaModel:
    """Learned ensemble combiner that produces institutional-grade forecasts.

    Combines expert predictions through a learned linear layer conditioned
    on market regime, instrument, timeframe, and expert performance.

    Falls back to weighted averaging until enough training samples exist.

    Args:
        learning_rate: SGD learning rate.
        regularization: L2 regularization.
        min_samples_to_activate: Samples before meta layer is used.
        meta_version: Version string for this meta model.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        regularization: float = 1e-4,
        min_samples_to_activate: int = 50,
        meta_version: str = "1.0",
    ) -> None:
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.min_samples = min_samples_to_activate
        self.meta_version = meta_version
        self.feature_builder = MetaFeatureBuilder()
        self._fallback = WeightedAveragingStrategy()
        self._n_features: int | None = None
        self._layer: LinearMetaLayer | None = None
        self._n_training_samples = 0
        self._training_loss_history: list[float] = []
        self.model_id: str = str(uuid.uuid4())

    def _ensure_layer(self, n_features: int) -> LinearMetaLayer:
        if self._layer is None or self._layer.n_features != n_features:
            self._layer = LinearMetaLayer(n_features, self.learning_rate, self.regularization)
        return self._layer

    def predict(
        self,
        predictions: list[ExpertPrediction],
        weights: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> MetaPrediction:
        """Produce a meta-model prediction.

        Falls back to weighted averaging if not enough training data yet.
        """
        ctx = context or {}
        regime = ctx.get("regime", "unknown")
        instrument = ctx.get("instrument", "")
        timeframe = ctx.get("timeframe", "")
        n = len(predictions)

        if not predictions:
            return MetaPrediction(
                regime=regime,
                instrument=instrument,
                timeframe=timeframe,
                n_training_samples=self._n_training_samples,
                meta_version=self.meta_version,
            )

        agreement = self._compute_agreement(predictions)

        if self._layer is None or self._n_training_samples < self.min_samples:
            fallback = self._fallback.combine(predictions, weights)
            return MetaPrediction(
                buy_prob=fallback.buy_prob,
                sell_prob=fallback.sell_prob,
                no_trade_prob=fallback.no_trade_prob,
                confidence=fallback.confidence,
                uncertainty=fallback.uncertainty,
                strategy="meta_learning_fallback",
                n_experts=n,
                contributing_experts=[p.expert_id for p in predictions],
                weight_map=weights,
                agreement_score=agreement,
                regime=regime,
                instrument=instrument,
                timeframe=timeframe,
                meta_version=self.meta_version,
                n_training_samples=self._n_training_samples,
            )

        features = self.feature_builder.build(predictions, weights, ctx)
        probs = self._layer.predict(features)
        buy, sell, no_trade = probs[0], probs[1], probs[2]

        avg_conf = sum(weights.get(p.expert_id, 1.0 / n) * p.confidence for p in predictions)
        avg_unc = sum(weights.get(p.expert_id, 1.0 / n) * p.uncertainty for p in predictions)
        conf = avg_conf * (0.5 + 0.5 * agreement)

        return MetaPrediction(
            buy_prob=buy,
            sell_prob=sell,
            no_trade_prob=no_trade,
            confidence=conf,
            uncertainty=avg_unc,
            strategy="meta_learning",
            n_experts=n,
            contributing_experts=[p.expert_id for p in predictions],
            weight_map=weights,
            agreement_score=agreement,
            regime=regime,
            instrument=instrument,
            timeframe=timeframe,
            meta_version=self.meta_version,
            n_training_samples=self._n_training_samples,
        )

    def update(
        self,
        predictions: list[ExpertPrediction],
        weights: dict[str, float],
        true_class: str,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Online update with one labeled example. Returns cross-entropy loss."""
        class_map = {"buy": 0, "sell": 1, "no_trade": 2}
        target_idx = class_map.get(true_class, 2)
        features = self.feature_builder.build(predictions, weights, context)
        n_features = len(features)

        if self._layer is None or self._layer.n_features != n_features:
            self._n_features = n_features
            self._layer = self._ensure_layer(n_features)

        loss = self._layer.update(features, target_idx)
        self._n_training_samples += 1
        self._training_loss_history.append(loss)
        return loss

    def fit(self, examples: list[dict[str, Any]], epochs: int = 1) -> list[float]:
        """Batch fit from list of examples.

        Each example: {predictions: [...], weights: {...}, context: {...}, true_class: str}

        Returns list of per-example losses.
        """
        losses = []
        for _ in range(epochs):
            for ex in examples:
                preds = [
                    ExpertPrediction.from_dict(p) if isinstance(p, dict) else p
                    for p in ex.get("predictions", [])
                ]
                wts = ex.get("weights", {})
                ctx = ex.get("context", {})
                tc = ex.get("true_class", "no_trade")
                loss = self.update(preds, wts, tc, ctx)
                losses.append(loss)
        return losses

    def _compute_agreement(self, predictions: list[ExpertPrediction]) -> float:
        if not predictions:
            return 0.0
        classes = [p.predicted_class for p in predictions]
        majority = max(set(classes), key=classes.count)
        return classes.count(majority) / len(classes)

    @property
    def is_trained(self) -> bool:
        return self._n_training_samples >= self.min_samples

    @property
    def training_loss(self) -> float:
        if not self._training_loss_history:
            return 0.0
        recent = self._training_loss_history[-min(100, len(self._training_loss_history)) :]
        return sum(recent) / len(recent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "meta_version": self.meta_version,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "min_samples": self.min_samples,
            "n_training_samples": self._n_training_samples,
            "n_features": self._n_features,
            "layer": self._layer.to_dict() if self._layer else None,
            "training_loss_history_tail": self._training_loss_history[-100:],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EnsembleMetaModel:
        m = EnsembleMetaModel(
            learning_rate=d.get("learning_rate", 0.01),
            regularization=d.get("regularization", 1e-4),
            min_samples_to_activate=d.get("min_samples", 50),
            meta_version=d.get("meta_version", "1.0"),
        )
        m.model_id = d.get("model_id", str(uuid.uuid4()))
        m._n_training_samples = d.get("n_training_samples", 0)
        m._n_features = d.get("n_features")
        m._training_loss_history = d.get("training_loss_history_tail", [])
        if d.get("layer"):
            m._layer = LinearMetaLayer.from_dict(d["layer"])
        return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(variance)
