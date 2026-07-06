"""Permutation Feature Importance for QuantModelV1.

Shuffles each feature column independently, measures the resulting
degradation in model accuracy (or increase in loss), and ranks features by
their contribution to correct predictions.

Works with ANY model that implements ``forward(tensor) → logits``.
No gradients required — fully black-box.

Usage::

    pi = PermutationImportance(metric="accuracy", n_repeats=5)
    result = pi.compute(model, inputs, labels, feature_names=names)
    for name, score in result.top_k(10):
        print(f"{name}: {score:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FeatureImportanceResult:
    """Ranked feature importance scores from permutation testing.

    Attributes:
        feature_names: Feature column names (same order as inputs).
        importances: Mean importance per feature — drop in metric when
            that feature is permuted.  Higher = more important.
        std: Standard deviation of importance across repeat permutations.
        baseline_metric: Metric value on unpermuted (original) data.
        metric_name: Name of the metric used (e.g. "accuracy").
    """

    feature_names: list[str]
    importances: list[float]
    std: list[float]
    baseline_metric: float
    metric_name: str

    def top_k(self, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k most important features (highest metric drop).

        Args:
            k: Number of features to return.

        Returns:
            ``[(feature_name, importance_score), ...]`` sorted descending.
        """
        ranked = sorted(
            zip(self.feature_names, self.importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]

    def bottom_k(self, k: int = 10) -> list[tuple[str, float]]:
        """Return k least important features (lowest metric drop).

        Args:
            k: Number of features to return.

        Returns:
            ``[(feature_name, importance_score), ...]`` sorted ascending.
        """
        ranked = sorted(
            zip(self.feature_names, self.importances),
            key=lambda x: x[1],
        )
        return ranked[:k]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "feature_names": self.feature_names,
            "importances": self.importances,
            "std": self.std,
            "baseline_metric": self.baseline_metric,
            "metric_name": self.metric_name,
            "top_10": self.top_k(10),
        }


# ---------------------------------------------------------------------------
# Permutation Importance estimator
# ---------------------------------------------------------------------------


class PermutationImportance:
    """Permutation Feature Importance estimator.

    Args:
        metric: ``"accuracy"`` — drop in accuracy when feature is shuffled.
               ``"loss"``     — increase in cross-entropy loss.
        n_repeats: Number of random shuffles per feature (more = lower
            variance; 5–10 is usually sufficient).
        seed: Optional random seed for reproducibility.

    Usage::

        pi = PermutationImportance(metric="accuracy", n_repeats=5, seed=42)
        result = pi.compute(model, inputs, labels, feature_names=col_names)
        print(result.top_k(10))
    """

    _VALID_METRICS = frozenset({"accuracy", "loss"})

    def __init__(
        self,
        metric: str = "accuracy",
        n_repeats: int = 5,
        seed: int | None = None,
    ) -> None:
        if metric not in self._VALID_METRICS:
            raise ValueError(f"metric must be one of {sorted(self._VALID_METRICS)}, got {metric!r}")
        self._metric = metric
        self._n_repeats = n_repeats
        self._seed = seed

    def compute(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        feature_names: list[str] | None = None,
    ) -> FeatureImportanceResult:
        """Compute permutation importance for all input features.

        Args:
            model: Model implementing ``forward(tensor) → logits (batch, C)``.
            inputs: ``(batch, seq_len, n_features)`` input tensor.
            labels: ``(batch,)`` ground-truth class indices.
            feature_names: Optional list of feature column names.  Falls back
                to ``feature_0, feature_1, ...`` when omitted.

        Returns:
            ``FeatureImportanceResult`` with importances in input order.
        """
        n_features = inputs.shape[-1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        if len(names) != n_features:
            raise ValueError(f"feature_names length ({len(names)}) != n_features ({n_features})")

        rng = np.random.default_rng(self._seed)
        was_training = model.training
        model.eval()

        try:
            baseline = self._eval_metric(model, inputs, labels)
            logger.debug(f"PermutationImportance: baseline {self._metric}={baseline:.4f}")

            importances: list[float] = []
            stds: list[float] = []

            for feat_idx in range(n_features):
                drops = []
                for _ in range(self._n_repeats):
                    shuffled = inputs.clone()
                    perm = rng.permutation(shuffled.shape[0])
                    shuffled[:, :, feat_idx] = shuffled[perm, :, feat_idx]
                    score = self._eval_metric(model, shuffled, labels)
                    drops.append(float(baseline - score))

                importances.append(float(np.mean(drops)))
                stds.append(float(np.std(drops)))
                logger.debug(
                    f"  [{feat_idx:4d}] {names[feat_idx]:<30s} "
                    f"importance={importances[-1]:+.4f} ± {stds[-1]:.4f}"
                )

        finally:
            if was_training:
                model.train()

        return FeatureImportanceResult(
            feature_names=names,
            importances=importances,
            std=stds,
            baseline_metric=float(baseline),
            metric_name=self._metric,
        )

    @torch.no_grad()
    def _eval_metric(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Evaluate the chosen metric on a single batch.

        Returns a scalar — higher is always better (loss is negated).
        """
        logits = model(inputs)
        if self._metric == "accuracy":
            preds = logits.argmax(dim=-1)
            return float((preds == labels).float().mean().item())
        else:  # loss — negate so higher = better
            ce = torch.nn.functional.cross_entropy(logits, labels)
            return -float(ce.item())
