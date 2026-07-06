"""Phase 19.3 — HRSA Level 1→2: Normalisation.

Provides multiple normalisation methods for converting raw metric
values into [0, 1] (or [0, 100]) comparable scores.

Supported methods:
  MinMax     — linear map from [lo, hi] → [0, 1]
  ZScore     — (x - μ) / σ, clamped to [-3, 3] then mapped to [0, 1]
  Percentile — rank fraction within a reference distribution
  Rank       — rank / n, smooth rank scaling
  Robust     — (x - median) / IQR
  Custom     — caller-supplied transform function

All normalisers are pure functions — no side effects.

Usage::

    n = Normaliser()

    # Scalar normalisation
    score = n.normalise(1.82, lo=0.5, hi=2.5,
                        method="min_max", direction="higher")

    # Batch normalisation (provides relative scores)
    scores = n.normalise_batch([0.5, 1.2, 1.8, 2.1],
                               method="z_score", direction="higher")
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Method constants
# ---------------------------------------------------------------------------


class NormMethod:
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    PERCENTILE = "percentile"
    RANK = "rank_scaling"
    ROBUST = "robust_scaling"
    CUSTOM = "custom"
    ALL = [MIN_MAX, Z_SCORE, PERCENTILE, RANK, ROBUST, CUSTOM]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _iqr(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[3 * n // 4]
    return q3 - q1


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: List[float]) -> float:
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# NormResult — result of a normalisation operation
# ---------------------------------------------------------------------------


class NormResult:
    """Result of normalising one metric value."""

    __slots__ = ("raw_value", "normalised", "direction", "method")

    def __init__(
        self,
        raw_value: float,
        normalised: float,
        direction: str,
        method: str,
    ) -> None:
        self.raw_value = raw_value
        self.normalised = normalised  # [0, 1]
        self.direction = direction
        self.method = method

    @property
    def score_100(self) -> float:
        """Normalised value scaled to [0, 100]."""
        return self.normalised * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_value": self.raw_value,
            "normalised": self.normalised,
            "score_100": self.score_100,
            "direction": self.direction,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Normaliser — stateless normalisation engine
# ---------------------------------------------------------------------------


class Normaliser:
    """Apply one of 6 normalisation methods to convert a raw metric value
    into a [0, 1] score.

    All methods handle direction inversion: if direction is
    ``"lower_is_better"``, the score is flipped (1 - norm).
    """

    def normalise(
        self,
        value: float,
        *,
        lo: float = 0.0,
        hi: float = 1.0,
        method: str = NormMethod.MIN_MAX,
        direction: str = "higher_is_better",
        distribution: Optional[List[float]] = None,
        custom_fn: Optional[Callable[[float], float]] = None,
    ) -> NormResult:
        """Normalise a single value.

        Args:
            value:        The raw metric value.
            lo:           Lower bound of the expected range (for min-max).
            hi:           Upper bound of the expected range (for min-max).
            method:       One of NormMethod constants.
            direction:    "higher_is_better" or "lower_is_better".
            distribution: Reference values (required for z_score, percentile, rank, robust).
            custom_fn:    For NormMethod.CUSTOM — callable (value) → float in [0, 1].
        """
        norm = self._apply(
            value, lo=lo, hi=hi, method=method, distribution=distribution, custom_fn=custom_fn
        )
        if direction == "lower_is_better":
            norm = 1.0 - norm
        return NormResult(
            raw_value=value, normalised=_clamp01(norm), direction=direction, method=method
        )

    def normalise_batch(
        self,
        values: List[float],
        *,
        lo: float = 0.0,
        hi: float = 1.0,
        method: str = NormMethod.MIN_MAX,
        direction: str = "higher_is_better",
        custom_fn: Optional[Callable[[float], float]] = None,
    ) -> List[NormResult]:
        """Normalise a list of values, using the list itself as the reference distribution."""
        if not values:
            return []
        results = []
        for v in values:
            r = self.normalise(
                v,
                lo=lo,
                hi=hi,
                method=method,
                direction=direction,
                distribution=values,
                custom_fn=custom_fn,
            )
            results.append(r)
        return results

    def _apply(
        self,
        value: float,
        *,
        lo: float,
        hi: float,
        method: str,
        distribution: Optional[List[float]],
        custom_fn: Optional[Callable[[float], float]],
    ) -> float:
        if method == NormMethod.MIN_MAX:
            return self._min_max(value, lo, hi)
        if method == NormMethod.Z_SCORE:
            return self._z_score(value, distribution)
        if method == NormMethod.PERCENTILE:
            return self._percentile(value, distribution)
        if method == NormMethod.RANK:
            return self._rank(value, distribution)
        if method == NormMethod.ROBUST:
            return self._robust(value, distribution)
        if method == NormMethod.CUSTOM:
            if custom_fn is None:
                raise ValueError("custom_fn required for NormMethod.CUSTOM")
            return _clamp01(custom_fn(value))
        raise ValueError(f"Unknown normalisation method: {method!r}")

    # ---- Method implementations ------------------------------------------

    def _min_max(self, v: float, lo: float, hi: float) -> float:
        """Linear map [lo, hi] → [0, 1]."""
        if hi <= lo:
            return 0.0
        return _clamp01((v - lo) / (hi - lo))

    def _z_score(self, v: float, distribution: Optional[List[float]]) -> float:
        """Z-score normalisation: (v - μ) / σ, clamped to [-3, 3] → [0, 1]."""
        if not distribution or len(distribution) < 2:
            return 0.5
        mu = _mean(distribution)
        sd = _std(distribution)
        if sd < 1e-12:
            return 0.5
        z = (v - mu) / sd
        # Map [-3, 3] → [0, 1]
        return _clamp01((z + 3.0) / 6.0)

    def _percentile(self, v: float, distribution: Optional[List[float]]) -> float:
        """Rank fraction: fraction of distribution values ≤ v."""
        if not distribution:
            return 0.5
        n = len(distribution)
        rank = sum(1 for x in distribution if x <= v)
        return _clamp01(rank / n)

    def _rank(self, v: float, distribution: Optional[List[float]]) -> float:
        """Smooth rank scaling: rank (1-indexed) / n, interpolated."""
        if not distribution:
            return 0.5
        sorted_d = sorted(set(distribution))
        n = len(sorted_d)
        if n == 1:
            return 0.5
        # Linear interpolation of rank
        for i, x in enumerate(sorted_d):
            if v <= x:
                return _clamp01(i / (n - 1))
        return 1.0

    def _robust(self, v: float, distribution: Optional[List[float]]) -> float:
        """Robust scaling: (v - median) / IQR, mapped to [0, 1]."""
        if not distribution or len(distribution) < 4:
            return 0.5
        med = _median(distribution)
        iq = _iqr(distribution)
        if iq < 1e-12:
            return 0.5
        robust_z = (v - med) / iq
        # Map [-2, 2] → [0, 1]
        return _clamp01((robust_z + 2.0) / 4.0)


# ---------------------------------------------------------------------------
# CategoryNormaliser — normalises all metrics in a category
# ---------------------------------------------------------------------------


class CategoryNormaliser:
    """Normalise all raw metric values for one category, returning
    per-metric [0, 100] scores and an overall weighted mean."""

    def __init__(self, normaliser: Optional[Normaliser] = None) -> None:
        self._n = normaliser or Normaliser()

    def normalise_category(
        self,
        metric_values: Dict[str, float],
        metric_definitions: List[Any],  # List[MetricDefinition]
        method: str = NormMethod.MIN_MAX,
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, NormResult]]:
        """
        Normalise all metrics in a category.

        Returns:
            (category_score_0_100, {metric_name: NormResult})
        """
        results: Dict[str, NormResult] = {}
        weighted_scores: List[Tuple[float, float]] = []  # (score, weight)
        distribution = (
            list(metric_values.values())
            if method
            in (NormMethod.Z_SCORE, NormMethod.PERCENTILE, NormMethod.RANK, NormMethod.ROBUST)
            else None
        )

        for defn in metric_definitions:
            raw = metric_values.get(defn.name)
            if raw is None:
                continue
            result = self._n.normalise(
                raw,
                lo=defn.normalisation_lo,
                hi=defn.normalisation_hi,
                method=method,
                direction=defn.direction,
                distribution=distribution,
            )
            results[defn.name] = result
            w = (metric_weights or {}).get(defn.name, 1.0)
            weighted_scores.append((result.score_100, w))

        if not weighted_scores:
            return 0.0, results

        total_w = sum(w for _, w in weighted_scores)
        if total_w <= 0:
            return 0.0, results
        score = sum(s * w for s, w in weighted_scores) / total_w
        return max(0.0, min(100.0, score)), results
