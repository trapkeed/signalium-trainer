"""Phase 24 — Correlation Engine.

Evaluates: Rolling Correlations, Dynamic Correlations,
Correlation Clusters, Hidden Exposure, Currency Concentration,
Cross-Asset Correlation, Tail Correlation.

Pure Python implementation (no numpy).
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Set, Tuple

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


def _mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def _pearson(a: List[float], b: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    ma, mb = _mean(a[:n]), _mean(b[:n])
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((v - ma) ** 2 for v in a[:n]))
    db = math.sqrt(sum((v - mb) ** 2 for v in b[:n]))
    denom = da * db
    if denom == 0:
        return 0.0
    return max(-1.0, min(1.0, num / denom))


def _tail_correlation(a: List[float], b: List[float], percentile: float = 0.10) -> float:
    """Correlation in the joint lower tail (both below p-th percentile)."""
    n = min(len(a), len(b))
    if n < 10:
        return 0.0
    pairs_sorted_a = sorted(range(n), key=lambda i: a[i])
    cutoff = int(n * percentile)
    tail_idx = set(pairs_sorted_a[: max(1, cutoff)])
    # Also include b tail
    pairs_sorted_b = sorted(range(n), key=lambda i: b[i])
    tail_idx |= set(pairs_sorted_b[: max(1, cutoff)])
    idx = list(tail_idx)
    if len(idx) < 2:
        return 0.0
    return _pearson([a[i] for i in idx], [b[i] for i in idx])


class CorrelationEngine:
    """Rolling and dynamic correlation analysis."""

    def __init__(
        self,
        rolling_window: int = 30,
        cluster_threshold: float = 0.70,
    ):
        self._window = rolling_window
        self._threshold = cluster_threshold

    # ── Rolling correlation ───────────────────────────────────────────────────

    def rolling_correlation(
        self,
        returns_a: List[float],
        returns_b: List[float],
        window: Optional[int] = None,
    ) -> List[float]:
        """Compute rolling Pearson correlation with given window."""
        w = window or self._window
        n = min(len(returns_a), len(returns_b))
        result = []
        for i in range(w - 1, n):
            a = returns_a[i - w + 1 : i + 1]
            b = returns_b[i - w + 1 : i + 1]
            result.append(round(_pearson(a, b), 4))
        return result

    def current_correlation(self, returns_a: List[float], returns_b: List[float]) -> float:
        """Correlation over the most recent `window` observations."""
        w = min(self._window, len(returns_a), len(returns_b))
        return round(_pearson(returns_a[-w:], returns_b[-w:]), 4)

    # ── Correlation matrix ────────────────────────────────────────────────────

    def correlation_matrix(
        self,
        returns: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute the full pairwise correlation matrix."""
        instruments = list(returns.keys())
        matrix: Dict[str, Dict[str, float]] = {}
        for i, ins_a in enumerate(instruments):
            matrix[ins_a] = {}
            for j, ins_b in enumerate(instruments):
                if i == j:
                    matrix[ins_a][ins_b] = 1.0
                elif ins_b in matrix and ins_a in matrix.get(ins_b, {}):
                    matrix[ins_a][ins_b] = matrix[ins_b][ins_a]
                else:
                    matrix[ins_a][ins_b] = self.current_correlation(returns[ins_a], returns[ins_b])
        return matrix

    # ── Dynamic correlation ───────────────────────────────────────────────────

    def dynamic_correlation(
        self,
        returns: Dict[str, List[float]],
        lookback: int = 60,
    ) -> Dict[str, float]:
        """Average pairwise correlation for each instrument vs the portfolio."""
        instruments = list(returns.keys())
        result = {}
        for ins in instruments:
            others = [k for k in instruments if k != ins]
            if not others:
                result[ins] = 0.0
                continue
            corrs = [
                self.current_correlation(returns[ins][-lookback:], returns[o][-lookback:])
                for o in others
            ]
            result[ins] = round(sum(corrs) / len(corrs), 4)
        return result

    # ── Clustering ────────────────────────────────────────────────────────────

    def correlation_clusters(
        self,
        returns: Dict[str, List[float]],
    ) -> List[Set[str]]:
        """Identify groups of highly correlated instruments.

        Uses single-linkage clustering at `self._threshold`.
        """
        instruments = list(returns.keys())
        matrix = self.correlation_matrix(returns)

        # Assign each to its own cluster, then merge
        clusters: List[Set[str]] = [{ins} for ins in instruments]

        changed = True
        while changed:
            changed = False
            new_clusters: List[Set[str]] = []
            merged: Set[int] = set()
            for i in range(len(clusters)):
                if i in merged:
                    continue
                cluster = set(clusters[i])
                for j in range(i + 1, len(clusters)):
                    if j in merged:
                        continue
                    # Check if any pair across the two clusters exceeds threshold
                    for a in cluster:
                        for b in clusters[j]:
                            if a != b and abs(matrix.get(a, {}).get(b, 0.0)) >= self._threshold:
                                cluster |= clusters[j]
                                merged.add(j)
                                changed = True
                                break
                        else:
                            continue
                        break
                new_clusters.append(cluster)
            clusters = new_clusters

        return clusters

    # ── Hidden exposure ───────────────────────────────────────────────────────

    def hidden_exposure(
        self,
        weights: Dict[str, float],
        returns: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """Effective exposure accounting for correlation."""
        matrix = self.correlation_matrix(returns)
        instruments = list(weights.keys())
        hidden: Dict[str, float] = {}
        for ins in instruments:
            w = weights.get(ins, 0.0)
            corrs = [
                abs(matrix.get(ins, {}).get(o, 0.0)) * weights.get(o, 0.0)
                for o in instruments
                if o != ins
            ]
            hidden[ins] = round(w + sum(corrs), 4)
        return hidden

    # ── Currency concentration ────────────────────────────────────────────────

    def currency_concentration(
        self,
        weights: Dict[str, float],
        currencies: Dict[str, str],  # instrument → base currency
    ) -> Dict[str, float]:
        """Aggregate exposure by base currency."""
        ccy: Dict[str, float] = {}
        for ins, w in weights.items():
            c = currencies.get(ins, "UNKNOWN")
            ccy[c] = ccy.get(c, 0.0) + w
        return {k: round(v, 4) for k, v in sorted(ccy.items(), key=lambda x: -x[1])}

    # ── Tail correlation ──────────────────────────────────────────────────────

    def tail_correlations(
        self,
        returns: Dict[str, List[float]],
        percentile: float = 0.10,
    ) -> Dict[str, Dict[str, float]]:
        """Pairwise tail correlations in the lower `percentile` of both series."""
        instruments = list(returns.keys())
        result: Dict[str, Dict[str, float]] = {}
        for a in instruments:
            result[a] = {}
            for b in instruments:
                if a == b:
                    result[a][b] = 1.0
                else:
                    tc = _tail_correlation(returns[a], returns[b], percentile)
                    result[a][b] = round(tc, 4)
        return result

    # ── Portfolio average correlation ─────────────────────────────────────────

    def average_correlation(self, returns: Dict[str, List[float]]) -> float:
        """Average pairwise correlation across all instrument pairs."""
        instruments = list(returns.keys())
        if len(instruments) < 2:
            return 0.0
        pairs = []
        for i in range(len(instruments)):
            for j in range(i + 1, len(instruments)):
                pairs.append(
                    self.current_correlation(returns[instruments[i]], returns[instruments[j]])
                )
        return round(sum(pairs) / len(pairs), 4) if pairs else 0.0
