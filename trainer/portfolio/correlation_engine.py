"""Phase 20 — Correlation Engine.

Computes dynamic correlations between instrument forecasts/returns.

Supported methods:
  - Rolling correlations (Pearson)
  - Rank correlations (Spearman)
  - Tail correlations (lower/upper tail)
  - Conditional correlations (conditioned on regime)
  - Regime-aware correlations
  - Time-varying correlations (exponentially weighted)

Zero new runtime dependencies — stdlib math only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class CorrelationMethod(str, Enum):
    ROLLING = "rolling"
    RANK = "rank"
    TAIL_LOWER = "tail_lower"
    TAIL_UPPER = "tail_upper"
    CONDITIONAL = "conditional"
    EXPONENTIAL = "exponential"


@dataclass
class CorrelationResult:
    instrument_a: str
    instrument_b: str
    correlation: float
    confidence: float
    method: str
    n_observations: int
    p_value: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "instrument_a": self.instrument_a,
            "instrument_b": self.instrument_b,
            "correlation": round(self.correlation, 4),
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "n_observations": self.n_observations,
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
        }


@dataclass
class CorrelationMatrix:
    instruments: List[str]
    matrix: Dict[Tuple[str, str], float]
    method: str
    timestamp: str = ""

    @classmethod
    def from_results(
        cls, results: List[CorrelationResult], method: str, timestamp: str = ""
    ) -> "CorrelationMatrix":
        instruments: List[str] = []
        for r in results:
            if r.instrument_a not in instruments:
                instruments.append(r.instrument_a)
            if r.instrument_b not in instruments:
                instruments.append(r.instrument_b)
        matrix: Dict[Tuple[str, str], float] = {}
        for r in results:
            matrix[(r.instrument_a, r.instrument_b)] = r.correlation
            matrix[(r.instrument_b, r.instrument_a)] = r.correlation
        for i in instruments:
            matrix[(i, i)] = 1.0
        return cls(instruments=instruments, matrix=matrix, method=method, timestamp=timestamp)

    def get(self, a: str, b: str) -> float:
        return self.matrix.get((a, b), 0.0)

    def to_heatmap_data(self) -> List[List[float]]:
        n = len(self.instruments)
        data = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(self.get(self.instruments[i], self.instruments[j]))
            data.append(row)
        return data

    def high_correlation_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        seen = set()
        result = []
        for (a, b), corr in self.matrix.items():
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)
            if abs(corr) >= threshold:
                result.append((a, b, corr))
        return sorted(result, key=lambda x: abs(x[2]), reverse=True)

    def to_dict(self) -> dict:
        pairs = {}
        seen = set()
        for (a, b), corr in self.matrix.items():
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                pairs[f"{a}_{b}"] = round(corr, 4)
        return {
            "instruments": self.instruments,
            "method": self.method,
            "timestamp": self.timestamp,
            "pairs": pairs,
            "heatmap": self.to_heatmap_data(),
        }


class CorrelationEngine:
    """Dynamic correlation computation for portfolio instruments."""

    def __init__(self, window: int = 60, decay: float = 0.94) -> None:
        self._window = max(5, window)
        self._decay = max(0.5, min(0.999, decay))
        self._series: Dict[str, List[float]] = {}

    def add_observation(self, instrument: str, value: float) -> None:
        series = self._series.setdefault(instrument, [])
        series.append(value)
        if len(series) > self._window * 3:
            self._series[instrument] = series[-self._window * 3 :]

    def add_observations(self, observations: Dict[str, float]) -> None:
        for instrument, value in observations.items():
            self.add_observation(instrument, value)

    def instruments(self) -> List[str]:
        return list(self._series.keys())

    def _series_pair(
        self, a: str, b: str, window: Optional[int] = None
    ) -> Tuple[List[float], List[float]]:
        w = window or self._window
        sa = self._series.get(a, [])[-w:]
        sb = self._series.get(b, [])[-w:]
        n = min(len(sa), len(sb))
        return sa[-n:], sb[-n:]

    # ── Pearson rolling correlation ──────────────────────────────────────────

    @staticmethod
    def _pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
        n = len(x)
        if n < 3:
            return 0.0, 0.0
        mx = sum(x) / n
        my = sum(y) / n
        sx = math.sqrt(sum((v - mx) ** 2 for v in x) / n + 1e-12)
        sy = math.sqrt(sum((v - my) ** 2 for v in y) / n + 1e-12)
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
        corr = max(-1.0, min(1.0, cov / (sx * sy)))
        conf = min(1.0, n / 30.0)
        return corr, conf

    def rolling_correlation(
        self, a: str, b: str, window: Optional[int] = None
    ) -> CorrelationResult:
        x, y = self._series_pair(a, b, window)
        corr, conf = self._pearson(x, y)
        return CorrelationResult(a, b, corr, conf, "rolling", len(x))

    # ── Spearman rank correlation ────────────────────────────────────────────

    @staticmethod
    def _rank(series: List[float]) -> List[float]:
        n = len(series)
        sorted_vals = sorted(range(n), key=lambda i: series[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_vals, 1):
            ranks[idx] = float(rank)
        return ranks

    def rank_correlation(self, a: str, b: str, window: Optional[int] = None) -> CorrelationResult:
        x, y = self._series_pair(a, b, window)
        if len(x) < 3:
            return CorrelationResult(a, b, 0.0, 0.0, "rank", len(x))
        rx = self._rank(x)
        ry = self._rank(y)
        corr, conf = self._pearson(rx, ry)
        return CorrelationResult(a, b, corr, conf, "rank", len(x))

    # ── Tail correlations ────────────────────────────────────────────────────

    def tail_correlation(
        self,
        a: str,
        b: str,
        tail: str = "lower",
        quantile: float = 0.20,
        window: Optional[int] = None,
    ) -> CorrelationResult:
        x, y = self._series_pair(a, b, window)
        if len(x) < 6:
            return CorrelationResult(a, b, 0.0, 0.0, f"tail_{tail}", len(x))
        combined = list(zip(x, y))
        if tail == "lower":
            ref = sorted(x)[int(len(x) * quantile)]
            subset = [(xi, yi) for xi, yi in combined if xi <= ref]
        else:
            ref = sorted(x)[int(len(x) * (1 - quantile))]
            subset = [(xi, yi) for xi, yi in combined if xi >= ref]
        if len(subset) < 3:
            return CorrelationResult(a, b, 0.0, 0.0, f"tail_{tail}", len(x))
        xs, ys = zip(*subset)
        corr, conf = self._pearson(list(xs), list(ys))
        conf *= min(1.0, len(subset) / 15.0)
        return CorrelationResult(a, b, corr, conf, f"tail_{tail}", len(subset))

    # ── Exponentially weighted (time-varying) ────────────────────────────────

    def exponential_correlation(self, a: str, b: str) -> CorrelationResult:
        x, y = self._series_pair(a, b)
        n = len(x)
        if n < 3:
            return CorrelationResult(a, b, 0.0, 0.0, "exponential", n)
        lam = self._decay
        weights = [lam ** (n - 1 - i) for i in range(n)]
        w_sum = sum(weights)
        wx = sum(weights[i] * x[i] for i in range(n)) / w_sum
        wy = sum(weights[i] * y[i] for i in range(n)) / w_sum
        var_x = sum(weights[i] * (x[i] - wx) ** 2 for i in range(n)) / w_sum
        var_y = sum(weights[i] * (y[i] - wy) ** 2 for i in range(n)) / w_sum
        cov = sum(weights[i] * (x[i] - wx) * (y[i] - wy) for i in range(n)) / w_sum
        denom = math.sqrt(max(var_x, 1e-12) * max(var_y, 1e-12))
        corr = max(-1.0, min(1.0, cov / denom))
        conf = min(1.0, n / 30.0)
        return CorrelationResult(a, b, corr, conf, "exponential", n)

    # ── Conditional correlation ──────────────────────────────────────────────

    def conditional_correlation(
        self,
        a: str,
        b: str,
        condition_series: str,
        condition_value: str = "positive",
        window: Optional[int] = None,
    ) -> CorrelationResult:
        x, y = self._series_pair(a, b, window)
        cond_raw = (self._series.get(condition_series) or [])[-len(x) :]
        n = min(len(x), len(y), len(cond_raw))
        x, y, cond = x[-n:], y[-n:], cond_raw[-n:]
        if condition_value == "positive":
            mask = [c > 0 for c in cond]
        elif condition_value == "negative":
            mask = [c < 0 for c in cond]
        else:
            mask = [c != 0 for c in cond]
        xs = [x[i] for i in range(n) if mask[i]]
        ys = [y[i] for i in range(n) if mask[i]]
        if len(xs) < 3:
            return CorrelationResult(a, b, 0.0, 0.0, "conditional", n)
        corr, conf = self._pearson(xs, ys)
        conf *= min(1.0, len(xs) / 20.0)
        return CorrelationResult(a, b, corr, conf, "conditional", len(xs))

    # ── Full matrix computation ──────────────────────────────────────────────

    def compute_matrix(
        self,
        instruments: Optional[List[str]] = None,
        method: CorrelationMethod = CorrelationMethod.ROLLING,
        timestamp: str = "",
    ) -> CorrelationMatrix:
        syms = instruments or self.instruments()
        results: List[CorrelationResult] = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a, b = syms[i], syms[j]
                if method == CorrelationMethod.ROLLING:
                    r = self.rolling_correlation(a, b)
                elif method == CorrelationMethod.RANK:
                    r = self.rank_correlation(a, b)
                elif method == CorrelationMethod.TAIL_LOWER:
                    r = self.tail_correlation(a, b, tail="lower")
                elif method == CorrelationMethod.TAIL_UPPER:
                    r = self.tail_correlation(a, b, tail="upper")
                elif method == CorrelationMethod.EXPONENTIAL:
                    r = self.exponential_correlation(a, b)
                else:
                    r = self.rolling_correlation(a, b)
                results.append(r)
        return CorrelationMatrix.from_results(results, method=method.value, timestamp=timestamp)

    def regime_aware_matrix(
        self, instruments: Optional[List[str]] = None, regime: str = "normal", timestamp: str = ""
    ) -> CorrelationMatrix:
        if regime in ("risk_off", "high_volatility", "crash"):
            method = CorrelationMethod.TAIL_LOWER
        elif regime in ("risk_on", "trending"):
            method = CorrelationMethod.ROLLING
        else:
            method = CorrelationMethod.EXPONENTIAL
        return self.compute_matrix(instruments, method, timestamp)

    def correlation_confidence(self, a: str, b: str) -> float:
        x, y = self._series_pair(a, b)
        n = min(len(x), len(y))
        return min(1.0, n / self._window)
