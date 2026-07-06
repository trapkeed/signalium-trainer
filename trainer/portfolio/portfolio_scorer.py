"""Phase 20 — Portfolio Scorer.

Computes portfolio-level metrics:
  - Expected Return
  - Expected Volatility
  - Expected Drawdown (approximate)
  - Expected Sharpe
  - Expected Sortino
  - Diversification Score
  - Correlation Risk
  - Concentration Score
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from trainer.portfolio.portfolio_instruments import ModelForecast
from trainer.portfolio.correlation_engine import CorrelationMatrix


@dataclass
class PortfolioMetrics:
    expected_return: float
    expected_volatility: float
    expected_drawdown: float
    expected_sharpe: float
    expected_sortino: float
    diversification_score: float
    correlation_risk: float
    concentration_score: float
    instrument_count: int
    long_count: int
    short_count: int
    neutral_count: int
    weighted_confidence: float
    return_to_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.expected_drawdown > 1e-9:
            self.return_to_drawdown = self.expected_return / self.expected_drawdown
        if self.expected_drawdown > 1e-9:
            self.calmar_ratio = self.expected_return / self.expected_drawdown

    def to_dict(self) -> dict:
        return {
            "expected_return": round(self.expected_return, 4),
            "expected_volatility": round(self.expected_volatility, 4),
            "expected_drawdown": round(self.expected_drawdown, 4),
            "expected_sharpe": round(self.expected_sharpe, 4),
            "expected_sortino": round(self.expected_sortino, 4),
            "diversification_score": round(self.diversification_score, 4),
            "correlation_risk": round(self.correlation_risk, 4),
            "concentration_score": round(self.concentration_score, 4),
            "instrument_count": self.instrument_count,
            "long_count": self.long_count,
            "short_count": self.short_count,
            "neutral_count": self.neutral_count,
            "weighted_confidence": round(self.weighted_confidence, 4),
            "return_to_drawdown": round(self.return_to_drawdown, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
        }

    def grade(self) -> str:
        s = self.expected_sharpe
        if s >= 2.0:
            return "Exceptional"
        if s >= 1.5:
            return "Strong"
        if s >= 1.0:
            return "Good"
        if s >= 0.5:
            return "Adequate"
        if s >= 0.0:
            return "Weak"
        return "Negative"

    def summary(self) -> str:
        return (
            f"Portfolio ({self.instrument_count} instruments): "
            f"E[R]={self.expected_return:.2%} "
            f"σ={self.expected_volatility:.2%} "
            f"Sharpe={self.expected_sharpe:.2f} "
            f"Div={self.diversification_score:.2f} "
            f"Grade={self.grade()}"
        )


class PortfolioScorer:
    """Scores a portfolio given forecasts and optional correlation matrix."""

    RISK_FREE_RATE = 0.045

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self._rfr = risk_free_rate

    def score(
        self,
        forecasts: List[ModelForecast],
        weights: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[CorrelationMatrix] = None,
    ) -> PortfolioMetrics:
        if not forecasts:
            return self._empty_metrics()

        if weights is None:
            ew = 1.0 / len(forecasts)
            weights = {f.instrument: ew for f in forecasts}

        # Normalize weights to sum to 1 for return/vol calculation
        total_w = sum(abs(w) for w in weights.values())
        if total_w < 1e-9:
            return self._empty_metrics()
        norm_w = {k: v / total_w for k, v in weights.items()}

        expected_return = self._portfolio_return(forecasts, norm_w)
        expected_vol = self._portfolio_volatility(forecasts, norm_w, correlation_matrix)
        expected_drawdown = self._expected_drawdown(expected_vol)
        sharpe = (expected_return - self._rfr) / max(expected_vol, 1e-6)
        sortino = self._sortino(expected_return, forecasts, norm_w)
        div_score = self._diversification_score(norm_w, forecasts, correlation_matrix)
        corr_risk = self._correlation_risk(forecasts, correlation_matrix)
        concentration = self._concentration_score(norm_w)
        w_conf = self._weighted_confidence(forecasts, norm_w)

        long_count = sum(1 for f in forecasts if f.direction == "BUY")
        short_count = sum(1 for f in forecasts if f.direction == "SELL")
        neutral = sum(1 for f in forecasts if f.direction == "NEUTRAL")

        return PortfolioMetrics(
            expected_return=expected_return,
            expected_volatility=expected_vol,
            expected_drawdown=expected_drawdown,
            expected_sharpe=sharpe,
            expected_sortino=sortino,
            diversification_score=div_score,
            correlation_risk=corr_risk,
            concentration_score=concentration,
            instrument_count=len(forecasts),
            long_count=long_count,
            short_count=short_count,
            neutral_count=neutral,
            weighted_confidence=w_conf,
        )

    @staticmethod
    def _portfolio_return(forecasts: List[ModelForecast], norm_w: Dict[str, float]) -> float:
        total = 0.0
        for fc in forecasts:
            w = norm_w.get(fc.instrument, 0.0)
            total += w * fc.signed_return()
        return total

    @staticmethod
    def _portfolio_volatility(
        forecasts: List[ModelForecast],
        norm_w: Dict[str, float],
        matrix: Optional[CorrelationMatrix],
    ) -> float:
        n = len(forecasts)
        if n == 0:
            return 0.0

        # Portfolio variance = Σ_i Σ_j w_i * w_j * σ_i * σ_j * ρ_ij
        total_var = 0.0
        for i, fc_i in enumerate(forecasts):
            wi = abs(norm_w.get(fc_i.instrument, 0.0))
            si = fc_i.expected_risk
            for j, fc_j in enumerate(forecasts):
                wj = abs(norm_w.get(fc_j.instrument, 0.0))
                sj = fc_j.expected_risk
                if matrix is not None and fc_i.instrument != fc_j.instrument:
                    rho = matrix.get(fc_i.instrument, fc_j.instrument)
                else:
                    rho = 1.0 if i == j else 0.0
                total_var += wi * wj * si * sj * rho

        return math.sqrt(max(total_var, 0.0))

    @staticmethod
    def _expected_drawdown(expected_vol: float, factor: float = 1.5) -> float:
        return expected_vol * factor

    def _sortino(
        self, expected_return: float, forecasts: List[ModelForecast], norm_w: Dict[str, float]
    ) -> float:
        downside_var = 0.0
        for fc in forecasts:
            w = abs(norm_w.get(fc.instrument, 0.0))
            if fc.signed_return() < self._rfr:
                downside_var += (w * fc.expected_risk) ** 2
        downside_vol = math.sqrt(downside_var) if downside_var > 0 else 1e-6
        return (expected_return - self._rfr) / downside_vol

    @staticmethod
    def _diversification_score(
        norm_w: Dict[str, float],
        forecasts: List[ModelForecast],
        matrix: Optional[CorrelationMatrix],
    ) -> float:
        if not forecasts or not matrix:
            # Fallback: 1/n approximation
            n = len(forecasts)
            if n == 0:
                return 0.0
            weights = [abs(norm_w.get(f.instrument, 0.0)) for f in forecasts]
            hhi = sum(w**2 for w in weights)
            return round(1.0 - hhi, 4)

        # Portfolio vol / weighted average individual vol
        weighted_vol_sum = sum(
            abs(norm_w.get(fc.instrument, 0.0)) * fc.expected_risk for fc in forecasts
        )
        port_vol = PortfolioScorer._portfolio_volatility(forecasts, norm_w, matrix)
        if weighted_vol_sum < 1e-9:
            return 0.0
        ratio = 1.0 - port_vol / weighted_vol_sum
        return max(0.0, min(1.0, ratio))

    @staticmethod
    def _correlation_risk(
        forecasts: List[ModelForecast], matrix: Optional[CorrelationMatrix]
    ) -> float:
        if matrix is None or len(forecasts) < 2:
            return 0.0
        syms = [fc.instrument for fc in forecasts]
        n = len(syms)
        pairs = 0
        total_abs_corr = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total_abs_corr += abs(matrix.get(syms[i], syms[j]))
                pairs += 1
        if pairs == 0:
            return 0.0
        return round(total_abs_corr / pairs, 4)

    @staticmethod
    def _concentration_score(norm_w: Dict[str, float]) -> float:
        if not norm_w:
            return 0.0
        weights = [abs(v) for v in norm_w.values()]
        total = sum(weights)
        if total < 1e-9:
            return 0.0
        fracs = [w / total for w in weights]
        hhi = sum(f**2 for f in fracs)
        n = len(fracs)
        max_hhi = 1.0
        min_hhi = 1.0 / n if n > 0 else 1.0
        if max_hhi - min_hhi < 1e-9:
            return 0.0
        return round((hhi - min_hhi) / (max_hhi - min_hhi), 4)

    @staticmethod
    def _weighted_confidence(forecasts: List[ModelForecast], norm_w: Dict[str, float]) -> float:
        total = 0.0
        for fc in forecasts:
            w = abs(norm_w.get(fc.instrument, 0.0))
            total += w * fc.confidence
        return round(total, 4)

    @staticmethod
    def _empty_metrics() -> PortfolioMetrics:
        return PortfolioMetrics(
            expected_return=0.0,
            expected_volatility=0.0,
            expected_drawdown=0.0,
            expected_sharpe=0.0,
            expected_sortino=0.0,
            diversification_score=0.0,
            correlation_risk=0.0,
            concentration_score=0.0,
            instrument_count=0,
            long_count=0,
            short_count=0,
            neutral_count=0,
            weighted_confidence=0.0,
        )

    def compare(self, metrics_a: PortfolioMetrics, metrics_b: PortfolioMetrics) -> dict:
        return {
            "return_delta": round(metrics_b.expected_return - metrics_a.expected_return, 4),
            "vol_delta": round(metrics_b.expected_volatility - metrics_a.expected_volatility, 4),
            "sharpe_delta": round(metrics_b.expected_sharpe - metrics_a.expected_sharpe, 4),
            "div_delta": round(
                metrics_b.diversification_score - metrics_a.diversification_score, 4
            ),
            "improved": metrics_b.expected_sharpe > metrics_a.expected_sharpe,
        }
