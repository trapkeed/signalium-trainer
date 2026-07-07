"""Phase 24 — Portfolio Risk Engine.

Calculates: Portfolio VaR, Conditional VaR (CVaR), Expected Shortfall,
Maximum Expected Drawdown, Portfolio Volatility, Risk Contribution,
Marginal Risk Contribution, Diversification Ratio, Risk Concentration.

All calculations are pure Python (no numpy).
"""

from __future__ import annotations

import math
import statistics

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.icari.icari_models import PortfolioRisk


def _mean(data: list[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def _std(data: list[float]) -> float:
    if len(data) < 2:
        return 0.0
    return statistics.stdev(data)


def _percentile(sorted_data: list[float], p: float) -> float:
    """p in [0, 100]. sorted_data must be ascending-sorted."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


class PortfolioRiskEngine:
    """Historical simulation + parametric portfolio risk metrics."""

    def __init__(self, confidence_95: float = 0.95, confidence_99: float = 0.99):
        self._c95 = confidence_95
        self._c99 = confidence_99

    def calculate(
        self,
        returns: dict[str, list[float]],  # instrument → list of daily returns
        weights: dict[str, float],  # instrument → weight (must sum ≤ 1)
    ) -> PortfolioRisk:
        """Calculate full portfolio risk metrics from historical returns."""
        if not returns or not weights:
            return PortfolioRisk()

        instruments = list(weights.keys())
        port_returns = self._portfolio_returns(returns, weights)

        if not port_returns:
            return PortfolioRisk()

        # Sort ascending for percentile calculations
        sorted_r = sorted(port_returns)

        # VaR (losses are negative returns)
        var_95 = -_percentile(sorted_r, (1 - self._c95) * 100)
        var_99 = -_percentile(sorted_r, (1 - self._c99) * 100)

        # CVaR / Expected Shortfall (mean of worst losses)
        cutoff_95 = _percentile(sorted_r, (1 - self._c95) * 100)
        tail_95 = [r for r in sorted_r if r <= cutoff_95]
        cvar_95 = -_mean(tail_95) if tail_95 else var_95

        # Portfolio volatility (annualised from daily)
        vol = _std(port_returns) * math.sqrt(252)

        # Max expected drawdown — approximation: 2 * daily_std * sqrt(T)
        daily_std = _std(port_returns)
        max_dd = 2.0 * daily_std * math.sqrt(max(len(port_returns), 1))
        max_dd = min(1.0, max_dd)

        # Risk contribution: VaR_i / sum(VaR_i)
        risk_contribution = self._risk_contribution(returns, weights)

        # Marginal risk contribution
        marginal_risk = self._marginal_risk(returns, weights)

        # Diversification ratio: (weighted sum of individual vols) / portfolio vol
        indiv_vols = {ins: _std(returns.get(ins, [0.0])) for ins in instruments}
        weighted_vol_sum = sum(weights.get(ins, 0.0) * indiv_vols[ins] for ins in instruments)
        port_daily_vol = _std(port_returns) if port_returns else 0.0
        div_ratio = weighted_vol_sum / max(port_daily_vol, 1e-9)
        div_ratio = max(1.0, min(10.0, div_ratio))

        # Risk concentration (Herfindahl index of risk contributions)
        rc_values = list(risk_contribution.values())
        hhi = sum(v**2 for v in rc_values) if rc_values else 0.0

        result = PortfolioRisk(
            var_95=round(max(0.0, var_95), 4),
            var_99=round(max(0.0, var_99), 4),
            cvar_95=round(max(0.0, cvar_95), 4),
            expected_shortfall=round(max(0.0, cvar_95), 4),
            max_expected_drawdown=round(max_dd, 4),
            portfolio_volatility=round(vol, 4),
            risk_contribution={k: round(v, 4) for k, v in risk_contribution.items()},
            marginal_risk={k: round(v, 4) for k, v in marginal_risk.items()},
            diversification_ratio=round(div_ratio, 4),
            risk_concentration=round(hhi, 4),
        )

        logger.debug(
            f"PortfolioRisk: var95={result.var_95:.4f} var99={result.var_99:.4f} "
            f"vol={result.portfolio_volatility:.4f}"
        )
        return result

    def _portfolio_returns(
        self,
        returns: dict[str, list[float]],
        weights: dict[str, float],
    ) -> list[float]:
        """Compute weighted portfolio return series (minimum common length)."""
        instruments = list(weights.keys())
        if not instruments:
            return []

        min_len = min(len(returns.get(ins, [])) for ins in instruments)
        if min_len == 0:
            return []

        port = []
        for i in range(min_len):
            r = sum(
                weights.get(ins, 0.0) * returns[ins][i] for ins in instruments if ins in returns
            )
            port.append(r)
        return port

    def _risk_contribution(
        self,
        returns: dict[str, list[float]],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Risk contribution as fraction of total risk."""
        instruments = list(weights.keys())
        indiv_vols = {ins: _std(returns.get(ins, [0.0])) for ins in instruments}
        total_risk = sum(weights.get(ins, 0.0) * indiv_vols[ins] for ins in instruments)
        if total_risk == 0:
            return {ins: 0.0 for ins in instruments}
        return {ins: (weights.get(ins, 0.0) * indiv_vols[ins]) / total_risk for ins in instruments}

    def _marginal_risk(
        self,
        returns: dict[str, list[float]],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Marginal risk contribution — change in portfolio vol per unit weight."""
        instruments = list(weights.keys())
        port_returns = self._portfolio_returns(returns, weights)
        port_vol = _std(port_returns) if port_returns else 0.0
        if port_vol == 0:
            return {ins: 0.0 for ins in instruments}

        marginals = {}
        for ins in instruments:
            ins_returns = returns.get(ins, [])
            if not ins_returns or not port_returns:
                marginals[ins] = 0.0
                continue
            min_len = min(len(ins_returns), len(port_returns))
            cov = _covariance(ins_returns[:min_len], port_returns[:min_len])
            marginals[ins] = cov / max(port_vol, 1e-9)
        return marginals

    def var_parametric(self, mean_r: float, vol: float, confidence: float = 0.95) -> float:
        """Parametric VaR using Normal distribution approximation."""
        # z-scores for common confidence levels
        z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
        return max(0.0, -(mean_r - z * vol))

    def stress_var(self, base_var: float, stress_factor: float = 2.0) -> float:
        """Stressed VaR = base_var * stress_factor."""
        return round(base_var * stress_factor, 4)


def _covariance(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)
