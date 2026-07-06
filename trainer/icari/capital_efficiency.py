"""Phase 24 — Capital Efficiency Engine.

Estimates: Capital Utilization, Idle Capital, Risk Efficiency,
Return Efficiency, Exposure Efficiency, Capital Turnover.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class CapitalEfficiencyEngine:
    """Measures how efficiently capital is deployed."""

    def __init__(self, capital: float = 100_000.0):
        self._capital = capital
        self._history: List[Dict] = []

    def analyse(
        self,
        allocated_usd: float,
        at_risk_usd: float,
        realised_pnl: float,
        unrealised_pnl: float,
        num_open_trades: int,
        num_closed_trades: int = 0,
        period_days: float = 1.0,
    ) -> Dict:
        """Compute all capital efficiency metrics."""
        cap = max(self._capital, 1.0)

        # 1. Capital Utilization — fraction of capital allocated
        utilisation = min(1.0, allocated_usd / cap)

        # 2. Idle Capital — fraction sitting uninvested
        idle = max(0.0, 1.0 - utilisation)

        # 3. Risk Efficiency — return per unit of risk taken
        risk_adj_return = 0.0
        if at_risk_usd > 0:
            risk_adj_return = realised_pnl / at_risk_usd

        # 4. Return Efficiency — return on allocated capital
        return_efficiency = 0.0
        if allocated_usd > 0:
            return_efficiency = (realised_pnl + unrealised_pnl) / allocated_usd

        # 5. Exposure Efficiency — how much open pnl per unit of exposure
        exposure_efficiency = 0.0
        if allocated_usd > 0:
            exposure_efficiency = min(1.0, max(-1.0, unrealised_pnl / allocated_usd))

        # 6. Capital Turnover — number of full capital cycles per day
        capital_turnover = 0.0
        if cap > 0 and period_days > 0:
            # Approximated as: (allocated / capital) * (num_closed / period_days)
            if num_closed_trades > 0:
                capital_turnover = (allocated_usd / cap) * (num_closed_trades / period_days)

        result = {
            "capital": round(self._capital, 2),
            "allocated_usd": round(allocated_usd, 2),
            "at_risk_usd": round(at_risk_usd, 2),
            "utilisation": round(utilisation, 4),
            "idle_fraction": round(idle, 4),
            "risk_efficiency": round(risk_adj_return, 4),
            "return_efficiency": round(return_efficiency, 4),
            "exposure_efficiency": round(exposure_efficiency, 4),
            "capital_turnover": round(capital_turnover, 4),
            "num_open_trades": num_open_trades,
            "num_closed_trades": num_closed_trades,
        }

        self._history.append(result)
        logger.debug(f"CapitalEfficiency: utilisation={utilisation:.2%} idle={idle:.2%}")
        return result

    def efficiency_score(self, metrics: Dict) -> float:
        """Aggregate efficiency score 0–1 from metrics dict."""
        u = metrics.get("utilisation", 0.0)  # want ~0.6–0.8
        ri = metrics.get("risk_efficiency", 0.0)  # higher is better
        re = metrics.get("return_efficiency", 0.0)  # higher is better
        ee = metrics.get("exposure_efficiency", 0.0)  # signed

        # Utilisation score: peak at 70%
        u_score = 1.0 - abs(u - 0.70) / 0.70

        # Risk efficiency score: capped
        ri_score = min(1.0, max(0.0, ri * 10 + 0.5))

        # Return efficiency score
        re_score = min(1.0, max(0.0, re * 20 + 0.5))

        # Exposure efficiency
        ee_score = min(1.0, max(0.0, ee + 0.5))

        score = 0.35 * u_score + 0.25 * ri_score + 0.25 * re_score + 0.15 * ee_score
        return round(max(0.0, min(1.0, score)), 4)

    def utilisation_score(self, allocated_usd: float) -> float:
        u = min(1.0, allocated_usd / max(self._capital, 1.0))
        return round(1.0 - abs(u - 0.70) / 0.70, 4)

    def history(self, last_n: int = 30) -> List[Dict]:
        return self._history[-last_n:]

    def trend(self) -> Dict:
        if len(self._history) < 2:
            return {"trend": "insufficient_data"}
        first = self._history[0]["utilisation"]
        last = self._history[-1]["utilisation"]
        delta = last - first
        direction = "improving" if delta > 0.02 else ("declining" if delta < -0.02 else "stable")
        return {
            "utilisation_first": round(first, 4),
            "utilisation_last": round(last, 4),
            "delta": round(delta, 4),
            "direction": direction,
        }
