"""Phase 24 — Capital Health Engine.

Generates 7 health scores → Overall Capital Health Score:
Capital Efficiency Score, Risk Score, Allocation Quality,
Diversification Score, Portfolio Stability, Capital Utilization Score,
Overall Capital Health Score.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.icari.icari_types import CapitalHealthGrade, health_grade, DrawdownMode
from trainer.icari.icari_models import CapitalHealth, PortfolioRisk


# Score weights (must sum to 1.0)
_HEALTH_WEIGHTS = {
    "efficiency":    0.20,
    "risk":          0.20,
    "allocation":    0.15,
    "diversification": 0.15,
    "stability":     0.15,
    "utilisation":   0.15,
}
assert abs(sum(_HEALTH_WEIGHTS.values()) - 1.0) < 1e-9


class CapitalHealthEngine:
    """Computes composite capital health scores."""

    def __init__(self):
        self._history: List[CapitalHealth] = []

    def score(
        self,
        portfolio_risk:     PortfolioRisk,
        utilisation:        float,            # 0–1 capital utilisation
        drawdown_mode:      DrawdownMode      = DrawdownMode.NORMAL,
        current_drawdown:   float             = 0.0,
        num_instruments:    int               = 1,
        avg_correlation:    float             = 0.0,
        portfolio_vol:      Optional[float]   = None,
        recent_returns:     Optional[List[float]] = None,
        allocation_results: Optional[List[Dict]] = None,
    ) -> CapitalHealth:
        """Compute all health scores and return a CapitalHealth object."""

        efficiency_score = self._efficiency_score(utilisation, portfolio_risk.risk_concentration)
        risk_score       = self._risk_score(portfolio_risk, drawdown_mode, current_drawdown)
        alloc_quality    = self._allocation_quality(allocation_results)
        diversification  = self._diversification_score(portfolio_risk.diversification_ratio, num_instruments, avg_correlation)
        stability        = self._stability_score(portfolio_risk, recent_returns, portfolio_vol)
        util_score       = self._utilisation_score(utilisation)

        overall = (
            _HEALTH_WEIGHTS["efficiency"]     * efficiency_score +
            _HEALTH_WEIGHTS["risk"]           * risk_score +
            _HEALTH_WEIGHTS["allocation"]     * alloc_quality +
            _HEALTH_WEIGHTS["diversification"] * diversification +
            _HEALTH_WEIGHTS["stability"]      * stability +
            _HEALTH_WEIGHTS["utilisation"]    * util_score
        )
        overall = max(0.0, min(1.0, overall))

        health = CapitalHealth(
            efficiency_score   = round(efficiency_score, 4),
            risk_score         = round(risk_score, 4),
            allocation_quality = round(alloc_quality, 4),
            diversification    = round(diversification, 4),
            stability          = round(stability, 4),
            utilisation_score  = round(util_score, 4),
            overall_score      = round(overall, 4),
            grade              = health_grade(overall),
        )
        self._history.append(health)
        logger.debug(f"CapitalHealth: overall={overall:.4f} grade={health.grade.value}")
        return health

    # ── Sub-scores ────────────────────────────────────────────────────────────

    def _efficiency_score(self, utilisation: float, concentration: float) -> float:
        """Peak efficiency at 70% utilisation; penalise high concentration."""
        u_score = 1.0 - abs(utilisation - 0.70) / 0.70
        c_score = 1.0 - min(1.0, concentration * 2)   # HHI→ 0 is good, 1 is bad
        return max(0.0, 0.6 * u_score + 0.4 * c_score)

    def _risk_score(
        self,
        pr:         PortfolioRisk,
        mode:       DrawdownMode,
        drawdown:   float,
    ) -> float:
        """Higher VaR and drawdown → lower risk score."""
        mode_penalty = {
            DrawdownMode.NORMAL:             0.0,
            DrawdownMode.SOFT_LIMIT:         0.10,
            DrawdownMode.HARD_LIMIT:         0.30,
            DrawdownMode.PRESERVATION:       0.40,
            DrawdownMode.RECOVERY:           0.20,
            DrawdownMode.EMERGENCY_LOCKDOWN: 0.60,
        }.get(mode, 0.0)

        var_penalty    = min(0.30, pr.var_95 * 3)
        dd_penalty     = min(0.30, drawdown * 3)

        score = 1.0 - mode_penalty - var_penalty - dd_penalty
        return max(0.0, score)

    def _allocation_quality(self, results: Optional[List[Dict]]) -> float:
        """Quality based on proportion of APPROVED vs DENIED allocations."""
        if not results:
            return 0.70   # neutral if no data
        approved = sum(1 for r in results if r.get("status") == "APPROVED")
        total    = len(results)
        if total == 0:
            return 0.70
        approval_rate = approved / total
        # Average confidence of approved
        confidences = [r.get("confidence", 0.5) for r in results if r.get("status") == "APPROVED"]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        return max(0.0, 0.5 * approval_rate + 0.5 * avg_conf)

    def _diversification_score(
        self,
        div_ratio:   float,
        n_instruments: int,
        avg_corr:    float,
    ) -> float:
        """Higher diversification ratio and more instruments → better score."""
        n_score    = min(1.0, (n_instruments - 1) / 9.0)    # 10+ instruments → full
        dr_score   = min(1.0, (div_ratio - 1.0) / 4.0)      # div_ratio 5 → full
        corr_score = max(0.0, 1.0 - avg_corr)
        return max(0.0, 0.35 * n_score + 0.35 * dr_score + 0.30 * corr_score)

    def _stability_score(
        self,
        pr:           PortfolioRisk,
        recent_r:     Optional[List[float]],
        port_vol:     Optional[float],
    ) -> float:
        """Stable returns and low max_dd → high stability score."""
        vol = port_vol if port_vol is not None else pr.portfolio_volatility
        vol_score = max(0.0, 1.0 - vol * 5)   # 0% vol → 1.0; 20% → 0.0
        dd_score  = max(0.0, 1.0 - pr.max_expected_drawdown * 5)

        if recent_r and len(recent_r) >= 5:
            neg_days  = sum(1 for r in recent_r if r < 0)
            pos_rate  = 1.0 - neg_days / len(recent_r)
        else:
            pos_rate = 0.60

        return max(0.0, 0.40 * vol_score + 0.40 * dd_score + 0.20 * pos_rate)

    def _utilisation_score(self, utilisation: float) -> float:
        """Optimal utilisation ~0.65–0.75; too low = idle, too high = overexposed."""
        return max(0.0, 1.0 - abs(utilisation - 0.70) / 0.70)

    # ── History & trends ─────────────────────────────────────────────────────

    def history(self, last_n: int = 30) -> List[CapitalHealth]:
        return self._history[-last_n:]

    def trend(self) -> Dict:
        h = self._history
        if len(h) < 2:
            return {"trend": "insufficient_data"}
        first = h[0].overall_score
        last  = h[-1].overall_score
        delta = last - first
        direction = "improving" if delta > 0.02 else ("declining" if delta < -0.02 else "stable")
        return {
            "first_score": round(first, 4),
            "last_score":  round(last, 4),
            "delta":       round(delta, 4),
            "direction":   direction,
        }

    def latest(self) -> Optional[CapitalHealth]:
        return self._history[-1] if self._history else None
