"""Phase 24 — ICARI Dashboard.

7 dashboard views:
1. Capital Allocation Dashboard
2. Risk Dashboard
3. Exposure Dashboard
4. Drawdown Dashboard
5. Capital Health Dashboard
6. Optimization Dashboard
7. Scenario Analysis Dashboard
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.icari.icari_models import (
    AllocationResult, PortfolioRisk, CapitalHealth,
    DrawdownEvent, OptimizationResult, ScenarioResult, RiskBudget,
)
from trainer.icari.icari_types import AllocationStatus, DrawdownMode


def _bucket(ts: float) -> str:
    t = time.gmtime(ts)
    return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"


class IcariDashboard:
    """Generates the 7 ICARI dashboard views."""

    # ── 1. Capital Allocation ─────────────────────────────────────────────────

    def capital_allocation(self, results: List[AllocationResult]) -> Dict:
        if not results:
            return {"view": "CAPITAL_ALLOCATION", "total": 0, "approved": 0, "denied": 0}
        approved  = [r for r in results if r.status == AllocationStatus.APPROVED]
        reduced   = [r for r in results if r.status == AllocationStatus.REDUCED]
        denied    = [r for r in results if r.status == AllocationStatus.DENIED]
        total_usd = sum(r.allocation_usd for r in approved + reduced)
        avg_conf  = sum(r.confidence for r in results) / len(results)

        by_instrument: Dict[str, float] = {}
        for r in results:
            by_instrument[r.instrument] = by_instrument.get(r.instrument, 0.0) + r.allocation_usd

        return {
            "view":         "CAPITAL_ALLOCATION",
            "total":        len(results),
            "approved":     len(approved),
            "reduced":      len(reduced),
            "denied":       len(denied),
            "approval_rate": round(len(approved) / len(results), 4),
            "total_allocated_usd": round(total_usd, 2),
            "avg_confidence": round(avg_conf, 4),
            "by_instrument": {k: round(v, 2) for k, v in sorted(by_instrument.items(), key=lambda x: -x[1])[:10]},
        }

    # ── 2. Risk Dashboard ─────────────────────────────────────────────────────

    def risk_dashboard(
        self,
        portfolio_risk: PortfolioRisk,
        budgets:        List[RiskBudget],
    ) -> Dict:
        breached   = [b for b in budgets if b.is_breached]
        near_limit = [b for b in budgets if b.utilisation >= 0.80 and not b.is_breached]
        return {
            "view":           "RISK_DASHBOARD",
            "var_95":         round(portfolio_risk.var_95, 4),
            "var_99":         round(portfolio_risk.var_99, 4),
            "cvar_95":        round(portfolio_risk.cvar_95, 4),
            "portfolio_vol":  round(portfolio_risk.portfolio_volatility, 4),
            "div_ratio":      round(portfolio_risk.diversification_ratio, 4),
            "concentration":  round(portfolio_risk.risk_concentration, 4),
            "max_exp_dd":     round(portfolio_risk.max_expected_drawdown, 4),
            "budgets_total":  len(budgets),
            "budgets_breached": len(breached),
            "budgets_near_limit": len(near_limit),
            "breached_types": [b.budget_type.value for b in breached],
        }

    # ── 3. Exposure Dashboard ─────────────────────────────────────────────────

    def exposure_dashboard(
        self,
        weights:     Dict[str, float],
        risk_contrib: Dict[str, float],
        correlations: Optional[Dict[str, float]] = None,
    ) -> Dict:
        return {
            "view":             "EXPOSURE_DASHBOARD",
            "instruments":      len(weights),
            "weights":          {k: round(v, 4) for k, v in sorted(weights.items(), key=lambda x: -x[1])},
            "risk_contribution": {k: round(v, 4) for k, v in sorted(risk_contrib.items(), key=lambda x: -x[1])},
            "avg_correlation":  round(sum(correlations.values()) / max(len(correlations), 1), 4) if correlations else 0.0,
            "max_weight":       round(max(weights.values()), 4) if weights else 0.0,
            "min_weight":       round(min(weights.values()), 4) if weights else 0.0,
            "herfindahl":       round(sum(v ** 2 for v in weights.values()), 4),
        }

    # ── 4. Drawdown Dashboard ─────────────────────────────────────────────────

    def drawdown_dashboard(
        self,
        events:          List[DrawdownEvent],
        current_dd:      float,
        dd_mode:         DrawdownMode,
        risk_multiplier: float,
    ) -> Dict:
        recovered = [e for e in events if e.recovered]
        active    = [e for e in events if not e.recovered]
        max_dd    = max((e.drawdown for e in events), default=current_dd)

        return {
            "view":            "DRAWDOWN_DASHBOARD",
            "current_drawdown": round(current_dd, 4),
            "mode":            dd_mode.value,
            "risk_multiplier": round(risk_multiplier, 4),
            "can_trade":       dd_mode != DrawdownMode.EMERGENCY_LOCKDOWN,
            "max_dd_historical": round(max_dd, 4),
            "total_events":    len(events),
            "recovered":       len(recovered),
            "active_events":   len(active),
            "avg_dd":          round(sum(e.drawdown for e in events) / max(len(events), 1), 4),
        }

    # ── 5. Capital Health Dashboard ───────────────────────────────────────────

    def capital_health_dashboard(self, health: CapitalHealth) -> Dict:
        return {
            "view":             "CAPITAL_HEALTH",
            "grade":            health.grade.value,
            "overall_score":    round(health.overall_score, 4),
            "efficiency":       round(health.efficiency_score, 4),
            "risk_score":       round(health.risk_score, 4),
            "allocation_q":     round(health.allocation_quality, 4),
            "diversification":  round(health.diversification, 4),
            "stability":        round(health.stability, 4),
            "utilisation":      round(health.utilisation_score, 4),
            "alerts":           self._health_alerts(health),
        }

    def _health_alerts(self, h: CapitalHealth) -> List[str]:
        alerts = []
        if h.overall_score < 0.40:
            alerts.append("CRITICAL: Overall health below 40%")
        if h.risk_score < 0.50:
            alerts.append("WARNING: Risk score below 50%")
        if h.diversification < 0.40:
            alerts.append("WARNING: Portfolio under-diversified")
        if h.efficiency_score < 0.40:
            alerts.append("WARNING: Capital efficiency low")
        if h.stability < 0.40:
            alerts.append("WARNING: Portfolio stability low")
        return alerts

    # ── 6. Optimization Dashboard ─────────────────────────────────────────────

    def optimization_dashboard(self, results: List[OptimizationResult]) -> Dict:
        if not results:
            return {"view": "OPTIMIZATION", "runs": 0}
        latest = results[-1]
        best_sharpe = max(results, key=lambda r: r.metrics.get("sharpe", -99))
        return {
            "view":              "OPTIMIZATION",
            "runs":              len(results),
            "latest_objective":  latest.objective.value,
            "latest_converged":  latest.converged,
            "latest_weights":    latest.weights,
            "latest_metrics":    latest.metrics,
            "best_sharpe":       round(best_sharpe.metrics.get("sharpe", 0.0), 4),
            "best_sharpe_obj":   best_sharpe.objective.value,
        }

    # ── 7. Scenario Analysis Dashboard ───────────────────────────────────────

    def scenario_dashboard(self, results: List[ScenarioResult]) -> Dict:
        if not results:
            return {"view": "SCENARIO_ANALYSIS", "scenarios": 0}
        worst   = min(results, key=lambda r: r.resilience_score)
        best    = max(results, key=lambda r: r.resilience_score)
        avg_res = sum(r.resilience_score for r in results) / len(results)
        return {
            "view":             "SCENARIO_ANALYSIS",
            "scenarios_run":    len(results),
            "avg_resilience":   round(avg_res, 4),
            "worst_scenario":   worst.scenario.value,
            "worst_resilience": round(worst.resilience_score, 4),
            "worst_return":     round(worst.portfolio_return, 4),
            "best_scenario":    best.scenario.value,
            "best_resilience":  round(best.resilience_score, 4),
            "all_scenarios":    [
                {"scenario": r.scenario.value, "resilience": round(r.resilience_score, 4),
                 "return": round(r.portfolio_return, 4)}
                for r in sorted(results, key=lambda r: r.resilience_score)
            ],
        }

    def all_dashboards(
        self,
        results:        List[AllocationResult],
        portfolio_risk: PortfolioRisk,
        budgets:        List[RiskBudget],
        weights:        Dict[str, float],
        risk_contrib:   Dict[str, float],
        dd_events:      List[DrawdownEvent],
        current_dd:     float,
        dd_mode:        DrawdownMode,
        risk_mult:      float,
        health:         CapitalHealth,
        opt_results:    List[OptimizationResult],
        scenario_results: List[ScenarioResult],
    ) -> Dict:
        return {
            "capital_allocation": self.capital_allocation(results),
            "risk_dashboard":     self.risk_dashboard(portfolio_risk, budgets),
            "exposure_dashboard": self.exposure_dashboard(weights, risk_contrib),
            "drawdown_dashboard": self.drawdown_dashboard(dd_events, current_dd, dd_mode, risk_mult),
            "capital_health":     self.capital_health_dashboard(health),
            "optimization":       self.optimization_dashboard(opt_results),
            "scenario_analysis":  self.scenario_dashboard(scenario_results),
        }
