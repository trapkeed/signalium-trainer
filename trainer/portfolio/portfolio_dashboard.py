"""Phase 20 — Portfolio Dashboard.

Generates visualisation data structures for:
  1. Correlation Heatmap
  2. Exposure Charts (asset class, currency, region)
  3. Risk Allocation Chart
  4. Dependency Graph (nodes + edges)
  5. Portfolio Timeline
  6. Optimization Dashboard
  7. Scenario Dashboard
  8. Portfolio Scorecard
"""

from __future__ import annotations

import math

from trainer.portfolio.correlation_engine import CorrelationMatrix
from trainer.portfolio.dependency_graph import DependencyGraphSnapshot
from trainer.portfolio.exposure_engine import ExposureReport
from trainer.portfolio.portfolio_database import PortfolioDatabase
from trainer.portfolio.portfolio_instruments import ModelForecast
from trainer.portfolio.portfolio_scorer import PortfolioMetrics


def _colour(value: float, lo: float = -1.0, hi: float = 1.0) -> str:
    """Map value in [lo,hi] to a colour label."""
    norm = (value - lo) / max(hi - lo, 1e-9)
    if norm >= 0.75:
        return "red"
    if norm >= 0.55:
        return "orange"
    if norm >= 0.45:
        return "grey"
    if norm >= 0.25:
        return "lightblue"
    return "blue"


class PortfolioDashboard:
    """Produces visualisation-ready data for all Phase 20 charts."""

    def __init__(self, db: PortfolioDatabase | None = None) -> None:
        self._db = db

    # ── 1. Correlation Heatmap ─────────────────────────────────────────────

    def correlation_heatmap(self, matrix: CorrelationMatrix | None) -> dict:
        if matrix is None or not matrix.instruments:
            return {"type": "correlation_heatmap", "instruments": [], "cells": []}
        cells = []
        for i, a in enumerate(matrix.instruments):
            for j, b in enumerate(matrix.instruments):
                corr = matrix.get(a, b)
                cells.append(
                    {
                        "row": i,
                        "col": j,
                        "instrument_a": a,
                        "instrument_b": b,
                        "correlation": round(corr, 4),
                        "colour": _colour(corr, -1.0, 1.0),
                        "label": f"{corr:+.2f}",
                    }
                )
        return {
            "type": "correlation_heatmap",
            "instruments": matrix.instruments,
            "cells": cells,
            "high_correlation_pairs": [
                {"a": a, "b": b, "correlation": round(c, 4)}
                for a, b, c in matrix.high_correlation_pairs(0.7)
            ],
        }

    # ── 2. Exposure Charts ─────────────────────────────────────────────────

    def exposure_chart(self, exposure: ExposureReport | None) -> dict:
        if exposure is None:
            return {
                "type": "exposure_chart",
                "by_asset_class": [],
                "by_currency": [],
                "by_region": [],
            }

        def _bars(d: dict[str, float], title: str) -> list[dict]:
            return sorted(
                [
                    {
                        "label": k,
                        "value": round(v, 4),
                        "abs_value": round(abs(v), 4),
                        "direction": "long" if v > 0 else ("short" if v < 0 else "neutral"),
                    }
                    for k, v in d.items()
                    if abs(v) > 1e-6
                ],
                key=lambda x: abs(x["value"]),
                reverse=True,
            )

        return {
            "type": "exposure_chart",
            "by_asset_class": _bars(exposure.by_asset_class, "Asset Class"),
            "by_currency": _bars(exposure.by_currency, "Currency"),
            "by_region": _bars(exposure.by_region, "Region"),
            "by_instrument": _bars(exposure.by_instrument, "Instrument"),
            "summary": {
                "gross_exposure": round(exposure.gross_exposure, 4),
                "net_exposure": round(exposure.net_exposure, 4),
                "long_exposure": round(exposure.long_exposure, 4),
                "short_exposure": round(exposure.short_exposure, 4),
                "diversification_ratio": round(exposure.diversification_ratio, 4),
                "alert_count": len(exposure.alerts),
            },
            "alerts": [a.to_dict() for a in exposure.alerts],
        }

    # ── 3. Risk Allocation Chart ───────────────────────────────────────────

    def risk_allocation_chart(
        self,
        forecasts: list[ModelForecast],
        weights: dict[str, float] | None = None,
    ) -> dict:
        if not forecasts:
            return {"type": "risk_allocation_chart", "segments": []}
        if weights is None:
            w = 1.0 / len(forecasts)
            weights = {f.instrument: w for f in forecasts}
        total_var = sum(
            (abs(weights.get(fc.instrument, 0.0)) * fc.expected_risk) ** 2 for fc in forecasts
        )
        total_vol = math.sqrt(max(total_var, 0.0))
        segments = []
        for fc in forecasts:
            w = abs(weights.get(fc.instrument, 0.0))
            risk_contrib = w * fc.expected_risk
            pct = risk_contrib / max(total_vol, 1e-9)
            segments.append(
                {
                    "instrument": fc.instrument,
                    "weight": round(w, 4),
                    "individual_risk": round(fc.expected_risk, 4),
                    "risk_contribution": round(risk_contrib, 4),
                    "risk_pct": round(pct, 4),
                    "direction": fc.direction,
                }
            )
        segments.sort(key=lambda x: x["risk_contribution"], reverse=True)
        return {
            "type": "risk_allocation_chart",
            "total_portfolio_vol": round(total_vol, 4),
            "segments": segments,
        }

    # ── 4. Dependency Graph ────────────────────────────────────────────────

    def dependency_graph_chart(self, graph: DependencyGraphSnapshot | None) -> dict:
        if graph is None:
            return {"type": "dependency_graph", "nodes": [], "edges": [], "clusters": []}
        return {
            "type": "dependency_graph",
            "nodes": graph.node_list(),
            "edges": graph.edge_list(),
            "clusters": graph.cluster_list(),
            "stats": {
                "n_nodes": len(graph.nodes),
                "n_edges": len(graph.edges),
                "n_clusters": len(graph.clusters),
            },
        }

    # ── 5. Portfolio Timeline ──────────────────────────────────────────────

    def portfolio_timeline(self, experiment_id: str | None = None) -> dict:
        if self._db is None:
            return {"type": "portfolio_timeline", "points": []}
        history = self._db.recommendations_history(limit=50)
        points = [
            {
                "timestamp": r["timestamp"],
                "expected_sharpe": r["expected_sharpe"],
                "expected_return": r["expected_return"],
                "instrument_count": r["instrument_count"],
                "regime": r["regime"],
                "method": r["method"],
            }
            for r in history
        ]
        return {
            "type": "portfolio_timeline",
            "points": list(reversed(points)),
            "count": len(points),
        }

    # ── 6. Optimization Dashboard ──────────────────────────────────────────

    def optimization_dashboard(self) -> dict:
        if self._db is None:
            return {"type": "optimization_dashboard", "history": [], "best_by_objective": {}}
        history = self._db.optimization_history(limit=50)
        by_obj: dict[str, dict] = {}
        for r in history:
            obj = r["objective"]
            if obj not in by_obj or r["expected_sharpe"] > by_obj[obj]["expected_sharpe"]:
                by_obj[obj] = r
        return {
            "type": "optimization_dashboard",
            "history": history,
            "best_by_objective": by_obj,
            "objective_comparison": [
                {
                    "objective": k,
                    "sharpe": round(v["expected_sharpe"], 4),
                    "return": round(v["expected_return"], 4),
                    "volatility": round(v["expected_volatility"], 4),
                }
                for k, v in by_obj.items()
            ],
        }

    # ── 7. Scenario Dashboard ──────────────────────────────────────────────

    def scenario_dashboard(self) -> dict:
        if self._db is None:
            return {"type": "scenario_dashboard", "scenarios": [], "resilience_ranking": []}
        history = self._db.scenario_history(limit=50)
        resilience_ranking = sorted(history, key=lambda r: r["resilience_score"], reverse=True)
        worst = min(history, key=lambda r: r["stressed_sharpe"]) if history else None
        best = max(history, key=lambda r: r["stressed_sharpe"]) if history else None
        return {
            "type": "scenario_dashboard",
            "scenarios": history,
            "resilience_ranking": [
                {
                    "scenario_type": r["scenario_type"],
                    "resilience_score": r["resilience_score"],
                    "stressed_sharpe": r["stressed_sharpe"],
                }
                for r in resilience_ranking[:10]
            ],
            "worst_scenario": worst,
            "best_scenario": best,
            "average_resilience": (
                round(sum(r["resilience_score"] for r in history) / len(history), 4)
                if history
                else None
            ),
        }

    # ── 8. Portfolio Scorecard ─────────────────────────────────────────────

    def portfolio_scorecard(
        self,
        metrics: PortfolioMetrics,
        exposure: ExposureReport | None = None,
    ) -> dict:
        def _grade_value(v: float, lo: float, hi: float, higher_better: bool = True) -> str:
            norm = (v - lo) / max(hi - lo, 1e-9)
            norm = max(0.0, min(1.0, norm))
            if not higher_better:
                norm = 1.0 - norm
            if norm >= 0.80:
                return "Exceptional"
            if norm >= 0.60:
                return "Strong"
            if norm >= 0.40:
                return "Adequate"
            if norm >= 0.20:
                return "Weak"
            return "Critical"

        cards = [
            {
                "metric": "Expected Sharpe",
                "value": round(metrics.expected_sharpe, 2),
                "grade": _grade_value(metrics.expected_sharpe, -0.5, 2.5),
            },
            {
                "metric": "Expected Return",
                "value": f"{metrics.expected_return:.2%}",
                "grade": _grade_value(metrics.expected_return, -0.05, 0.30),
            },
            {
                "metric": "Expected Volatility",
                "value": f"{metrics.expected_volatility:.2%}",
                "grade": _grade_value(metrics.expected_volatility, 0.05, 0.50, False),
            },
            {
                "metric": "Expected Drawdown",
                "value": f"{metrics.expected_drawdown:.2%}",
                "grade": _grade_value(metrics.expected_drawdown, 0.05, 0.50, False),
            },
            {
                "metric": "Diversification Score",
                "value": round(metrics.diversification_score, 3),
                "grade": _grade_value(metrics.diversification_score, 0.0, 1.0),
            },
            {
                "metric": "Correlation Risk",
                "value": round(metrics.correlation_risk, 3),
                "grade": _grade_value(metrics.correlation_risk, 0.0, 1.0, False),
            },
            {
                "metric": "Concentration Score",
                "value": round(metrics.concentration_score, 3),
                "grade": _grade_value(metrics.concentration_score, 0.0, 1.0, False),
            },
            {
                "metric": "Weighted Confidence",
                "value": round(metrics.weighted_confidence, 3),
                "grade": _grade_value(metrics.weighted_confidence, 0.3, 0.95),
            },
        ]
        if exposure:
            cards.append(
                {
                    "metric": "Exposure Alerts",
                    "value": len(exposure.alerts),
                    "grade": (
                        "Critical"
                        if any(a.severity == "high" for a in exposure.alerts)
                        else ("Adequate" if exposure.alerts else "Strong")
                    ),
                }
            )

        return {
            "type": "portfolio_scorecard",
            "overall_grade": metrics.grade(),
            "summary": metrics.summary(),
            "cards": cards,
            "instrument_count": metrics.instrument_count,
            "long_short_breakdown": {
                "long": metrics.long_count,
                "short": metrics.short_count,
                "neutral": metrics.neutral_count,
            },
        }

    # ── Build All ──────────────────────────────────────────────────────────

    def build_all(
        self,
        forecasts: list[ModelForecast],
        weights: dict[str, float] | None = None,
        metrics: PortfolioMetrics | None = None,
        exposure: ExposureReport | None = None,
        matrix: CorrelationMatrix | None = None,
        graph: DependencyGraphSnapshot | None = None,
    ) -> dict:
        return {
            "correlation_heatmap": self.correlation_heatmap(matrix),
            "exposure_chart": self.exposure_chart(exposure),
            "risk_allocation": self.risk_allocation_chart(forecasts, weights),
            "dependency_graph": self.dependency_graph_chart(graph),
            "portfolio_timeline": self.portfolio_timeline(),
            "optimization_dashboard": self.optimization_dashboard(),
            "scenario_dashboard": self.scenario_dashboard(),
            "portfolio_scorecard": self.portfolio_scorecard(metrics, exposure) if metrics else {},
        }
