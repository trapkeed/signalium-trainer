"""Phase 21 — Execution Dashboard.

Generates chart-ready data structures for 8 dashboard views:
  Execution Dashboard, Risk Dashboard, Trade Timeline,
  Probability Charts, Stop Distribution, Target Distribution,
  Execution Scorecards, Fill Quality Analysis.

All output is pure data (dicts/lists) — no rendering dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.execution_planner import ExecutionPlan


class ExecutionDashboard:
    """Converts a list of ExecutionPlan objects into dashboard data."""

    def __init__(self, plans: Optional[List[ExecutionPlan]] = None):
        self._plans: List[ExecutionPlan] = plans or []

    def add_plan(self, plan: ExecutionPlan) -> None:
        self._plans.append(plan)

    # ── 1. Execution Dashboard ─────────────────────────────────────────────────

    def execution_dashboard(self) -> Dict:
        """High-level summary of all active plans."""
        rows = []
        for p in self._plans:
            rows.append(
                {
                    "instrument": p.instrument,
                    "direction": p.direction,
                    "grade": p.score.grade.value,
                    "score": round(p.score.overall_score, 4),
                    "executable": p.is_executable(),
                    "timing": p.timing.recommendation.value,
                    "entry": round(p.zones.primary_entry, 5),
                    "stop": round(p.stop_loss.initial_stop, 5),
                    "tp1": round(p.take_profit.tp1, 5),
                    "lots": p.sizing.lots,
                    "risk_pct": round(p.sizing.risk_pct, 4),
                }
            )
        rows.sort(key=lambda r: r["score"], reverse=True)

        executable = [r for r in rows if r["executable"]]
        grade_counts = {}
        for r in rows:
            grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1

        return {
            "chart_type": "execution_dashboard",
            "total_plans": len(rows),
            "executable": len(executable),
            "grade_distribution": grade_counts,
            "plans": rows,
        }

    # ── 2. Risk Dashboard ─────────────────────────────────────────────────────

    def risk_dashboard(self) -> Dict:
        """Per-trade risk metrics for portfolio risk monitoring."""
        rows = []
        for p in self._plans:
            rows.append(
                {
                    "instrument": p.instrument,
                    "direction": p.direction,
                    "risk_pct": round(p.sizing.risk_pct, 4),
                    "prob_success": round(p.risk.prob_success, 4),
                    "prob_failure": round(p.risk.prob_failure, 4),
                    "rr_ratio": round(p.risk.risk_reward_ratio, 3),
                    "expected_value": round(p.risk.expected_value, 5),
                    "tail_risk": round(p.risk.tail_risk, 5),
                    "worst_case": round(p.risk.worst_case, 5),
                    "best_case": round(p.risk.best_case, 5),
                    "expected_drawdown": round(p.risk.expected_drawdown, 5),
                    "stop_hit_prob": round(p.stop_loss.expected_hit_probability, 4),
                }
            )

        total_risk = round(sum(r["risk_pct"] for r in rows), 4)
        avg_rr = round(sum(r["rr_ratio"] for r in rows) / max(len(rows), 1), 3)

        return {
            "chart_type": "risk_dashboard",
            "total_portfolio_risk_pct": total_risk,
            "average_rr_ratio": avg_rr,
            "plan_count": len(rows),
            "risks": rows,
        }

    # ── 3. Trade Timeline ─────────────────────────────────────────────────────

    def trade_timeline(self) -> Dict:
        """Timeline of entry / stop / target levels for all plans."""
        events = []
        for i, p in enumerate(self._plans):
            base_t = i * 100  # relative time offset per plan
            events.extend(
                [
                    {
                        "t": base_t,
                        "instrument": p.instrument,
                        "event": "ENTRY_ZONE",
                        "price": round(p.zones.primary_entry, 5),
                    },
                    {
                        "t": base_t + 10,
                        "instrument": p.instrument,
                        "event": "STOP",
                        "price": round(p.stop_loss.initial_stop, 5),
                    },
                    {
                        "t": base_t + 30,
                        "instrument": p.instrument,
                        "event": "TP1",
                        "price": round(p.take_profit.tp1, 5),
                    },
                    {
                        "t": base_t + 60,
                        "instrument": p.instrument,
                        "event": "TP2",
                        "price": round(p.take_profit.tp2, 5),
                    },
                    {
                        "t": base_t + 100,
                        "instrument": p.instrument,
                        "event": "RUNNER",
                        "price": round(p.take_profit.runner_target, 5),
                    },
                ]
            )
        return {
            "chart_type": "trade_timeline",
            "events": events,
            "plan_count": len(self._plans),
        }

    # ── 4. Probability Charts ─────────────────────────────────────────────────

    def probability_charts(self) -> Dict:
        """TP / stop hit probabilities per instrument."""
        rows = []
        for p in self._plans:
            rows.append(
                {
                    "instrument": p.instrument,
                    "prob_success": round(p.risk.prob_success, 4),
                    "prob_failure": round(p.risk.prob_failure, 4),
                    "tp1_prob": round(p.take_profit.tp1_hit_prob, 4),
                    "tp2_prob": round(p.take_profit.tp2_hit_prob, 4),
                    "tp3_prob": round(p.take_profit.tp3_hit_prob, 4),
                    "runner_prob": round(p.take_profit.runner_hit_prob, 4),
                    "stop_prob": round(p.stop_loss.expected_hit_probability, 4),
                    "fill_conf": round(p.order.fill_confidence, 4),
                }
            )
        return {
            "chart_type": "probability_charts",
            "instruments": rows,
        }

    # ── 5. Stop Distribution ──────────────────────────────────────────────────

    def stop_distribution(self) -> Dict:
        """All stop levels across instruments for distribution analysis."""
        rows = []
        for p in self._plans:
            sl = p.stop_loss
            rows.append(
                {
                    "instrument": p.instrument,
                    "direction": p.direction,
                    "initial": round(sl.initial_stop, 5),
                    "adaptive": round(sl.adaptive_stop, 5),
                    "atr": round(sl.atr_stop, 5),
                    "structure": round(sl.structure_stop, 5),
                    "volatility": round(sl.volatility_stop, 5),
                    "emergency": round(sl.emergency_stop, 5),
                    "pips_initial": round(sl.pips_initial, 5),
                    "pips_structure": round(sl.pips_structure, 5),
                    "stop_confidence": round(sl.stop_confidence, 4),
                }
            )
        return {
            "chart_type": "stop_distribution",
            "instruments": rows,
        }

    # ── 6. Target Distribution ────────────────────────────────────────────────

    def target_distribution(self) -> Dict:
        """All take profit levels across instruments."""
        rows = []
        for p in self._plans:
            tp = p.take_profit
            rows.append(
                {
                    "instrument": p.instrument,
                    "direction": p.direction,
                    "tp1": round(tp.tp1, 5),
                    "tp2": round(tp.tp2, 5),
                    "tp3": round(tp.tp3, 5),
                    "runner": round(tp.runner_target, 5),
                    "adaptive": round(tp.adaptive_target, 5),
                    "rr_tp1": tp.rr_tp1,
                    "rr_tp2": tp.rr_tp2,
                    "rr_runner": tp.rr_runner,
                    "best_level": tp.best_expectancy_level(),
                }
            )
        return {
            "chart_type": "target_distribution",
            "instruments": rows,
        }

    # ── 7. Execution Scorecards ───────────────────────────────────────────────

    def execution_scorecards(self) -> Dict:
        """Detailed scorecard per instrument."""
        cards = []
        for p in self._plans:
            s = p.score
            cards.append(
                {
                    "instrument": p.instrument,
                    "direction": p.direction,
                    "grade": s.grade.value,
                    "overall": round(s.overall_score, 4),
                    "forecast_quality": round(s.forecast_quality, 4),
                    "entry_quality": round(s.entry_quality, 4),
                    "risk": round(s.risk_score, 4),
                    "liquidity": round(s.liquidity_score, 4),
                    "spread": round(s.spread_score, 4),
                    "volatility": round(s.volatility_score, 4),
                    "reward": round(s.reward_score, 4),
                    "portfolio_impact": round(s.portfolio_impact, 4),
                    "notes": s.notes,
                    "executable": p.is_executable(),
                    "order_type": p.order.primary_type.value,
                }
            )
        cards.sort(key=lambda c: c["overall"], reverse=True)
        return {
            "chart_type": "execution_scorecards",
            "scorecards": cards,
            "best_instrument": cards[0]["instrument"] if cards else None,
        }

    # ── 8. Fill Quality Analysis ──────────────────────────────────────────────

    def fill_quality_analysis(self) -> Dict:
        """Simulation fill quality and slippage analysis."""
        rows = []
        for p in self._plans:
            sim = p.simulation
            rows.append(
                {
                    "instrument": p.instrument,
                    "direction": p.direction,
                    "expected_fill": round(sim.expected_fill, 5),
                    "slippage": round(sim.expected_slippage, 5),
                    "fill_quality": sim.fill_quality,
                    "expected_pnl_pct": round(sim.expected_pnl_pct, 4),
                    "pnl_std_dev": round(sim.pnl_std_dev, 4),
                    "max_adverse": round(sim.max_adverse_excursion, 5),
                    "max_favourable": round(sim.max_favourable_excursion, 5),
                    "session": p.timing.session,
                }
            )

        quality_counts: Dict[str, int] = {}
        for r in rows:
            q = r["fill_quality"]
            quality_counts[q] = quality_counts.get(q, 0) + 1

        return {
            "chart_type": "fill_quality_analysis",
            "instruments": rows,
            "quality_summary": quality_counts,
        }

    # ── Full dashboard ────────────────────────────────────────────────────────

    def full_dashboard(self) -> Dict:
        """All 8 views in a single response."""
        return {
            "execution_dashboard": self.execution_dashboard(),
            "risk_dashboard": self.risk_dashboard(),
            "trade_timeline": self.trade_timeline(),
            "probability_charts": self.probability_charts(),
            "stop_distribution": self.stop_distribution(),
            "target_distribution": self.target_distribution(),
            "execution_scorecards": self.execution_scorecards(),
            "fill_quality_analysis": self.fill_quality_analysis(),
        }
