"""Phase 21.6 — Memory Dashboard.

8 dashboard views:
1. Decision Timeline
2. Decision Tree
3. Decision Quality Distribution
4. Failure Analysis
5. Counterfactual Explorer
6. Knowledge Graph
7. Learning Timeline
8. Decision Heatmaps
"""

from __future__ import annotations

import collections
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_models import DecisionRecord
from trainer.memory.decision_types import NEGATIVE_OUTCOMES, POSITIVE_OUTCOMES


def _bucket(ts: float, granularity: str = "day") -> str:
    import time

    t = time.gmtime(ts)
    if granularity == "hour":
        return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d} {t.tm_hour:02d}:00"
    if granularity == "month":
        return f"{t.tm_year}-{t.tm_mon:02d}"
    return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"


class MemoryDashboard:
    """Generates all 8 decision memory dashboard views."""

    # ── 1. Decision Timeline ──────────────────────────────────────────────────

    def decision_timeline(
        self,
        records: list[DecisionRecord],
        granularity: str = "day",
    ) -> dict:
        buckets: dict[str, dict] = {}
        for r in sorted(records, key=lambda r: r.timestamp):
            b = _bucket(r.timestamp, granularity)
            if b not in buckets:
                buckets[b] = {"total": 0, "by_type": {}, "with_outcome": 0}
            buckets[b]["total"] += 1
            dt = r.decision_type.value
            buckets[b]["by_type"][dt] = buckets[b]["by_type"].get(dt, 0) + 1
            if r.has_outcome:
                buckets[b]["with_outcome"] += 1

        return {
            "view": "DECISION_TIMELINE",
            "granularity": granularity,
            "periods": list(buckets.keys()),
            "data": buckets,
            "total": len(records),
        }

    # ── 2. Decision Tree ──────────────────────────────────────────────────────

    def decision_tree(self, records: list[DecisionRecord]) -> dict:
        """Build a tree showing decision flow per instrument."""
        tree: dict[str, Any] = {}
        for r in records:
            instr = r.instrument
            dtype = r.decision_type.value
            if instr not in tree:
                tree[instr] = {}
            if dtype not in tree[instr]:
                tree[instr][dtype] = {"count": 0, "outcomes": {}}
            tree[instr][dtype]["count"] += 1
            if r.has_outcome:
                ot = r.outcome.outcome_type.value
                tree[instr][dtype]["outcomes"][ot] = tree[instr][dtype]["outcomes"].get(ot, 0) + 1

        return {
            "view": "DECISION_TREE",
            "instruments": list(tree.keys()),
            "tree": tree,
            "total": len(records),
        }

    # ── 3. Decision Quality Distribution ─────────────────────────────────────

    def quality_distribution(self, records: list[DecisionRecord]) -> dict:
        scored = [r for r in records if r.has_score]
        grade_dist: dict[str, int] = collections.Counter(r.score.grade.value for r in scored)
        bins = [0.0, 0.2, 0.4, 0.55, 0.70, 0.85, 1.0]
        bin_labels = ["0-0.20", "0.20-0.40", "0.40-0.55", "0.55-0.70", "0.70-0.85", "0.85-1.0"]
        bin_counts = [0] * (len(bins) - 1)
        for r in scored:
            v = r.score.overall
            for i in range(len(bin_counts)):
                if bins[i] <= v < bins[i + 1]:
                    bin_counts[i] += 1
                    break
            else:
                bin_counts[-1] += 1

        avg_q = round(sum(r.score.overall for r in scored) / len(scored), 4) if scored else 0.0

        return {
            "view": "QUALITY_DISTRIBUTION",
            "total_scored": len(scored),
            "total_unscored": len(records) - len(scored),
            "grade_distribution": dict(grade_dist),
            "score_histogram": {"labels": bin_labels, "counts": bin_counts},
            "average_quality": avg_q,
        }

    # ── 4. Failure Analysis ───────────────────────────────────────────────────

    def failure_analysis(self, records: list[DecisionRecord]) -> dict:
        completed = [r for r in records if r.has_outcome]
        failures = [r for r in completed if r.outcome.outcome_type in NEGATIVE_OUTCOMES]

        by_regime = collections.Counter(r.snapshot.market_regime for r in failures)
        by_session = collections.Counter(r.snapshot.session for r in failures)
        by_type = collections.Counter(r.decision_type.value for r in failures)
        by_outcome = collections.Counter(r.outcome.outcome_type.value for r in failures)
        sl_hits = [r for r in failures if r.outcome.hit_sl]
        overconf = [r for r in failures if r.snapshot.forecast_confidence > 0.75]

        failure_rate = round(len(failures) / len(completed), 4) if completed else 0.0

        return {
            "view": "FAILURE_ANALYSIS",
            "total_failures": len(failures),
            "failure_rate": failure_rate,
            "by_regime": dict(by_regime),
            "by_session": dict(by_session),
            "by_type": dict(by_type),
            "by_outcome": dict(by_outcome),
            "sl_hit_count": len(sl_hits),
            "overconfident_failures": len(overconf),
            "overconfidence_rate": round(len(overconf) / len(failures), 4) if failures else 0.0,
        }

    # ── 5. Counterfactual Explorer ────────────────────────────────────────────

    def counterfactual_explorer(self, records: list[DecisionRecord]) -> dict:
        with_cf = [r for r in records if r.counterfactuals]
        if not with_cf:
            return {"view": "COUNTERFACTUAL_EXPLORER", "total": 0, "scenarios": []}

        scenario_stats: dict[str, dict] = {}
        for r in with_cf:
            for cf in r.counterfactuals:
                name = cf.alternative_action
                if name not in scenario_stats:
                    scenario_stats[name] = {
                        "count": 0,
                        "better_count": 0,
                        "avg_delta_pnl": 0.0,
                        "_pnls": [],
                    }
                scenario_stats[name]["count"] += 1
                if cf.better_than_actual:
                    scenario_stats[name]["better_count"] += 1
                scenario_stats[name]["_pnls"].append(cf.delta_pnl)

        for name, d in scenario_stats.items():
            pnls = d.pop("_pnls", [])
            d["avg_delta_pnl"] = round(sum(pnls) / len(pnls), 5) if pnls else 0.0
            d["better_rate"] = round(d["better_count"] / d["count"], 4)

        return {
            "view": "COUNTERFACTUAL_EXPLORER",
            "total": len(with_cf),
            "total_scenarios": sum(len(r.counterfactuals) for r in with_cf),
            "scenario_summary": scenario_stats,
        }

    # ── 6. Knowledge Graph ────────────────────────────────────────────────────

    def knowledge_graph(self, records: list[DecisionRecord]) -> dict:
        nodes = []
        edges = []
        seen_instruments: dict[str, int] = {}
        seen_regimes: dict[str, int] = {}

        for r in records:
            nodes.append(
                {
                    "id": r.decision_id[:8],
                    "type": "DECISION",
                    "label": f"{r.decision_type.value}@{r.instrument}",
                }
            )
            instr = r.instrument
            if instr not in seen_instruments:
                seen_instruments[instr] = len(nodes)
                nodes.append({"id": f"instr_{instr}", "type": "INSTRUMENT", "label": instr})
            edges.append(
                {
                    "from": r.decision_id[:8],
                    "to": f"instr_{instr}",
                    "rel": "ON",
                }
            )
            reg = r.snapshot.market_regime
            if reg not in seen_regimes:
                seen_regimes[reg] = len(nodes)
                nodes.append({"id": f"reg_{reg}", "type": "REGIME", "label": reg})
            edges.append(
                {
                    "from": r.decision_id[:8],
                    "to": f"reg_{reg}",
                    "rel": "DURING",
                }
            )

        return {
            "view": "KNOWLEDGE_GRAPH",
            "nodes": nodes,
            "edges": edges,
        }

    # ── 7. Learning Timeline ──────────────────────────────────────────────────

    def learning_timeline(
        self,
        records: list[DecisionRecord],
        granularity: str = "month",
    ) -> dict:
        scored_records = sorted(
            [r for r in records if r.has_score],
            key=lambda r: r.timestamp,
        )
        if not scored_records:
            return {"view": "LEARNING_TIMELINE", "periods": [], "data": {}}

        periods: dict[str, dict] = {}
        for r in scored_records:
            b = _bucket(r.timestamp, granularity)
            if b not in periods:
                periods[b] = {"quality_scores": [], "win_count": 0, "total": 0}
            periods[b]["quality_scores"].append(r.score.overall)
            periods[b]["total"] += 1
            if r.has_outcome and r.outcome.outcome_type in POSITIVE_OUTCOMES:
                periods[b]["win_count"] += 1

        timeline = {}
        for b, d in periods.items():
            n = len(d["quality_scores"])
            timeline[b] = {
                "avg_quality": round(sum(d["quality_scores"]) / n, 4),
                "win_rate": round(d["win_count"] / d["total"], 4) if d["total"] else 0.0,
                "total": d["total"],
            }

        return {
            "view": "LEARNING_TIMELINE",
            "granularity": granularity,
            "periods": list(timeline.keys()),
            "data": timeline,
        }

    # ── 8. Decision Heatmaps ──────────────────────────────────────────────────

    def decision_heatmaps(self, records: list[DecisionRecord]) -> dict:
        """Win rate heatmap: session × market_regime."""
        completed = [r for r in records if r.has_outcome]
        grid: dict[str, dict[str, dict]] = {}

        for r in completed:
            sess = r.snapshot.session
            reg = r.snapshot.market_regime
            if sess not in grid:
                grid[sess] = {}
            if reg not in grid[sess]:
                grid[sess][reg] = {"wins": 0, "total": 0}
            grid[sess][reg]["total"] += 1
            if r.outcome.outcome_type in POSITIVE_OUTCOMES:
                grid[sess][reg]["wins"] += 1

        # Compute win rates
        for sess in grid:
            for reg in grid[sess]:
                t = grid[sess][reg]["total"]
                w = grid[sess][reg]["wins"]
                grid[sess][reg]["win_rate"] = round(w / t, 4) if t else 0.0

        sessions = sorted(grid.keys())
        regimes = sorted({reg for sess in grid for reg in grid[sess]})

        return {
            "view": "DECISION_HEATMAPS",
            "sessions": sessions,
            "regimes": regimes,
            "grid": grid,
            "total": len(completed),
        }

    def all_dashboards(self, records: list[DecisionRecord]) -> dict:
        return {
            "decision_timeline": self.decision_timeline(records),
            "decision_tree": self.decision_tree(records),
            "quality_distribution": self.quality_distribution(records),
            "failure_analysis": self.failure_analysis(records),
            "counterfactual_explorer": self.counterfactual_explorer(records),
            "knowledge_graph": self.knowledge_graph(records),
            "learning_timeline": self.learning_timeline(records),
            "decision_heatmaps": self.decision_heatmaps(records),
        }
