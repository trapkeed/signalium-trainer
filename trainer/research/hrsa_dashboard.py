"""Phase 19.3 — HRSA Visualisation Dashboard.

Generates text-based visualisations and structured data for all eight
HRSA chart types:

  1. Hierarchical Score Tree    — 3-level score decomposition
  2. Radar Chart                — category profile polygon (ASCII)
  3. Category Heatmap           — colour-coded category grid
  4. Metric Heatmap             — metric-level detail
  5. Score Timeline             — composite score over time
  6. Improvement Timeline       — score deltas between periods
  7. Benchmark Comparison       — side-by-side bar chart
  8. Score Breakdown            — weighted contribution waterfall

All outputs are pure Python dicts — no matplotlib, no heavy deps.
Callers may render the data in any frontend.

Usage::

    db  = HRSADatabase(":memory:")
    api = HRSAApi(db_path=":memory:")
    dashboard = HRSADashboard(api, db)

    tree    = dashboard.hierarchical_tree(hs)
    radar   = dashboard.radar_chart(hs)
    heatmap = dashboard.category_heatmap(hs)
    timeline = dashboard.score_timeline(experiment_id)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Colour scale helper
# ---------------------------------------------------------------------------

def _score_colour(score: float) -> str:
    if score >= 90:
        return "gold"
    if score >= 80:
        return "green"
    if score >= 70:
        return "blue"
    if score >= 60:
        return "cyan"
    if score >= 50:
        return "yellow"
    return "red"


def _ascii_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# HRSADashboard
# ---------------------------------------------------------------------------


class HRSADashboard:
    """Generate structured dashboard data for all 8 HRSA visualisation types."""

    def __init__(
        self,
        api: Optional[Any] = None,
        db: Optional[Any] = None,
    ) -> None:
        self._api = api
        self._db  = db

    # 1. Hierarchical Score Tree ─────────────────────────────────────────────

    def hierarchical_tree(self, hs: Any) -> Dict[str, Any]:
        """3-level score tree suitable for a tree/sunburst visualisation."""
        l2_nodes: List[Dict[str, Any]] = []
        for cat, result in sorted(
            hs.level2_categories.items(),
            key=lambda x: x[1].score, reverse=True
        ):
            if result.metric_count == 0:
                continue
            metric_nodes = [
                {
                    "name": metric,
                    "score": round(score, 1),
                    "colour": _score_colour(score),
                    "bar": _ascii_bar(score, 10),
                }
                for metric, score in sorted(
                    result.metric_scores.items(),
                    key=lambda x: x[1], reverse=True
                )
            ]
            l2_nodes.append({
                "name": result.display_name,
                "category_key": cat,
                "score": round(result.score, 1),
                "confidence": round(result.confidence, 3),
                "weight": result.weight,
                "weighted_contribution": round(result.score * result.weight, 2),
                "grade": result.grade,
                "colour": _score_colour(result.score),
                "bar": _ascii_bar(result.score),
                "metric_count": result.metric_count,
                "metrics": metric_nodes,
            })

        return {
            "type": "hierarchical_tree",
            "root": {
                "name": "Institutional Research Score",
                "score": round(hs.level3_composite, 1),
                "confidence": round(hs.confidence, 3),
                "institutional_grade": hs.institutional_grade,
                "research_grade": hs.research_grade,
                "colour": _score_colour(hs.level3_composite),
                "bar": _ascii_bar(hs.level3_composite),
            },
            "level2_nodes": l2_nodes,
            "total_categories": len(l2_nodes),
            "experiment": hs.name or hs.experiment_id,
        }

    # 2. Radar Chart ─────────────────────────────────────────────────────────

    def radar_chart(
        self,
        hs: Any,
        reference: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Category radar/spider chart data."""
        axes: List[Dict[str, Any]] = []
        for cat, result in hs.level2_categories.items():
            if result.metric_count == 0:
                continue
            entry: Dict[str, Any] = {
                "axis": result.display_name,
                "value": round(result.score, 1),
                "max": 100.0,
                "colour": _score_colour(result.score),
            }
            if reference:
                ref_cat = reference.level2_categories.get(cat)
                if ref_cat and ref_cat.metric_count > 0:
                    entry["reference_value"] = round(ref_cat.score, 1)
            axes.append(entry)

        return {
            "type": "radar_chart",
            "experiment": hs.name or hs.experiment_id,
            "composite": round(hs.level3_composite, 1),
            "grade": hs.institutional_grade,
            "axes": axes,
            "has_reference": reference is not None,
        }

    # 3. Category Heatmap ────────────────────────────────────────────────────

    def category_heatmap(self, hs: Any) -> Dict[str, Any]:
        """Colour-coded 5×3 category grid."""
        cells: List[Dict[str, Any]] = []
        for cat, result in hs.level2_categories.items():
            cells.append({
                "category": result.display_name,
                "score": round(result.score, 1),
                "confidence": round(result.confidence, 3),
                "grade": result.grade,
                "colour": _score_colour(result.score),
                "bar": _ascii_bar(result.score, 10),
                "coverage": round(result.coverage, 2),
                "metric_count": result.metric_count,
            })
        cells.sort(key=lambda c: c["score"], reverse=True)

        return {
            "type": "category_heatmap",
            "experiment": hs.name or hs.experiment_id,
            "composite": round(hs.level3_composite, 1),
            "cells": cells,
            "colour_scale": [
                {"label": "≥90 Exceptional", "colour": "gold"},
                {"label": "≥80 Strong",      "colour": "green"},
                {"label": "≥70 Good",        "colour": "blue"},
                {"label": "≥60 Adequate",    "colour": "cyan"},
                {"label": "≥50 Weak",        "colour": "yellow"},
                {"label": "<50 Critical",    "colour": "red"},
            ],
        }

    # 4. Metric Heatmap ──────────────────────────────────────────────────────

    def metric_heatmap(
        self, hs: Any, category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Metric-level heatmap, optionally filtered to one category."""
        rows: List[Dict[str, Any]] = []
        cats = hs.level2_categories
        if category:
            cats = {k: v for k, v in cats.items() if k == category}

        for cat, result in cats.items():
            for metric, score in sorted(
                result.metric_scores.items(),
                key=lambda x: x[1], reverse=True
            ):
                raw = result.raw_values.get(metric)
                rows.append({
                    "category": result.display_name,
                    "metric": metric,
                    "normalised_score": round(score, 1),
                    "raw_value": raw,
                    "colour": _score_colour(score),
                    "bar": _ascii_bar(score, 10),
                })

        return {
            "type": "metric_heatmap",
            "experiment": hs.name or hs.experiment_id,
            "filter_category": category,
            "rows": rows,
            "total_metrics": len(rows),
        }

    # 5. Score Timeline ──────────────────────────────────────────────────────

    def score_timeline(self, experiment_id: str) -> Dict[str, Any]:
        """Composite score history from DB."""
        if self._db is None:
            return {"type": "score_timeline", "error": "No database configured"}
        history = self._db.composite_history(experiment_id)
        if not history:
            return {
                "type": "score_timeline",
                "experiment_id": experiment_id,
                "points": [],
            }
        composites = [r["composite"] for r in history]
        return {
            "type": "score_timeline",
            "experiment_id": experiment_id,
            "points": [
                {
                    "date": r["scored_at"][:10],
                    "composite": round(r["composite"], 1),
                    "institutional_grade": r["institutional_grade"],
                    "colour": _score_colour(r["composite"]),
                    "bar": _ascii_bar(r["composite"], 10),
                }
                for r in history
            ],
            "latest": round(composites[-1], 1),
            "peak": round(max(composites), 1),
            "trough": round(min(composites), 1),
        }

    # 6. Improvement Timeline ────────────────────────────────────────────────

    def improvement_timeline(self, experiment_id: str) -> Dict[str, Any]:
        """Period-over-period composite deltas."""
        if self._db is None:
            return {"type": "improvement_timeline", "error": "No database configured"}
        history = self._db.composite_history(experiment_id)
        if len(history) < 2:
            return {
                "type": "improvement_timeline",
                "experiment_id": experiment_id,
                "deltas": [],
            }
        deltas = []
        for i in range(1, len(history)):
            delta = history[i]["composite"] - history[i - 1]["composite"]
            deltas.append({
                "period": f"{history[i-1]['scored_at'][:10]} → {history[i]['scored_at'][:10]}",
                "delta": round(delta, 2),
                "direction": "up" if delta > 0 else ("flat" if delta == 0 else "down"),
                "colour": "green" if delta > 0 else ("gray" if delta == 0 else "red"),
            })
        return {
            "type": "improvement_timeline",
            "experiment_id": experiment_id,
            "deltas": deltas,
            "total_improvement": round(
                sum(d["delta"] for d in deltas), 2
            ),
            "avg_improvement": round(
                sum(d["delta"] for d in deltas) / len(deltas), 3
            ),
        }

    # 7. Benchmark Comparison ────────────────────────────────────────────────

    def benchmark_comparison(
        self,
        subject_hs: Any,
        benchmark_records: List[Any],
    ) -> Dict[str, Any]:
        """Side-by-side bar chart data for subject vs benchmarks."""
        bars: List[Dict[str, Any]] = []

        # Subject bar
        bars.append({
            "name": subject_hs.name or subject_hs.experiment_id,
            "role": "subject",
            "composite": round(subject_hs.level3_composite, 1),
            "colour": "blue",
            "bar": _ascii_bar(subject_hs.level3_composite),
        })

        for rec in benchmark_records:
            delta = subject_hs.level3_composite - rec.composite
            bars.append({
                "name": rec.name or rec.experiment_id,
                "role": rec.role,
                "composite": round(rec.composite, 1),
                "delta_vs_subject": round(delta, 2),
                "colour": _score_colour(rec.composite),
                "bar": _ascii_bar(rec.composite),
            })

        return {
            "type": "benchmark_comparison",
            "experiment": subject_hs.name or subject_hs.experiment_id,
            "subject_score": round(subject_hs.level3_composite, 1),
            "bars": sorted(bars, key=lambda b: b["composite"], reverse=True),
            "beats_all": all(
                subject_hs.level3_composite >= rec.composite
                for rec in benchmark_records
            ),
        }

    # 8. Score Breakdown (waterfall) ─────────────────────────────────────────

    def score_breakdown(self, hs: Any) -> Dict[str, Any]:
        """Weighted contribution waterfall from 0 to composite."""
        segments: List[Dict[str, Any]] = []
        running = 0.0

        sorted_cats = sorted(
            [
                (cat, r)
                for cat, r in hs.level2_categories.items()
                if r.metric_count > 0
            ],
            key=lambda x: x[1].weighted_score, reverse=True,
        )

        for cat, result in sorted_cats:
            effective_w = result.weight * result.coverage
            total_w = hs.total_weight_used
            contribution = (
                (result.score * effective_w / total_w)
                if total_w > 0 else 0.0
            )
            segments.append({
                "category": result.display_name,
                "category_score": round(result.score, 1),
                "weight": result.weight,
                "effective_weight": round(effective_w, 4),
                "contribution": round(contribution, 2),
                "cumulative": round(running + contribution, 2),
                "colour": _score_colour(result.score),
                "bar": _ascii_bar(contribution * 5, 10),  # scaled for visibility
            })
            running += contribution

        return {
            "type": "score_breakdown",
            "experiment": hs.name or hs.experiment_id,
            "composite": round(hs.level3_composite, 1),
            "segments": segments,
            "total_contribution": round(running, 2),
        }

    # ── All dashboards ────────────────────────────────────────────────────────

    def build_all(
        self,
        hs: Any,
        experiment_id: Optional[str] = None,
        reference: Optional[Any] = None,
        benchmark_records: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        exp_id = experiment_id or hs.experiment_id
        return {
            "hierarchical_tree":    self.hierarchical_tree(hs),
            "radar_chart":          self.radar_chart(hs, reference),
            "category_heatmap":     self.category_heatmap(hs),
            "metric_heatmap":       self.metric_heatmap(hs),
            "score_timeline":       self.score_timeline(exp_id),
            "improvement_timeline": self.improvement_timeline(exp_id),
            "benchmark_comparison": self.benchmark_comparison(
                hs, benchmark_records or []
            ),
            "score_breakdown":      self.score_breakdown(hs),
        }
