"""Phase 19.2 — IRSE Score Dashboards.

Generates structured dashboard data for the Research Score system.

Seven dashboard types:

  1. research_score_timeline     — composite score over time
  2. radar_chart                 — 8-axis category radar for one experiment
  3. category_breakdown          — bar chart data: all categories for N experiments
  4. benchmark_comparison        — side-by-side category scores for compared models
  5. leaderboard_dashboard       — top-N table with rank, grade, score
  6. score_heatmap               — experiment × category score matrix
  7. historical_evolution        — champion score progression over time

All dashboards return JSON-serialisable dicts.

Usage::

    dash = IRSEScoreDashboard(scorer, leaderboard, trend_tracker, benchmarker)
    timeline = dash.research_score_timeline()
    radar = dash.radar_chart("exp_001")
    all_db = dash.build_all()
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# IRSEScoreDashboard
# ---------------------------------------------------------------------------


class IRSEScoreDashboard:
    """Generates IRSE dashboard payloads."""

    def __init__(
        self,
        leaderboard: Any,
        trend_tracker: Any,
        benchmarker: Any,
        database: Any,
    ) -> None:
        self._lb = leaderboard
        self._trends = trend_tracker
        self._bm = benchmarker
        self._db = database

    # ------------------------------------------------------------------
    # 1. Research Score Timeline
    # ------------------------------------------------------------------

    def research_score_timeline(
        self,
        experiment_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Composite score over time — suitable for a line chart."""
        history = self._db.get_score_history(experiment_id, limit=limit)
        series = [
            {
                "x": snap["recorded_at"],
                "y": snap["composite"],
                "grade": snap["grade_label"],
                "name": snap["name"],
                "is_champion": snap["is_champion"],
            }
            for snap in history
        ]
        return {
            "type": "research_score_timeline",
            "title": "Research Score Timeline",
            "experiment_id": experiment_id,
            "series": series,
            "x_label": "Date",
            "y_label": "Research Score (0–100)",
            "generated_at": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # 2. Radar Chart
    # ------------------------------------------------------------------

    def radar_chart(self, experiment_id: str) -> dict[str, Any]:
        """8-axis radar chart for one experiment's category scores."""
        entry = self._lb.get_by_experiment(experiment_id)
        if entry is None:
            return {
                "type": "radar_chart",
                "error": f"Experiment not in leaderboard: {experiment_id}",
            }

        axes = [
            {"axis": "Validation", "key": "validation_quality_score"},
            {"axis": "Trading", "key": "trading_performance_score"},
            {"axis": "Robustness", "key": "robustness_score"},
            {"axis": "Calibration", "key": "calibration_score"},
            {"axis": "Generalization", "key": "generalization_score"},
            {"axis": "Efficiency", "key": "efficiency_score"},
            {"axis": "Explainability", "key": "explainability_score"},
            {"axis": "Operational", "key": "operational_readiness_score"},
        ]

        points = [
            {
                "axis": ax["axis"],
                "value": entry.get(ax["key"], 0.0),
                "max": 100.0,
            }
            for ax in axes
        ]

        return {
            "type": "radar_chart",
            "title": f"Category Radar — {entry.get('name', experiment_id)}",
            "experiment_id": experiment_id,
            "composite_score": entry.get("composite", 0.0),
            "grade_label": entry.get("grade_label", ""),
            "axes": points,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # 3. Category Breakdown
    # ------------------------------------------------------------------

    def category_breakdown(self, top_n: int = 10) -> dict[str, Any]:
        """Bar chart data: category scores for top-N experiments."""
        top_entries = self._lb.top(top_n)
        categories = [
            "validation_quality_score",
            "trading_performance_score",
            "robustness_score",
            "calibration_score",
            "generalization_score",
            "efficiency_score",
            "explainability_score",
            "operational_readiness_score",
        ]
        series = []
        for cat_key in categories:
            label = cat_key.replace("_score", "").replace("_", " ").title()
            series.append(
                {
                    "category": label,
                    "values": [
                        {
                            "experiment": e.get("name", e.get("experiment_id", "")),
                            "score": e.get(cat_key, 0.0),
                        }
                        for e in top_entries
                    ],
                }
            )

        return {
            "type": "category_breakdown",
            "title": f"Category Breakdown — Top {top_n} Experiments",
            "top_n": top_n,
            "series": series,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # 4. Benchmark Comparison
    # ------------------------------------------------------------------

    def benchmark_comparison(self, subject_id: str | None = None) -> dict[str, Any]:
        """Side-by-side category scores across all benchmark roles."""
        report = self._bm.full_benchmark_report()
        records = report.get("ranking", [])

        categories = [
            "validation_quality",
            "trading_performance",
            "robustness",
            "calibration",
            "generalization",
            "efficiency",
            "explainability",
            "operational_readiness",
        ]

        models_data = []
        for rec in records[:8]:
            cat_scores = rec.get("category_scores", {})
            models_data.append(
                {
                    "experiment_id": rec["experiment_id"],
                    "name": rec.get("name", rec["experiment_id"]),
                    "role": rec.get("role", ""),
                    "composite": rec.get("composite", 0.0),
                    "category_scores": {cat: cat_scores.get(cat, 0.0) for cat in categories},
                }
            )

        return {
            "type": "benchmark_comparison",
            "title": "Benchmark Comparison",
            "subject_id": subject_id,
            "champion_id": report.get("champion_id"),
            "models": models_data,
            "categories": categories,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # 5. Leaderboard Dashboard
    # ------------------------------------------------------------------

    def leaderboard_dashboard(self, top_n: int = 20) -> dict[str, Any]:
        """Top-N leaderboard table with rank, grade, score, categories."""
        report = self._lb.leaderboard_report(top_n=top_n)
        return {
            "type": "leaderboard",
            "title": f"IRSE Leaderboard — Top {top_n}",
            **report,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # 6. Score Heatmap
    # ------------------------------------------------------------------

    def score_heatmap(self, top_n: int = 20) -> dict[str, Any]:
        """Experiment × Category score matrix for a heatmap visualisation."""
        entries = self._lb.top(top_n)
        categories = [
            "validation_quality_score",
            "trading_performance_score",
            "robustness_score",
            "calibration_score",
            "generalization_score",
            "efficiency_score",
            "explainability_score",
            "operational_readiness_score",
        ]
        cat_labels = [
            "Validation",
            "Trading",
            "Robustness",
            "Calibration",
            "Generalization",
            "Efficiency",
            "Explainability",
            "Operational",
        ]

        rows = []
        for e in entries:
            rows.append(
                {
                    "experiment": e.get("name", e.get("experiment_id", "")),
                    "composite": e.get("composite", 0.0),
                    "scores": [e.get(cat, 0.0) for cat in categories],
                }
            )

        return {
            "type": "score_heatmap",
            "title": f"Score Heatmap — Top {top_n}",
            "category_labels": cat_labels,
            "rows": rows,
            "scale": {"min": 0.0, "max": 100.0},
            "generated_at": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # 7. Historical Evolution
    # ------------------------------------------------------------------

    def historical_evolution(self, limit: int = 200) -> dict[str, Any]:
        """Champion score progression over time — suitable for an area chart."""
        history = self._db.get_score_history(limit=limit)
        champion_snapshots = [
            {
                "x": snap["recorded_at"],
                "y": snap["composite"],
                "name": snap["name"],
                "grade": snap["grade_label"],
            }
            for snap in history
            if snap.get("is_champion")
        ]
        all_series = [{"x": snap["recorded_at"], "y": snap["composite"]} for snap in history]

        dist = self._trends.score_distribution()

        return {
            "type": "historical_evolution",
            "title": "Historical Score Evolution",
            "champion_series": champion_snapshots,
            "all_series": all_series,
            "score_distribution": dist,
            "champion_changes": self._trends.champion_timeline(),
            "generated_at": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # Build all
    # ------------------------------------------------------------------

    def build_all(
        self,
        top_n: int = 10,
        sample_experiment_id: str | None = None,
    ) -> dict[str, Any]:
        """Build all 7 dashboards and return in a bundle."""
        dashboards: dict[str, Any] = {
            "research_score_timeline": self.research_score_timeline(limit=top_n * 3),
            "category_breakdown": self.category_breakdown(top_n=top_n),
            "benchmark_comparison": self.benchmark_comparison(),
            "leaderboard": self.leaderboard_dashboard(top_n=top_n),
            "score_heatmap": self.score_heatmap(top_n=top_n),
            "historical_evolution": self.historical_evolution(),
        }
        if sample_experiment_id:
            dashboards["radar_chart"] = self.radar_chart(sample_experiment_id)
        return {
            "dashboards": dashboards,
            "count": len(dashboards),
            "generated_at": datetime.now(UTC).isoformat(),
        }
