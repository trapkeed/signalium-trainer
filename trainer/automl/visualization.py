"""Phase 16 — AutoML Visualization Data Builders.

Generates JSON-safe chart data dicts for the AutoML platform.

No matplotlib/plotly dependencies — all outputs are pure Python dicts
that can be rendered by any frontend charting library.

Generates:
  - Optimization history     : score over trials
  - Parameter importance     : feature importance from score correlation
  - Search convergence       : running best score
  - Leaderboard progression  : leaderboard updates over time
  - Experiment timelines     : Gantt-style trial start/end data
  - Resource utilization     : CPU/GPU/RAM usage over time
  - Objective trade-off      : Pareto front 2D scatter data

Usage::

    viz = AutoMLVisualizationBuilder()
    chart = viz.optimization_history(records)
    pareto = viz.objective_tradeoff(records, "sharpe_ratio", "max_drawdown")
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Visualization builder
# ---------------------------------------------------------------------------


class AutoMLVisualizationBuilder:
    """Builds JSON-safe chart data dicts for the AutoML platform."""

    # -----------------------------------------------------------------------
    # Optimization history
    # -----------------------------------------------------------------------

    def optimization_history(
        self, records: list[dict[str, Any]], score_key: str = "primary_score"
    ) -> dict[str, Any]:
        """Trial-by-trial score over the search.

        Returns a line chart data dict with x = trial index, y = score.
        """
        completed = [r for r in records if r.get("status") == "completed"]
        completed.sort(key=lambda r: r.get("created_at", ""))
        points = []
        for i, r in enumerate(completed):
            points.append(
                {
                    "trial_index": i + 1,
                    "trial_id": r.get("trial_id", ""),
                    "score": r.get(score_key) or r.get("composite_research_score", 0.0),
                    "backbone": r.get("backbone", ""),
                    "created_at": r.get("created_at", ""),
                }
            )
        return {
            "chart_type": "line",
            "title": "Optimization History",
            "x_label": "Trial Index",
            "y_label": "Score",
            "series": [{"name": "Trial Score", "data": points}],
        }

    # -----------------------------------------------------------------------
    # Search convergence (running best)
    # -----------------------------------------------------------------------

    def search_convergence(
        self, records: list[dict[str, Any]], score_key: str = "primary_score"
    ) -> dict[str, Any]:
        """Running best score over the search (convergence curve)."""
        completed = [r for r in records if r.get("status") == "completed"]
        completed.sort(key=lambda r: r.get("created_at", ""))
        points = []
        best = None
        for i, r in enumerate(completed):
            score = r.get(score_key) or r.get("composite_research_score", 0.0)
            if score is not None:
                if best is None or score > best:
                    best = score
            points.append(
                {
                    "trial_index": i + 1,
                    "running_best": best,
                    "created_at": r.get("created_at", ""),
                }
            )
        return {
            "chart_type": "line",
            "title": "Search Convergence",
            "x_label": "Trial Index",
            "y_label": "Running Best Score",
            "series": [{"name": "Best Score", "data": points}],
        }

    # -----------------------------------------------------------------------
    # Parameter importance
    # -----------------------------------------------------------------------

    def parameter_importance(
        self, records: list[dict[str, Any]], score_key: str = "primary_score"
    ) -> dict[str, Any]:
        """Estimate parameter importance via Pearson correlation with score.

        Returns a bar chart sorted by absolute correlation.
        """
        completed = [
            r for r in records if r.get("status") == "completed" and r.get(score_key) is not None
        ]
        if len(completed) < 3:
            return {
                "chart_type": "bar",
                "title": "Parameter Importance",
                "x_label": "Parameter",
                "y_label": "|Correlation with Score|",
                "series": [{"name": "Importance", "data": []}],
                "note": "Requires at least 3 completed trials.",
            }

        scores = [r.get(score_key) or 0.0 for r in completed]
        configs = [r.get("config") or {} for r in completed]

        param_names: list[str] = []
        for c in configs:
            for k in c:
                if k not in param_names:
                    param_names.append(k)

        importances = []
        for param in param_names:
            values = []
            valid_scores = []
            for r, c in zip(completed, configs):
                v = c.get(param)
                if isinstance(v, (int, float)) and v is not None:
                    values.append(float(v))
                    valid_scores.append(r.get(score_key) or 0.0)
            if len(values) < 3:
                continue
            corr = _pearson(values, valid_scores)
            importances.append(
                {
                    "parameter": param,
                    "importance": round(abs(corr), 4),
                    "correlation": round(corr, 4),
                }
            )

        importances.sort(key=lambda x: x["importance"], reverse=True)
        return {
            "chart_type": "bar",
            "title": "Parameter Importance",
            "x_label": "Parameter",
            "y_label": "|Correlation with Score|",
            "series": [{"name": "Importance", "data": importances}],
        }

    # -----------------------------------------------------------------------
    # Leaderboard progression
    # -----------------------------------------------------------------------

    def leaderboard_progression(
        self, records: list[dict[str, Any]], top_n: int = 5
    ) -> dict[str, Any]:
        """Show how the top-N models evolved over the search."""
        completed = [r for r in records if r.get("status") == "completed"]
        completed.sort(key=lambda r: r.get("primary_score") or 0.0, reverse=True)
        top = completed[:top_n]
        series = []
        for entry in top:
            series.append(
                {
                    "trial_id": entry.get("trial_id", ""),
                    "backbone": entry.get("backbone", ""),
                    "score": entry.get("primary_score") or 0.0,
                    "rank": completed.index(entry) + 1,
                    "created_at": entry.get("created_at", ""),
                }
            )
        return {
            "chart_type": "bar",
            "title": f"Top-{top_n} Leaderboard",
            "x_label": "Trial",
            "y_label": "Score",
            "series": [{"name": "Top Models", "data": series}],
        }

    # -----------------------------------------------------------------------
    # Experiment timeline (Gantt)
    # -----------------------------------------------------------------------

    def experiment_timeline(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Gantt-style trial timeline data."""
        entries = []
        for r in records:
            start = r.get("created_at", "")
            duration = (r.get("training_duration_seconds") or 0.0) + (
                r.get("validation_duration_seconds") or 0.0
            )
            entries.append(
                {
                    "trial_id": r.get("trial_id", ""),
                    "backbone": r.get("backbone", ""),
                    "status": r.get("status", ""),
                    "start_time": start,
                    "duration_seconds": duration,
                    "pruned": r.get("pruned", False),
                    "score": r.get("primary_score"),
                }
            )
        entries.sort(key=lambda x: x["start_time"])
        return {
            "chart_type": "gantt",
            "title": "Experiment Timeline",
            "x_label": "Time",
            "y_label": "Trial",
            "entries": entries,
        }

    # -----------------------------------------------------------------------
    # Resource utilization
    # -----------------------------------------------------------------------

    def resource_utilization(self, usage_report: dict[str, Any]) -> dict[str, Any]:
        """Build a resource utilization summary chart."""
        return {
            "chart_type": "gauge",
            "title": "Resource Utilization",
            "metrics": {
                "active_workers": {
                    "value": usage_report.get("active_workers", 0),
                    "max": usage_report.get("max_workers", 1),
                    "unit": "workers",
                },
                "peak_cpu_percent": {
                    "value": usage_report.get("peak_cpu_percent", 0.0),
                    "max": 100.0,
                    "unit": "%",
                },
                "peak_ram_mb": {
                    "value": usage_report.get("peak_ram_mb", 0.0),
                    "max": usage_report.get("max_ram_gb", 32.0) * 1024,
                    "unit": "MB",
                },
                "peak_gpu_percent": {
                    "value": usage_report.get("peak_gpu_percent", 0.0),
                    "max": 100.0,
                    "unit": "%",
                },
                "peak_vram_mb": {
                    "value": usage_report.get("peak_vram_mb", 0.0),
                    "max": usage_report.get("gpu_memory_gb", 0.0) * 1024,
                    "unit": "MB",
                },
            },
        }

    # -----------------------------------------------------------------------
    # Objective trade-off (Pareto front)
    # -----------------------------------------------------------------------

    def objective_tradeoff(
        self,
        records: list[dict[str, Any]],
        objective_x: str = "sharpe_ratio",
        objective_y: str = "max_drawdown",
        pareto_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """2D scatter of two objectives, highlighting the Pareto front."""
        completed = [r for r in records if r.get("status") == "completed"]
        points = []
        for i, r in enumerate(completed):
            metrics = r.get("validation_results") or r.get("metrics") or {}
            x_val = _extract_nested(metrics, objective_x)
            y_val = _extract_nested(metrics, objective_y)
            points.append(
                {
                    "trial_id": r.get("trial_id", ""),
                    "backbone": r.get("backbone", ""),
                    "x": x_val,
                    "y": y_val,
                    "score": r.get("primary_score"),
                    "is_pareto": (i in pareto_indices) if pareto_indices else False,
                }
            )
        return {
            "chart_type": "scatter",
            "title": f"Objective Trade-off: {objective_x} vs {objective_y}",
            "x_label": objective_x,
            "y_label": objective_y,
            "series": [{"name": "Trials", "data": points}],
        }

    # -----------------------------------------------------------------------
    # Score distribution
    # -----------------------------------------------------------------------

    def score_distribution(
        self, records: list[dict[str, Any]], n_bins: int = 20, score_key: str = "primary_score"
    ) -> dict[str, Any]:
        """Histogram of trial scores."""
        scores = [r.get(score_key) or 0.0 for r in records if r.get("status") == "completed"]
        if not scores:
            return {
                "chart_type": "histogram",
                "title": "Score Distribution",
                "bins": [],
                "note": "No completed trials.",
            }
        low = min(scores)
        high = max(scores)
        if high == low:
            high = low + 1e-9
        bin_width = (high - low) / n_bins
        bins = [
            {"bin_start": low + i * bin_width, "bin_end": low + (i + 1) * bin_width, "count": 0}
            for i in range(n_bins)
        ]
        for s in scores:
            idx = min(int((s - low) / bin_width), n_bins - 1)
            bins[idx]["count"] += 1
        return {
            "chart_type": "histogram",
            "title": "Score Distribution",
            "x_label": "Score",
            "y_label": "Count",
            "bins": bins,
            "n_trials": len(scores),
            "mean_score": sum(scores) / len(scores),
            "best_score": max(scores),
        }

    # -----------------------------------------------------------------------
    # Full dashboard
    # -----------------------------------------------------------------------

    def full_dashboard(
        self,
        records: list[dict[str, Any]],
        usage_report: dict[str, Any] | None = None,
        pareto_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Return all charts in one dashboard dict."""
        return {
            "optimization_history": self.optimization_history(records),
            "search_convergence": self.search_convergence(records),
            "parameter_importance": self.parameter_importance(records),
            "leaderboard_progression": self.leaderboard_progression(records),
            "experiment_timeline": self.experiment_timeline(records),
            "resource_utilization": (
                self.resource_utilization(usage_report) if usage_report else None
            ),
            "objective_tradeoff": self.objective_tradeoff(
                records, "sharpe_ratio", "max_drawdown", pareto_indices
            ),
            "score_distribution": self.score_distribution(records),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
    if std_x * std_y < 1e-12:
        return 0.0
    return cov / (n * std_x * std_y)


def _extract_nested(d: dict[str, Any], key: str) -> float | None:
    val = d.get(key)
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    for v in d.values():
        if isinstance(v, dict):
            result = _extract_nested(v, key)
            if result is not None:
                return result
    return None
