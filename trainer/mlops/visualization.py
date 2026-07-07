"""Phase 18 — MLOps Visualization Builder.

Generates JSON-serializable chart data for the MLOps dashboard.

No matplotlib/plotly dependency — all output is pure Python dicts
for rendering by any frontend charting library.

Charts:
  - registry_status_chart        : Model count per lifecycle state (pie/bar)
  - model_health_chart           : Drift scores per model (bar)
  - deployment_history_chart     : Deployments over time (line)
  - champion_comparison_chart    : Champion vs challenger metric comparison (bar)
  - alert_distribution_chart     : Alert count by category and level (bar)
  - confidence_trend_chart       : Avg confidence over time per model (line)
  - drift_heatmap                : Drift scores heatmap (model × drift_type)
  - latency_chart                : Latency trend per model (line)
  - full_dashboard               : All charts in one call
"""

from __future__ import annotations

from typing import Any


class MLOpsVisualizationBuilder:
    """Generates all MLOps dashboard charts as JSON-safe dicts."""

    # -----------------------------------------------------------------------
    # Registry status
    # -----------------------------------------------------------------------

    def registry_status_chart(
        self,
        models: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Pie/bar chart of model counts per lifecycle state."""
        state_counts: dict[str, int] = {}
        for m in models:
            state = m.get("lifecycle_state", "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1
        data = [{"state": k, "count": v} for k, v in sorted(state_counts.items())]
        return {
            "chart_type": "bar",
            "title": "Model Registry — Lifecycle State Distribution",
            "x_label": "Lifecycle State",
            "y_label": "Model Count",
            "series": [{"name": "Models", "data": data}],
            "total_models": len(models),
            "state_breakdown": state_counts,
        }

    # -----------------------------------------------------------------------
    # Model health
    # -----------------------------------------------------------------------

    def model_health_chart(
        self,
        health_reports: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Grouped bar chart of drift scores per model."""
        drift_keys = [
            "confidence_drift",
            "prediction_drift",
            "calibration_drift",
            "performance_degradation",
            "latency_degradation",
            "distribution_drift",
        ]
        series = []
        for key in drift_keys:
            data = [
                {"model_id": mid, "value": round(report.get(key, 0.0), 4)}
                for mid, report in health_reports.items()
            ]
            series.append({"name": key, "data": data})
        return {
            "chart_type": "bar_grouped",
            "title": "Model Health — Drift Scores",
            "x_label": "Model",
            "y_label": "Drift Score",
            "series": series,
            "n_models": len(health_reports),
            "unhealthy": [
                mid for mid, r in health_reports.items() if not r.get("is_healthy", True)
            ],
        }

    # -----------------------------------------------------------------------
    # Deployment history
    # -----------------------------------------------------------------------

    def deployment_history_chart(
        self,
        deployments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Line/bar chart of deployment events over time."""
        by_type: dict[str, int] = {}
        by_target: dict[str, int] = {}
        by_status: dict[str, int] = {}
        timeline = []
        for d in deployments:
            t = d.get("deploy_type", "manual")
            tgt = d.get("target", "unknown")
            st = d.get("status", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
            by_target[tgt] = by_target.get(tgt, 0) + 1
            by_status[st] = by_status.get(st, 0) + 1
            timeline.append(
                {
                    "deployed_at": d.get("deployed_at", ""),
                    "model_id": d.get("model_id", ""),
                    "target": tgt,
                    "status": st,
                    "type": t,
                }
            )
        return {
            "chart_type": "timeline",
            "title": "Deployment History",
            "timeline": sorted(timeline, key=lambda x: x["deployed_at"]),
            "by_type": by_type,
            "by_target": by_target,
            "by_status": by_status,
            "total_deployments": len(deployments),
        }

    # -----------------------------------------------------------------------
    # Champion comparison
    # -----------------------------------------------------------------------

    def champion_comparison_chart(
        self,
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        """Grouped bar chart comparing champion vs challenger metrics."""
        metrics = comparison.get("metrics", {})
        model_a = comparison.get("model_a", {})
        model_b = comparison.get("model_b", {})
        data_a = []
        data_b = []
        for metric, vals in metrics.items():
            data_a.append(
                {
                    "metric": metric,
                    "value": vals.get("model_a"),
                    "is_winner": vals.get("winner") == model_a.get("model_id"),
                }
            )
            data_b.append(
                {
                    "metric": metric,
                    "value": vals.get("model_b"),
                    "is_winner": vals.get("winner") == model_b.get("model_id"),
                }
            )
        return {
            "chart_type": "bar_grouped",
            "title": f"Champion Comparison: {model_a.get('name', 'A')} vs "
            f"{model_b.get('name', 'B')}",
            "x_label": "Metric",
            "y_label": "Value",
            "series": [
                {"name": model_a.get("name", "Model A"), "data": data_a},
                {"name": model_b.get("name", "Model B"), "data": data_b},
            ],
            "overall_winner": comparison.get("overall_winner_name", ""),
        }

    # -----------------------------------------------------------------------
    # Alert distribution
    # -----------------------------------------------------------------------

    def alert_distribution_chart(
        self,
        alerts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Stacked bar chart of alert counts by category and level."""
        by_category: dict[str, int] = {}
        by_level: dict[str, int] = {}
        for a in alerts:
            cat = a.get("category", "unknown")
            lvl = a.get("level", "WARNING")
            by_category[cat] = by_category.get(cat, 0) + 1
            by_level[lvl] = by_level.get(lvl, 0) + 1
        cat_data = [
            {"category": k, "count": v} for k, v in sorted(by_category.items(), key=lambda x: -x[1])
        ]
        return {
            "chart_type": "bar_stacked",
            "title": "Alert Distribution by Category",
            "x_label": "Category",
            "y_label": "Count",
            "series": [{"name": "Alerts", "data": cat_data}],
            "by_level": by_level,
            "by_category": by_category,
            "total_alerts": len(alerts),
        }

    # -----------------------------------------------------------------------
    # Confidence trend
    # -----------------------------------------------------------------------

    def confidence_trend_chart(
        self,
        model_id: str,
        confidence_history: list[float],
        window: int = 20,
    ) -> dict[str, Any]:
        """Rolling confidence line chart for a single model."""
        data = []
        for i, conf in enumerate(confidence_history):
            w = confidence_history[max(0, i - window) : i + 1]
            avg = sum(w) / len(w)
            data.append({"idx": i, "raw_confidence": round(conf, 4), "rolling_avg": round(avg, 4)})
        return {
            "chart_type": "line",
            "title": f"Confidence Trend — {model_id[:8]}",
            "model_id": model_id,
            "x_label": "Prediction Index",
            "y_label": "Confidence",
            "series": [
                {
                    "name": "Raw Confidence",
                    "data": [{"idx": d["idx"], "value": d["raw_confidence"]} for d in data],
                },
                {
                    "name": f"Rolling Avg (w={window})",
                    "data": [{"idx": d["idx"], "value": d["rolling_avg"]} for d in data],
                },
            ],
            "n_points": len(data),
        }

    # -----------------------------------------------------------------------
    # Drift heatmap
    # -----------------------------------------------------------------------

    def drift_heatmap(
        self,
        health_reports: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Heatmap of drift scores: models × drift types."""
        drift_types = [
            "confidence_drift",
            "prediction_drift",
            "calibration_drift",
            "performance_degradation",
            "latency_degradation",
            "distribution_drift",
        ]
        model_ids = list(health_reports.keys())
        matrix = {
            mid: {dtype: round(health_reports[mid].get(dtype, 0.0), 4) for dtype in drift_types}
            for mid in model_ids
        }
        return {
            "chart_type": "heatmap",
            "title": "Drift Heatmap — Models × Drift Types",
            "x_axis": drift_types,
            "y_axis": model_ids,
            "matrix": matrix,
            "color_scale": "green_red",
        }

    # -----------------------------------------------------------------------
    # Latency chart
    # -----------------------------------------------------------------------

    def latency_chart(
        self,
        monitoring_snapshots: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Bar chart of avg/p95/p99 latency per model."""
        data = [
            {
                "model_id": mid,
                "avg_latency_ms": snap.get("avg_latency_ms", 0),
                "p95_latency_ms": snap.get("p95_latency_ms", 0),
                "p99_latency_ms": snap.get("p99_latency_ms", 0),
                "latency_alert": snap.get("latency_alert", False),
            }
            for mid, snap in monitoring_snapshots.items()
        ]
        return {
            "chart_type": "bar_grouped",
            "title": "Latency by Model (avg / p95 / p99)",
            "x_label": "Model",
            "y_label": "Latency (ms)",
            "series": [
                {
                    "name": "Avg",
                    "data": [
                        {"model_id": d["model_id"], "value": d["avg_latency_ms"]} for d in data
                    ],
                },
                {
                    "name": "P95",
                    "data": [
                        {"model_id": d["model_id"], "value": d["p95_latency_ms"]} for d in data
                    ],
                },
                {
                    "name": "P99",
                    "data": [
                        {"model_id": d["model_id"], "value": d["p99_latency_ms"]} for d in data
                    ],
                },
            ],
            "alerts": [d["model_id"] for d in data if d["latency_alert"]],
        }

    # -----------------------------------------------------------------------
    # Full dashboard
    # -----------------------------------------------------------------------

    def full_dashboard(
        self,
        models: list[dict[str, Any]],
        health_reports: dict[str, dict[str, Any]],
        deployments: list[dict[str, Any]],
        comparison: dict[str, Any] | None,
        alerts: list[dict[str, Any]],
        monitoring_snapshots: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build all dashboard charts in one call."""
        charts: dict[str, Any] = {}
        charts["registry_status"] = self.registry_status_chart(models)
        charts["model_health"] = self.model_health_chart(health_reports)
        charts["deployment_history"] = self.deployment_history_chart(deployments)
        if comparison:
            charts["champion_comparison"] = self.champion_comparison_chart(comparison)
        charts["alert_distribution"] = self.alert_distribution_chart(alerts)
        charts["drift_heatmap"] = self.drift_heatmap(health_reports)
        charts["latency"] = self.latency_chart(monitoring_snapshots)
        return charts
