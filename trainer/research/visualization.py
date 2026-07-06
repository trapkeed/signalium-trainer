"""Phase 19 — Research Intelligence Visualization Builder.

Generates JSON-serializable chart data for the Research dashboard.

No matplotlib/plotly dependency — pure Python dicts for any frontend.

Charts:
  1. market_evolution_timeline    : Volatility/trend timeline per instrument
  2. feature_lifecycle_chart      : Feature health scores by classification
  3. dataset_evolution_chart      : Dataset quality / sample count timeline
  4. recommendation_queue_chart   : Retraining recommendations by priority/trigger
  5. knowledge_graph_summary      : Knowledge base entries by type and impact
  6. research_timeline_chart      : Experiment/finding/deployment timeline
  7. label_quality_chart          : Label quality scores by version
  8. model_health_evolution_chart : Long-term accuracy/confidence trends
  9. full_dashboard               : All charts combined
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ResearchVisualizationBuilder:
    """Generates all research dashboard charts as JSON-safe dicts."""

    # -----------------------------------------------------------------------
    # Market evolution timeline
    # -----------------------------------------------------------------------

    def market_evolution_timeline(
        self,
        snapshots_by_instrument: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Line chart of volatility and trend strength over time per instrument."""
        series = []
        for instrument, snaps in snapshots_by_instrument.items():
            vol_series = [
                {"ts": s.get("timestamp", ""), "value": s.get("volatility", 0)} for s in snaps
            ]
            trend_series = [
                {"ts": s.get("timestamp", ""), "value": s.get("trend_strength", 0)} for s in snaps
            ]
            series.append(
                {
                    "instrument": instrument,
                    "volatility": vol_series,
                    "trend_strength": trend_series,
                    "n_snapshots": len(snaps),
                }
            )
        detections_count = sum(
            1 for snaps in snapshots_by_instrument.values() for s in snaps if s.get("regime_change")
        )
        return {
            "chart_type": "multi_line",
            "title": "Market Evolution Timeline",
            "x_label": "Timestamp",
            "y_label": "Value",
            "instruments": list(snapshots_by_instrument.keys()),
            "series": series,
            "regime_changes_detected": detections_count,
        }

    # -----------------------------------------------------------------------
    # Feature lifecycle chart
    # -----------------------------------------------------------------------

    def feature_lifecycle_chart(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Bar chart of feature health scores grouped by classification."""
        by_cls: Dict[str, List[Dict]] = {}
        for e in evaluations:
            cls = e.get("classification", "unknown")
            if cls not in by_cls:
                by_cls[cls] = []
            by_cls[cls].append(
                {
                    "name": e.get("name", ""),
                    "health_score": e.get("health_score", 0.0),
                    "recommendation": e.get("recommendation", ""),
                    "avg_importance": e.get("avg_importance", 0.0),
                }
            )
        recommendation_counts: Dict[str, int] = {}
        for e in evaluations:
            rec = e.get("recommendation", "monitor")
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        return {
            "chart_type": "bar_grouped",
            "title": "Feature Lifecycle Health Scores",
            "x_label": "Feature",
            "y_label": "Health Score (0–1)",
            "by_classification": by_cls,
            "recommendation_counts": recommendation_counts,
            "n_features": len(evaluations),
            "color_scheme": {
                "experimental": "blue",
                "candidate": "orange",
                "approved": "green",
                "deprecated": "yellow",
                "retired": "gray",
            },
        }

    # -----------------------------------------------------------------------
    # Dataset evolution chart
    # -----------------------------------------------------------------------

    def dataset_evolution_chart(
        self,
        quality_timeline: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Timeline of dataset quality scores and sample counts."""
        quality_data = [
            {"version": d.get("version", ""), "value": d.get("quality_score", 0.0)}
            for d in quality_timeline
        ]
        sample_data = [
            {"version": d.get("version", ""), "value": d.get("n_samples", 0)}
            for d in quality_timeline
        ]
        instrument_data = [
            {"version": d.get("version", ""), "value": d.get("n_instruments", 0)}
            for d in quality_timeline
        ]
        return {
            "chart_type": "multi_line",
            "title": "Dataset Evolution",
            "x_label": "Dataset Version",
            "series": [
                {"name": "Quality Score", "data": quality_data, "y_axis": "quality"},
                {"name": "Sample Count", "data": sample_data, "y_axis": "samples"},
                {"name": "Instrument Count", "data": instrument_data, "y_axis": "instruments"},
            ],
            "n_versions": len(quality_timeline),
        }

    # -----------------------------------------------------------------------
    # Recommendation queue chart
    # -----------------------------------------------------------------------

    def recommendation_queue_chart(
        self,
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Bar chart of pending recommendations by trigger and priority."""
        by_trigger: Dict[str, int] = {}
        by_priority: Dict[int, int] = {}
        by_model: Dict[str, int] = {}
        for rec in recommendations:
            trig = rec.get("trigger", "unknown")
            pri = rec.get("priority", 3)
            mid = rec.get("model_id", "unknown")
            by_trigger[trig] = by_trigger.get(trig, 0) + 1
            by_priority[pri] = by_priority.get(pri, 0) + 1
            by_model[mid] = by_model.get(mid, 0) + 1
        return {
            "chart_type": "bar",
            "title": "Retraining Recommendation Queue",
            "by_trigger": [
                {"trigger": k, "count": v}
                for k, v in sorted(by_trigger.items(), key=lambda x: -x[1])
            ],
            "by_priority": [{"priority": k, "count": v} for k, v in sorted(by_priority.items())],
            "by_model": by_model,
            "total_recommendations": len(recommendations),
            "note": "ADVISORY ONLY",
        }

    # -----------------------------------------------------------------------
    # Knowledge graph summary
    # -----------------------------------------------------------------------

    def knowledge_graph_summary(
        self,
        knowledge_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summary chart of knowledge base entries by type and impact."""
        by_type = knowledge_report.get("by_type", {})
        by_impact = knowledge_report.get("by_impact", {})
        type_data = [
            {"type": k, "count": v} for k, v in sorted(by_type.items(), key=lambda x: -x[1])
        ]
        impact_data = [{"impact": k, "count": v} for k, v in by_impact.items()]
        return {
            "chart_type": "bar_grouped",
            "title": "Knowledge Base — Entries by Type and Impact",
            "by_type": type_data,
            "by_impact": impact_data,
            "total_entries": knowledge_report.get("total_entries", 0),
        }

    # -----------------------------------------------------------------------
    # Research timeline
    # -----------------------------------------------------------------------

    def research_timeline_chart(
        self,
        experiments: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Timeline of experiments and findings ordered by creation date."""
        events = []
        for e in experiments:
            events.append(
                {
                    "type": "experiment",
                    "title": e.get("title", ""),
                    "category": e.get("category", ""),
                    "status": e.get("status", ""),
                    "priority": e.get("priority", 3),
                    "created_at": e.get("created_at", ""),
                }
            )
        for f in findings:
            events.append(
                {
                    "type": "finding",
                    "title": f.get("title", ""),
                    "impact": f.get("impact_level", ""),
                    "entry_type": f.get("entry_type", ""),
                    "created_at": f.get("created_at", ""),
                }
            )
        events.sort(key=lambda x: x.get("created_at", ""))
        by_month: Dict[str, int] = {}
        for ev in events:
            ts = ev.get("created_at", "")[:7]
            by_month[ts] = by_month.get(ts, 0) + 1
        return {
            "chart_type": "timeline",
            "title": "Research Activity Timeline",
            "events": events[-50:],  # last 50 events
            "total_events": len(events),
            "by_month": by_month,
        }

    # -----------------------------------------------------------------------
    # Label quality chart
    # -----------------------------------------------------------------------

    def label_quality_chart(
        self,
        label_reports: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Bar chart of label quality scores by version."""
        data = [
            {
                "version": version,
                "quality_score": report.get("quality_score", 0.0),
                "quality_grade": report.get("quality_grade", "?"),
                "avg_noise": report.get("avg_noise", 0.0),
                "avg_consistency": report.get("avg_consistency", 1.0),
                "n_issues": len(report.get("issues", [])),
            }
            for version, report in sorted(label_reports.items())
        ]
        return {
            "chart_type": "bar",
            "title": "Label Quality Scores by Version",
            "x_label": "Label Version",
            "y_label": "Quality Score (0–1)",
            "data": data,
            "grade_color_scheme": {
                "A": "green",
                "B": "teal",
                "C": "yellow",
                "D": "orange",
                "F": "red",
            },
        }

    # -----------------------------------------------------------------------
    # Model health evolution chart
    # -----------------------------------------------------------------------

    def model_health_evolution_chart(
        self,
        health_trends: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Multi-model health trend comparison chart."""
        data = [
            {
                "model_id": mid,
                "avg_accuracy": trend.get("avg_accuracy", 0.0),
                "accuracy_delta": trend.get("accuracy_delta", 0.0),
                "overall_health": trend.get("overall_health", "stable"),
                "accuracy_trend": trend.get("accuracy_trend", "stable"),
                "confidence_trend": trend.get("confidence_trend", "stable"),
                "n_points": trend.get("n_points", 0),
            }
            for mid, trend in health_trends.items()
        ]
        status_counts = {
            "improving": sum(1 for d in data if d["overall_health"] == "improving"),
            "stable": sum(1 for d in data if d["overall_health"] == "stable"),
            "degrading": sum(1 for d in data if d["overall_health"] == "degrading"),
        }
        return {
            "chart_type": "bar_grouped",
            "title": "Model Health Evolution",
            "x_label": "Model",
            "y_label": "Accuracy",
            "data": data,
            "status_summary": status_counts,
            "color_scheme": {"improving": "green", "stable": "blue", "degrading": "red"},
        }

    # -----------------------------------------------------------------------
    # Full dashboard
    # -----------------------------------------------------------------------

    def full_dashboard(
        self,
        snapshots_by_instrument: Optional[Dict[str, List[Dict]]] = None,
        feature_evaluations: Optional[List[Dict]] = None,
        quality_timeline: Optional[List[Dict]] = None,
        recommendations: Optional[List[Dict]] = None,
        knowledge_report: Optional[Dict] = None,
        experiments: Optional[List[Dict]] = None,
        findings: Optional[List[Dict]] = None,
        label_reports: Optional[Dict[str, Dict]] = None,
        health_trends: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """Build all research dashboard charts in one call."""
        charts: Dict[str, Any] = {}
        if snapshots_by_instrument:
            charts["market_evolution"] = self.market_evolution_timeline(snapshots_by_instrument)
        if feature_evaluations:
            charts["feature_lifecycle"] = self.feature_lifecycle_chart(feature_evaluations)
        if quality_timeline:
            charts["dataset_evolution"] = self.dataset_evolution_chart(quality_timeline)
        if recommendations:
            charts["recommendation_queue"] = self.recommendation_queue_chart(recommendations)
        if knowledge_report:
            charts["knowledge_graph"] = self.knowledge_graph_summary(knowledge_report)
        if experiments or findings:
            charts["research_timeline"] = self.research_timeline_chart(
                experiments or [], findings or []
            )
        if label_reports:
            charts["label_quality"] = self.label_quality_chart(label_reports)
        if health_trends:
            charts["model_health_evolution"] = self.model_health_evolution_chart(health_trends)
        return charts
