"""Phase 17 — AEIP Ensemble Visualization Builder.

Generates all chart data for the AEIP dashboard.

All methods return JSON-serializable Python dicts — no matplotlib or
plotly dependency. Frontend can render with any charting library.

Charts:
  - expert_weight_chart          : Current and historical weights per expert
  - voting_distribution          : Class vote distribution across experts
  - agreement_matrix             : Pairwise expert agreement heatmap
  - disagreement_matrix          : Pairwise expert disagreement heatmap
  - confidence_distribution      : Distribution of ensemble confidence values
  - weight_evolution             : Weight trajectories per expert over time
  - expert_performance_timeline  : Expert accuracy over time
  - ensemble_vs_expert_accuracy  : Ensemble vs individual expert accuracy bar chart
  - full_dashboard               : All charts in one call

Usage::

    viz = EnsembleVisualizationBuilder()
    chart = viz.expert_weight_chart(weight_history)
    dashboard = viz.full_dashboard(...)
"""

from __future__ import annotations

from typing import Any


class EnsembleVisualizationBuilder:
    """Builds JSON-safe chart data for the AEIP visualization layer."""

    # -----------------------------------------------------------------------
    # Expert weight chart
    # -----------------------------------------------------------------------

    def expert_weight_chart(
        self,
        current_weights: dict[str, float],
        expert_names: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Bar chart of current expert weights."""
        names = expert_names or {}
        data = [
            {
                "expert_id": eid,
                "name": names.get(eid, eid[:8]),
                "weight": round(w, 4),
                "weight_pct": round(w * 100, 2),
            }
            for eid, w in sorted(current_weights.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return {
            "chart_type": "bar",
            "title": "Current Expert Weights",
            "x_label": "Expert",
            "y_label": "Weight",
            "series": [{"name": "Weight", "data": data}],
            "n_experts": len(data),
        }

    # -----------------------------------------------------------------------
    # Voting distribution
    # -----------------------------------------------------------------------

    def voting_distribution(
        self,
        vote_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Stacked bar chart of vote distribution per expert."""
        if not vote_records:
            return {
                "chart_type": "bar_stacked",
                "title": "Voting Distribution",
                "series": [],
                "note": "No voting records provided.",
            }
        classes = ["buy", "sell", "no_trade"]
        expert_votes: dict[str, dict[str, int]] = {}
        for record in vote_records:
            eid = record.get("expert_id", "unknown")
            cls = record.get("voted_class", "no_trade")
            if eid not in expert_votes:
                expert_votes[eid] = {"buy": 0, "sell": 0, "no_trade": 0}
            if cls in expert_votes[eid]:
                expert_votes[eid][cls] += 1

        series = []
        for cls in classes:
            series.append(
                {
                    "name": cls,
                    "data": [
                        {"expert_id": eid, "count": expert_votes[eid].get(cls, 0)}
                        for eid in expert_votes
                    ],
                }
            )
        return {
            "chart_type": "bar_stacked",
            "title": "Expert Voting Distribution",
            "x_label": "Expert",
            "y_label": "Vote Count",
            "series": series,
            "classes": classes,
            "n_experts": len(expert_votes),
            "total_votes": len(vote_records),
        }

    # -----------------------------------------------------------------------
    # Agreement matrix
    # -----------------------------------------------------------------------

    def agreement_matrix(
        self,
        matrix_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Heatmap of pairwise expert agreement."""
        matrix_data["chart_type"] = "heatmap"
        matrix_data["title"] = matrix_data.get("title", "Expert Agreement Matrix")
        matrix_data["color_scale"] = "green_red"
        return matrix_data

    def disagreement_matrix(
        self,
        matrix_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Heatmap of pairwise expert disagreement."""
        matrix_data["chart_type"] = "heatmap"
        matrix_data["title"] = matrix_data.get("title", "Expert Disagreement Matrix")
        matrix_data["color_scale"] = "red_green"
        return matrix_data

    # -----------------------------------------------------------------------
    # Confidence distribution
    # -----------------------------------------------------------------------

    def confidence_distribution(
        self,
        ensemble_results: list[dict[str, Any]],
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """Histogram of ensemble confidence values."""
        if not ensemble_results:
            return {
                "chart_type": "histogram",
                "title": "Ensemble Confidence Distribution",
                "bins": [],
                "n_samples": 0,
            }
        confidences = [r.get("confidence", 0.5) for r in ensemble_results]
        bins = []
        for i in range(n_bins):
            low = i / n_bins
            high = (i + 1) / n_bins
            count = sum(1 for c in confidences if low <= c < high)
            bins.append(
                {
                    "bin_low": round(low, 3),
                    "bin_high": round(high, 3),
                    "bin_mid": round((low + high) / 2, 3),
                    "count": count,
                }
            )
        return {
            "chart_type": "histogram",
            "title": "Ensemble Confidence Distribution",
            "x_label": "Confidence",
            "y_label": "Count",
            "bins": bins,
            "n_samples": len(confidences),
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
        }

    # -----------------------------------------------------------------------
    # Weight evolution
    # -----------------------------------------------------------------------

    def weight_evolution(
        self,
        weight_history: dict[str, list[float]],
        expert_names: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Line chart showing expert weight trajectories over time."""
        names = expert_names or {}
        series = []
        for expert_id, weights in weight_history.items():
            series.append(
                {
                    "name": names.get(expert_id, expert_id[:8]),
                    "expert_id": expert_id,
                    "data": [{"step": i, "weight": round(w, 4)} for i, w in enumerate(weights)],
                }
            )
        return {
            "chart_type": "line",
            "title": "Expert Weight Evolution",
            "x_label": "Step",
            "y_label": "Weight",
            "series": series,
            "n_experts": len(series),
            "n_steps": max((len(s["data"]) for s in series), default=0),
        }

    # -----------------------------------------------------------------------
    # Expert performance timeline
    # -----------------------------------------------------------------------

    def expert_performance_timeline(
        self,
        performance_records: list[dict[str, Any]],
        window: int = 50,
    ) -> dict[str, Any]:
        """Line chart of rolling accuracy per expert over time."""
        if not performance_records:
            return {
                "chart_type": "line",
                "title": "Expert Performance Timeline",
                "series": [],
                "note": "No performance records provided.",
            }

        by_expert: dict[str, list[dict[str, Any]]] = {}
        for r in performance_records:
            eid = r.get("expert_id", "unknown")
            if eid not in by_expert:
                by_expert[eid] = []
            by_expert[eid].append(r)

        series = []
        for expert_id, records in by_expert.items():
            data = []
            buf: list = []
            for i, rec in enumerate(records):
                is_correct = rec.get("is_correct")
                if is_correct is not None:
                    buf.append(int(is_correct))
                    if len(buf) > window:
                        buf.pop(0)
                    rolling_acc = sum(buf) / len(buf) if buf else 0.0
                    data.append({"step": i, "rolling_accuracy": round(rolling_acc, 4)})
            series.append({"name": expert_id[:8], "expert_id": expert_id, "data": data})

        return {
            "chart_type": "line",
            "title": f"Expert Performance Timeline (window={window})",
            "x_label": "Prediction Index",
            "y_label": f"Rolling Accuracy (window={window})",
            "series": series,
            "window": window,
        }

    # -----------------------------------------------------------------------
    # Ensemble vs expert accuracy
    # -----------------------------------------------------------------------

    def ensemble_vs_expert_accuracy(
        self,
        expert_metrics: dict[str, dict[str, float]],
        ensemble_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Bar chart comparing ensemble accuracy vs each expert."""
        data = []
        for eid, met in sorted(expert_metrics.items()):
            data.append(
                {
                    "entity": eid[:8],
                    "entity_id": eid,
                    "accuracy": round(met.get("accuracy", 0.0), 4),
                    "type": "expert",
                }
            )
        if ensemble_metrics:
            data.append(
                {
                    "entity": "Ensemble",
                    "entity_id": "ensemble",
                    "accuracy": round(ensemble_metrics.get("accuracy", 0.0), 4),
                    "type": "ensemble",
                }
            )
        data.sort(key=lambda d: d["accuracy"], reverse=True)
        return {
            "chart_type": "bar",
            "title": "Ensemble vs Expert Accuracy",
            "x_label": "Entity",
            "y_label": "Accuracy",
            "series": [{"name": "Accuracy", "data": data}],
        }

    # -----------------------------------------------------------------------
    # Regime performance heatmap
    # -----------------------------------------------------------------------

    def regime_performance_heatmap(
        self,
        regime_accuracy: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Heatmap of expert accuracy per regime."""
        expert_ids = list(regime_accuracy.keys())
        regimes = sorted({r for m in regime_accuracy.values() for r in m})
        matrix = {eid: {r: regime_accuracy[eid].get(r) for r in regimes} for eid in expert_ids}
        return {
            "chart_type": "heatmap",
            "title": "Expert Accuracy by Market Regime",
            "x_axis": regimes,
            "y_axis": expert_ids,
            "matrix": matrix,
            "color_scale": "blue_red",
        }

    # -----------------------------------------------------------------------
    # Full dashboard
    # -----------------------------------------------------------------------

    def full_dashboard(
        self,
        current_weights: dict[str, float],
        weight_history: dict[str, list[float]],
        vote_records: list[dict[str, Any]],
        agreement_matrix_data: dict[str, Any],
        ensemble_results: list[dict[str, Any]],
        performance_records: list[dict[str, Any]],
        expert_metrics: dict[str, dict[str, float]] | None = None,
        ensemble_metrics: dict[str, float] | None = None,
        expert_names: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build all dashboard charts in one call."""
        return {
            "expert_weight_chart": self.expert_weight_chart(current_weights, expert_names),
            "voting_distribution": self.voting_distribution(vote_records),
            "agreement_matrix": self.agreement_matrix(dict(agreement_matrix_data)),
            "disagreement_matrix": self.disagreement_matrix(dict(agreement_matrix_data)),
            "confidence_distribution": self.confidence_distribution(ensemble_results),
            "weight_evolution": self.weight_evolution(weight_history, expert_names),
            "expert_performance_timeline": self.expert_performance_timeline(performance_records),
            "ensemble_vs_expert_accuracy": self.ensemble_vs_expert_accuracy(
                expert_metrics or {}, ensemble_metrics
            ),
        }
