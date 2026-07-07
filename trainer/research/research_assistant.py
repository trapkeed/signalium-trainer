"""Phase 19 — AI Research Assistant.

Generates periodic institutional research summaries:

  - Weekly Research Report
  - Monthly Model Health Report
  - Feature Quality Report
  - Dataset Quality Report
  - Validation Summary
  - Research Roadmap

All reports are advisory documents for human review.

Usage::

    assistant = ResearchAssistant(
        observatory=obs, health_tracker=mhe,
        feature_manager=flm, label_monitor=lqm,
        dataset_tracker=det, advisor=advisor,
        knowledge_base=kb,
    )
    weekly = assistant.weekly_research_report()
    monthly = assistant.monthly_model_health_report("model_01")
    roadmap = assistant.research_roadmap()
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


class ReportType:
    WEEKLY_RESEARCH = "weekly_research_report"
    MONTHLY_MODEL_HEALTH = "monthly_model_health_report"
    FEATURE_QUALITY = "feature_quality_report"
    DATASET_QUALITY = "dataset_quality_report"
    VALIDATION_SUMMARY = "validation_summary"
    RESEARCH_ROADMAP = "research_roadmap"
    MARKET_EVOLUTION = "market_evolution_report"
    LABEL_QUALITY = "label_quality_report"

    ALL = [
        WEEKLY_RESEARCH,
        MONTHLY_MODEL_HEALTH,
        FEATURE_QUALITY,
        DATASET_QUALITY,
        VALIDATION_SUMMARY,
        RESEARCH_ROADMAP,
        MARKET_EVOLUTION,
        LABEL_QUALITY,
    ]


# ---------------------------------------------------------------------------
# Research assistant
# ---------------------------------------------------------------------------


class ResearchAssistant:
    """Generates structured institutional research reports.

    Args:
        observatory: ResearchObservatory instance.
        health_tracker: ModelHealthEvolutionTracker instance.
        feature_manager: FeatureLifecycleManager instance.
        label_monitor: LabelQualityMonitor instance.
        dataset_tracker: DatasetEvolutionTracker instance.
        advisor: RetrainingAdvisor instance.
        experiment_engine: ExperimentRecommendationEngine instance.
        knowledge_base: KnowledgeBase instance.
        market_monitor: MarketEvolutionMonitor instance.
        registry: ModelRegistry (optional).
    """

    def __init__(
        self,
        observatory=None,
        health_tracker=None,
        feature_manager=None,
        label_monitor=None,
        dataset_tracker=None,
        advisor=None,
        experiment_engine=None,
        knowledge_base=None,
        market_monitor=None,
        registry=None,
    ) -> None:
        self.observatory = observatory
        self.health_tracker = health_tracker
        self.feature_manager = feature_manager
        self.label_monitor = label_monitor
        self.dataset_tracker = dataset_tracker
        self.advisor = advisor
        self.experiment_engine = experiment_engine
        self.kb = knowledge_base
        self.market_monitor = market_monitor
        self.registry = registry

    def weekly_research_report(self) -> dict[str, Any]:
        """Generate the weekly research summary."""
        sections: dict[str, Any] = {
            "report_type": ReportType.WEEKLY_RESEARCH,
            "report_id": self._report_id(),
            "period": "weekly",
            "generated_at": self._now(),
            "note": "ADVISORY — for human review only.",
        }

        if self.observatory:
            obs_report = self.observatory.get_observatory_report()
            sections["production_overview"] = {
                "n_models_observed": obs_report.get("n_models_observed", 0),
                "total_snapshots": obs_report.get("total_snapshots", 0),
                "total_outcomes": obs_report.get("total_outcomes", 0),
            }

        if self.health_tracker:
            summary = self.health_tracker.global_summary()
            sections["model_health_summary"] = summary

        if self.feature_manager:
            feat_summary = self.feature_manager.summary()
            promotions = self.feature_manager.promotion_candidates()
            retirements = self.feature_manager.retirement_candidates()
            sections["feature_lifecycle"] = {
                **feat_summary,
                "promotion_candidates": [e.to_dict() for e in promotions[:5]],
                "retirement_candidates": [e.to_dict() for e in retirements[:5]],
            }

        if self.label_monitor:
            sections["label_quality"] = self.label_monitor.summary()

        if self.advisor:
            sections["retraining_recommendations"] = {
                **self.advisor.stats(),
                "pending": self.advisor.pending()[:5],
            }

        if self.experiment_engine:
            sections["experiment_suggestions"] = {
                **self.experiment_engine.stats(),
                "top_pending": self.experiment_engine.pending()[:5],
            }

        if self.market_monitor:
            sections["market_evolution"] = self.market_monitor.global_report()

        if self.kb:
            sections["knowledge_base_summary"] = {
                "total_entries": self.kb.n_entries,
                "recent_findings": self.kb.findings(limit=3),
            }

        return sections

    def monthly_model_health_report(self, model_id: str | None = None) -> dict[str, Any]:
        """Generate monthly model health report."""
        report: dict[str, Any] = {
            "report_type": ReportType.MONTHLY_MODEL_HEALTH,
            "report_id": self._report_id(),
            "model_id": model_id or "all",
            "period": "monthly",
            "generated_at": self._now(),
        }

        if self.health_tracker:
            if model_id:
                trend = self.health_tracker.get_trend(model_id)
                report["health_trend"] = trend.to_dict()
                report["evolution"] = self.health_tracker.evolution_report(model_id)
            else:
                report["global_summary"] = self.health_tracker.global_summary()
                report["degrading_models"] = self.health_tracker.degrading_models()
                report["improving_models"] = self.health_tracker.improving_models()

        if self.observatory and model_id:
            report["signal_stats"] = self.observatory.get_signal_stats(model_id)
            report["production_trend"] = self.observatory.get_production_trend(model_id)

        if self.advisor and model_id:
            report["retraining_recommendations"] = self.advisor.pending(model_id=model_id)

        return report

    def feature_quality_report(self) -> dict[str, Any]:
        """Generate feature quality report."""
        report: dict[str, Any] = {
            "report_type": ReportType.FEATURE_QUALITY,
            "report_id": self._report_id(),
            "generated_at": self._now(),
        }

        if self.feature_manager:
            evaluations = self.feature_manager.evaluate_all()
            by_cls: dict[str, int] = {}
            by_rec: dict[str, int] = {}
            for e in evaluations:
                by_cls[e.classification] = by_cls.get(e.classification, 0) + 1
                by_rec[e.recommendation] = by_rec.get(e.recommendation, 0) + 1
            report["summary"] = self.feature_manager.summary()
            report["by_classification"] = by_cls
            report["by_recommendation"] = by_rec
            report["evaluations"] = [
                e.to_dict() for e in sorted(evaluations, key=lambda e: -e.health_score)[:20]
            ]
            report["promotion_candidates"] = [
                e.to_dict() for e in self.feature_manager.promotion_candidates()
            ]
            report["retirement_candidates"] = [
                e.to_dict() for e in self.feature_manager.retirement_candidates()
            ]
        return report

    def dataset_quality_report(self) -> dict[str, Any]:
        """Generate dataset quality report."""
        report: dict[str, Any] = {
            "report_type": ReportType.DATASET_QUALITY,
            "report_id": self._report_id(),
            "generated_at": self._now(),
        }
        if self.dataset_tracker:
            report["evolution_report"] = self.dataset_tracker.evolution_report()
        if self.label_monitor:
            all_reports = self.label_monitor.all_reports()
            report["label_quality"] = {v: r.to_dict() for v, r in all_reports.items()}
            report["low_quality_versions"] = self.label_monitor.low_quality_versions()
        return report

    def validation_summary(
        self,
        validation_records: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Generate validation summary from recent records."""
        report: dict[str, Any] = {
            "report_type": ReportType.VALIDATION_SUMMARY,
            "report_id": self._report_id(),
            "generated_at": self._now(),
        }
        records = validation_records or []
        if records:
            accs = [r.get("accuracy", 0.0) for r in records]
            sharpes = [r.get("sharpe_ratio", 0.0) for r in records]
            eces = [r.get("calibration_error", 0.0) for r in records]
            n = len(records)
            report["n_records"] = n
            report["avg_accuracy"] = round(sum(accs) / n, 4)
            report["avg_sharpe_ratio"] = round(sum(sharpes) / n, 4)
            report["avg_calibration_error"] = round(sum(eces) / n, 4)
            report["best_accuracy"] = round(max(accs), 4)
            report["worst_accuracy"] = round(min(accs), 4)
            report["records"] = records
        else:
            if self.observatory:
                obs = self.observatory.get_observatory_report()
                report["observatory"] = obs
        return report

    def research_roadmap(self, horizon_months: int = 3) -> dict[str, Any]:
        """Generate a research roadmap for the next N months."""
        roadmap: dict[str, Any] = {
            "report_type": ReportType.RESEARCH_ROADMAP,
            "report_id": self._report_id(),
            "horizon_months": horizon_months,
            "generated_at": self._now(),
        }

        immediate = []
        short_term = []
        medium_term = []

        if self.advisor:
            pending = self.advisor.pending()
            for rec in pending:
                item = {
                    "type": "retraining_recommendation",
                    "priority": rec["priority"],
                    "title": f"Retrain: {rec['trigger']}",
                    "reason": rec["reason"],
                    "expected_benefit": rec["expected_benefit"],
                }
                if rec["priority"] <= 1:
                    immediate.append(item)
                elif rec["priority"] <= 2:
                    short_term.append(item)
                else:
                    medium_term.append(item)

        if self.experiment_engine:
            pending_exp = self.experiment_engine.pending()
            for exp in pending_exp[:10]:
                item = {
                    "type": "experiment",
                    "priority": exp["priority"],
                    "title": exp["title"],
                    "category": exp["category"],
                    "expected_benefit": exp["expected_benefit"],
                }
                if exp["priority"] <= 1:
                    immediate.append(item)
                elif exp["priority"] <= 2:
                    short_term.append(item)
                else:
                    medium_term.append(item)

        if self.feature_manager:
            promotions = self.feature_manager.promotion_candidates()
            retirements = self.feature_manager.retirement_candidates()
            for e in promotions[:3]:
                medium_term.append(
                    {
                        "type": "feature_promotion",
                        "priority": 3,
                        "title": f"Promote feature: {e.name}",
                        "recommendation": e.recommendation,
                    }
                )
            for e in retirements[:3]:
                short_term.append(
                    {
                        "type": "feature_retirement",
                        "priority": 3,
                        "title": f"Retire feature: {e.name}",
                        "recommendation": e.recommendation,
                    }
                )

        roadmap["immediate"] = sorted(immediate, key=lambda x: x["priority"])
        roadmap["short_term"] = sorted(short_term, key=lambda x: x["priority"])
        roadmap["medium_term"] = sorted(medium_term, key=lambda x: x["priority"])
        roadmap["n_immediate"] = len(immediate)
        roadmap["n_short_term"] = len(short_term)
        roadmap["n_medium_term"] = len(medium_term)
        return roadmap

    def market_evolution_report(self) -> dict[str, Any]:
        """Generate market evolution report."""
        report: dict[str, Any] = {
            "report_type": ReportType.MARKET_EVOLUTION,
            "report_id": self._report_id(),
            "generated_at": self._now(),
        }
        if self.market_monitor:
            detections = self.market_monitor.detect_all()
            report["global_report"] = self.market_monitor.global_report()
            report["new_detections"] = [d.to_dict() for d in detections]
            report["n_new_detections"] = len(detections)
        return report

    def label_quality_report(self) -> dict[str, Any]:
        """Generate label quality report."""
        report: dict[str, Any] = {
            "report_type": ReportType.LABEL_QUALITY,
            "report_id": self._report_id(),
            "generated_at": self._now(),
        }
        if self.label_monitor:
            all_reports = self.label_monitor.all_reports()
            report["label_versions"] = {v: r.to_dict() for v, r in all_reports.items()}
            report["summary"] = self.label_monitor.summary()
            report["low_quality"] = self.label_monitor.low_quality_versions()
        return report

    def full_intelligence_report(self) -> dict[str, Any]:
        """Generate the complete institutional research intelligence report."""
        return {
            "report_type": "research_intelligence_report",
            "report_id": self._report_id(),
            "generated_at": self._now(),
            "weekly_summary": self.weekly_research_report(),
            "feature_quality": self.feature_quality_report(),
            "dataset_quality": self.dataset_quality_report(),
            "market_evolution": self.market_evolution_report(),
            "research_roadmap": self.research_roadmap(),
            "note": "ADVISORY — all recommendations require human review.",
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _report_id() -> str:
        import uuid

        return str(uuid.uuid4())
