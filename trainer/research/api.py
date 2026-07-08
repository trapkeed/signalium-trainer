"""Phase 19 — Research Intelligence API.

Standard endpoint layer for the Research Intelligence Platform.

All responses use the standard envelope::

    {
      "status": "ok" | "error",
      "code": "...",
      "message": "...",
      "data": { ... },
      "timestamp": "2026-07-05T12:00:00Z"
    }

Endpoint groups:
  Observatory:    get_observatory_report, record_production_snapshot,
                  record_signal_outcome
  Market:         get_market_evolution_report, detect_market_evolution,
                  record_regime_snapshot
  Features:       get_feature_evaluation, get_feature_summary,
                  classify_feature, list_features_by_classification
  Labels:         get_label_quality_report, record_label_batch,
                  get_all_label_reports
  Datasets:       register_dataset, get_dataset, list_datasets,
                  get_dataset_evolution_report, compare_datasets
  ModelHealth:    record_health_point, get_health_trend,
                  get_evolution_report, get_degrading_models
  Advisor:        evaluate_retraining, list_recommendations,
                  acknowledge_recommendation, dismiss_recommendation
  Experiments:    generate_experiments, list_experiments, update_status
  KnowledgeBase:  add_entry, get_entry, query_entries, search_entries,
                  get_knowledge_report
  Reports:        weekly_report, monthly_health_report,
                  feature_quality_report, dataset_quality_report,
                  validation_summary, research_roadmap,
                  full_intelligence_report
  Stats:          get_stats
"""

from __future__ import annotations

import traceback
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from trainer.research.database import ResearchDatabase
from trainer.research.dataset_evolution import DatasetEvolutionTracker, DatasetRecord
from trainer.research.experiment_engine import ExperimentRecommendationEngine
from trainer.research.feature_lifecycle import FeatureLifecycleManager, FeatureRecord
from trainer.research.knowledge_base import KnowledgeBase, KnowledgeEntry
from trainer.research.label_quality import LabelQualityMonitor
from trainer.research.market_evolution import MarketEvolutionMonitor
from trainer.research.model_health_evolution import ModelHealthEvolutionTracker
from trainer.research.observatory import ResearchObservatory
from trainer.research.research_assistant import ResearchAssistant
from trainer.research.retraining_advisor import RetrainingAdvisor


def _ts() -> str:
    return datetime.now(UTC).isoformat()


def _ok(data: Any, message: str = "ok") -> dict[str, Any]:
    return {"status": "ok", "message": message, "data": data, "timestamp": _ts()}


def _err(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "code": code, "message": message, "data": None, "timestamp": _ts()}


class ResearchAPI:
    """Research Intelligence Platform API."""

    def __init__(
        self,
        db: ResearchDatabase,
        observatory: ResearchObservatory,
        market_monitor: MarketEvolutionMonitor,
        feature_manager: FeatureLifecycleManager,
        label_monitor: LabelQualityMonitor,
        dataset_tracker: DatasetEvolutionTracker,
        health_tracker: ModelHealthEvolutionTracker,
        advisor: RetrainingAdvisor,
        experiment_engine: ExperimentRecommendationEngine,
        knowledge_base: KnowledgeBase,
        assistant: ResearchAssistant,
    ) -> None:
        self.db = db
        self.observatory = observatory
        self.market = market_monitor
        self.features = feature_manager
        self.labels = label_monitor
        self.datasets = dataset_tracker
        self.health = health_tracker
        self.advisor = advisor
        self.experiments = experiment_engine
        self.kb = knowledge_base
        self.assistant = assistant

    # -----------------------------------------------------------------------
    # Observatory
    # -----------------------------------------------------------------------

    def get_observatory_report(self) -> dict[str, Any]:
        try:
            return _ok(self.observatory.get_observatory_report())
        except Exception as e:
            return _err("OBSERVATORY_ERROR", str(e))

    def record_production_snapshot(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        try:
            snap = self.observatory.record_production_snapshot(model_id, **kwargs)
            return _ok(snap.to_dict(), "Snapshot recorded.")
        except Exception as e:
            return _err("SNAPSHOT_ERROR", str(e))

    def record_signal_outcome(
        self, model_id: str, signal: str, outcome: str, pnl: float = 0.0, **kwargs: Any
    ) -> dict[str, Any]:
        try:
            so = self.observatory.record_signal_outcome(
                model_id, signal=signal, outcome=outcome, pnl=pnl, **kwargs
            )
            return _ok(so.to_dict(), "Signal outcome recorded.")
        except Exception as e:
            return _err("OUTCOME_ERROR", str(e))

    def get_signal_stats(self, model_id: str) -> dict[str, Any]:
        try:
            return _ok(self.observatory.get_signal_stats(model_id))
        except Exception as e:
            return _err("SIGNAL_STATS_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Market evolution
    # -----------------------------------------------------------------------

    def record_regime_snapshot(
        self, instrument: str, timeframe: str, **kwargs: Any
    ) -> dict[str, Any]:
        try:
            snap = self.market.record_regime_snapshot(instrument, timeframe, **kwargs)
            self.db.save_market_snapshot(snap.to_dict())
            return _ok(snap.to_dict(), "Regime snapshot recorded.")
        except Exception as e:
            return _err("REGIME_SNAPSHOT_ERROR", str(e))

    def detect_market_evolution(self, instrument: str, timeframe: str) -> dict[str, Any]:
        try:
            detections = self.market.detect_evolution(instrument, timeframe)
            return _ok(
                {
                    "detections": [d.to_dict() for d in detections],
                    "n_detections": len(detections),
                }
            )
        except Exception as e:
            return _err("DETECT_EVOLUTION_ERROR", str(e))

    def get_market_evolution_report(
        self, instrument: str, timeframe: str | None = None
    ) -> dict[str, Any]:
        try:
            return _ok(self.market.evolution_report(instrument, timeframe))
        except Exception as e:
            return _err("MARKET_REPORT_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Features
    # -----------------------------------------------------------------------

    def register_feature(
        self,
        name: str,
        category: str = "technical",
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            record = FeatureRecord(
                name=name, category=category, description=description, tags=tags or []
            )
            self.features.register_feature(record)
            return _ok(record.to_dict(), f"Feature {name!r} registered.")
        except Exception as e:
            return _err("REGISTER_FEATURE_ERROR", str(e))

    def record_feature_observation(self, name: str, **kwargs: Any) -> dict[str, Any]:
        try:
            obs = self.features.record_observation(name, **kwargs)
            self.db.save_feature_observation(obs.to_dict())
            return _ok(obs.to_dict(), "Feature observation recorded.")
        except Exception as e:
            return _err("FEATURE_OBS_ERROR", str(e))

    def get_feature_evaluation(self, name: str) -> dict[str, Any]:
        try:
            ev = self.features.evaluate(name)
            return _ok(ev.to_dict())
        except Exception as e:
            return _err("FEATURE_EVAL_ERROR", str(e))

    def get_feature_summary(self) -> dict[str, Any]:
        try:
            return _ok(self.features.summary())
        except Exception as e:
            return _err("FEATURE_SUMMARY_ERROR", str(e))

    def classify_feature(
        self, name: str, classification: str, actor: str = "system", reason: str = ""
    ) -> dict[str, Any]:
        try:
            ok = self.features.classify(name, classification, actor=actor, reason=reason)
            if not ok:
                return _err("NOT_FOUND", f"Feature {name!r} not found.")
            return _ok({"name": name, "classification": classification})
        except Exception as e:
            return _err("CLASSIFY_FEATURE_ERROR", str(e))

    def list_features_by_classification(self, classification: str) -> dict[str, Any]:
        try:
            features = self.features.features_by_classification(classification)
            return _ok({"features": [f.to_dict() for f in features], "count": len(features)})
        except Exception as e:
            return _err("LIST_FEATURES_ERROR", str(e))

    def get_promotion_candidates(self) -> dict[str, Any]:
        try:
            candidates = self.features.promotion_candidates()
            return _ok({"candidates": [e.to_dict() for e in candidates], "count": len(candidates)})
        except Exception as e:
            return _err("PROMOTION_CANDIDATES_ERROR", str(e))

    def get_retirement_candidates(self) -> dict[str, Any]:
        try:
            candidates = self.features.retirement_candidates()
            return _ok({"candidates": [e.to_dict() for e in candidates], "count": len(candidates)})
        except Exception as e:
            return _err("RETIREMENT_CANDIDATES_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------------

    def record_label_batch(
        self,
        label_version: str,
        labels: list[str] | None = None,
        confidences: list[float] | None = None,
        predicted: list[str] | None = None,
        dataset_version: str = "",
    ) -> dict[str, Any]:
        try:
            batch = self.labels.record_batch(
                label_version,
                labels=labels,
                confidences=confidences,
                predicted=predicted,
                dataset_version=dataset_version,
            )
            self.db.save_label_batch(batch.to_dict())
            return _ok(batch.to_dict(), "Label batch recorded.")
        except Exception as e:
            return _err("LABEL_BATCH_ERROR", str(e))

    def get_label_quality_report(self, label_version: str) -> dict[str, Any]:
        try:
            report = self.labels.get_quality_report(label_version)
            return _ok(report.to_dict())
        except Exception as e:
            return _err("LABEL_QUALITY_ERROR", str(e))

    def get_all_label_reports(self) -> dict[str, Any]:
        try:
            reports = {v: r.to_dict() for v, r in self.labels.all_reports().items()}
            return _ok({"reports": reports, "count": len(reports)})
        except Exception as e:
            return _err("ALL_LABEL_REPORTS_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------

    def register_dataset(self, dataset_dict: dict[str, Any]) -> dict[str, Any]:
        try:
            record = DatasetRecord(
                **{
                    k: v
                    for k, v in dataset_dict.items()
                    if hasattr(DatasetRecord, k)
                    or k
                    in (
                        "dataset_id",
                        "version",
                        "description",
                        "date_start",
                        "date_end",
                        "n_instruments",
                        "n_samples",
                        "n_features",
                        "n_labels",
                        "timeframes",
                        "instruments",
                        "regimes_covered",
                        "feature_schema_version",
                        "label_schema_version",
                        "data_quality_score",
                        "completeness",
                        "status",
                        "parent_version",
                        "changes_from_parent",
                        "metadata",
                    )
                }
            )
            self.datasets.register(record)
            self.db.save_dataset(record.to_dict())
            return _ok(record.to_dict(), "Dataset registered.")
        except Exception as e:
            logger.error(f"register_dataset: {e}")
            return _err("REGISTER_DATASET_ERROR", str(e))

    def get_dataset(self, version: str) -> dict[str, Any]:
        try:
            record = self.datasets.get(version)
            if record is None:
                return _err("NOT_FOUND", f"Dataset version {version!r} not found.")
            return _ok(record.to_dict())
        except Exception as e:
            return _err("GET_DATASET_ERROR", str(e))

    def list_datasets_api(self, status: str | None = None) -> dict[str, Any]:
        try:
            datasets = (
                self.datasets.active_datasets()
                if status == "active"
                else list(self.datasets._datasets.values())
            )
            return _ok({"datasets": [d.to_dict() for d in datasets], "count": len(datasets)})
        except Exception as e:
            return _err("LIST_DATASETS_ERROR", str(e))

    def get_dataset_evolution_report(self) -> dict[str, Any]:
        try:
            return _ok(self.datasets.evolution_report())
        except Exception as e:
            return _err("DATASET_EVOLUTION_ERROR", str(e))

    def compare_datasets(self, version_a: str, version_b: str) -> dict[str, Any]:
        try:
            return _ok(self.datasets.compare(version_a, version_b))
        except Exception as e:
            return _err("COMPARE_DATASETS_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Model health
    # -----------------------------------------------------------------------

    def record_health_point(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        try:
            pt = self.health.record_health_point(model_id, **kwargs)
            self.db.save_health_point(pt.to_dict())
            return _ok(pt.to_dict(), "Health point recorded.")
        except Exception as e:
            return _err("HEALTH_POINT_ERROR", str(e))

    def get_health_trend(self, model_id: str) -> dict[str, Any]:
        try:
            trend = self.health.get_trend(model_id)
            return _ok(trend.to_dict())
        except Exception as e:
            return _err("HEALTH_TREND_ERROR", str(e))

    def get_evolution_report(self, model_id: str) -> dict[str, Any]:
        try:
            return _ok(self.health.evolution_report(model_id))
        except Exception as e:
            return _err("EVOLUTION_REPORT_ERROR", str(e))

    def get_degrading_models(self) -> dict[str, Any]:
        try:
            degrading = self.health.degrading_models()
            improving = self.health.improving_models()
            return _ok(
                {
                    "degrading": degrading,
                    "improving": improving,
                    "summary": self.health.global_summary(),
                }
            )
        except Exception as e:
            return _err("DEGRADING_MODELS_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Advisor
    # -----------------------------------------------------------------------

    def evaluate_retraining(
        self, model_id: str, context: dict[str, Any], force_manual: bool = False
    ) -> dict[str, Any]:
        try:
            recs = self.advisor.evaluate(model_id, context, force_manual=force_manual)
            for rec in recs:
                self.db.save_recommendation(rec.to_dict())
            return _ok(
                {
                    "recommendations": [r.to_dict() for r in recs],
                    "count": len(recs),
                    "note": "ADVISORY ONLY — no automatic retraining.",
                }
            )
        except Exception as e:
            return _err("EVALUATE_RETRAINING_ERROR", str(e))

    def list_recommendations(
        self, model_id: str | None = None, status: str = "pending"
    ) -> dict[str, Any]:
        try:
            recs = self.advisor.pending(model_id=model_id)
            return _ok({"recommendations": recs, "count": len(recs)})
        except Exception as e:
            return _err("LIST_RECOMMENDATIONS_ERROR", str(e))

    def acknowledge_recommendation(
        self, recommendation_id: str, acted_by: str = "human"
    ) -> dict[str, Any]:
        try:
            ok = self.advisor.acknowledge(recommendation_id, acted_by=acted_by)
            if not ok:
                return _err("NOT_FOUND", f"Recommendation {recommendation_id!r} not found.")
            return _ok({"recommendation_id": recommendation_id, "status": "acted_upon"})
        except Exception as e:
            return _err("ACK_RECOMMENDATION_ERROR", str(e))

    def dismiss_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        try:
            ok = self.advisor.dismiss(recommendation_id)
            if not ok:
                return _err("NOT_FOUND", f"Recommendation {recommendation_id!r} not found.")
            return _ok({"recommendation_id": recommendation_id, "status": "dismissed"})
        except Exception as e:
            return _err("DISMISS_RECOMMENDATION_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Experiments
    # -----------------------------------------------------------------------

    def generate_experiments(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            suggestions = self.experiments.generate(context)
            for s in suggestions:
                self.db.save_experiment(s.to_dict())
            return _ok(
                {
                    "suggestions": [s.to_dict() for s in suggestions],
                    "count": len(suggestions),
                    "note": "ADVISORY ONLY — no automatic execution.",
                }
            )
        except Exception as e:
            return _err("GENERATE_EXPERIMENTS_ERROR", str(e))

    def list_experiments_api(
        self, status: str | None = None, category: str | None = None
    ) -> dict[str, Any]:
        try:
            exps = self.db.list_experiments(status=status, category=category)
            return _ok({"experiments": exps, "count": len(exps)})
        except Exception as e:
            return _err("LIST_EXPERIMENTS_ERROR", str(e))

    def update_experiment_status(self, suggestion_id: str, status: str) -> dict[str, Any]:
        try:
            ok = self.experiments.update_status(suggestion_id, status)
            if not ok:
                return _err("NOT_FOUND", f"Experiment {suggestion_id!r} not found.")
            return _ok({"suggestion_id": suggestion_id, "status": status})
        except Exception as e:
            return _err("UPDATE_EXPERIMENT_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Knowledge base
    # -----------------------------------------------------------------------

    def add_knowledge_entry(self, entry_dict: dict[str, Any]) -> dict[str, Any]:
        try:
            entry = KnowledgeEntry(
                **{
                    k: v
                    for k, v in entry_dict.items()
                    if k
                    in (
                        "entry_id",
                        "entry_type",
                        "title",
                        "body",
                        "tags",
                        "model_id",
                        "feature_name",
                        "dataset_version",
                        "impact_level",
                        "author",
                        "references",
                        "metadata",
                    )
                }
            )
            eid = self.kb.add(entry)
            self.db.save_knowledge_entry(entry.to_dict())
            return _ok(entry.to_dict(), "Knowledge entry added.")
        except Exception as e:
            return _err("ADD_ENTRY_ERROR", str(e))

    def get_knowledge_entry(self, entry_id: str) -> dict[str, Any]:
        try:
            entry = self.kb.get(entry_id)
            if entry is None:
                return _err("NOT_FOUND", f"Entry {entry_id!r} not found.")
            return _ok(entry.to_dict())
        except Exception as e:
            return _err("GET_ENTRY_ERROR", str(e))

    def query_knowledge_entries(self, **kwargs: Any) -> dict[str, Any]:
        try:
            entries = self.kb.query(**kwargs)
            return _ok({"entries": entries, "count": len(entries)})
        except Exception as e:
            return _err("QUERY_ENTRIES_ERROR", str(e))

    def search_knowledge(self, query_text: str, limit: int = 20) -> dict[str, Any]:
        try:
            results = self.kb.search(query_text, limit=limit)
            return _ok({"results": results, "count": len(results), "query": query_text})
        except Exception as e:
            return _err("SEARCH_KNOWLEDGE_ERROR", str(e))

    def get_knowledge_report(self) -> dict[str, Any]:
        try:
            return _ok(self.kb.knowledge_report())
        except Exception as e:
            return _err("KNOWLEDGE_REPORT_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Reports
    # -----------------------------------------------------------------------

    def weekly_report(self) -> dict[str, Any]:
        try:
            return _ok(self.assistant.weekly_research_report())
        except Exception as e:
            logger.error(f"weekly_report: {traceback.format_exc()}")
            return _err("WEEKLY_REPORT_ERROR", str(e))

    def monthly_health_report(self, model_id: str | None = None) -> dict[str, Any]:
        try:
            return _ok(self.assistant.monthly_model_health_report(model_id))
        except Exception as e:
            return _err("MONTHLY_HEALTH_REPORT_ERROR", str(e))

    def feature_quality_report(self) -> dict[str, Any]:
        try:
            return _ok(self.assistant.feature_quality_report())
        except Exception as e:
            return _err("FEATURE_QUALITY_REPORT_ERROR", str(e))

    def dataset_quality_report(self) -> dict[str, Any]:
        try:
            return _ok(self.assistant.dataset_quality_report())
        except Exception as e:
            return _err("DATASET_QUALITY_REPORT_ERROR", str(e))

    def research_roadmap(self, horizon_months: int = 3) -> dict[str, Any]:
        try:
            return _ok(self.assistant.research_roadmap(horizon_months))
        except Exception as e:
            return _err("ROADMAP_ERROR", str(e))

    def full_intelligence_report(self) -> dict[str, Any]:
        try:
            return _ok(self.assistant.full_intelligence_report())
        except Exception as e:
            logger.error(f"full_intelligence_report: {traceback.format_exc()}")
            return _err("INTELLIGENCE_REPORT_ERROR", str(e))

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        try:
            return _ok(
                {
                    "database": self.db.stats(),
                    "observatory": {
                        "n_models": self.observatory.n_models,
                    },
                    "features": self.features.summary(),
                    "labels": self.labels.summary(),
                    "datasets": self.datasets.summary(),
                    "health": self.health.global_summary(),
                    "advisor": self.advisor.stats(),
                    "experiments": self.experiments.stats(),
                    "knowledge_base": {"total_entries": self.kb.n_entries},
                    "market": self.market.global_report(),
                }
            )
        except Exception as e:
            return _err("STATS_ERROR", str(e))
