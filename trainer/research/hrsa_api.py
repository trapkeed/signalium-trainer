"""Phase 19.3 — HRSA API.

Single entry-point wrapping the full HRSA scoring pipeline:

  score()              — score from raw metrics bundle
  get_score()          — retrieve stored score
  category_scores()    — retrieve Level 2 breakdown
  metric_scores()      — retrieve Level 1 raw values
  benchmarks()         — benchmark comparisons
  historical_trends()  — score evolution
  recommendations()    — improvement actions
  what_if()            — counterfactual simulation
  compare()            — compare two experiments

All responses follow the envelope: {"status": "ok", "data": ...}
Errors:  {"status": "error", "error": "..."}

Usage::

    api = HRSAApi()
    resp = api.score(bundle, name="my_model")
    resp = api.what_if(experiment_id, [
        {"category": "calibration", "new_score": 85.0}
    ])
    resp = api.leaderboard(top_n=10)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trainer.research.hrsa_benchmarks import HRSABenchmarkEngine
from trainer.research.hrsa_categories import HRSACategoryScorer
from trainer.research.hrsa_database import HRSADatabase
from trainer.research.hrsa_explanations import HRSAExplainer
from trainer.research.hrsa_grades import HRSAGradeEngine
from trainer.research.hrsa_scorer import HierarchicalResearchScorer
from trainer.research.hrsa_trends import HRSATrendEngine
from trainer.research.hrsa_whatif import HRSAWhatIfSimulator
from trainer.research.raw_metrics import RawMetricsBundle, RawMetricsRegistry

# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _ok(data: Any) -> Dict[str, Any]:
    return {"status": "ok", "data": data}


def _err(msg: str) -> Dict[str, Any]:
    return {"status": "error", "error": msg}


# ---------------------------------------------------------------------------
# HRSAApi
# ---------------------------------------------------------------------------


class HRSAApi:
    """Unified API for the Hierarchical Research Scoring Architecture."""

    def __init__(
        self,
        db_path: str = ":memory:",
        registry: Optional[RawMetricsRegistry] = None,
        scorer: Optional[HierarchicalResearchScorer] = None,
        grade_engine: Optional[HRSAGradeEngine] = None,
    ) -> None:
        self._registry = registry or RawMetricsRegistry.default()
        self._scorer = scorer or HierarchicalResearchScorer()
        self._grade_engine = grade_engine or HRSAGradeEngine.default()
        self._db = HRSADatabase(db_path)
        self._trends = HRSATrendEngine()
        self._benchmarks = HRSABenchmarkEngine()
        self._explainer = HRSAExplainer()
        self._simulator = HRSAWhatIfSimulator(scorer=self._scorer)
        self._bundles: Dict[str, RawMetricsBundle] = {}  # experiment_id → bundle

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        bundle: RawMetricsBundle,
        *,
        name: str = "",
        instrument: str = "",
        timeframe: str = "",
        backbone: str = "",
        model_family: str = "",
        trend_slope: float = 0.0,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Score a RawMetricsBundle through the full 3-level pipeline."""
        try:
            hs = self._scorer.score(
                bundle,
                name=name,
                instrument=instrument,
                timeframe=timeframe,
                backbone=backbone,
                model_family=model_family,
                trend_slope=trend_slope,
            )
            if save:
                self._db.save(hs)
                self._trends.record(hs)
            self._bundles[bundle.experiment_id] = bundle
            return _ok(hs.to_dict())
        except Exception as exc:
            return _err(str(exc))

    def get_score(self, score_id: str) -> Dict[str, Any]:
        row = self._db.get_score(score_id)
        if row is None:
            return _err(f"Score not found: {score_id}")
        return _ok(row)

    def get_latest(self, experiment_id: str) -> Dict[str, Any]:
        row = self._db.get_latest(experiment_id)
        if row is None:
            return _err(f"No scores for experiment: {experiment_id}")
        return _ok(row)

    # ------------------------------------------------------------------
    # Category & metric details
    # ------------------------------------------------------------------

    def category_scores(self, score_id: str) -> Dict[str, Any]:
        cats = self._db.category_scores_for(score_id)
        if not cats:
            return _err(f"No category scores for: {score_id}")
        return _ok({"score_id": score_id, "categories": cats})

    def metric_scores(
        self, experiment_id: str, metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if metric_name:
            history = self._db.metric_history(experiment_id, metric_name)
            return _ok({"experiment_id": experiment_id, "metric": metric_name, "history": history})
        # Return all latest raw values from trends
        return _ok(
            {
                "experiment_id": experiment_id,
                "metrics_tracked": list(self._trends._metric_series.keys()),
            }
        )

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def register_benchmark(
        self,
        experiment_id: str,
        bundle: RawMetricsBundle,
        role: str = "candidate",
        name: str = "",
        generation: str = "",
    ) -> Dict[str, Any]:
        try:
            hs = self._scorer.score(bundle, name=name)
            rec = self._benchmarks.register(experiment_id, hs, role=role, generation=generation)
            self._db.save_benchmark(rec)
            return _ok(rec.to_dict())
        except Exception as exc:
            return _err(str(exc))

    def compare_to_benchmarks(
        self,
        experiment_id: str,
        bundle: RawMetricsBundle,
        roles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        try:
            hs = self._scorer.score(bundle)
            suite = self._benchmarks.compare(experiment_id, hs, roles=roles)
            return _ok(suite.to_dict())
        except Exception as exc:
            return _err(str(exc))

    def benchmark_report(self) -> Dict[str, Any]:
        return _ok(self._benchmarks.full_report())

    # ------------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------------

    def historical_trends(self, experiment_id: str, window: int = 0) -> Dict[str, Any]:
        history = self._db.composite_history(experiment_id)
        cat_trend = self._trends.category_trends(window)
        comp_trend = self._trends.composite_trend(window)
        regressions = self._trends.detect_regressions()
        return _ok(
            {
                "experiment_id": experiment_id,
                "composite_history": history,
                "composite_trend": comp_trend.to_dict(),
                "category_trends": {k: v.to_dict() for k, v in cat_trend.items()},
                "regression_alerts": [r.to_dict() for r in regressions],
                "improvement_rate": self._trends.improvement_rate(window or 10),
            }
        )

    def trend_report(self, window: int = 0) -> Dict[str, Any]:
        return _ok(self._trends.trend_report(window))

    # ------------------------------------------------------------------
    # Explanations & recommendations
    # ------------------------------------------------------------------

    def explain(self, score_id: str, bundle: Optional[RawMetricsBundle] = None) -> Dict[str, Any]:
        row = self._db.get_score(score_id)
        if row is None:
            return _err(f"Score not found: {score_id}")
        cached = self._db.get_explanation(score_id)
        if cached:
            return _ok(cached)
        # Rebuild HierarchicalScore from scratch for explanation
        exp_id = row["experiment_id"]
        b = bundle or self._bundles.get(exp_id)
        if b is None:
            return _err("Bundle required for explanation (not cached)")
        hs = self._scorer.score(b, name=row["name"])
        explanation = self._explainer.explain(hs)
        self._db.save_explanation(explanation)
        return _ok(explanation.to_dict())

    def recommendations(
        self, experiment_id: str, bundle: Optional[RawMetricsBundle] = None
    ) -> Dict[str, Any]:
        b = bundle or self._bundles.get(experiment_id)
        if b is None:
            return _err("Bundle required for recommendations (not cached)")
        hs = self._scorer.score(b)
        explanation = self._explainer.explain(hs)
        return _ok(
            {
                "experiment_id": experiment_id,
                "recommendations": [r.to_dict() for r in explanation.recommendations],
                "improvement_potential": explanation.improvement_potential,
            }
        )

    # ------------------------------------------------------------------
    # What-If
    # ------------------------------------------------------------------

    def what_if(
        self,
        experiment_id: str,
        scenarios: List[Dict[str, Any]],
        bundle: Optional[RawMetricsBundle] = None,
    ) -> Dict[str, Any]:
        """Run category-level what-if scenarios.

        Each scenario: {"name": "...", "changes": {"category": new_score, ...}}
        """
        b = bundle or self._bundles.get(experiment_id)
        if b is None:
            return _err("Bundle required for what-if (not cached)")
        try:
            hs = self._scorer.score(b)
            results = self._simulator.run_batch(hs, scenarios)
            for r in results:
                self._db.save_whatif(experiment_id, r)
            return _ok(
                {
                    "experiment_id": experiment_id,
                    "base_composite": round(hs.level3_composite, 2),
                    "simulations": [r.to_dict() for r in results],
                }
            )
        except Exception as exc:
            return _err(str(exc))

    def what_if_optimal(
        self,
        experiment_id: str,
        bundle: Optional[RawMetricsBundle] = None,
        budget: int = 3,
    ) -> Dict[str, Any]:
        b = bundle or self._bundles.get(experiment_id)
        if b is None:
            return _err("Bundle required for optimal what-if (not cached)")
        try:
            hs = self._scorer.score(b)
            results = self._simulator.optimal_improvements(hs, budget_categories=budget)
            for r in results:
                self._db.save_whatif(experiment_id, r)
            return _ok(
                {
                    "experiment_id": experiment_id,
                    "base_composite": round(hs.level3_composite, 2),
                    "optimal_improvements": [r.to_dict() for r in results],
                    "budget": budget,
                }
            )
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Leaderboard & comparison
    # ------------------------------------------------------------------

    def leaderboard(
        self,
        top_n: int = 20,
        instrument: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        rows = self._db.leaderboard(top_n, instrument, min_confidence)
        return _ok({"leaderboard": rows, "count": len(rows)})

    def compare(
        self,
        experiment_a: str,
        bundle_a: RawMetricsBundle,
        experiment_b: str,
        bundle_b: RawMetricsBundle,
    ) -> Dict[str, Any]:
        try:
            hs_a = self._scorer.score(bundle_a, name=experiment_a)
            hs_b = self._scorer.score(bundle_b, name=experiment_b)
            comparison = self._explainer.compare_explanations(hs_a, hs_b)
            return _ok(
                {
                    "a": {"id": experiment_a, "score": hs_a.level3_composite},
                    "b": {"id": experiment_b, "score": hs_b.level3_composite},
                    "comparison": comparison,
                }
            )
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    def registry_info(self) -> Dict[str, Any]:
        return _ok(
            {
                "total_metrics": self._registry.total(),
                "categories": self._registry.all_categories(),
                "metric_count_per_category": {
                    cat: len(self._registry.for_category(cat))
                    for cat in self._registry.all_categories()
                },
            }
        )

    def grade_system(self) -> Dict[str, Any]:
        return _ok(
            {
                "institutional_grades": self._grade_engine.all_institutional_grades(),
                "research_grades": self._grade_engine.all_research_grades(),
            }
        )

    def system_stats(self) -> Dict[str, Any]:
        db_stats = self._db.stats()
        return _ok(
            {
                "database": db_stats,
                "experiments_in_memory": len(self._bundles),
                "categories_tracked_in_trends": len(self._trends._category_series),
                "metrics_tracked_in_trends": len(self._trends._metric_series),
                "benchmarks_registered": len(self._benchmarks.all_records()),
            }
        )
