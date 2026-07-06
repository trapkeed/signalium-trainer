"""Phase 19.2 — IRSE Structured API.

Provides a structured, JSON-serialisable API layer over the core IRSE
objects.  Every endpoint returns a consistent envelope::

    {
        "status": "ok" | "error",
        "data":   { ... },
        "error":  null | "message"
    }

Endpoints:

  Research Scores
    score_experiment(inputs)              — compute + store a new score
    get_score(score_id)                   — retrieve by ID
    get_latest_score(experiment_id)       — latest score for an experiment
    score_breakdown(score_id)             — full category breakdown

  Leaderboards
    get_leaderboard(axis, filters, limit) — ranked leaderboard
    get_promotion_candidates()            — promotion-eligible list
    get_category_leaders()                — leader per category

  Benchmarking
    register_benchmark(experiment_id, role)   — register for benchmarking
    compare_experiments(subject, reference)   — pairwise comparison
    benchmark_report(experiment_id)           — full report
    full_benchmark()                          — all registered models

  Historical Trends
    score_history(experiment_id)          — time-series snapshots
    trend_report()                        — improvement trend + distribution
    detect_regressions(window, threshold) — regression alerts

  Weight Management
    set_weights(weights_dict)             — update scorer weights
    get_weights()                         — current weights
    save_weight_preset(name, weights)     — save a named preset
    list_weight_presets()                 — all saved presets

  Export
    export_leaderboard(fmt)               — JSON/CSV
    export_score(score_id, fmt)           — full score export

Usage::

    api = IRSEApi(scorer, leaderboard, trend_tracker, benchmarker, db)
    response = api.score_experiment(inputs)
    print(response["data"]["composite"])
"""

from __future__ import annotations

import csv
import io
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------


def _ok(data: Any = None) -> Dict[str, Any]:
    return {"status": "ok", "data": data, "error": None}


def _err(msg: str) -> Dict[str, Any]:
    return {"status": "error", "data": None, "error": msg}


# ---------------------------------------------------------------------------
# IRSEApi
# ---------------------------------------------------------------------------


class IRSEApi:
    """Structured API over the IRSE scoring system."""

    def __init__(
        self,
        scorer: Any,
        leaderboard: Any,
        trend_tracker: Any,
        benchmarker: Any,
        database: Any,
    ) -> None:
        self._scorer = scorer
        self._lb = leaderboard
        self._trends = trend_tracker
        self._bm = benchmarker
        self._db = database

    # ------------------------------------------------------------------
    # Research Scores
    # ------------------------------------------------------------------

    def score_experiment(self, inputs: Any) -> Dict[str, Any]:
        """Compute a Research Score and persist + index it."""
        try:
            rs = self._scorer.score(inputs)
            rs_dict = rs.to_dict()

            self._db.save_research_score(rs_dict)
            self._lb.add(rs)
            self._trends.record_from_research_score(rs)

            snap = self._trends.record_from_research_score(rs)
            self._db.save_score_snapshot(snap.to_dict())

            return _ok(
                {
                    "score_id": rs.score_id,
                    "experiment_id": rs.experiment_id,
                    "composite": rs.composite,
                    "grade_label": rs.grade_label,
                    "promotion_eligible": rs.promotion_eligible,
                    "deployment_eligible": rs.deployment_eligible,
                    "confidence_score": rs.confidence_score(),
                }
            )
        except Exception as exc:
            return _err(f"Scoring failed: {exc}\n{traceback.format_exc()}")

    def score_and_get_full(self, inputs: Any) -> Dict[str, Any]:
        """Score and return the full ResearchScore dict."""
        try:
            rs = self._scorer.score(inputs)
            self._db.save_research_score(rs.to_dict())
            self._lb.add(rs)
            return _ok(rs.to_dict())
        except Exception as exc:
            return _err(str(exc))

    def get_score(self, score_id: str) -> Dict[str, Any]:
        try:
            row = self._db.get_research_score(score_id)
            if row is None:
                return _err(f"Score not found: {score_id}")
            return _ok(row)
        except Exception as exc:
            return _err(str(exc))

    def get_latest_score(self, experiment_id: str) -> Dict[str, Any]:
        try:
            row = self._db.get_latest_for_experiment(experiment_id)
            if row is None:
                return _err(f"No scores found for experiment: {experiment_id}")
            return _ok(row)
        except Exception as exc:
            return _err(str(exc))

    def score_breakdown(self, score_id: str) -> Dict[str, Any]:
        try:
            row = self._db.get_research_score(score_id)
            if row is None:
                return _err(f"Score not found: {score_id}")
            return _ok(
                {
                    "research_score": row,
                    "weights": self._scorer.weights.as_dict(),
                    "promotion_threshold": 80.0,
                    "deployment_threshold": 85.0,
                }
            )
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Leaderboards
    # ------------------------------------------------------------------

    def get_leaderboard(
        self,
        axis: str = "research_score",
        limit: int = 20,
        instrument: Optional[str] = None,
        timeframe: Optional[str] = None,
        backbone: Optional[str] = None,
        architecture: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_version: Optional[str] = None,
        model_family: Optional[str] = None,
        grade_label: Optional[str] = None,
        promotion_eligible: Optional[bool] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            lb = self._lb.filter(
                instrument=instrument,
                timeframe=timeframe,
                backbone=backbone,
                architecture=architecture,
                dataset_version=dataset_version,
                feature_version=feature_version,
                model_family=model_family,
                grade_label=grade_label,
                promotion_eligible=promotion_eligible,
                min_score=min_score,
            )
            report = lb.leaderboard_report(top_n=limit, axis=axis)
            return _ok(report)
        except Exception as exc:
            return _err(str(exc))

    def get_promotion_candidates(self) -> Dict[str, Any]:
        try:
            candidates = self._lb.promotion_candidates()
            return _ok(
                {
                    "count": len(candidates),
                    "candidates": candidates,
                }
            )
        except Exception as exc:
            return _err(str(exc))

    def get_category_leaders(self) -> Dict[str, Any]:
        try:
            return _ok(self._lb.category_leaders())
        except Exception as exc:
            return _err(str(exc))

    def get_grade_distribution(self) -> Dict[str, Any]:
        try:
            return _ok(self._lb.grade_distribution())
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def register_benchmark(
        self,
        experiment_id: str,
        research_score: Any,
        role: str = "experimental",
    ) -> Dict[str, Any]:
        try:
            record = self._bm.register(experiment_id, research_score, role)
            self._db.save_benchmark_record(record.to_dict())
            return _ok(
                {
                    "experiment_id": experiment_id,
                    "role": role,
                    "composite": record.composite,
                }
            )
        except Exception as exc:
            return _err(str(exc))

    def compare_experiments(
        self,
        subject_id: str,
        reference_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            comp = self._bm.compare(subject_id, against=reference_id)
            comp_dict = comp.to_dict()
            self._db.save_benchmark_comparison(comp_dict)
            return _ok(comp_dict)
        except Exception as exc:
            return _err(str(exc))

    def benchmark_report(self, experiment_id: str) -> Dict[str, Any]:
        try:
            report = self._bm.benchmark_report(experiment_id)
            return _ok(report)
        except Exception as exc:
            return _err(str(exc))

    def full_benchmark(self) -> Dict[str, Any]:
        try:
            return _ok(self._bm.full_benchmark_report())
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Historical Trends
    # ------------------------------------------------------------------

    def score_history(
        self, experiment_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        try:
            history = self._db.get_score_history(experiment_id, limit=limit)
            return _ok({"count": len(history), "history": history})
        except Exception as exc:
            return _err(str(exc))

    def trend_report(self) -> Dict[str, Any]:
        try:
            return _ok(self._trends.trend_report())
        except Exception as exc:
            return _err(str(exc))

    def detect_regressions(self, window: int = 5, threshold: float = 5.0) -> Dict[str, Any]:
        try:
            alerts = self._trends.detect_regressions(window, threshold)
            return _ok(
                {
                    "count": len(alerts),
                    "regressions": [a.to_dict() for a in alerts],
                }
            )
        except Exception as exc:
            return _err(str(exc))

    def champion_timeline(self) -> Dict[str, Any]:
        try:
            return _ok(self._trends.champion_timeline())
        except Exception as exc:
            return _err(str(exc))

    def score_distribution(self) -> Dict[str, Any]:
        try:
            return _ok(self._trends.score_distribution())
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, Any]:
        try:
            return _ok(self._scorer.weights.as_dict())
        except Exception as exc:
            return _err(str(exc))

    def set_weights(self, weights_dict: Dict[str, float]) -> Dict[str, Any]:
        try:
            from trainer.research.weights import ScoringWeights

            new_weights = ScoringWeights.from_dict(weights_dict)
            new_weights.validate()
            self._scorer.update_weights(new_weights)
            return _ok({"weights": new_weights.as_dict(), "applied": True})
        except Exception as exc:
            return _err(str(exc))

    def save_weight_preset(
        self, name: str, weights: Dict[str, float], description: str = ""
    ) -> Dict[str, Any]:
        try:
            config_id = self._db.save_weight_configuration(name, weights, description)
            return _ok({"config_id": config_id, "name": name})
        except Exception as exc:
            return _err(str(exc))

    def list_weight_presets(self) -> Dict[str, Any]:
        try:
            presets = self._db.list_weight_configurations()
            return _ok({"count": len(presets), "presets": presets})
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Export helpers (JSON / CSV)
    # ------------------------------------------------------------------

    def export_leaderboard(self, fmt: str = "json", limit: int = 100) -> Dict[str, Any]:
        try:
            entries = self._lb.rank_by("research_score")[:limit]
            if fmt == "json":
                return _ok({"format": "json", "data": json.dumps(entries, indent=2)})
            elif fmt == "csv":
                if not entries:
                    return _ok({"format": "csv", "data": ""})
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=list(entries[0].keys()))
                writer.writeheader()
                for e in entries:
                    row = {
                        k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                        for k, v in e.items()
                    }
                    writer.writerow(row)
                return _ok({"format": "csv", "data": buf.getvalue()})
            else:
                return _err(f"Unsupported format: {fmt}. Use 'json' or 'csv'.")
        except Exception as exc:
            return _err(str(exc))

    def export_score(self, score_id: str, fmt: str = "json") -> Dict[str, Any]:
        try:
            row = self._db.get_research_score(score_id)
            if row is None:
                return _err(f"Score not found: {score_id}")
            if fmt == "json":
                return _ok({"format": "json", "data": json.dumps(row, indent=2)})
            elif fmt == "csv":
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(["field", "value"])
                for k, v in row.items():
                    writer.writerow([k, json.dumps(v) if isinstance(v, (list, dict)) else v])
                return _ok({"format": "csv", "data": buf.getvalue()})
            else:
                return _err(f"Unsupported format: {fmt}")
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    def system_stats(self) -> Dict[str, Any]:
        try:
            return _ok(
                {
                    "database": self._db.stats(),
                    "leaderboard_entries": len(self._lb.all_entries()),
                    "trend_snapshots": len(self._trends.all_snapshots()),
                    "weights": self._scorer.weights.as_dict(),
                }
            )
        except Exception as exc:
            return _err(str(exc))
