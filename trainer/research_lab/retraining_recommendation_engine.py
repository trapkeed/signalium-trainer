"""S20 — Retraining Recommendation Engine.

Orchestrates all five evaluation dimensions, applies deduplication against
open recommendations, generates RetrainingRecommendation objects, and persists
them to the Research Lab DB.

Usage::

    engine = RetrainingRecommendationEngine(warehouse, db)
    results = engine.evaluate(model_version="v2.1.0")
    # → list[EvaluationResult]

    recs = engine.list_recommendations(status="PROPOSED")
    engine.approve(recommendation_id, resolved_by="head_quant")
    engine.dismiss(recommendation_id, resolved_by="head_quant", notes="false alarm")
"""

from __future__ import annotations

from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.research_lab.dimension_evaluators import (
    ConfidenceDriftEvaluator,
    PortfolioInefficiencyEvaluator,
    ReasoningDriftEvaluator,
    RegimeUnderperformEvaluator,
    SharpeDegradationEvaluator,
)
from trainer.research_lab.recommendation_models import (
    EvaluationResult,
    RecommendationConfig,
    RecommendationStatus,
    RecommendationUrgency,
    RetrainingRecommendation,
    _elevate_urgency,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.rl_models import ExperimentRecord, ExperimentStatus
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse


class RetrainingRecommendationEngine:
    """Evaluates champion model health across five dimensions and generates
    RetrainingRecommendation objects for admin review."""

    # S24 §3: RecommendedScope -> Experiment hyperparameter hints. Purely
    # advisory metadata attached to the auto-created experiment; it does not
    # change how the (still admin-gated) training pipeline executes.
    _SCOPE_HYPERPARAMS: dict[str, dict[str, Any]] = {
        "FULL_RETRAIN": {"retrain_mode": "full"},
        "FINE_TUNE": {"retrain_mode": "fine_tune"},
        "CALIBRATION_ONLY": {"retrain_mode": "calibration_only"},
        "HEAD_RETRAIN": {"retrain_mode": "head_retrain"},
    }

    def __init__(
        self,
        warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase,
        config: RecommendationConfig | None = None,
    ) -> None:
        self._wh = warehouse
        self._db = db
        self._cfg = config or RecommendationConfig()

        self._evaluators = [
            SharpeDegradationEvaluator(warehouse, self._cfg),
            ConfidenceDriftEvaluator(warehouse, self._cfg),
            ReasoningDriftEvaluator(warehouse, self._cfg),
            RegimeUnderperformEvaluator(warehouse, self._cfg),
            PortfolioInefficiencyEvaluator(warehouse, self._cfg),
        ]

    # ── Public: evaluation ────────────────────────────────────────────────────

    def evaluate(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[EvaluationResult]:
        """Run all five evaluators and return their results.

        For any dimension that triggers, also persist a deduplicated
        RetrainingRecommendation in PROPOSED status.
        """
        results: list[EvaluationResult] = []
        for ev in self._evaluators:
            try:
                r = ev.evaluate(model_version=model_version, since=since, until=until)
                results.append(r)
            except Exception as exc:
                logger.warning(
                    f"RetrainingRecommendationEngine: evaluator "
                    f"{type(ev).__name__} raised {exc!r} — skipping"
                )

        triggered = [r for r in results if r.triggered]
        n_triggered = len(triggered)

        # Build evaluation summary for embedding in each recommendation
        eval_summary = {r.dimension.value: r.to_dict() for r in results}

        mv = model_version or ""
        for result in triggered:
            # Adjust urgency upward when multiple dimensions fire
            urgency = _elevate_urgency(result.urgency, n_triggered)
            try:
                self._persist_recommendation(
                    model_version=mv,
                    result=result,
                    urgency=urgency,
                    evaluation_summary=eval_summary,
                )
            except Exception as exc:
                logger.warning(
                    f"RetrainingRecommendationEngine: failed to persist "
                    f"recommendation for {result.dimension.value}: {exc!r}"
                )

        return results

    # ── Public: lifecycle actions ─────────────────────────────────────────────

    def approve(self, recommendation_id: str, resolved_by: str = "admin") -> dict[str, Any]:
        """Approve a PROPOSED recommendation → APPROVED.

        S24 §3 (retraining trigger pipeline): approval also auto-creates a
        linked ``ExperimentRecord`` in PENDING status, bridging
        "recommendation approved" → "experiment exists" — this is a
        bookkeeping/proposal step only. It does **not** start training: an
        admin must still separately call ``start_experiment`` and clear the
        ``TRAINING_START`` approval gate before ``run_pipeline`` will
        execute the TRAINING stage (see ``pipeline_controller.py``). No
        code path here launches a pipeline job.
        """
        rec_dict = self._db.get_retraining_recommendation(recommendation_id)
        if rec_dict is None:
            raise ValueError(f"Recommendation {recommendation_id!r} not found")
        rec = _from_dict(rec_dict)
        rec.approve(resolved_by=resolved_by)
        self._db.upsert_retraining_recommendation(rec.to_dict())
        logger.info(f"RecommendationEngine: approved {recommendation_id} by {resolved_by}")

        experiment_id = self._maybe_create_linked_experiment(rec)
        result = rec.to_dict()
        result["experiment_id"] = experiment_id
        return result

    def _maybe_create_linked_experiment(self, rec: RetrainingRecommendation) -> str:
        """Create (or find) the ExperimentRecord bridging this approved
        recommendation to the training pipeline. Dedup'd on recommendation_id
        via experiment tags so re-approval attempts / repeated calls never
        create duplicates."""
        dedup_tag = f"from_recommendation:{rec.recommendation_id}"
        for status in ExperimentStatus:
            for existing in self._db.list_experiments(status=status.value):
                if dedup_tag in existing.get("tags", []):
                    return existing.get("experiment_id", "")

        hyperparameters = dict(self._SCOPE_HYPERPARAMS.get(rec.recommended_scope, {}))
        exp = ExperimentRecord.new(
            name=(
                f"Auto-proposed retraining ({rec.trigger_dimension}) for "
                f"{rec.model_version or 'unknown'}"
            ),
            qde_version=rec.model_version or "unknown",
            training_dataset="pending_selection",
            researcher=rec.resolved_by or "system",
            tags=["auto-proposed", "retraining_recommendation", dedup_tag],
            hyperparameters=hyperparameters,
            notes=(
                f"Auto-created from approved RetrainingRecommendation "
                f"{rec.recommendation_id}. Trigger: {rec.trigger_dimension} — "
                f"{rec.trigger_reason}"
            ),
        )
        self._db.upsert_experiment(exp.to_dict())
        logger.info(
            f"RecommendationEngine: created linked experiment {exp.experiment_id} "
            f"from recommendation {rec.recommendation_id}"
        )
        return exp.experiment_id

    def dismiss(
        self,
        recommendation_id: str,
        resolved_by: str = "admin",
        notes: str = "",
    ) -> dict[str, Any]:
        """Dismiss a PROPOSED recommendation → DISMISSED."""
        rec_dict = self._db.get_retraining_recommendation(recommendation_id)
        if rec_dict is None:
            raise ValueError(f"Recommendation {recommendation_id!r} not found")
        rec = _from_dict(rec_dict)
        rec.dismiss(resolved_by=resolved_by, notes=notes)
        self._db.upsert_retraining_recommendation(rec.to_dict())
        logger.info(f"RecommendationEngine: dismissed {recommendation_id} by {resolved_by}")
        return rec.to_dict()

    # ── Public: queries ───────────────────────────────────────────────────────

    def list_recommendations(
        self,
        status: str | None = None,
        model_version: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._db.list_retraining_recommendations(
            status=status, model_version=model_version, limit=limit
        )

    def get_recommendation(self, recommendation_id: str) -> dict[str, Any] | None:
        return self._db.get_retraining_recommendation(recommendation_id)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _persist_recommendation(
        self,
        model_version: str,
        result: EvaluationResult,
        urgency: RecommendationUrgency,
        evaluation_summary: dict[str, Any],
    ) -> None:
        """Persist a recommendation, skipping if an open one already exists
        for the same model_version + trigger_dimension."""
        existing_open = self._db.list_retraining_recommendations(
            status=RecommendationStatus.PROPOSED.value,
            model_version=model_version,
            limit=500,
        )
        for existing in existing_open:
            if existing.get("trigger_dimension") == result.dimension.value:
                logger.debug(
                    f"RecommendationEngine: skipping duplicate recommendation "
                    f"for {result.dimension.value} model={model_version!r}"
                )
                return

        rec = RetrainingRecommendation.new(
            model_version=model_version,
            trigger_dimension=result.dimension,
            trigger_reason=result.summary,
            evidence=result.evidence,
            recommended_scope=result.recommended_scope,
            urgency=urgency,
            evaluation_summary=evaluation_summary,
        )
        self._db.upsert_retraining_recommendation(rec.to_dict())
        logger.info(
            f"RecommendationEngine: created {rec.recommendation_id} "
            f"dim={result.dimension.value} urgency={urgency.value} "
            f"model={model_version!r}"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _from_dict(d: dict[str, Any]) -> RetrainingRecommendation:
    """Reconstruct a RetrainingRecommendation from a persisted dict."""
    return RetrainingRecommendation(
        recommendation_id=d["recommendation_id"],
        model_version=d.get("model_version", ""),
        trigger_dimension=d.get("trigger_dimension", ""),
        trigger_reason=d.get("trigger_reason", ""),
        evidence=d.get("evidence", {}),
        recommended_scope=d.get("recommended_scope", "FINE_TUNE"),
        urgency=d.get("urgency", "LOW"),
        status=d.get("status", "PROPOSED"),
        created_at=d.get("created_at", 0.0),
        updated_at=d.get("updated_at", 0.0),
        resolved_by=d.get("resolved_by", ""),
        resolved_at=d.get("resolved_at"),
        notes=d.get("notes", ""),
        evaluation_summary=d.get("evaluation_summary", {}),
    )
