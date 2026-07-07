"""Phase 15 — Validation Platform API.

Internal REST-like API layer for the institutional validation platform.
Returns structured dict responses (not HTTP responses) — the actual HTTP
layer is left to the CLI or a FastAPI wrapper.

API endpoints (method-style):
  - ValidationAPI.run_validation()         → run a complete validation
  - ValidationAPI.get_report()             → retrieve a stored report
  - ValidationAPI.get_leaderboard()        → query the leaderboard
  - ValidationAPI.compare_models()         → head-to-head comparison
  - ValidationAPI.get_promotion_status()   → check promotion eligibility
  - ValidationAPI.get_walk_forward()       → walk-forward fold results
  - ValidationAPI.get_drift_report()       → drift analysis for a run
  - ValidationAPI.get_failure_analysis()   → failure analysis for a run
  - ValidationAPI.approve_promotion()      → manually approve a promotion
  - ValidationAPI.retire_model()           → retire a model from leaderboard
  - ValidationAPI.search_leaderboard()     → full-text search

Usage::

    api = ValidationAPI(db, leaderboard, policy)
    response = api.get_leaderboard(sort_by="sharpe_ratio", limit=20)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# API response wrapper
# ---------------------------------------------------------------------------


def _ok(data: Any, message: str = "") -> dict[str, Any]:
    return {
        "status": "ok",
        "message": message,
        "data": data,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _error(message: str, code: str = "ERROR") -> dict[str, Any]:
    return {
        "status": "error",
        "code": code,
        "message": message,
        "data": None,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Validation API
# ---------------------------------------------------------------------------


class ValidationAPI:
    """Internal API for the Institutional Validation Platform.

    All methods return ``{"status": "ok"|"error", "data": ..., ...}``.

    Args:
        database: ``ValidationDatabase`` instance.
        leaderboard: ``InstitutionalLeaderboard`` instance.
        promotion_policy: ``PromotionPolicy`` instance.
    """

    def __init__(
        self,
        database: Any,
        leaderboard: Any,
        promotion_policy: Any,
    ) -> None:
        self._db = database
        self._lb = leaderboard
        self._policy = promotion_policy

    # ── Validation Runs ───────────────────────────────────────────────────

    def get_report(self, run_id: str) -> dict[str, Any]:
        """Retrieve a complete validation run report.

        Args:
            run_id: UUID of the validation run.

        Returns:
            API response with ``ValidationRunRecord`` data.
        """
        try:
            record = self._db.get_run(run_id)
            if not record:
                return _error(f"Run {run_id!r} not found.", code="NOT_FOUND")
            return _ok(
                {
                    "run_id": record.run_id,
                    "model_id": record.model_id,
                    "model_version": record.model_version,
                    "run_timestamp": record.run_timestamp,
                    "status": record.status,
                    "statistical_metrics": record.statistical_metrics,
                    "trading_metrics": record.trading_metrics,
                    "calibration": record.calibration,
                    "drift": record.drift,
                    "monte_carlo": record.monte_carlo,
                    "robustness": record.robustness,
                    "promotion_decision": record.promotion_decision,
                    "leaderboard_score": record.leaderboard_score,
                }
            )
        except Exception as exc:
            return _error(str(exc))

    def list_reports(
        self,
        model_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List validation runs with optional filtering."""
        try:
            records = self._db.list_runs(
                model_id=model_id, status=status, limit=limit, offset=offset
            )
            total = self._db.count_runs(model_id)
            return _ok(
                {
                    "runs": [
                        {
                            "run_id": r.run_id,
                            "model_id": r.model_id,
                            "model_version": r.model_version,
                            "run_timestamp": r.run_timestamp,
                            "status": r.status,
                            "leaderboard_score": r.leaderboard_score,
                        }
                        for r in records
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )
        except Exception as exc:
            return _error(str(exc))

    # ── Leaderboard ───────────────────────────────────────────────────────

    def get_leaderboard(
        self,
        sort_by: str = "overall_score",
        limit: int = 50,
        promotion_status: str | None = None,
        architecture: str | None = None,
        min_accuracy: float | None = None,
    ) -> dict[str, Any]:
        """Query the institutional leaderboard.

        Args:
            sort_by: Field to sort by.
            limit: Max entries.
            promotion_status: Filter by status.
            architecture: Filter by architecture.
            min_accuracy: Filter by minimum accuracy.

        Returns:
            API response with ranked leaderboard entries.
        """
        try:
            entries = self._lb.filter(
                promotion_status=promotion_status,
                architecture=architecture,
                min_accuracy=min_accuracy,
            )
            entries.sort(
                key=lambda e: float(getattr(e, sort_by, 0.0)),
                reverse=True,
            )
            return _ok(
                {
                    "entries": [e.to_dict() for e in entries[:limit]],
                    "total": len(entries),
                    "sort_by": sort_by,
                    "champion": self._lb.champion().to_dict() if self._lb.champion() else None,
                }
            )
        except Exception as exc:
            return _error(str(exc))

    def search_leaderboard(self, query: str) -> dict[str, Any]:
        """Full-text search across leaderboard entries."""
        try:
            results = self._lb.search(query)
            return _ok({"results": [e.to_dict() for e in results], "query": query})
        except Exception as exc:
            return _error(str(exc))

    # ── Model Comparison ──────────────────────────────────────────────────

    def compare_models(self, model_id_a: str, model_id_b: str) -> dict[str, Any]:
        """Compare two models from the leaderboard.

        Args:
            model_id_a, model_id_b: Model IDs to compare.

        Returns:
            API response with side-by-side comparison.
        """
        try:
            result = self._lb.compare(model_id_a, model_id_b)
            return _ok(result)
        except Exception as exc:
            return _error(str(exc))

    # ── Promotion ─────────────────────────────────────────────────────────

    def get_promotion_status(self, run_id: str) -> dict[str, Any]:
        """Get the promotion eligibility for a validation run.

        Args:
            run_id: Validation run UUID.

        Returns:
            API response with promotion decision details.
        """
        try:
            record = self._db.get_run(run_id)
            if not record:
                return _error(f"Run {run_id!r} not found.", code="NOT_FOUND")
            return _ok(
                {
                    "run_id": run_id,
                    "model_id": record.model_id,
                    "promotion_decision": record.promotion_decision,
                }
            )
        except Exception as exc:
            return _error(str(exc))

    def approve_promotion(
        self,
        run_id: str,
        approver: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        """Manually approve promotion for an eligible model.

        Args:
            run_id: Validation run UUID.
            approver: Approver identifier.
            notes: Approval notes.

        Returns:
            API response confirming approval.
        """
        try:
            record = self._db.get_run(run_id)
            if not record:
                return _error(f"Run {run_id!r} not found.", code="NOT_FOUND")
            promo = record.promotion_decision
            if not promo.get("is_eligible", False):
                return _error(
                    f"Model {record.model_id!r} is NOT eligible for promotion.",
                    code="NOT_ELIGIBLE",
                )
            promo["is_promoted"] = True
            promo["approved_by"] = approver
            promo["approval_notes"] = notes
            promo["approved_at"] = datetime.now(UTC).isoformat()
            self._db.update_status(run_id, "completed")

            # Update leaderboard
            self._lb.update(
                record.model_id,
                {
                    "promotion_status": "champion",
                    "is_champion": True,
                },
            )
            logger.info(
                f"ValidationAPI: model {record.model_id!r} approved for promotion by {approver!r}"
            )
            return _ok({"run_id": run_id, "model_id": record.model_id, "approved": True})
        except Exception as exc:
            return _error(str(exc))

    def retire_model(self, model_id: str, reason: str = "") -> dict[str, Any]:
        """Retire a model from the leaderboard.

        Args:
            model_id: Model to retire.
            reason: Retirement reason.

        Returns:
            API response confirming retirement.
        """
        try:
            self._lb.update(
                model_id,
                {
                    "promotion_status": "retired",
                    "is_retired": True,
                    "is_champion": False,
                    "notes": reason,
                },
            )
            logger.info(f"ValidationAPI: model {model_id!r} retired. Reason: {reason}")
            return _ok({"model_id": model_id, "status": "retired"})
        except Exception as exc:
            return _error(str(exc))

    # ── Drift & Failure Analysis ──────────────────────────────────────────

    def get_drift_report(self, run_id: str) -> dict[str, Any]:
        """Get drift analysis for a specific validation run."""
        try:
            record = self._db.get_run(run_id)
            if not record:
                return _error(f"Run {run_id!r} not found.", code="NOT_FOUND")
            return _ok({"run_id": run_id, "drift": record.drift})
        except Exception as exc:
            return _error(str(exc))

    def get_failure_analysis(self, run_id: str) -> dict[str, Any]:
        """Get failure analysis for a specific validation run."""
        try:
            record = self._db.get_run(run_id)
            if not record:
                return _error(f"Run {run_id!r} not found.", code="NOT_FOUND")
            robustness = record.robustness
            return _ok(
                {
                    "run_id": run_id,
                    "model_id": record.model_id,
                    "robustness_score": robustness.get("overall_robustness_score", 0.0),
                    "failed_scenarios": [
                        s for s in robustness.get("scenarios", []) if not s.get("passed", True)
                    ],
                }
            )
        except Exception as exc:
            return _error(str(exc))

    def get_walk_forward_results(self, run_id: str) -> dict[str, Any]:
        """Get walk-forward fold metrics for a run."""
        try:
            record = self._db.get_run(run_id)
            if not record:
                return _error(f"Run {run_id!r} not found.", code="NOT_FOUND")
            viz = record.visualization_data
            return _ok(
                {
                    "run_id": run_id,
                    "walk_forward_timeline": viz.get("walk_forward_timeline"),
                    "performance_trend": viz.get("performance_trend"),
                    "n_folds": record.model_metadata.get("n_validation_folds", 0),
                }
            )
        except Exception as exc:
            return _error(str(exc))
