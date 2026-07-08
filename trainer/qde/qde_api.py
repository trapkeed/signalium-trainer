"""Phase 27.6 — QDE Lifecycle API.

Public interface for the Quant Decision Engine lifecycle management system.

Endpoints:
 1.  register_version        — Register a new QDE version
 2.  get_version             — Retrieve a QDE version
 3.  list_versions           — List all QDE versions (optionally filtered by stage)
 4.  start_pipeline          — Start the training pipeline for a version
 5.  advance_pipeline        — Advance to the next pipeline stage
 6.  fail_pipeline           — Fail the current pipeline stage
 7.  skip_pipeline_stage     — Skip the current pipeline stage
 8.  get_pipeline_run        — Get pipeline run status
 9.  evaluate_promotion      — Evaluate whether a version can be promoted
10.  approve_promotion       — Manually approve a promotion decision
11.  auto_promote            — Evaluate and auto-promote if eligible
12.  rollback_to             — Immediate rollback to a specific version
13.  rollback_to_previous    — Rollback to the most recent previous version
14.  rollback_history        — Full rollback audit trail
15.  eligible_rollback_targets — List rollback-eligible versions
16.  create_shadow_test      — Create a shadow deployment test
17.  create_side_by_side_test — Create a side-by-side evaluation
18.  record_ab_results       — Record A/B test metrics
19.  conclude_ab_test        — Conclude an A/B test with a winner
20.  compare_ab_test         — Statistical comparison of A/B test results
21.  list_ab_tests           — List all A/B tests
22.  active_version          — Get the current active QDE version
23.  registry_summary        — High-level registry statistics
24.  version_lineage         — Ancestry chain for a version
25.  promotion_history       — All past promotion decisions
26.  promotion_criteria      — Current promotion criteria configuration
27.  dispatch_shadow_decision — Dual-dispatch one live input to champion + challenger
28.  evaluate_shadow_promotion — Feed a concluded shadow test's live-traffic signal
                                 into the backtest-based promotion engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.qde_ab_testing import QDEABTestingEngine
from trainer.qde.qde_models import QDEABTestType, QDEPromotionCriteria, QDEStage, QDEVersion
from trainer.qde.qde_pipeline import QDEPipeline
from trainer.qde.qde_promotion import QDEPromotionEngine
from trainer.qde.qde_registry import QDERegistry
from trainer.qde.qde_rollback import QDERollbackManager
from trainer.qde.qde_shadow import ShadowDispatcher, evaluate_and_maybe_promote
from trainer.research_lab.rl_models import ShadowTestRecommendation, ShadowTestRun, ShadowTestStatus


@dataclass
class ApiResponse:
    success: bool
    endpoint: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "success": self.success,
            "endpoint": self.endpoint,
            "data": self.data,
        }
        if self.error:
            d["error"] = self.error
        if self.warnings:
            d["warnings"] = self.warnings
        return d


def _ok(endpoint: str, data: dict[str, Any]) -> ApiResponse:
    return ApiResponse(True, endpoint, data)


def _err(endpoint: str, error: str) -> ApiResponse:
    return ApiResponse(False, endpoint, error=error)


class QDELifecycleAPI:
    """Public API for the QDE Lifecycle Management system."""

    def __init__(
        self,
        storage_path: str | None = None,
        criteria: QDEPromotionCriteria | None = None,
        auto_promote: bool = False,
    ) -> None:
        self._registry = QDERegistry(storage_path=storage_path)
        self._pipeline = QDEPipeline(self._registry)
        self._promotion = QDEPromotionEngine(
            self._registry,
            criteria=criteria,
            auto_promote=auto_promote,
        )
        self._rollback = QDERollbackManager(self._registry)
        self._ab = QDEABTestingEngine(self._registry)
        self._shadow_dispatcher = ShadowDispatcher(self._registry)

    # ── 1. Register Version ───────────────────────────────────────────────────

    def register_version(
        self,
        label: str = "",
        parent_version_id: str | None = None,
        tags: list[str] | None = None,
        description: str = "",
        artifact_uri: str = "",
        version_number: int | None = None,
    ) -> ApiResponse:
        try:
            vn = version_number or self._registry.next_version_number()
            version = QDEVersion.new(
                version_number=vn,
                label=label,
                parent_version_id=parent_version_id,
                tags=tags,
                description=description,
                artifact_uri=artifact_uri,
            )
            self._registry.register(version)
            logger.info(f"QDELifecycleAPI.register_version: {version.version_id}")
            return _ok("register_version", version.to_dict())
        except Exception as exc:
            logger.error(f"register_version error: {exc}")
            return _err("register_version", str(exc))

    # ── 2. Get Version ────────────────────────────────────────────────────────

    def get_version(self, version_id: str) -> ApiResponse:
        version = self._registry.get(version_id)
        if version is None:
            return _err("get_version", f"Version {version_id!r} not found")
        return _ok("get_version", version.to_dict())

    # ── 3. List Versions ──────────────────────────────────────────────────────

    def list_versions(self, stage: str | None = None) -> ApiResponse:
        try:
            if stage:
                versions = self._registry.list_by_stage(QDEStage(stage))
            else:
                versions = self._registry.list_all()
            return _ok(
                "list_versions",
                {
                    "count": len(versions),
                    "versions": [v.to_dict() for v in versions],
                    "filter_stage": stage,
                },
            )
        except ValueError as exc:
            return _err("list_versions", f"Invalid stage: {exc}")

    # ── 4. Start Pipeline ─────────────────────────────────────────────────────

    def start_pipeline(self, version_id: str) -> ApiResponse:
        try:
            run = self._pipeline.start(version_id)
            # Auto-start first step
            first = run.current_step()
            if first:
                import time

                first.started_at = time.time()
                first.status = "RUNNING"
            return _ok("start_pipeline", run.to_dict())
        except Exception as exc:
            logger.error(f"start_pipeline error: {exc}")
            return _err("start_pipeline", str(exc))

    # ── 5. Advance Pipeline ───────────────────────────────────────────────────

    def advance_pipeline(
        self,
        run_id: str,
        metrics: dict[str, float] | None = None,
        logs: list[str] | None = None,
    ) -> ApiResponse:
        try:
            step = self._pipeline.advance(run_id, metrics=metrics, logs=logs)
            run = self._pipeline.get_run(run_id)
            return _ok(
                "advance_pipeline",
                {
                    "completed_step": step.to_dict(),
                    "run": run.to_dict() if run else {},
                },
            )
        except Exception as exc:
            logger.error(f"advance_pipeline error: {exc}")
            return _err("advance_pipeline", str(exc))

    # ── 6. Fail Pipeline ──────────────────────────────────────────────────────

    def fail_pipeline(
        self,
        run_id: str,
        error: str,
        logs: list[str] | None = None,
    ) -> ApiResponse:
        try:
            step = self._pipeline.fail(run_id, error=error, logs=logs)
            run = self._pipeline.get_run(run_id)
            return _ok(
                "fail_pipeline",
                {
                    "failed_step": step.to_dict(),
                    "run": run.to_dict() if run else {},
                },
            )
        except Exception as exc:
            return _err("fail_pipeline", str(exc))

    # ── 7. Skip Pipeline Stage ────────────────────────────────────────────────

    def skip_pipeline_stage(self, run_id: str, reason: str = "") -> ApiResponse:
        try:
            step = self._pipeline.skip(run_id, reason=reason)
            run = self._pipeline.get_run(run_id)
            return _ok(
                "skip_pipeline_stage",
                {
                    "skipped_step": step.to_dict(),
                    "run": run.to_dict() if run else {},
                },
            )
        except Exception as exc:
            return _err("skip_pipeline_stage", str(exc))

    # ── 8. Get Pipeline Run ───────────────────────────────────────────────────

    def get_pipeline_run(self, run_id: str) -> ApiResponse:
        run = self._pipeline.get_run(run_id)
        if run is None:
            return _err("get_pipeline_run", f"Run {run_id!r} not found")
        return _ok("get_pipeline_run", run.to_dict())

    def list_pipeline_runs(self) -> ApiResponse:
        return _ok("list_pipeline_runs", {"runs": self._pipeline.all_runs()})

    # ── 9. Evaluate Promotion ─────────────────────────────────────────────────

    def evaluate_promotion(
        self,
        challenger_version_id: str,
        notes: str = "",
    ) -> ApiResponse:
        try:
            decision = self._promotion.evaluate(challenger_version_id, notes=notes)
            return _ok("evaluate_promotion", decision.to_dict())
        except Exception as exc:
            logger.error(f"evaluate_promotion error: {exc}")
            return _err("evaluate_promotion", str(exc))

    # ── 10. Approve Promotion ─────────────────────────────────────────────────

    def approve_promotion(
        self,
        challenger_version_id: str,
        approver: str = "",
        notes: str = "",
    ) -> ApiResponse:
        try:
            decision = self._promotion.evaluate(challenger_version_id, notes=notes)
            if not decision.is_eligible:
                return _ok(
                    "approve_promotion",
                    {
                        "approved": False,
                        "reason": "Not eligible",
                        "failed_dimensions": decision.failed_dimensions,
                        "decision": decision.to_dict(),
                    },
                )
            self._promotion.approve(decision, approver=approver)
            return _ok("approve_promotion", decision.to_dict())
        except Exception as exc:
            logger.error(f"approve_promotion error: {exc}")
            return _err("approve_promotion", str(exc))

    # ── 11. Auto-Promote ──────────────────────────────────────────────────────

    def auto_promote(
        self,
        challenger_version_id: str,
        notes: str = "",
    ) -> ApiResponse:
        try:
            decision = self._promotion.auto_evaluate_and_promote(challenger_version_id, notes=notes)
            return _ok("auto_promote", decision.to_dict())
        except Exception as exc:
            logger.error(f"auto_promote error: {exc}")
            return _err("auto_promote", str(exc))

    # ── 12. Rollback To ───────────────────────────────────────────────────────

    def rollback_to(
        self,
        target_version_id: str,
        reason: str,
        initiated_by: str = "system",
        notes: str = "",
    ) -> ApiResponse:
        try:
            record = self._rollback.rollback_to(
                target_version_id=target_version_id,
                reason=reason,
                initiated_by=initiated_by,
                notes=notes,
            )
            active = self._registry.get_active()
            return _ok(
                "rollback_to",
                {
                    "rollback_record": record.to_dict(),
                    "active_version": active.to_dict() if active else None,
                },
            )
        except Exception as exc:
            logger.error(f"rollback_to error: {exc}")
            return _err("rollback_to", str(exc))

    # ── 13. Rollback To Previous ──────────────────────────────────────────────

    def rollback_to_previous(
        self,
        reason: str,
        initiated_by: str = "system",
        notes: str = "",
    ) -> ApiResponse:
        try:
            record = self._rollback.rollback_to_previous(
                reason=reason,
                initiated_by=initiated_by,
                notes=notes,
            )
            active = self._registry.get_active()
            return _ok(
                "rollback_to_previous",
                {
                    "rollback_record": record.to_dict(),
                    "active_version": active.to_dict() if active else None,
                },
            )
        except Exception as exc:
            logger.error(f"rollback_to_previous error: {exc}")
            return _err("rollback_to_previous", str(exc))

    # ── 14. Rollback History ──────────────────────────────────────────────────

    def rollback_history(self) -> ApiResponse:
        history = self._rollback.rollback_history()
        return _ok("rollback_history", {"count": len(history), "history": history})

    # ── 15. Eligible Rollback Targets ─────────────────────────────────────────

    def eligible_rollback_targets(self) -> ApiResponse:
        targets = self._rollback.eligible_targets()
        return _ok(
            "eligible_rollback_targets",
            {"count": len(targets), "targets": targets},
        )

    # ── 15b. Archive Version ──────────────────────────────────────────────────

    def archive_version(
        self,
        version_id: str,
        archiver: str = "",
        reason: str = "",
    ) -> ApiResponse:
        try:
            version = self._registry.archive(version_id)
            logger.info(
                f"QDELifecycleAPI.archive_version: {version_id} archiver={archiver} reason={reason}"
            )
            return _ok("archive_version", version.to_dict())
        except Exception as exc:
            logger.error(f"archive_version error: {exc}")
            return _err("archive_version", str(exc))

    # ── 15c. Retire Version ───────────────────────────────────────────────────

    def retire_version(
        self,
        version_id: str,
        retirer: str = "",
        reason: str = "",
    ) -> ApiResponse:
        try:
            version = self._registry.retire(version_id)
            logger.info(
                f"QDELifecycleAPI.retire_version: {version_id} retirer={retirer} reason={reason}"
            )
            return _ok("retire_version", version.to_dict())
        except Exception as exc:
            logger.error(f"retire_version error: {exc}")
            return _err("retire_version", str(exc))

    # ── 16. Create Shadow Test ────────────────────────────────────────────────

    def create_shadow_test(
        self,
        challenger_version_id: str,
        control_version_id: str | None = None,
    ) -> ApiResponse:
        try:
            test = self._ab.create_shadow(
                challenger_version_id=challenger_version_id,
                control_version_id=control_version_id,
            )
            return _ok("create_shadow_test", test.to_dict())
        except Exception as exc:
            logger.error(f"create_shadow_test error: {exc}")
            return _err("create_shadow_test", str(exc))

    # ── 17. Create Side-by-Side Test ──────────────────────────────────────────

    def create_side_by_side_test(
        self,
        challenger_version_id: str,
        control_version_id: str | None = None,
    ) -> ApiResponse:
        try:
            test = self._ab.create_side_by_side(
                challenger_version_id=challenger_version_id,
                control_version_id=control_version_id,
            )
            return _ok("create_side_by_side_test", test.to_dict())
        except Exception as exc:
            logger.error(f"create_side_by_side_test error: {exc}")
            return _err("create_side_by_side_test", str(exc))

    # ── 19. Record A/B Results ────────────────────────────────────────────────

    def record_ab_results(
        self,
        test_id: str,
        control_metrics: dict[str, float],
        challenger_metrics: dict[str, float],
    ) -> ApiResponse:
        try:
            test = self._ab.record_results(
                test_id=test_id,
                control_metrics=control_metrics,
                challenger_metrics=challenger_metrics,
            )
            return _ok("record_ab_results", test.to_dict())
        except Exception as exc:
            return _err("record_ab_results", str(exc))

    # ── 20. Conclude A/B Test ─────────────────────────────────────────────────

    def conclude_ab_test(
        self,
        test_id: str,
        winner: str,
        conclusion: str,
    ) -> ApiResponse:
        try:
            test = self._ab.conclude(test_id=test_id, winner=winner, conclusion=conclusion)
            return _ok("conclude_ab_test", test.to_dict())
        except Exception as exc:
            return _err("conclude_ab_test", str(exc))

    # ── 21. Compare A/B Test ──────────────────────────────────────────────────

    def compare_ab_test(self, test_id: str) -> ApiResponse:
        try:
            comparison = self._ab.compare(test_id)
            return _ok("compare_ab_test", comparison)
        except Exception as exc:
            return _err("compare_ab_test", str(exc))

    # ── 22. List A/B Tests ────────────────────────────────────────────────────

    def list_ab_tests(
        self,
        status: str | None = None,
        test_type: str | None = None,
    ) -> ApiResponse:
        try:
            tt = QDEABTestType(test_type) if test_type else None
            tests = self._ab.list_tests(status=status, test_type=tt)
            return _ok("list_ab_tests", {"count": len(tests), "tests": tests})
        except ValueError as exc:
            return _err("list_ab_tests", f"Invalid test_type: {exc}")

    # ── 23. Active Version ────────────────────────────────────────────────────

    def active_version(self) -> ApiResponse:
        active = self._registry.get_active()
        if active is None:
            return _ok("active_version", {"active_version": None})
        return _ok("active_version", {"active_version": active.to_dict()})

    # ── 24. Registry Summary ──────────────────────────────────────────────────

    def registry_summary(self) -> ApiResponse:
        return _ok("registry_summary", self._registry.summary())

    # ── 25. Version Lineage ───────────────────────────────────────────────────

    def version_lineage(self, version_id: str) -> ApiResponse:
        chain = self._registry.lineage(version_id)
        if not chain:
            return _err("version_lineage", f"Version {version_id!r} not found")
        return _ok(
            "version_lineage",
            {
                "version_id": version_id,
                "depth": len(chain),
                "lineage": [v.to_dict() for v in chain],
            },
        )

    # ── 26. Promotion History ─────────────────────────────────────────────────

    def promotion_history(self) -> ApiResponse:
        history = self._promotion.history()
        return _ok("promotion_history", {"count": len(history), "history": history})

    # ── 26. Promotion Criteria ────────────────────────────────────────────────

    def promotion_criteria(self) -> ApiResponse:
        return _ok("promotion_criteria", {"criteria": self._promotion.criteria()})

    # ── 27. Dispatch Shadow Decision ──────────────────────────────────────────

    def dispatch_shadow_decision(
        self,
        test_id: str,
        challenger_version_id: str,
        inputs: Any,
        regime: str = "",
    ) -> ApiResponse:
        """Dual-dispatch one live `QDEInferenceInput` to champion + challenger.

        Both models decide on the exact same input; only the champion's
        decision is ever executed elsewhere (e.g. `ExecutionPlanner`). The
        resulting comparison — including the computed `promotion_score` — is
        returned so a caller can persist it via the Research Lab API
        (`ResearchLabAPI.record_dual_dispatch_decision`).
        """
        try:
            result = self._shadow_dispatcher.dispatch(
                test_id=test_id,
                challenger_version_id=challenger_version_id,
                inputs=inputs,
                regime=regime,
            )
            return _ok("dispatch_shadow_decision", result.to_dict())
        except Exception as exc:
            logger.error(f"dispatch_shadow_decision error: {exc}")
            return _err("dispatch_shadow_decision", str(exc))

    # ── 28. Evaluate Shadow Promotion ─────────────────────────────────────────

    def evaluate_shadow_promotion(
        self,
        shadow_test: dict[str, Any],
        notes: str = "",
    ) -> ApiResponse:
        """Feed a concluded Research Lab shadow test's live-traffic signal
        (`avg_promotion_score` / `recommendation`) into this registry's
        backtest-based `QDEPromotionEngine`, alongside — not instead of — its
        existing five-dimension backtest gates.

        `shadow_test` is the dict produced by
        `ResearchLabAPI.conclude_shadow_test` (a `ShadowTestRun.to_dict()`).
        """
        try:
            # `ShadowTestRun.to_dict()` does not serialise the per-decision
            # `decisions` list (only the aggregated summary), and
            # `evaluate_and_maybe_promote` only reads that summary — so it is
            # reconstructed here without decisions.
            test_obj = ShadowTestRun(
                test_id=shadow_test["test_id"],
                name=shadow_test["name"],
                challenger_version=shadow_test["challenger_version"],
                champion_version=shadow_test["champion_version"],
                instruments=shadow_test.get("instruments", []),
                status=ShadowTestStatus(shadow_test["status"]),
                started_at=shadow_test["started_at"],
                min_decisions=shadow_test.get("min_decisions", 100),
                concluded_at=shadow_test.get("concluded_at"),
                notes=shadow_test.get("notes", ""),
                total_decisions=shadow_test.get("total_decisions", 0),
                agreement_rate=shadow_test.get("agreement_rate", 0.0),
                avg_promotion_score=shadow_test.get("avg_promotion_score", 0.0),
                recommendation=ShadowTestRecommendation(shadow_test["recommendation"]),
                conclusion_notes=shadow_test.get("conclusion_notes", ""),
            )
            result = evaluate_and_maybe_promote(test_obj, self._promotion, notes=notes)
            return _ok("evaluate_shadow_promotion", result)
        except Exception as exc:
            logger.error(f"evaluate_shadow_promotion error: {exc}")
            return _err("evaluate_shadow_promotion", str(exc))
