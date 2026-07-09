"""Phase 28.5 — Research Lab API.

Administrator-only interface for the Research Lab subsystem.

Responsibilities:
 1.  Dataset Generation    — register datasets and track their generation pipeline
 2.  Feature Discovery     — track and rank discovered features
 3.  Experiment Tracking   — full lifecycle tracking for all experiments
 4.  Training Runs         — start / complete / fail training jobs
 5.  Validation            — deployment gate evaluation
 6.  Backtesting           — register backtest runs against experiments
 7.  Shadow Testing        — observe challenger QDE against champion without execution
 8.  Version Promotion     — gate-controlled promotion approval
 9.  Rollback Approval     — formal approval workflow for version rollbacks
10.  Failure Analysis      — record and investigate failure events
11.  Version Comparisons   — side-by-side benchmarking across QDE versions
12.  Parameter Search      — hyperparameter search runs
13.  Market Regime Analysis — per-regime performance breakdown

ACCESS MODEL
------------
All methods in this class are for administrator use only.
This class is instantiated once in research_lab_server.py and is never imported
by retail-facing code. No retail user may call these methods.

PERSISTENCE
-----------
All state is persisted to SQLite via ResearchLabDatabase. No in-memory-only
stores exist. A server restart does not lose research state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.cert.institutional_certification import InstitutionalCertificationReport
from trainer.research_lab.decision_traceability import (
    DecisionTraceRecord,
    LineageResolver,
    ReplayResult,
    reconstruct_historical_decision,
    replay_decision,
)
from trainer.research_lab.deployment_approval import (
    DeploymentApprovalWorkflow,
    ReviewDecision,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.rl_models import (
    DatasetGenerationJob,
    DatasetGenerationStatus,
    DeploymentStatus,
    DiscoveredFeature,
    ExperimentRecord,
    ExperimentStatus,
    FailureCategory,
    FailureEvent,
    FailureSeverity,
    LifecycleAction,
    LifecycleAuthorization,
    LifecycleAuthorizationStatus,
    ModelComparisonResult,
    ParameterSearchRun,
    RegimeAnalysisReport,
    ResearchDataset,
    ResearchNote,
    RollbackRequest,
    RollbackStatus,
    SearchStrategy,
    ShadowDecisionRecord,
    ShadowTestRun,
    ShadowTestStatus,
)
from trainer.research_lab.sso import Role

# ── API Response ──────────────────────────────────────────────────────────────


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


# ── Research Lab API ──────────────────────────────────────────────────────────


class ResearchLabAPI:
    """Administrator-only Research Lab API.

    All state is persisted to SQLite via ResearchLabDatabase. Accepts an
    optional db argument for dependency injection (useful in tests). If no db
    is provided, the default path is used.

    This class is the sole internal implementation of all Research Lab
    responsibilities. It is never imported by retail-facing code.
    """

    def __init__(self, db: ResearchLabDatabase | None = None) -> None:
        self._db = db or ResearchLabDatabase()

    # ── System Summary ────────────────────────────────────────────────────────

    def system_summary(self) -> ApiResponse:
        stats = self._db.stats()
        return _ok("system_summary", stats)

    # ── 1. Dataset Generation ────────────────────────────────────────────────

    def create_dataset(
        self,
        name: str,
        version: str,
        source: str,
        instruments: list[str],
        start_date: str,
        end_date: str,
        row_count: int = 0,
        feature_count: int = 0,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> ApiResponse:
        try:
            ds = ResearchDataset.new(
                name=name,
                version=version,
                source=source,
                instruments=instruments,
                start_date=start_date,
                end_date=end_date,
                row_count=row_count,
                feature_count=feature_count,
                tags=tags,
                notes=notes,
            )
            self._db.upsert_dataset(ds.to_dict())
            logger.info(f"ResearchLab.create_dataset: {ds.dataset_id} ({ds.name} v{ds.version})")
            return _ok("create_dataset", ds.to_dict())
        except Exception as exc:
            return _err("create_dataset", str(exc))

    def get_dataset(self, dataset_id: str) -> ApiResponse:
        record = self._db.get_dataset(dataset_id)
        if not record:
            return _err("get_dataset", f"Dataset {dataset_id!r} not found")
        return _ok("get_dataset", record)

    def list_datasets(self) -> ApiResponse:
        items = self._db.list_datasets()
        return _ok("list_datasets", {"datasets": items, "total": len(items)})

    def create_generation_job(
        self,
        dataset_id: str,
        name: str,
        instruments: list[str],
        start_date: str,
        end_date: str,
        feature_schema: dict[str, Any] | None = None,
        label_schema: dict[str, Any] | None = None,
        requested_by: str = "",
        notes: str = "",
    ) -> ApiResponse:
        try:
            existing = self._db.get_dataset(dataset_id)
            if not existing:
                return _err(
                    "create_generation_job",
                    f"Dataset {dataset_id!r} not found — register the dataset first",
                )
            job = DatasetGenerationJob.new(
                dataset_id=dataset_id,
                name=name,
                instruments=instruments,
                start_date=start_date,
                end_date=end_date,
                feature_schema=feature_schema,
                label_schema=label_schema,
                requested_by=requested_by,
                notes=notes,
            )
            self._db.upsert_generation_job(job.to_dict())
            logger.info(f"ResearchLab.create_generation_job: {job.job_id} for dataset {dataset_id}")
            return _ok("create_generation_job", job.to_dict())
        except Exception as exc:
            return _err("create_generation_job", str(exc))

    def start_generation_job(self, job_id: str) -> ApiResponse:
        record = self._db.get_generation_job(job_id)
        if not record:
            return _err("start_generation_job", f"Generation job {job_id!r} not found")
        if record.get("status") != DatasetGenerationStatus.PENDING.value:
            return _err(
                "start_generation_job",
                f"Job {job_id} is {record['status']} — only PENDING jobs can be started",
            )
        record["status"] = DatasetGenerationStatus.GENERATING.value
        record["started_at"] = time.time()
        self._db.upsert_generation_job(record)
        logger.info(f"ResearchLab.start_generation_job: {job_id}")
        return _ok("start_generation_job", record)

    def validating_generation_job(
        self, job_id: str, row_count: int = 0, feature_count: int = 0
    ) -> ApiResponse:
        record = self._db.get_generation_job(job_id)
        if not record:
            return _err("validating_generation_job", f"Generation job {job_id!r} not found")
        if record.get("status") != DatasetGenerationStatus.GENERATING.value:
            return _err(
                "validating_generation_job",
                f"Job {job_id} is {record['status']} — expected GENERATING",
            )
        record["status"] = DatasetGenerationStatus.VALIDATING.value
        record["row_count"] = row_count
        record["feature_count"] = feature_count
        self._db.upsert_generation_job(record)
        logger.info(f"ResearchLab.validating_generation_job: {job_id} rows={row_count}")
        return _ok("validating_generation_job", record)

    def complete_generation_job(
        self, job_id: str, quality_metrics: dict[str, float] | None = None
    ) -> ApiResponse:
        record = self._db.get_generation_job(job_id)
        if not record:
            return _err("complete_generation_job", f"Generation job {job_id!r} not found")
        if record.get("status") != DatasetGenerationStatus.VALIDATING.value:
            return _err(
                "complete_generation_job",
                f"Job {job_id} is {record['status']} — expected VALIDATING",
            )
        record["status"] = DatasetGenerationStatus.READY.value
        record["completed_at"] = time.time()
        if quality_metrics:
            record["quality_metrics"] = quality_metrics
        self._db.upsert_generation_job(record)
        logger.info(f"ResearchLab.complete_generation_job: {job_id} → READY")
        return _ok("complete_generation_job", record)

    def fail_generation_job(self, job_id: str, error: str) -> ApiResponse:
        record = self._db.get_generation_job(job_id)
        if not record:
            return _err("fail_generation_job", f"Generation job {job_id!r} not found")
        record["status"] = DatasetGenerationStatus.FAILED.value
        record["completed_at"] = time.time()
        record["error"] = error
        self._db.upsert_generation_job(record)
        logger.warning(f"ResearchLab.fail_generation_job: {job_id} — {error}")
        return _ok("fail_generation_job", record)

    def get_generation_job(self, job_id: str) -> ApiResponse:
        record = self._db.get_generation_job(job_id)
        if not record:
            return _err("get_generation_job", f"Generation job {job_id!r} not found")
        return _ok("get_generation_job", record)

    def list_generation_jobs(
        self,
        dataset_id: str | None = None,
        status: str | None = None,
    ) -> ApiResponse:
        items = self._db.list_generation_jobs(dataset_id=dataset_id, status=status)
        return _ok("list_generation_jobs", {"jobs": items, "total": len(items)})

    # ── 2. Feature Discovery ──────────────────────────────────────────────────

    def record_feature(
        self,
        name: str,
        source: str,
        importance_score: float,
        method: str = "",
        notes: str = "",
    ) -> ApiResponse:
        try:
            combined_notes = f"[{method}] {notes}".strip() if method else notes
            feat = DiscoveredFeature.new(
                name=name,
                category=source,
                importance_score=importance_score,
                notes=combined_notes,
            )
            self._db.upsert_feature(feat.to_dict())
            logger.info(f"ResearchLab.record_feature: {feat.feature_id} ({feat.name})")
            return _ok("record_feature", feat.to_dict())
        except Exception as exc:
            return _err("record_feature", str(exc))

    def accept_feature(self, feature_id: str, notes: str = "") -> ApiResponse:
        record = self._db.get_feature(feature_id)
        if not record:
            return _err("accept_feature", f"Feature {feature_id!r} not found")
        record["status"] = "ACCEPTED"
        if notes:
            record["notes"] = notes
        self._db.upsert_feature(record)
        return _ok("accept_feature", record)

    def reject_feature(self, feature_id: str, notes: str = "") -> ApiResponse:
        record = self._db.get_feature(feature_id)
        if not record:
            return _err("reject_feature", f"Feature {feature_id!r} not found")
        record["status"] = "REJECTED"
        if notes:
            record["notes"] = notes
        self._db.upsert_feature(record)
        return _ok("reject_feature", record)

    def list_features(self, status: str | None = None) -> ApiResponse:
        items = self._db.list_features(status=status)
        return _ok("list_features", {"features": items, "total": len(items)})

    # ── 3. Experiment Tracking ────────────────────────────────────────────────

    def create_experiment(
        self,
        name: str,
        qde_version: str,
        training_dataset: str,
        researcher: str = "",
        tags: list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        notes: str = "",
    ) -> ApiResponse:
        try:
            exp = ExperimentRecord.new(
                name=name,
                qde_version=qde_version,
                training_dataset=training_dataset,
                researcher=researcher,
                tags=tags,
                hyperparameters=hyperparameters,
                notes=notes,
            )
            self._db.upsert_experiment(exp.to_dict())
            logger.info(
                f"ResearchLab.create_experiment: {exp.experiment_id} ({exp.name})"
                f" qde={qde_version}"
            )
            return _ok("create_experiment", exp.to_dict())
        except Exception as exc:
            return _err("create_experiment", str(exc))

    def start_experiment(self, experiment_id: str) -> ApiResponse:
        record = self._db.get_experiment(experiment_id)
        if not record:
            return _err("start_experiment", f"Experiment {experiment_id!r} not found")
        if record.get("status") != ExperimentStatus.PENDING.value:
            return _err(
                "start_experiment",
                f"Experiment {experiment_id} is {record['status']} — only PENDING can start",
            )
        record["status"] = ExperimentStatus.RUNNING.value
        record["training_started_at"] = time.time()
        record["updated_at"] = record["training_started_at"]
        self._db.upsert_experiment(record)
        logger.info(f"ResearchLab.start_experiment: {experiment_id}")
        return _ok("start_experiment", record)

    def complete_experiment(
        self, experiment_id: str, metrics: dict[str, float], notes: str = ""
    ) -> ApiResponse:
        record = self._db.get_experiment(experiment_id)
        if not record:
            return _err("complete_experiment", f"Experiment {experiment_id!r} not found")
        if record.get("status") != ExperimentStatus.RUNNING.value:
            return _err(
                "complete_experiment",
                f"Experiment {experiment_id} is {record['status']} — only RUNNING can complete",
            )
        now = time.time()
        record["status"] = ExperimentStatus.COMPLETED.value
        record["training_completed_at"] = now
        record["updated_at"] = now
        started = record.get("training_started_at")
        if started:
            record["training_duration_s"] = round(now - started, 1)
        record.setdefault("validation_metrics", {}).update(metrics)
        for attr in (
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "avg_drawdown",
            "profit_factor",
            "win_rate",
            "expected_value",
            "expected_value_per_trade",
            "decision_quality",
            "capital_efficiency",
            "recovery_factor",
        ):
            if attr in metrics:
                record[attr] = metrics[attr]
        if notes:
            record["notes"] = notes
        self._db.upsert_experiment(record)
        logger.info(
            f"ResearchLab.complete_experiment: {experiment_id}"
            f" sharpe={metrics.get('sharpe_ratio')}"
        )
        return _ok("complete_experiment", record)

    def fail_experiment(self, experiment_id: str, error: str) -> ApiResponse:
        record = self._db.get_experiment(experiment_id)
        if not record:
            return _err("fail_experiment", f"Experiment {experiment_id!r} not found")
        record["status"] = ExperimentStatus.FAILED.value
        record["error"] = error
        record["updated_at"] = time.time()
        self._db.upsert_experiment(record)
        logger.warning(f"ResearchLab.fail_experiment: {experiment_id} — {error}")
        return _ok("fail_experiment", record)

    def get_experiment(self, experiment_id: str) -> ApiResponse:
        record = self._db.get_experiment(experiment_id)
        if not record:
            return _err("get_experiment", f"Experiment {experiment_id!r} not found")
        return _ok("get_experiment", record)

    def list_experiments(
        self,
        qde_version: str | None = None,
        status: str | None = None,
        researcher: str | None = None,
    ) -> ApiResponse:
        items = self._db.list_experiments(
            qde_version=qde_version, status=status, researcher=researcher
        )
        by_status: dict[str, int] = {}
        for item in items:
            s = item.get("status", "UNKNOWN")
            by_status[s] = by_status.get(s, 0) + 1
        return _ok(
            "list_experiments",
            {"experiments": items, "total": len(items), "by_status": by_status},
        )

    def experiment_summary(self) -> ApiResponse:
        all_items = self._db.list_experiments()
        by_status: dict[str, int] = {}
        for item in all_items:
            s = item.get("status", "UNKNOWN")
            by_status[s] = by_status.get(s, 0) + 1
        completed = [
            item for item in all_items if item.get("status") == ExperimentStatus.COMPLETED.value
        ]
        sharpe_values = [
            item["sharpe_ratio"] for item in completed if item.get("sharpe_ratio") is not None
        ]
        avg_sharpe = round(sum(sharpe_values) / len(sharpe_values), 4) if sharpe_values else None
        return _ok(
            "experiment_summary",
            {
                "total": len(all_items),
                "by_status": by_status,
                "completed": len(completed),
                "avg_sharpe_ratio": avg_sharpe,
                "top_experiments": sorted(
                    completed,
                    key=lambda x: x.get("sharpe_ratio") or 0,
                    reverse=True,
                )[:5],
            },
        )

    # ── 4. Backtesting ────────────────────────────────────────────────────────

    def run_benchmark(
        self,
        name: str,
        qde_version: str,
        benchmark_type: str,
        parameters: dict[str, Any] | None = None,
        results: dict[str, Any] | None = None,
    ) -> ApiResponse:
        try:
            import uuid

            benchmark_id = f"bm-{str(uuid.uuid4())[:12]}"
            record: dict[str, Any] = {
                "benchmark_id": benchmark_id,
                "name": name,
                "qde_version": qde_version,
                "benchmark_type": benchmark_type,
                "parameters": parameters or {},
                "results": results or {},
                "created_at": time.time(),
            }
            self._db.upsert_benchmark(record)
            logger.info(f"ResearchLab.run_benchmark: {benchmark_id} ({name})")
            return _ok("run_benchmark", record)
        except Exception as exc:
            return _err("run_benchmark", str(exc))

    def get_benchmark(self, benchmark_id: str) -> ApiResponse:
        record = self._db.get_benchmark(benchmark_id)
        if not record:
            return _err("get_benchmark", f"Benchmark {benchmark_id!r} not found")
        return _ok("get_benchmark", record)

    def list_benchmarks(self) -> ApiResponse:
        items = self._db.list_benchmarks()
        return _ok("list_benchmarks", {"benchmarks": items, "total": len(items)})

    def register_backtest(
        self,
        experiment_id: str,
        start_date: str,
        end_date: str,
        instruments: list[str],
        metrics: dict[str, float],
        regime_breakdown: dict[str, Any] | None = None,
        notes: str = "",
    ) -> ApiResponse:
        try:
            exp = self._db.get_experiment(experiment_id)
            if not exp:
                return _err("register_backtest", f"Experiment {experiment_id!r} not found")
            import uuid

            backtest_id = f"bt-{str(uuid.uuid4())[:12]}"
            record: dict[str, Any] = {
                "backtest_id": backtest_id,
                "experiment_id": experiment_id,
                "qde_version": exp.get("qde_version", ""),
                "start_date": start_date,
                "end_date": end_date,
                "instruments": instruments,
                "metrics": metrics,
                "regime_breakdown": regime_breakdown or {},
                "notes": notes,
                "created_at": time.time(),
            }
            self._db.upsert_backtest(record)
            logger.info(
                f"ResearchLab.register_backtest: {backtest_id}" f" for experiment {experiment_id}"
            )
            return _ok("register_backtest", record)
        except Exception as exc:
            return _err("register_backtest", str(exc))

    def list_backtests(self, experiment_id: str | None = None) -> ApiResponse:
        items = self._db.list_backtests(experiment_id=experiment_id)
        return _ok("list_backtests", {"backtests": items, "total": len(items)})

    # ── 5. Shadow Testing ─────────────────────────────────────────────────────

    def create_shadow_test(
        self,
        name: str,
        challenger_version: str,
        champion_version: str,
        instruments: list[str] | None = None,
        min_decisions: int = 100,
        notes: str = "",
        min_duration_seconds: float = 0.0,
    ) -> ApiResponse:
        try:
            test = ShadowTestRun.new(
                name=name,
                challenger_version=challenger_version,
                champion_version=champion_version,
                instruments=instruments,
                min_decisions=min_decisions,
                notes=notes,
                min_duration_seconds=min_duration_seconds,
            )
            self._db.upsert_shadow_test(test.to_dict())
            logger.info(
                f"ResearchLab.create_shadow_test: {test.test_id}"
                f" challenger={challenger_version} vs champion={champion_version}"
            )
            return _ok("create_shadow_test", test.to_dict())
        except Exception as exc:
            return _err("create_shadow_test", str(exc))

    def record_shadow_decision(
        self,
        test_id: str,
        instrument: str,
        champion_decision: str,
        challenger_decision: str,
        champion_confidence: float,
        challenger_confidence: float,
        regime: str | None = None,
        pnl_diff: float = 0.0,
        latency_diff_ms: float = 0.0,
        risk_diff: float = 0.0,
    ) -> ApiResponse:
        """Record a manually-supplied champion/challenger comparison.

        For decisions produced by live dual dispatch (both models invoked
        together on the same input), prefer `record_dual_dispatch_decision`,
        which persists an already-built `ShadowDecisionRecord` — this method
        remains for callers that only have the raw decision strings/confidences
        on hand and still benefits from the same widened comparison fields and
        computed `promotion_score`.
        """
        row = self._db.get_shadow_test(test_id)
        if not row:
            return _err("record_shadow_decision", f"Shadow test {test_id!r} not found")
        if row.get("status") != ShadowTestStatus.RUNNING.value:
            return _err(
                "record_shadow_decision",
                f"Shadow test {test_id} is {row['status']} — can only record to RUNNING tests",
            )

        decision = ShadowDecisionRecord.new(
            test_id=test_id,
            instrument=instrument,
            champion_decision_type=champion_decision,
            challenger_decision_type=challenger_decision,
            champion_confidence=champion_confidence,
            challenger_confidence=challenger_confidence,
            regime=regime or "",
            pnl_diff=pnl_diff,
            latency_diff_ms=latency_diff_ms,
            risk_diff=risk_diff,
        )
        self._db.save_shadow_decision(decision.to_dict())
        count = self._db.count_shadow_decisions(test_id)
        return _ok(
            "record_shadow_decision",
            {
                "record_id": decision.record_id,
                "total_decisions": count,
                "promotion_score": decision.promotion_score,
            },
        )

    def record_dual_dispatch_decision(self, record: ShadowDecisionRecord) -> ApiResponse:
        """Persist a `ShadowDecisionRecord` already built by live dual dispatch.

        See `trainer.qde.qde_shadow.ShadowDispatcher` — every live inference
        input is sent to both champion and challenger models simultaneously,
        and the resulting comparison (including `promotion_score`) is built
        there. This method is the Research Lab's side of that hand-off.
        """
        row = self._db.get_shadow_test(record.test_id)
        if not row:
            return _err(
                "record_dual_dispatch_decision", f"Shadow test {record.test_id!r} not found"
            )
        if row.get("status") != ShadowTestStatus.RUNNING.value:
            return _err(
                "record_dual_dispatch_decision",
                f"Shadow test {record.test_id} is {row['status']} — can only record to RUNNING tests",
            )
        self._db.save_shadow_decision(record.to_dict())
        count = self._db.count_shadow_decisions(record.test_id)
        return _ok(
            "record_dual_dispatch_decision",
            {
                "record_id": record.record_id,
                "total_decisions": count,
                "promotion_score": record.promotion_score,
            },
        )

    def conclude_shadow_test(self, test_id: str, notes: str = "") -> ApiResponse:
        row = self._db.get_shadow_test(test_id)
        if not row:
            return _err("conclude_shadow_test", f"Shadow test {test_id!r} not found")
        if row.get("status") != ShadowTestStatus.RUNNING.value:
            return _err(
                "conclude_shadow_test",
                f"Shadow test {test_id} is {row['status']} — can only conclude RUNNING tests",
            )
        decisions = self._db.list_shadow_decisions(test_id)
        decision_objs = [ShadowDecisionRecord(**d) for d in decisions]
        test_obj = ShadowTestRun(
            test_id=row["test_id"],
            name=row["name"],
            challenger_version=row["challenger_version"],
            champion_version=row["champion_version"],
            instruments=row.get("instruments", []),
            status=ShadowTestStatus.RUNNING,
            started_at=row["started_at"],
            min_decisions=row.get("min_decisions", 100),
            notes=row.get("notes", ""),
            decisions=decision_objs,
        )
        test_obj.conclude(notes=notes)
        self._db.upsert_shadow_test(test_obj.to_dict())
        logger.info(
            f"ResearchLab.conclude_shadow_test: {test_id}"
            f" recommendation={test_obj.recommendation.value}"
        )
        return _ok("conclude_shadow_test", test_obj.to_dict())

    def get_shadow_test(self, test_id: str) -> ApiResponse:
        row = self._db.get_shadow_test(test_id)
        if not row:
            return _err("get_shadow_test", f"Shadow test {test_id!r} not found")
        return _ok("get_shadow_test", row)

    def list_shadow_tests(
        self,
        challenger_version: str | None = None,
        champion_version: str | None = None,
        status: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> ApiResponse:
        """List shadow tests, queryable by champion/challenger version pair
        and a ``started_at`` time window (S22B: shadow observation hardening)."""
        items = self._db.list_shadow_tests(
            challenger_version=challenger_version,
            champion_version=champion_version,
            status=status,
            since=since,
            until=until,
        )
        return _ok("list_shadow_tests", {"shadow_tests": items, "total": len(items)})

    def list_shadow_decisions_by_version_pair(
        self,
        champion_version: str | None = None,
        challenger_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 500,
    ) -> ApiResponse:
        """List individual shadow-decision comparisons (not just test-level
        aggregates), queryable by champion/challenger version pair and a
        ``recorded_at`` time window (S22B: shadow observation hardening —
        per-decision, not just per-test, queryability)."""
        items = self._db.list_shadow_decisions_by_version_pair(
            champion_version=champion_version,
            challenger_version=challenger_version,
            since=since,
            until=until,
            limit=limit,
        )
        return _ok(
            "list_shadow_decisions_by_version_pair", {"decisions": items, "total": len(items)}
        )

    # ── 6. Parameter Search ───────────────────────────────────────────────────

    def create_search(
        self,
        name: str,
        qde_version: str,
        parameter_space: dict[str, Any],
        strategy: str = "BAYESIAN",
        max_trials: int = 50,
        objective: str = "score",
        maximize: bool = True,
        notes: str = "",
    ) -> ApiResponse:
        try:
            search_strategy = SearchStrategy(strategy)
        except ValueError:
            return _err(
                "create_search",
                f"Unknown strategy {strategy!r}. Valid: {[s.value for s in SearchStrategy]}",
            )
        try:
            search = ParameterSearchRun.new(
                name=name,
                strategy=search_strategy,
                search_space=parameter_space,
                objective=objective,
                maximize=maximize,
                n_trials=max_trials,
                qde_version=qde_version,
                notes=notes,
            )
            self._db.upsert_search(search.to_dict())
            logger.info(f"ResearchLab.create_search: {search.search_id} ({name})")
            return _ok("create_search", search.to_dict())
        except Exception as exc:
            return _err("create_search", str(exc))

    def record_trial(
        self,
        search_id: str,
        parameters: dict[str, float],
        metrics: dict[str, float],
    ) -> ApiResponse:
        record = self._db.get_search(search_id)
        if not record:
            return _err("record_trial", f"Search {search_id!r} not found")
        trials: list[dict[str, Any]] = record.get("trials", [])
        trial: dict[str, Any] = {
            "trial_number": len(trials) + 1,
            "parameters": parameters,
            "metrics": metrics,
            "recorded_at": time.time(),
        }
        trials.append(trial)
        record["trials"] = trials
        record["total_trials"] = len(trials)
        if trials:
            best_score = record.get("best_score")
            score = metrics.get("sharpe_ratio", metrics.get("score", 0.0))
            if best_score is None or score > best_score:
                record["best_score"] = score
                record["best_parameters"] = parameters
        self._db.upsert_search(record)
        return _ok("record_trial", {"trial": trial, "total_trials": len(trials)})

    def complete_search(self, search_id: str) -> ApiResponse:
        record = self._db.get_search(search_id)
        if not record:
            return _err("complete_search", f"Search {search_id!r} not found")
        record["status"] = "COMPLETED"
        record["completed_at"] = time.time()
        self._db.upsert_search(record)
        logger.info(f"ResearchLab.complete_search: {search_id}")
        return _ok("complete_search", record)

    def get_search(self, search_id: str) -> ApiResponse:
        record = self._db.get_search(search_id)
        if not record:
            return _err("get_search", f"Search {search_id!r} not found")
        return _ok("get_search", record)

    def list_searches(self, qde_version: str | None = None) -> ApiResponse:
        items = self._db.list_searches(qde_version=qde_version)
        return _ok("list_searches", {"searches": items, "total": len(items)})

    # ── 7. Validation (Deployment Gates) ──────────────────────────────────────

    def evaluate_deployment_gates(
        self,
        experiment_id: str,
        baseline_version: str,
        min_sharpe: float = 0.5,
        max_drawdown_pct: float = 15.0,
        min_win_rate: float = 0.45,
        min_profit_factor: float = 1.2,
    ) -> ApiResponse:
        exp = self._db.get_experiment(experiment_id)
        if not exp:
            return _err("evaluate_deployment_gates", f"Experiment {experiment_id!r} not found")
        if exp.get("status") != ExperimentStatus.COMPLETED.value:
            return _err(
                "evaluate_deployment_gates",
                f"Experiment {experiment_id} is {exp['status']} — must be COMPLETED",
            )
        import uuid

        gate_id = f"gate-{str(uuid.uuid4())[:12]}"
        checks: list[dict[str, Any]] = [
            {
                "gate": "min_sharpe",
                "threshold": min_sharpe,
                "actual": exp.get("sharpe_ratio"),
                "passed": (exp.get("sharpe_ratio") or 0) >= min_sharpe,
            },
            {
                "gate": "max_drawdown",
                "threshold": max_drawdown_pct,
                "actual": exp.get("max_drawdown"),
                "passed": (exp.get("max_drawdown") or 100) <= max_drawdown_pct,
            },
            {
                "gate": "min_win_rate",
                "threshold": min_win_rate,
                "actual": exp.get("win_rate"),
                "passed": (exp.get("win_rate") or 0) >= min_win_rate,
            },
            {
                "gate": "min_profit_factor",
                "threshold": min_profit_factor,
                "actual": exp.get("profit_factor"),
                "passed": (exp.get("profit_factor") or 0) >= min_profit_factor,
            },
        ]
        all_passed = all(c["passed"] for c in checks)
        record: dict[str, Any] = {
            "gate_id": gate_id,
            "experiment_id": experiment_id,
            "qde_version": exp.get("qde_version", ""),
            "baseline_version": baseline_version,
            "passed": all_passed,
            "checks": checks,
            "evaluated_at": time.time(),
        }
        self._db.upsert_gate_result(record)
        logger.info(
            f"ResearchLab.evaluate_deployment_gates: {gate_id}"
            f" experiment={experiment_id} passed={all_passed}"
        )
        return _ok("evaluate_deployment_gates", record)

    def validate_retraining(self, experiment_id: str, baseline_version: str) -> ApiResponse:
        exp = self._db.get_experiment(experiment_id)
        if not exp:
            return _err("validate_retraining", f"Experiment {experiment_id!r} not found")
        gates = self._db.list_gate_results(experiment_id=experiment_id)
        latest_gate = gates[0] if gates else None
        return _ok(
            "validate_retraining",
            {
                "experiment_id": experiment_id,
                "baseline_version": baseline_version,
                "experiment_status": exp.get("status"),
                "gates_evaluated": latest_gate is not None,
                "gates_passed": latest_gate.get("passed") if latest_gate else None,
                "recommendation": (
                    "PROCEED" if latest_gate and latest_gate.get("passed") else "ABORT"
                ),
            },
        )

    # ── 8. Promotion Approval ─────────────────────────────────────────────────

    def approve_deployment(
        self,
        experiment_id: str,
        approver: str,
        notes: str = "",
    ) -> ApiResponse:
        exp = self._db.get_experiment(experiment_id)
        if not exp:
            return _err("approve_deployment", f"Experiment {experiment_id!r} not found")
        gates = self._db.list_gate_results(experiment_id=experiment_id)
        latest_gate = gates[0] if gates else None
        if not latest_gate or not latest_gate.get("passed"):
            return _err(
                "approve_deployment",
                f"Experiment {experiment_id} has not passed all deployment gates",
            )
        now = time.time()
        exp["deployment_status"] = DeploymentStatus.ACTIVE.value
        exp["deployment_notes"] = notes
        exp["updated_at"] = now
        self._db.upsert_experiment(exp)
        deployment_record: dict[str, Any] = {
            "action": "APPROVED",
            "experiment_id": experiment_id,
            "qde_version": exp.get("qde_version", ""),
            "approver": approver,
            "notes": notes,
            "timestamp": now,
        }
        self._db.save_deployment(deployment_record)
        auth = LifecycleAuthorization.new_approved(
            action=LifecycleAction.PROMOTE,
            version_id=exp.get("qde_version", ""),
            approved_by=approver,
            reason=notes,
        )
        self._db.upsert_lifecycle_authorization(auth.to_dict())
        logger.info(
            f"ResearchLab.approve_deployment: experiment={experiment_id}"
            f" version={exp.get('qde_version')} approver={approver}"
            f" authorization={auth.authorization_id}"
        )
        return _ok(
            "approve_deployment",
            {
                **exp,
                "deployment_record": deployment_record,
                "authorization": auth.to_dict(),
            },
        )

    def reject_deployment(self, experiment_id: str, rejector: str, reason: str) -> ApiResponse:
        exp = self._db.get_experiment(experiment_id)
        if not exp:
            return _err("reject_deployment", f"Experiment {experiment_id!r} not found")
        now = time.time()
        exp["deployment_status"] = DeploymentStatus.RETIRED.value
        exp["deployment_notes"] = reason
        exp["updated_at"] = now
        self._db.upsert_experiment(exp)
        deployment_record: dict[str, Any] = {
            "action": "REJECTED",
            "experiment_id": experiment_id,
            "qde_version": exp.get("qde_version", ""),
            "rejector": rejector,
            "reason": reason,
            "timestamp": now,
        }
        self._db.save_deployment(deployment_record)
        logger.info(
            f"ResearchLab.reject_deployment: experiment={experiment_id} rejector={rejector}"
        )
        return _ok("reject_deployment", {**exp, "deployment_record": deployment_record})

    def deployment_history(self) -> ApiResponse:
        items = self._db.list_deployments()
        return _ok("deployment_history", {"deployments": items, "total": len(items)})

    # ── 9. Rollback Approval ──────────────────────────────────────────────────

    def create_rollback_request(
        self,
        current_version_id: str,
        target_version_id: str,
        reason: str,
        requested_by: str,
    ) -> ApiResponse:
        """Create a formal request to roll the active QDE version back.

        The request begins in PENDING status. A second administrator must call
        approve_rollback() before any version change takes effect.
        """
        if not current_version_id:
            return _err("create_rollback_request", "current_version_id is required")
        if not target_version_id:
            return _err("create_rollback_request", "target_version_id is required")
        if current_version_id == target_version_id:
            return _err(
                "create_rollback_request",
                "current_version_id and target_version_id must differ",
            )
        if not reason:
            return _err("create_rollback_request", "reason is required")
        try:
            pending = self._db.list_rollback_requests(status=RollbackStatus.PENDING.value)
            for p in pending:
                if p.get("target_version_id") == target_version_id:
                    return _err(
                        "create_rollback_request",
                        f"A pending rollback to {target_version_id!r} already exists"
                        f" ({p['request_id']}). Approve, reject, or withdraw it first.",
                    )
            req = RollbackRequest.new(
                current_version_id=current_version_id,
                target_version_id=target_version_id,
                reason=reason,
                requested_by=requested_by,
            )
            self._db.upsert_rollback_request(req.to_dict())
            logger.info(
                f"ResearchLab.create_rollback_request: {req.request_id}"
                f" {current_version_id} → {target_version_id}"
                f" requested_by={requested_by}"
            )
            return _ok("create_rollback_request", req.to_dict())
        except Exception as exc:
            return _err("create_rollback_request", str(exc))

    def approve_rollback(self, request_id: str, approver: str, notes: str = "") -> ApiResponse:
        """Approve a pending rollback request.

        Records the approval. Does not automatically switch the QDE version —
        the caller is responsible for triggering the actual registry update
        after receiving a successful response.
        """
        record = self._db.get_rollback_request(request_id)
        if not record:
            return _err("approve_rollback", f"Rollback request {request_id!r} not found")
        if record.get("status") != RollbackStatus.PENDING.value:
            return _err(
                "approve_rollback",
                f"Request {request_id} is {record['status']} — only PENDING requests can be approved",
            )
        if record.get("requested_by") == approver:
            return _err(
                "approve_rollback",
                f"Approver {approver!r} cannot approve their own rollback request",
            )
        record["status"] = RollbackStatus.APPROVED.value
        record["resolved_at"] = time.time()
        record["resolved_by"] = approver
        record["resolution_notes"] = notes
        self._db.upsert_rollback_request(record)
        auth = LifecycleAuthorization.new_approved(
            action=LifecycleAction.ROLLBACK,
            version_id=record["current_version_id"],
            approved_by=approver,
            reason=notes,
            target_version_id=record["target_version_id"],
        )
        self._db.upsert_lifecycle_authorization(auth.to_dict())
        logger.info(
            f"ResearchLab.approve_rollback: {request_id}"
            f" {record['current_version_id']} → {record['target_version_id']}"
            f" approver={approver} authorization={auth.authorization_id}"
        )
        return _ok("approve_rollback", {**record, "authorization": auth.to_dict()})

    def reject_rollback(self, request_id: str, rejector: str, notes: str = "") -> ApiResponse:
        """Reject a pending rollback request."""
        record = self._db.get_rollback_request(request_id)
        if not record:
            return _err("reject_rollback", f"Rollback request {request_id!r} not found")
        if record.get("status") != RollbackStatus.PENDING.value:
            return _err(
                "reject_rollback",
                f"Request {request_id} is {record['status']} — only PENDING requests can be rejected",
            )
        record["status"] = RollbackStatus.REJECTED.value
        record["resolved_at"] = time.time()
        record["resolved_by"] = rejector
        record["resolution_notes"] = notes
        self._db.upsert_rollback_request(record)
        logger.info(f"ResearchLab.reject_rollback: {request_id} rejector={rejector}")
        return _ok("reject_rollback", record)

    def withdraw_rollback(self, request_id: str, notes: str = "") -> ApiResponse:
        """Withdraw a pending rollback request (by the original requester)."""
        record = self._db.get_rollback_request(request_id)
        if not record:
            return _err("withdraw_rollback", f"Rollback request {request_id!r} not found")
        if record.get("status") != RollbackStatus.PENDING.value:
            return _err(
                "withdraw_rollback",
                f"Request {request_id} is {record['status']} — only PENDING requests can be withdrawn",
            )
        record["status"] = RollbackStatus.WITHDRAWN.value
        record["resolved_at"] = time.time()
        record["resolution_notes"] = notes
        self._db.upsert_rollback_request(record)
        logger.info(f"ResearchLab.withdraw_rollback: {request_id}")
        return _ok("withdraw_rollback", record)

    def get_rollback_request(self, request_id: str) -> ApiResponse:
        record = self._db.get_rollback_request(request_id)
        if not record:
            return _err("get_rollback_request", f"Rollback request {request_id!r} not found")
        return _ok("get_rollback_request", record)

    def list_rollback_requests(self, status: str | None = None) -> ApiResponse:
        items = self._db.list_rollback_requests(status=status)
        pending = sum(1 for i in items if i.get("status") == RollbackStatus.PENDING.value)
        return _ok(
            "list_rollback_requests",
            {"requests": items, "total": len(items), "pending": pending},
        )

    # ── 9b. Lifecycle Authorizations ─────────────────────────────────────────
    #
    # Research Lab is the SOLE authority permitted to approve PROMOTE,
    # ROLLBACK, ARCHIVE, and RETIRE actions against QDE versions. PROMOTE and
    # ROLLBACK authorizations are issued automatically as a side effect of
    # `approve_deployment` / `approve_rollback` above (which already carry
    # their own gating criteria/approval workflow). ARCHIVE and RETIRE have
    # no separate workflow, so they are authorized directly here.

    def authorize_lifecycle_action(
        self,
        action: str,
        version_id: str,
        approver: str,
        reason: str = "",
    ) -> ApiResponse:
        try:
            act = LifecycleAction(action)
        except ValueError:
            return _err("authorize_lifecycle_action", f"Invalid action: {action!r}")
        if act in (LifecycleAction.PROMOTE, LifecycleAction.ROLLBACK):
            return _err(
                "authorize_lifecycle_action",
                f"{act.value} authorizations must go through approve_deployment / "
                "approve_rollback, not authorize_lifecycle_action",
            )
        auth = LifecycleAuthorization.new_approved(
            action=act,
            version_id=version_id,
            approved_by=approver,
            reason=reason,
        )
        self._db.upsert_lifecycle_authorization(auth.to_dict())
        logger.info(
            f"ResearchLab.authorize_lifecycle_action: {act.value} version={version_id}"
            f" approver={approver} authorization={auth.authorization_id}"
        )
        return _ok("authorize_lifecycle_action", auth.to_dict())

    def consume_lifecycle_authorization(self, version_id: str, action: str) -> ApiResponse:
        """Atomically find-and-consume the most recent APPROVED, unconsumed
        authorization for (version_id, action). Called by the gateway
        immediately before it executes the corresponding registry mutation;
        the mutation must only proceed if this returns success."""
        record = self._db.find_approved_lifecycle_authorization(version_id, action)
        if not record:
            return _err(
                "consume_lifecycle_authorization",
                f"No approved Research Lab authorization found for version={version_id!r} "
                f"action={action!r}",
            )
        record["status"] = LifecycleAuthorizationStatus.CONSUMED.value
        record["consumed_at"] = time.time()
        self._db.upsert_lifecycle_authorization(record)
        logger.info(
            f"ResearchLab.consume_lifecycle_authorization: {record['authorization_id']}"
            f" version={version_id} action={action}"
        )
        return _ok("consume_lifecycle_authorization", record)

    def get_lifecycle_authorization(self, authorization_id: str) -> ApiResponse:
        record = self._db.get_lifecycle_authorization(authorization_id)
        if not record:
            return _err(
                "get_lifecycle_authorization",
                f"Authorization {authorization_id!r} not found",
            )
        return _ok("get_lifecycle_authorization", record)

    def list_lifecycle_authorizations(
        self,
        version_id: str | None = None,
        action: str | None = None,
        status: str | None = None,
    ) -> ApiResponse:
        items = self._db.list_lifecycle_authorizations(
            version_id=version_id, action=action, status=status
        )
        return _ok(
            "list_lifecycle_authorizations",
            {"authorizations": items, "total": len(items)},
        )

    # ── 10. Failure Analysis ──────────────────────────────────────────────────

    def record_failure(
        self,
        qde_version: str,
        category: str,
        severity: str,
        description: str,
        detected_at: float | None = None,
        affected_instruments: list[str] | None = None,
        loss_pct: float | None = None,
        notes: str = "",
    ) -> ApiResponse:
        try:
            cat = FailureCategory(category)
            sev = FailureSeverity(severity)
        except ValueError as exc:
            return _err("record_failure", str(exc))
        try:
            failure = FailureEvent.new(
                category=cat,
                severity=sev,
                qde_version=qde_version,
                description=description,
                instrument=(affected_instruments[0] if affected_instruments else ""),
                loss_amount=loss_pct,
                tags=affected_instruments,
                metadata=(
                    {"notes": notes, "detected_at_override": detected_at}
                    if (notes or detected_at is not None)
                    else None
                ),
            )
            self._db.upsert_failure(failure.to_dict())
            logger.warning(
                f"ResearchLab.record_failure: {failure.failure_id}"
                f" {sev.value} {cat.value} on {qde_version}"
            )
            return _ok("record_failure", failure.to_dict())
        except Exception as exc:
            return _err("record_failure", str(exc))

    def investigate_failure(self, failure_id: str, root_cause: str, notes: str = "") -> ApiResponse:
        record = self._db.get_failure(failure_id)
        if not record:
            return _err("investigate_failure", f"Failure {failure_id!r} not found")
        record["root_cause"] = root_cause
        record["status"] = "INVESTIGATING"
        if notes:
            record["notes"] = notes
        self._db.upsert_failure(record)
        logger.info(f"ResearchLab.investigate_failure: {failure_id} — {root_cause[:80]}")
        return _ok("investigate_failure", record)

    def resolve_failure(self, failure_id: str, resolution: str, resolver: str = "") -> ApiResponse:
        record = self._db.get_failure(failure_id)
        if not record:
            return _err("resolve_failure", f"Failure {failure_id!r} not found")
        record["resolution"] = resolution
        record["resolver"] = resolver
        record["status"] = "RESOLVED"
        record["resolved_at"] = time.time()
        self._db.upsert_failure(record)
        logger.info(f"ResearchLab.resolve_failure: {failure_id} resolver={resolver}")
        return _ok("resolve_failure", record)

    def list_failures(
        self,
        category: str | None = None,
        severity: str | None = None,
        qde_version: str | None = None,
        status: str | None = None,
    ) -> ApiResponse:
        items = self._db.list_failures(
            category=category,
            severity=severity,
            qde_version=qde_version,
            status=status,
        )
        return _ok("list_failures", {"failures": items, "total": len(items)})

    def failure_summary(self) -> ApiResponse:
        all_failures = self._db.list_failures()
        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for f in all_failures:
            s = f.get("severity", "UNKNOWN")
            by_severity[s] = by_severity.get(s, 0) + 1
            c = f.get("category", "UNKNOWN")
            by_category[c] = by_category.get(c, 0) + 1
            st = f.get("status", "UNKNOWN")
            by_status[st] = by_status.get(st, 0) + 1
        critical = [f for f in all_failures if f.get("severity") == FailureSeverity.CRITICAL.value]
        return _ok(
            "failure_summary",
            {
                "total": len(all_failures),
                "by_severity": by_severity,
                "by_category": by_category,
                "by_status": by_status,
                "open_critical": len([f for f in critical if f.get("status") != "RESOLVED"]),
                "recent_critical": critical[:5],
            },
        )

    # ── 11. Market Regime Analysis ─────────────────────────────────────────────

    def analyze_regime(
        self,
        qde_version: str,
        regime_metrics: dict[str, Any],
        notes: str = "",
    ) -> ApiResponse:
        try:
            report = RegimeAnalysisReport.new(
                qde_version=qde_version,
                regime_metrics=regime_metrics,
                notes=notes,
            )
            self._db.upsert_regime_report(report.to_dict())
            logger.info(f"ResearchLab.analyze_regime: {report.report_id}" f" qde={qde_version}")
            return _ok("analyze_regime", report.to_dict())
        except Exception as exc:
            return _err("analyze_regime", str(exc))

    def get_regime_report(self, report_id: str) -> ApiResponse:
        record = self._db.get_regime_report(report_id)
        if not record:
            return _err("get_regime_report", f"Report {report_id!r} not found")
        return _ok("get_regime_report", record)

    def list_regime_reports(self, qde_version: str | None = None) -> ApiResponse:
        items = self._db.list_regime_reports(qde_version=qde_version)
        return _ok("list_regime_reports", {"reports": items, "total": len(items)})

    # ── 12. Version Comparisons ───────────────────────────────────────────────

    def compare_models(
        self,
        versions: list[str],
        metrics: list[str] | None = None,
        regime_filter: str | None = None,
        notes: str = "",
    ) -> ApiResponse:
        if len(versions) < 2:
            return _err("compare_models", "At least 2 versions are required for comparison")
        try:
            result = ModelComparisonResult.new(
                versions=versions,
                metrics=metrics,
                regime_filter=regime_filter,
                notes=notes,
            )
            exp_data: dict[str, Any] = {}
            for v in versions:
                exps = self._db.list_experiments(qde_version=v, status="COMPLETED")
                if exps:
                    best = max(exps, key=lambda e: e.get("sharpe_ratio") or 0)
                    exp_data[v] = {
                        "sharpe_ratio": best.get("sharpe_ratio"),
                        "max_drawdown": best.get("max_drawdown"),
                        "win_rate": best.get("win_rate"),
                        "profit_factor": best.get("profit_factor"),
                        "capital_efficiency": best.get("capital_efficiency"),
                        "decision_quality": best.get("decision_quality"),
                    }
            result.version_metrics = exp_data
            if exp_data:
                result.ranked_versions = sorted(
                    exp_data.keys(),
                    key=lambda v: exp_data[v].get("sharpe_ratio") or 0,
                    reverse=True,
                )
                result.recommended_version = result.ranked_versions[0]
            self._db.upsert_comparison(result.to_dict())
            logger.info(
                f"ResearchLab.compare_models: {result.comparison_id}" f" versions={versions}"
            )
            return _ok("compare_models", result.to_dict())
        except Exception as exc:
            return _err("compare_models", str(exc))

    def get_comparison(self, comparison_id: str) -> ApiResponse:
        record = self._db.get_comparison(comparison_id)
        if not record:
            return _err("get_comparison", f"Comparison {comparison_id!r} not found")
        return _ok("get_comparison", record)

    def list_comparisons(self) -> ApiResponse:
        items = self._db.list_comparisons()
        return _ok("list_comparisons", {"comparisons": items, "total": len(items)})

    # ── 13. Research Notes ────────────────────────────────────────────────────

    def create_note(
        self,
        content: str,
        author: str = "",
        linked_entity_type: str = "",
        linked_entity_id: str = "",
        tags: list[str] | None = None,
    ) -> ApiResponse:
        try:
            if not content:
                return _err("create_note", "content is required")
            note = ResearchNote.new(
                content=content,
                author=author,
                linked_entity_type=linked_entity_type,
                linked_entity_id=linked_entity_id,
                tags=tags,
            )
            self._db.upsert_research_note(note.to_dict())
            logger.info(f"ResearchLab.create_note: {note.note_id} author={author!r}")
            return _ok("create_note", note.to_dict())
        except Exception as exc:
            return _err("create_note", str(exc))

    def get_note(self, note_id: str) -> ApiResponse:
        record = self._db.get_research_note(note_id)
        if not record:
            return _err("get_note", f"Note {note_id!r} not found")
        return _ok("get_note", record)

    def list_notes(
        self,
        author: str | None = None,
        linked_entity_type: str | None = None,
        linked_entity_id: str | None = None,
    ) -> ApiResponse:
        items = self._db.list_research_notes(
            author=author,
            linked_entity_type=linked_entity_type,
            linked_entity_id=linked_entity_id,
        )
        return _ok("list_notes", {"notes": items, "total": len(items)})

    # ── 14. Institutional Certification (S27) ──────────────────────────────────

    def create_institutional_certification(
        self,
        qde_version: str,
        metrics: dict[str, float],
        certified_by: str = "AUTOMATED",
        notes: str = "",
    ) -> ApiResponse:
        """Run the five-domain (Performance/Risk/Operational/Research/Market)
        institutional certification for a QDE version and persist the result
        as a new, permanent report. Never overwrites a prior report for the
        same version — each call creates a new report_id, preserving full
        certification history.
        """
        try:
            report = InstitutionalCertificationReport.certify(
                qde_version=qde_version,
                metrics=metrics,
                certified_by=certified_by,
                notes=notes,
            )
            self._db.upsert_institutional_certification(report.to_dict())
            logger.info(
                f"ResearchLab.create_institutional_certification: {report.report_id}"
                f" version={qde_version} decision={report.decision.value}"
            )
            return _ok("create_institutional_certification", report.to_dict())
        except Exception as exc:
            return _err("create_institutional_certification", str(exc))

    def get_institutional_certification(self, report_id: str) -> ApiResponse:
        record = self._db.get_institutional_certification(report_id)
        if not record:
            return _err("get_institutional_certification", f"Report {report_id!r} not found")
        return _ok("get_institutional_certification", record)

    def list_institutional_certifications(
        self,
        qde_version: str | None = None,
        decision: str | None = None,
        limit: int = 100,
    ) -> ApiResponse:
        items = self._db.list_institutional_certifications(
            qde_version=qde_version, decision=decision, limit=limit
        )
        return _ok("list_institutional_certifications", {"reports": items, "total": len(items)})

    def get_latest_institutional_certification(self, qde_version: str) -> ApiResponse:
        record = self._db.latest_institutional_certification(qde_version)
        if not record:
            return _err(
                "get_latest_institutional_certification",
                f"No certification found for version {qde_version!r}",
            )
        return _ok("get_latest_institutional_certification", record)

    # ── 15. Multi-role Deployment Approval Workflow (S27) ───────────────────────

    def create_deployment_approval_workflow(
        self,
        qde_version: str,
        certification_report_id: str = "",
        requested_by: str = "",
        notes: str = "",
    ) -> ApiResponse:
        try:
            workflow = DeploymentApprovalWorkflow.new(
                qde_version=qde_version,
                certification_report_id=certification_report_id,
                requested_by=requested_by,
                notes=notes,
            )
            self._db.upsert_deployment_approval_workflow(workflow.to_dict())
            logger.info(
                f"ResearchLab.create_deployment_approval_workflow: {workflow.workflow_id}"
                f" version={qde_version}"
            )
            return _ok("create_deployment_approval_workflow", workflow.to_dict())
        except Exception as exc:
            return _err("create_deployment_approval_workflow", str(exc))

    def submit_deployment_review(
        self,
        workflow_id: str,
        reviewer: str,
        role: str,
        decision: str,
        certification_result: str = "",
        notes: str = "",
    ) -> ApiResponse:
        """Record one structured reviewer decision.

        `role` must be a valid `sso.Role` value. SuperAdmin/Admin and
        DeploymentApprover are the two mandatory approver roles; Founder,
        QuantResearcher, and MLResearcher may add an optional review. The
        workflow is APPROVED only once both mandatory roles have recorded
        APPROVE; any REJECT — from any role — rejects the whole workflow.
        """
        try:
            record = self._db.get_deployment_approval_workflow(workflow_id)
            if not record:
                return _err("submit_deployment_review", f"Workflow {workflow_id!r} not found")
            workflow = DeploymentApprovalWorkflow.from_dict(record)
            try:
                role_enum = Role(role)
            except ValueError:
                return _err("submit_deployment_review", f"Unknown role {role!r}")
            try:
                decision_enum = ReviewDecision(decision)
            except ValueError:
                return _err("submit_deployment_review", f"Unknown decision {decision!r}")

            workflow.add_review(
                reviewer=reviewer,
                role=role_enum,
                decision=decision_enum,
                certification_result=certification_result,
                notes=notes,
            )
            self._db.upsert_deployment_approval_workflow(workflow.to_dict())
            logger.info(
                f"ResearchLab.submit_deployment_review: {workflow_id} reviewer={reviewer!r}"
                f" role={role_enum.value} decision={decision_enum.value}"
                f" status={workflow.status.value}"
            )
            return _ok("submit_deployment_review", workflow.to_dict())
        except Exception as exc:
            return _err("submit_deployment_review", str(exc))

    def get_deployment_approval_workflow(self, workflow_id: str) -> ApiResponse:
        record = self._db.get_deployment_approval_workflow(workflow_id)
        if not record:
            return _err("get_deployment_approval_workflow", f"Workflow {workflow_id!r} not found")
        return _ok("get_deployment_approval_workflow", record)

    def list_deployment_approval_workflows(
        self,
        qde_version: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> ApiResponse:
        items = self._db.list_deployment_approval_workflows(
            qde_version=qde_version, status=status, limit=limit
        )
        return _ok("list_deployment_approval_workflows", {"workflows": items, "total": len(items)})

    # ── 16. Decision Traceability & Model Lineage (S28) ─────────────────────────
    #
    # Every method below is administrative-only. Visibility is enforced by the
    # caller (typically apps/trainer-api's require_visibility(Visibility.
    # DECISION_TRACEABILITY) dependency) — Admin, SuperAdmin, Founder,
    # QuantResearcher, MLResearcher, DeploymentApprover, and Auditor may all
    # see decision traces; no customer- or public-facing route may call these.

    def record_decision_trace(
        self,
        qde_version: str,
        output_decision: str,
        confidence: float,
        size_signal: float,
        risk_signal: float,
        exit_horizon: float,
        reasoning_summary: str,
        input_features: dict[str, Any] | None = None,
        dataset_version: str = "",
        feature_version: str = "",
        symbol: str = "",
        decision_id: str | None = None,
    ) -> ApiResponse:
        """Persist the full traceability record for one QDE decision.

        This is the single write path institutional-grade traceability
        requires: every live decision the QDE produces should call this
        (or have its caller call this) so it can later be explained,
        audited, and reproduced.
        """
        try:
            trace = DecisionTraceRecord.new(
                qde_version=qde_version,
                output_decision=output_decision,
                confidence=confidence,
                size_signal=size_signal,
                risk_signal=risk_signal,
                exit_horizon=exit_horizon,
                reasoning_summary=reasoning_summary,
                input_features=input_features,
                dataset_version=dataset_version,
                feature_version=feature_version,
                symbol=symbol,
                decision_id=decision_id,
            )
            self._db.upsert_decision_trace(trace.to_dict())
            logger.info(
                f"ResearchLab.record_decision_trace: {trace.decision_id}"
                f" version={qde_version} decision={output_decision} symbol={symbol!r}"
            )
            return _ok("record_decision_trace", trace.to_dict())
        except Exception as exc:
            return _err("record_decision_trace", str(exc))

    def get_decision_trace(self, decision_id: str) -> ApiResponse:
        record = self._db.get_decision_trace(decision_id)
        if not record:
            return _err("get_decision_trace", f"Decision {decision_id!r} not found")
        return _ok("get_decision_trace", record)

    def list_decision_traces(
        self,
        qde_version: str | None = None,
        symbol: str | None = None,
        input_feature_hash: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> ApiResponse:
        items = self._db.list_decision_traces(
            qde_version=qde_version,
            symbol=symbol,
            input_feature_hash=input_feature_hash,
            since=since,
            until=until,
            limit=limit,
        )
        return _ok("list_decision_traces", {"traces": items, "total": len(items)})

    def get_decision_lineage(self, decision_id: str) -> ApiResponse:
        """Resolve the full lineage chain for a decision:
        Decision -> Model Version -> Dataset Version -> Feature Set ->
        Training Run -> Experiment -> Git Commit — built against real
        persisted records, not bare identifier strings.
        """
        record = self._db.get_decision_trace(decision_id)
        if not record:
            return _err("get_decision_lineage", f"Decision {decision_id!r} not found")
        trace = DecisionTraceRecord.from_dict(record)
        chain = LineageResolver(self._db).resolve(trace)
        return _ok("get_decision_lineage", chain.to_dict())

    def replay_decision(self, decision_id: str, inference_fn: Any) -> ApiResponse:
        """Deterministic inference replay: re-run the QDE on a decision's
        exact recorded input features and confirm the same output/confidence
        is produced.

        `inference_fn` is caller-supplied (typically the live QDE inference
        entry point) so this module never has to import the model runtime
        directly. It must accept the input feature dict and return
        ``{"output_decision": str, "confidence": float}``.
        """
        record = self._db.get_decision_trace(decision_id)
        if not record:
            return _err("replay_decision", f"Decision {decision_id!r} not found")
        try:
            trace = DecisionTraceRecord.from_dict(record)
            result: ReplayResult = replay_decision(trace, inference_fn)
            logger.info(
                f"ResearchLab.replay_decision: {decision_id}"
                f" deterministic_match={result.deterministic_match}"
            )
            return _ok("replay_decision", result.to_dict())
        except Exception as exc:
            return _err("replay_decision", str(exc))

    def reconstruct_historical_decision(self, decision_id: str) -> ApiResponse:
        """Historical reconstruction: rebuild the full context (traceability
        record + resolved lineage chain) that produced a past decision,
        purely from persisted state.
        """
        context = reconstruct_historical_decision(decision_id, self._db)
        if context is None:
            return _err("reconstruct_historical_decision", f"Decision {decision_id!r} not found")
        return _ok("reconstruct_historical_decision", context.to_dict())

    # ── 17. Decision Audits (S28) ────────────────────────────────────────────────

    def record_decision_audit(
        self,
        audit_kind: str,
        summary: str,
        severity_score: float = 0.0,
        qde_version: str = "",
        related_decision_ids: list[str] | None = None,
        evidence: dict[str, Any] | None = None,
        investigated_by: str = "",
    ) -> ApiResponse:
        """Record an investigation into an unexpected drawdown, abnormal
        decision, failed deployment, model regression, or abnormal
        confidence shift.

        `audit_kind` is a free-form category label (e.g.
        "UNEXPECTED_DRAWDOWN", "ABNORMAL_CONFIDENCE_SHIFT",
        "FAILED_DEPLOYMENT", "MODEL_REGRESSION", or any of the six
        InvestigationFailureKind values from the S20 engine) — decision
        audits are intentionally broader than that engine's live-trading
        taxonomy, since deployment failures and model regressions aren't
        live-trading failure patterns.
        """
        import uuid as _uuid

        try:
            audit = {
                "audit_id": f"aud-{_uuid.uuid4().hex[:16]}",
                "audit_kind": audit_kind,
                "qde_version": qde_version,
                "severity_score": min(1.0, max(0.0, severity_score)),
                "summary": summary,
                "related_decision_ids": related_decision_ids or [],
                "evidence": evidence or {},
                "investigated_by": investigated_by,
                "created_at": time.time(),
            }
            self._db.upsert_decision_audit(audit)
            logger.info(
                f"ResearchLab.record_decision_audit: {audit['audit_id']}"
                f" kind={audit_kind} severity={audit['severity_score']}"
            )
            return _ok("record_decision_audit", audit)
        except Exception as exc:
            return _err("record_decision_audit", str(exc))

    def get_decision_audit(self, audit_id: str) -> ApiResponse:
        record = self._db.get_decision_audit(audit_id)
        if not record:
            return _err("get_decision_audit", f"Audit {audit_id!r} not found")
        return _ok("get_decision_audit", record)

    def list_decision_audits(
        self,
        audit_kind: str | None = None,
        qde_version: str | None = None,
        min_severity: float = 0.0,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> ApiResponse:
        items = self._db.list_decision_audits(
            audit_kind=audit_kind,
            qde_version=qde_version,
            min_severity=min_severity,
            since=since,
            until=until,
            limit=limit,
        )
        return _ok("list_decision_audits", {"audits": items, "total": len(items)})
