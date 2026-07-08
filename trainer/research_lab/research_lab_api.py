"""Phase 28.4 — Research Lab API.

Administrator-only interface for the Research Lab subsystem.

Responsibilities:
 1.  Dataset Creation          — register and manage research datasets
 2.  Feature Discovery         — track and rank discovered features
 3.  Experiment Tracking       — full lifecycle tracking for all experiments
 4.  Training Runs             — start / complete / fail training jobs
 5.  Validation                — deployment gate evaluation
 6.  Backtesting               — register backtest runs against experiments
 7.  Shadow Testing            — observe challenger QDE against champion without execution
 8.  Version Promotion         — gate-controlled deployment sign-off
 9.  Failure Analysis          — record and investigate failure events
10.  Deployment Approval       — approve or reject deployment after gates pass
11.  Parameter Search          — hyperparameter search runs
12.  Model Comparison          — side-by-side benchmarking across QDE versions
13.  Market Regime Analysis    — per-regime performance breakdown

ACCESS MODEL
------------
All methods in this class are for administrator use only.
This class is instantiated once in research_lab_server.py and is never imported
by retail-facing code. No retail user may call these methods.
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

from trainer.research_lab.rl_models import (
    COMPARISON_METRICS,
    DeploymentGateResult,
    DeploymentStatus,
    DiscoveredFeature,
    ExperimentRecord,
    ExperimentStatus,
    FailureCategory,
    FailureEvent,
    FailureSeverity,
    ModelComparisonResult,
    ParameterSearchRun,
    RegimeAnalysisReport,
    ResearchDataset,
    SearchStrategy,
    ShadowDecisionRecord,
    ShadowTestRecommendation,
    ShadowTestRun,
    ShadowTestStatus,
)


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

    All stores are in-memory; extend with a database adapter for persistence.

    This class is the sole internal implementation of all Research Lab
    responsibilities. It is never imported by retail-facing code.
    """

    def __init__(self) -> None:
        self._datasets: dict[str, ResearchDataset] = {}
        self._features: dict[str, DiscoveredFeature] = {}
        self._experiments: dict[str, ExperimentRecord] = {}
        self._benchmarks: dict[str, dict[str, Any]] = {}
        self._backtests: dict[str, dict[str, Any]] = {}
        self._searches: dict[str, ParameterSearchRun] = {}
        self._gate_results: dict[str, DeploymentGateResult] = {}
        self._failures: dict[str, FailureEvent] = {}
        self._regime_reports: dict[str, RegimeAnalysisReport] = {}
        self._comparisons: dict[str, ModelComparisonResult] = {}
        self._deployments: list[dict[str, Any]] = []
        self._shadow_tests: dict[str, ShadowTestRun] = {}

    # ── System Summary ────────────────────────────────────────────────────────

    def system_summary(self) -> ApiResponse:
        return _ok("system_summary", {
            "datasets": len(self._datasets),
            "features": len(self._features),
            "experiments": len(self._experiments),
            "benchmarks": len(self._benchmarks),
            "backtests": len(self._backtests),
            "parameter_searches": len(self._searches),
            "deployment_gate_runs": len(self._gate_results),
            "failures": len(self._failures),
            "regime_reports": len(self._regime_reports),
            "model_comparisons": len(self._comparisons),
            "deployments": len(self._deployments),
            "shadow_tests": len(self._shadow_tests),
        })

    # ── 1. Datasets ───────────────────────────────────────────────────────────

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
                name=name, version=version, source=source,
                instruments=instruments, start_date=start_date, end_date=end_date,
                row_count=row_count, feature_count=feature_count,
                tags=tags, notes=notes,
            )
            self._datasets[ds.dataset_id] = ds
            logger.info(f"ResearchLab.create_dataset: {ds.dataset_id} ({ds.name} v{ds.version})")
            return _ok("create_dataset", ds.to_dict())
        except Exception as exc:
            return _err("create_dataset", str(exc))

    def get_dataset(self, dataset_id: str) -> ApiResponse:
        ds = self._datasets.get(dataset_id)
        if not ds:
            return _err("get_dataset", f"Dataset {dataset_id!r} not found")
        return _ok("get_dataset", ds.to_dict())

    def list_datasets(self) -> ApiResponse:
        items = sorted(self._datasets.values(), key=lambda d: d.created_at)
        return _ok("list_datasets", {"count": len(items), "datasets": [d.to_dict() for d in items]})

    # ── 2. Feature Discovery ──────────────────────────────────────────────────

    def record_feature(self, name: str, category: str, importance_score: float, notes: str = "") -> ApiResponse:
        try:
            feat = DiscoveredFeature.new(name, category, importance_score, notes=notes)
            self._features[feat.feature_id] = feat
            logger.info(f"ResearchLab.record_feature: {feat.feature_id} ({feat.name})")
            return _ok("record_feature", feat.to_dict())
        except Exception as exc:
            return _err("record_feature", str(exc))

    def accept_feature(self, feature_id: str, notes: str = "") -> ApiResponse:
        feat = self._features.get(feature_id)
        if not feat:
            return _err("accept_feature", f"Feature {feature_id!r} not found")
        feat.status = "ACCEPTED"
        if notes:
            feat.notes = notes
        return _ok("accept_feature", feat.to_dict())

    def reject_feature(self, feature_id: str, notes: str = "") -> ApiResponse:
        feat = self._features.get(feature_id)
        if not feat:
            return _err("reject_feature", f"Feature {feature_id!r} not found")
        feat.status = "REJECTED"
        if notes:
            feat.notes = notes
        return _ok("reject_feature", feat.to_dict())

    def list_features(self, status: str | None = None) -> ApiResponse:
        items = list(self._features.values())
        if status:
            items = [f for f in items if f.status == status]
        items.sort(key=lambda f: f.importance_score, reverse=True)
        return _ok("list_features", {"count": len(items), "features": [f.to_dict() for f in items]})

    # ── 3 & 4. Experiment Tracking / Training Runs ────────────────────────────

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
                name=name, qde_version=qde_version, training_dataset=training_dataset,
                researcher=researcher, tags=tags, hyperparameters=hyperparameters, notes=notes,
            )
            self._experiments[exp.experiment_id] = exp
            logger.info(f"ResearchLab.create_experiment: {exp.experiment_id} ({exp.name})")
            return _ok("create_experiment", exp.to_dict())
        except Exception as exc:
            return _err("create_experiment", str(exc))

    def start_experiment(self, experiment_id: str) -> ApiResponse:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return _err("start_experiment", f"Experiment {experiment_id!r} not found")
        exp.start_training()
        return _ok("start_experiment", exp.to_dict())

    def complete_experiment(self, experiment_id: str, metrics: dict[str, float], notes: str = "") -> ApiResponse:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return _err("complete_experiment", f"Experiment {experiment_id!r} not found")
        exp.complete(metrics=metrics, notes=notes)
        logger.info(f"ResearchLab.complete_experiment: {experiment_id}")
        return _ok("complete_experiment", exp.to_dict())

    def fail_experiment(self, experiment_id: str, error: str) -> ApiResponse:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return _err("fail_experiment", f"Experiment {experiment_id!r} not found")
        exp.fail(error=error)
        return _ok("fail_experiment", exp.to_dict())

    def get_experiment(self, experiment_id: str) -> ApiResponse:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return _err("get_experiment", f"Experiment {experiment_id!r} not found")
        return _ok("get_experiment", exp.to_dict())

    def list_experiments(
        self,
        qde_version: str | None = None,
        status: str | None = None,
        researcher: str | None = None,
    ) -> ApiResponse:
        items = list(self._experiments.values())
        if qde_version:
            items = [e for e in items if e.qde_version == qde_version]
        if status:
            items = [e for e in items if e.status.value == status]
        if researcher:
            items = [e for e in items if e.researcher == researcher]
        items.sort(key=lambda e: e.created_at, reverse=True)
        return _ok("list_experiments", {"count": len(items), "experiments": [e.to_dict() for e in items]})

    def experiment_summary(self) -> ApiResponse:
        exps = list(self._experiments.values())
        by_status: dict[str, int] = {}
        by_version: dict[str, int] = {}
        for e in exps:
            by_status[e.status.value] = by_status.get(e.status.value, 0) + 1
            by_version[e.qde_version] = by_version.get(e.qde_version, 0) + 1
        completed = [e for e in exps if e.status == ExperimentStatus.COMPLETED]
        best = max(completed, key=lambda e: e.sharpe_ratio or 0.0) if completed else None
        return _ok("experiment_summary", {
            "total": len(exps),
            "by_status": by_status,
            "by_version": by_version,
            "best_experiment": best.to_dict() if best else None,
        })

    # ── Model Benchmarking ────────────────────────────────────────────────────

    def run_benchmark(
        self,
        name: str,
        version_ids: list[str],
        metrics_per_version: dict[str, dict[str, float]],
        notes: str = "",
    ) -> ApiResponse:
        try:
            import uuid as _uuid
            benchmark_id = f"bench-{str(_uuid.uuid4())[:10]}"
            result = {
                "benchmark_id": benchmark_id,
                "name": name,
                "version_ids": version_ids,
                "metrics": {
                    v: {k: round(val, 6) for k, val in m.items()}
                    for v, m in metrics_per_version.items()
                },
                "created_at": round(time.time(), 3),
                "notes": notes,
            }
            self._benchmarks[benchmark_id] = result
            logger.info(f"ResearchLab.run_benchmark: {benchmark_id} ({name})")
            return _ok("run_benchmark", result)
        except Exception as exc:
            return _err("run_benchmark", str(exc))

    def get_benchmark(self, benchmark_id: str) -> ApiResponse:
        b = self._benchmarks.get(benchmark_id)
        if not b:
            return _err("get_benchmark", f"Benchmark {benchmark_id!r} not found")
        return _ok("get_benchmark", b)

    def list_benchmarks(self) -> ApiResponse:
        items = sorted(self._benchmarks.values(), key=lambda b: b["created_at"], reverse=True)
        return _ok("list_benchmarks", {"count": len(items), "benchmarks": items})

    # ── 6. Backtesting ────────────────────────────────────────────────────────

    def register_backtest(
        self,
        experiment_id: str,
        dataset_id: str,
        metrics: dict[str, float],
        notes: str = "",
    ) -> ApiResponse:
        try:
            import uuid as _uuid
            bt_id = f"bt-{str(_uuid.uuid4())[:10]}"
            bt = {
                "backtest_id": bt_id,
                "experiment_id": experiment_id,
                "dataset_id": dataset_id,
                "metrics": {k: round(v, 6) for k, v in metrics.items()},
                "created_at": round(time.time(), 3),
                "notes": notes,
            }
            self._backtests[bt_id] = bt
            exp = self._experiments.get(experiment_id)
            if exp:
                exp.validation_metrics.update(metrics)
                for attr in ("sharpe_ratio", "max_drawdown", "profit_factor", "win_rate"):
                    if attr in metrics:
                        setattr(exp, attr, metrics[attr])
            logger.info(f"ResearchLab.register_backtest: {bt_id}")
            return _ok("register_backtest", bt)
        except Exception as exc:
            return _err("register_backtest", str(exc))

    def list_backtests(self, experiment_id: str | None = None) -> ApiResponse:
        items = list(self._backtests.values())
        if experiment_id:
            items = [b for b in items if b["experiment_id"] == experiment_id]
        return _ok("list_backtests", {"count": len(items), "backtests": items})

    # ── 7. Shadow Testing ─────────────────────────────────────────────────────

    def create_shadow_test(
        self,
        name: str,
        challenger_version: str,
        champion_version: str,
        instruments: list[str],
        min_decisions: int = 500,
        notes: str = "",
    ) -> ApiResponse:
        """Create a new shadow test run.

        The challenger receives live snapshots but its decisions are NEVER executed.
        All comparison is observational only.
        """
        try:
            run = ShadowTestRun.new(
                name=name,
                challenger_version=challenger_version,
                champion_version=champion_version,
                instruments=instruments,
                min_decisions=min_decisions,
                notes=notes,
            )
            self._shadow_tests[run.test_id] = run
            logger.info(
                f"ResearchLab.create_shadow_test: {run.test_id} "
                f"challenger={challenger_version} champion={champion_version}"
            )
            return _ok("create_shadow_test", run.to_dict())
        except Exception as exc:
            return _err("create_shadow_test", str(exc))

    def record_shadow_decision(
        self,
        test_id: str,
        instrument: str,
        champion_decision_type: str,
        challenger_decision_type: str,
        champion_confidence: float,
        challenger_confidence: float,
        regime: str = "",
        market_context: dict[str, Any] | None = None,
    ) -> ApiResponse:
        """Record one decision comparison in a running shadow test."""
        run = self._shadow_tests.get(test_id)
        if not run:
            return _err("record_shadow_decision", f"Shadow test {test_id!r} not found")
        if run.status != ShadowTestStatus.RUNNING:
            return _err("record_shadow_decision", f"Shadow test {test_id!r} is not RUNNING (status={run.status.value})")
        record = ShadowDecisionRecord.new(
            test_id=test_id,
            instrument=instrument,
            champion_decision_type=champion_decision_type,
            challenger_decision_type=challenger_decision_type,
            champion_confidence=champion_confidence,
            challenger_confidence=challenger_confidence,
            regime=regime,
            market_context=market_context,
        )
        run.add_decision(record)
        return _ok("record_shadow_decision", {
            "record": record.to_dict(),
            "total_decisions": len(run.decisions),
            "decisions_remaining": max(0, run.min_decisions - len(run.decisions)),
        })

    def conclude_shadow_test(self, test_id: str, notes: str = "") -> ApiResponse:
        """Conclude a shadow test and compute its recommendation."""
        run = self._shadow_tests.get(test_id)
        if not run:
            return _err("conclude_shadow_test", f"Shadow test {test_id!r} not found")
        if run.status != ShadowTestStatus.RUNNING:
            return _err("conclude_shadow_test", f"Shadow test {test_id!r} is not RUNNING")
        run.conclude(notes=notes)
        logger.info(
            f"ResearchLab.conclude_shadow_test: {test_id} "
            f"decisions={run.total_decisions} "
            f"agreement={run.agreement_rate:.1%} "
            f"recommendation={run.recommendation.value}"
        )
        return _ok("conclude_shadow_test", run.to_dict())

    def get_shadow_test(self, test_id: str) -> ApiResponse:
        run = self._shadow_tests.get(test_id)
        if not run:
            return _err("get_shadow_test", f"Shadow test {test_id!r} not found")
        return _ok("get_shadow_test", run.summary())

    def list_shadow_tests(
        self,
        challenger_version: str | None = None,
        status: str | None = None,
    ) -> ApiResponse:
        items = list(self._shadow_tests.values())
        if challenger_version:
            items = [r for r in items if r.challenger_version == challenger_version]
        if status:
            items = [r for r in items if r.status.value == status]
        items.sort(key=lambda r: r.started_at, reverse=True)
        return _ok("list_shadow_tests", {
            "count": len(items),
            "shadow_tests": [r.summary() for r in items],
        })

    # ── Parameter Search ──────────────────────────────────────────────────────

    def create_search(
        self,
        name: str,
        strategy: str,
        search_space: dict[str, Any],
        objective: str,
        maximize: bool = True,
        n_trials: int = 20,
        qde_version: str = "",
        notes: str = "",
    ) -> ApiResponse:
        try:
            run = ParameterSearchRun.new(
                name=name, strategy=SearchStrategy(strategy),
                search_space=search_space, objective=objective,
                maximize=maximize, n_trials=n_trials,
                qde_version=qde_version, notes=notes,
            )
            self._searches[run.search_id] = run
            logger.info(f"ResearchLab.create_search: {run.search_id} ({run.name})")
            return _ok("create_search", run.to_dict())
        except ValueError as exc:
            return _err("create_search", f"Invalid strategy: {exc}")

    def record_trial(
        self,
        search_id: str,
        params: dict[str, Any],
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> ApiResponse:
        run = self._searches.get(search_id)
        if not run:
            return _err("record_trial", f"Search {search_id!r} not found")
        run.record_trial(params, score, metadata)
        return _ok("record_trial", {"search_id": search_id, "trial": run.trials[-1], "best_score": run.best_score})

    def complete_search(self, search_id: str) -> ApiResponse:
        run = self._searches.get(search_id)
        if not run:
            return _err("complete_search", f"Search {search_id!r} not found")
        run.complete()
        return _ok("complete_search", run.to_dict())

    def get_search(self, search_id: str) -> ApiResponse:
        run = self._searches.get(search_id)
        if not run:
            return _err("get_search", f"Search {search_id!r} not found")
        return _ok("get_search", run.to_dict())

    def list_searches(self, qde_version: str | None = None) -> ApiResponse:
        items = list(self._searches.values())
        if qde_version:
            items = [s for s in items if s.qde_version == qde_version]
        items.sort(key=lambda s: s.created_at, reverse=True)
        return _ok("list_searches", {"count": len(items), "searches": [s.to_dict() for s in items]})

    # ── 5. Validation / Deployment Gates ──────────────────────────────────────

    def evaluate_deployment_gates(
        self,
        experiment_id: str,
        baseline_experiment_id: str | None = None,
        max_drawdown_tolerance: float = 0.02,
    ) -> ApiResponse:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return _err("evaluate_deployment_gates", f"Experiment {experiment_id!r} not found")
        baseline_metrics: dict[str, float] | None = None
        if baseline_experiment_id:
            b_exp = self._experiments.get(baseline_experiment_id)
            if b_exp:
                baseline_metrics = b_exp.validation_metrics
        result = DeploymentGateResult.evaluate(
            experiment=exp,
            baseline_metrics=baseline_metrics,
            max_drawdown_tolerance=max_drawdown_tolerance,
        )
        self._gate_results[result.gate_id] = result
        logger.info(
            f"ResearchLab.evaluate_deployment_gates: {result.gate_id} "
            f"experiment={experiment_id} passed={result.passed}"
        )
        return _ok("evaluate_deployment_gates", result.to_dict())

    def validate_retraining(
        self,
        experiment_id: str,
        incumbent_experiment_id: str,
    ) -> ApiResponse:
        return self.evaluate_deployment_gates(
            experiment_id=experiment_id,
            baseline_experiment_id=incumbent_experiment_id,
        )

    # ── 10. Deployment Approval ───────────────────────────────────────────────

    def approve_deployment(
        self,
        experiment_id: str,
        gate_id: str,
        approver: str,
        notes: str = "",
    ) -> ApiResponse:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return _err("approve_deployment", f"Experiment {experiment_id!r} not found")
        gate = self._gate_results.get(gate_id)
        if not gate:
            return _err("approve_deployment", f"Gate result {gate_id!r} not found")
        if not gate.passed:
            return _err(
                "approve_deployment",
                f"Cannot approve: deployment gates failed — {gate.failed_gates}",
            )
        exp.deployment_status = DeploymentStatus.ACTIVE
        exp.deployment_notes = f"Approved by {approver}. {notes}".strip()
        record = {
            "action": "APPROVED",
            "experiment_id": experiment_id,
            "gate_id": gate_id,
            "approver": approver,
            "notes": notes,
            "timestamp": round(time.time(), 3),
        }
        self._deployments.append(record)
        logger.info(f"ResearchLab.approve_deployment: {experiment_id} by {approver!r}")
        return _ok("approve_deployment", {"record": record, "experiment": exp.to_dict()})

    def reject_deployment(self, experiment_id: str, rejector: str, reason: str) -> ApiResponse:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return _err("reject_deployment", f"Experiment {experiment_id!r} not found")
        exp.deployment_status = DeploymentStatus.NOT_DEPLOYED
        exp.deployment_notes = f"Rejected by {rejector}: {reason}"
        record = {
            "action": "REJECTED",
            "experiment_id": experiment_id,
            "rejector": rejector,
            "reason": reason,
            "timestamp": round(time.time(), 3),
        }
        self._deployments.append(record)
        return _ok("reject_deployment", {"record": record, "experiment": exp.to_dict()})

    def deployment_history(self) -> ApiResponse:
        return _ok("deployment_history", {"count": len(self._deployments), "history": self._deployments})

    # ── 9. Failure Analysis ───────────────────────────────────────────────────

    def record_failure(
        self,
        category: str,
        severity: str,
        qde_version: str,
        description: str,
        instrument: str = "",
        regime: str = "",
        loss_amount: float | None = None,
        drawdown_magnitude: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ApiResponse:
        try:
            event = FailureEvent.new(
                category=FailureCategory(category),
                severity=FailureSeverity(severity),
                qde_version=qde_version,
                description=description,
                instrument=instrument,
                regime=regime,
                loss_amount=loss_amount,
                drawdown_magnitude=drawdown_magnitude,
                tags=tags,
                metadata=metadata,
            )
            self._failures[event.failure_id] = event
            logger.warning(
                f"ResearchLab.record_failure: {event.failure_id} "
                f"category={category} severity={severity} version={qde_version}"
            )
            return _ok("record_failure", event.to_dict())
        except ValueError as exc:
            return _err("record_failure", f"Invalid category or severity: {exc}")

    def investigate_failure(self, failure_id: str, root_cause: str, notes: str = "") -> ApiResponse:
        event = self._failures.get(failure_id)
        if not event:
            return _err("investigate_failure", f"Failure {failure_id!r} not found")
        event.root_cause = root_cause
        event.status = "INVESTIGATING"
        if notes:
            event.metadata["investigation_notes"] = notes
        return _ok("investigate_failure", event.to_dict())

    def resolve_failure(self, failure_id: str, resolution: str, resolver: str = "") -> ApiResponse:
        event = self._failures.get(failure_id)
        if not event:
            return _err("resolve_failure", f"Failure {failure_id!r} not found")
        event.resolution = resolution
        event.status = "RESOLVED"
        if resolver:
            event.metadata["resolver"] = resolver
        logger.info(f"ResearchLab.resolve_failure: {failure_id}")
        return _ok("resolve_failure", event.to_dict())

    def list_failures(
        self,
        category: str | None = None,
        severity: str | None = None,
        qde_version: str | None = None,
        status: str | None = None,
    ) -> ApiResponse:
        items = list(self._failures.values())
        if category:
            items = [f for f in items if f.category.value == category]
        if severity:
            items = [f for f in items if f.severity.value == severity]
        if qde_version:
            items = [f for f in items if f.qde_version == qde_version]
        if status:
            items = [f for f in items if f.status == status]
        items.sort(key=lambda f: f.detected_at, reverse=True)
        return _ok("list_failures", {"count": len(items), "failures": [f.to_dict() for f in items]})

    def failure_summary(self) -> ApiResponse:
        failures = list(self._failures.values())
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for f in failures:
            by_category[f.category.value] = by_category.get(f.category.value, 0) + 1
            by_severity[f.severity.value] = by_severity.get(f.severity.value, 0) + 1
            by_status[f.status] = by_status.get(f.status, 0) + 1
        return _ok("failure_summary", {
            "total": len(failures),
            "by_category": by_category,
            "by_severity": by_severity,
            "by_status": by_status,
            "open": by_status.get("OPEN", 0),
            "critical": by_severity.get("CRITICAL", 0),
        })

    # ── Market Regime Analysis ─────────────────────────────────────────────────

    def analyze_regime(
        self,
        qde_version: str,
        regime_results: dict[str, dict[str, float]],
        notes: str = "",
    ) -> ApiResponse:
        try:
            report = RegimeAnalysisReport.build(qde_version, regime_results, notes)
            self._regime_reports[report.report_id] = report
            logger.info(f"ResearchLab.analyze_regime: {report.report_id} version={qde_version}")
            return _ok("analyze_regime", report.to_dict())
        except Exception as exc:
            return _err("analyze_regime", str(exc))

    def get_regime_report(self, report_id: str) -> ApiResponse:
        r = self._regime_reports.get(report_id)
        if not r:
            return _err("get_regime_report", f"Report {report_id!r} not found")
        return _ok("get_regime_report", r.to_dict())

    def list_regime_reports(self, qde_version: str | None = None) -> ApiResponse:
        items = list(self._regime_reports.values())
        if qde_version:
            items = [r for r in items if r.qde_version == qde_version]
        items.sort(key=lambda r: r.created_at, reverse=True)
        return _ok("list_regime_reports", {"count": len(items), "reports": [r.to_dict() for r in items]})

    # ── Model Comparison ──────────────────────────────────────────────────────

    def compare_models(
        self,
        version_ids: list[str],
        metrics_per_version: dict[str, dict[str, float]],
        notes: str = "",
    ) -> ApiResponse:
        try:
            result = ModelComparisonResult.build(version_ids, metrics_per_version, notes)
            self._comparisons[result.comparison_id] = result
            logger.info(f"ResearchLab.compare_models: {result.comparison_id} winner={result.overall_winner}")
            return _ok("compare_models", result.to_dict())
        except Exception as exc:
            return _err("compare_models", str(exc))

    def get_comparison(self, comparison_id: str) -> ApiResponse:
        c = self._comparisons.get(comparison_id)
        if not c:
            return _err("get_comparison", f"Comparison {comparison_id!r} not found")
        return _ok("get_comparison", c.to_dict())

    def list_comparisons(self) -> ApiResponse:
        items = sorted(self._comparisons.values(), key=lambda c: c.created_at, reverse=True)
        return _ok("list_comparisons", {"count": len(items), "comparisons": [c.to_dict() for c in items]})
