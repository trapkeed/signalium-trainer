"""Phase 25 — Certification API (13-stage pipeline orchestrator)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.cert.cert_types import (
    CheckStatus,
    CertificationStage,
    CertificationStatus,
    CERT_THRESHOLDS,
    STAGE_WEIGHTS,
    clamp,
)
from trainer.cert.cert_models import (
    CertificationInput,
    CertificationRun,
    StageResult,
    CertificationDecision,
    AuditPackage,
)
from trainer.cert.cert_database import CertDatabase
from trainer.cert.validation_suite import ValidationSuite
from trainer.cert.stress_engine import StressEngine
from trainer.cert.monte_carlo_engine import MonteCarloEngine
from trainer.cert.benchmark_engine import BenchmarkEngine
from trainer.cert.reproducibility_engine import ReproducibilityEngine
from trainer.cert.security_engine import SecurityEngine
from trainer.cert.operational_engine import OperationalEngine
from trainer.cert.checklist_engine import ChecklistEngine
from trainer.cert.scorecard_engine import ScorecardEngine
from trainer.cert.cert_dashboard import CertDashboard


def _stage(
    stage: CertificationStage,
    score: float,
    findings: List[str] = None,
    warnings: List[str] = None,
    failures: List[str] = None,
    duration_ms: float = 0.0,
) -> StageResult:
    findings = findings or []
    warnings = warnings or []
    failures = failures or []
    if failures:
        status = CheckStatus.FAIL
    elif warnings:
        status = CheckStatus.WARNING
    else:
        status = CheckStatus.PASS
    return StageResult(
        stage=stage,
        status=status,
        score=round(score, 2),
        findings=findings,
        warnings=warnings,
        failures=failures,
        duration_ms=round(duration_ms, 2),
    )


class CertificationAPI:
    """
    Main certification orchestrator.

    Runs the complete 13-stage pipeline:
    RESEARCH → TRAINING → VALIDATION → WALK_FORWARD → CALIBRATION →
    STRESS_TESTING → MONTE_CARLO → REPRODUCIBILITY → BENCHMARK →
    SECURITY_REVIEW → OPERATIONAL_REVIEW → PERFORMANCE_REVIEW →
    CERTIFICATION_DECISION

    All results are persisted to the CertDatabase.
    """

    def __init__(self, db: Optional[CertDatabase] = None) -> None:
        self._db = db or CertDatabase()
        self._validation = ValidationSuite()
        self._stress = StressEngine()
        self._mc = MonteCarloEngine()
        self._benchmark = BenchmarkEngine()
        self._repro = ReproducibilityEngine()
        self._security = SecurityEngine()
        self._operational = OperationalEngine()
        self._checklist = ChecklistEngine()
        self._scorecard = ScorecardEngine()
        self._dashboard = CertDashboard()

    # ── Public API ─────────────────────────────────────────────────────────

    def certify(self, inp: CertificationInput) -> CertificationRun:
        """Run the full 13-stage certification pipeline."""
        run = CertificationRun(
            model_id=inp.model_id,
            model_name=inp.model_name,
            model_version=inp.model_version,
        )
        self._db.save_run(
            {
                "run_id": run.run_id,
                "model_id": inp.model_id,
                "model_name": inp.model_name,
                "model_version": inp.model_version,
                "requester": inp.requester,
                "started_at": run.started_at,
            }
        )
        logger.info(f"[CertAPI] Starting certification run_id={run.run_id} model={inp.model_name}")

        # ── Stage 1: Research ──────────────────────────────────────────────
        t0 = time.time()
        rq = inp.research_quality_score
        run.stages.append(
            _stage(
                CertificationStage.RESEARCH,
                score=clamp(rq),
                findings=[f"Research quality score: {rq:.1f}"],
                failures=["Research quality < 40"] if rq < 40 else [],
                warnings=["Research quality < 60"] if 40 <= rq < 60 else [],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 2: Training ──────────────────────────────────────────────
        t0 = time.time()
        training_ok = bool(inp.model_id) and bool(inp.git_commit)
        run.stages.append(
            _stage(
                CertificationStage.TRAINING,
                score=100.0 if training_ok else 40.0,
                failures=[] if training_ok else ["model_id or git_commit missing"],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 3: Validation ────────────────────────────────────────────
        t0 = time.time()
        val_results = self._validation.run_all(inp)
        run.validations = val_results
        val_pass = sum(1 for v in val_results if v.status == CheckStatus.PASS)
        val_fail = sum(1 for v in val_results if v.status == CheckStatus.FAIL)
        val_score = clamp(val_pass / max(len(val_results), 1) * 100.0)
        run.stages.append(
            _stage(
                CertificationStage.VALIDATION,
                score=val_score,
                findings=[f"{val_pass}/{len(val_results)} validation checks passed"],
                failures=[f"Validation failures: {val_fail}"] if val_fail else [],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 4: Walk-Forward ──────────────────────────────────────────
        t0 = time.time()
        wf_result = next((v for v in val_results if v.check.value == "WALK_FORWARD"), None)
        wf_score = wf_result.score if wf_result else 40.0
        run.stages.append(
            _stage(
                CertificationStage.WALK_FORWARD,
                score=wf_score,
                warnings=["Walk-forward data not provided"] if not inp.walk_forward_results else [],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 5: Calibration ───────────────────────────────────────────
        t0 = time.time()
        calib_check = next((v for v in val_results if v.check.value == "CALIBRATION"), None)
        calib_score = calib_check.score if calib_check else clamp(inp.calibration_score * 100)
        run.stages.append(
            _stage(
                CertificationStage.CALIBRATION,
                score=calib_score,
                failures=["Poor calibration (score < 40)"] if calib_score < 40 else [],
                warnings=["Calibration score < 60"] if 40 <= calib_score < 60 else [],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 6: Stress Testing ────────────────────────────────────────
        t0 = time.time()
        stress_results = self._stress.run_all(inp)
        run.stress_tests = stress_results
        st_pass = sum(1 for s in stress_results if s.status == CheckStatus.PASS)
        st_fail = sum(1 for s in stress_results if s.status == CheckStatus.FAIL)
        st_score = clamp(st_pass / max(len(stress_results), 1) * 100.0)
        run.stages.append(
            _stage(
                CertificationStage.STRESS_TESTING,
                score=st_score,
                findings=[f"{st_pass}/{len(stress_results)} stress scenarios passed"],
                failures=[f"{st_fail} critical stress failures"] if st_fail else [],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 7: Monte Carlo ───────────────────────────────────────────
        t0 = time.time()
        mc_results = self._mc.run_all(inp)
        run.monte_carlos = mc_results
        mc_pass = sum(1 for m in mc_results if m.status == CheckStatus.PASS)
        mc_score = clamp(mc_pass / max(len(mc_results), 1) * 100.0)
        run.stages.append(
            _stage(
                CertificationStage.MONTE_CARLO,
                score=mc_score,
                findings=[f"{mc_pass}/{len(mc_results)} MC techniques passed"],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 8: Reproducibility ───────────────────────────────────────
        t0 = time.time()
        repro_results = self._repro.run_all(inp)
        run.reproducibility = repro_results
        rp_pass = sum(1 for r in repro_results if r.status == CheckStatus.PASS)
        rp_fail = sum(1 for r in repro_results if r.status == CheckStatus.FAIL)
        rp_score = self._repro.overall_score(repro_results)
        run.stages.append(
            _stage(
                CertificationStage.REPRODUCIBILITY,
                score=rp_score,
                failures=[f"{rp_fail} reproducibility checks failed"] if rp_fail else [],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 9: Benchmark ─────────────────────────────────────────────
        t0 = time.time()
        bm_results = self._benchmark.run_all(inp)
        run.benchmarks = bm_results
        bm_score = self._benchmark.overall_score(bm_results)
        for bm in bm_results:
            self._db.save_benchmark(
                {
                    "run_id": run.run_id,
                    "metric": bm.metric.value,
                    "value": bm.value,
                    "unit": bm.unit,
                    "threshold": bm.threshold,
                    "status": bm.status.value,
                }
            )
        run.stages.append(
            _stage(
                CertificationStage.BENCHMARK,
                score=bm_score,
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 10: Security Review ──────────────────────────────────────
        t0 = time.time()
        sec_results = self._security.run_all(inp)
        run.security = sec_results
        sec_score = self._security.overall_score(sec_results)
        run.stages.append(
            _stage(
                CertificationStage.SECURITY_REVIEW,
                score=sec_score,
                failures=[
                    f"Critical security failure: {r.check_name}"
                    for r in sec_results
                    if r.status == CheckStatus.FAIL
                ],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 11: Operational Review ──────────────────────────────────
        t0 = time.time()
        op_results = self._operational.run_all(inp)
        run.operational = op_results
        op_score = self._operational.overall_score(op_results)
        run.stages.append(
            _stage(
                CertificationStage.OPERATIONAL_REVIEW,
                score=op_score,
                failures=[
                    f"Mandatory op failure: {r.check_name}"
                    for r in op_results
                    if r.status == CheckStatus.FAIL
                ],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Stage 12: Performance Review ──────────────────────────────────
        t0 = time.time()
        perf_score = clamp((val_score * 0.5 + bm_score * 0.3 + st_score * 0.2))
        run.stages.append(
            _stage(
                CertificationStage.PERFORMANCE_REVIEW,
                score=perf_score,
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Checklist ──────────────────────────────────────────────────────
        checklist = self._checklist.build(inp)
        run.checklist = checklist
        for item in checklist.items:
            self._db.save_checklist_item(run.run_id, item.to_dict())

        # ── Scorecard ──────────────────────────────────────────────────────
        scorecard = self._scorecard.compute(
            inp,
            val_results,
            stress_results,
            mc_results,
            bm_results,
            repro_results,
            sec_results,
            op_results,
        )
        run.scorecard = scorecard

        # ── Stage 13: Certification Decision ──────────────────────────────
        t0 = time.time()
        decision = self._make_decision(scorecard, checklist)
        run.decision = decision
        run.stages.append(
            _stage(
                CertificationStage.CERTIFICATION_DECISION,
                score=scorecard.overall,
                findings=[f"Status: {decision.status.value}", decision.rationale],
                duration_ms=(time.time() - t0) * 1000,
            )
        )

        # ── Audit Package ──────────────────────────────────────────────────
        audit = self._build_audit(run)
        run.audit = audit

        run.completed_at = time.time()

        # ── Persist run summary ────────────────────────────────────────────
        self._db.save_run(
            {
                "run_id": run.run_id,
                "model_id": inp.model_id,
                "model_name": inp.model_name,
                "model_version": inp.model_version,
                "requester": inp.requester,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "decision_status": decision.status.value,
                "overall_score": scorecard.overall,
            }
        )
        for sr in run.stages:
            self._db.save_stage_result(
                {
                    "run_id": run.run_id,
                    "stage": sr.stage.value,
                    "status": sr.status.value,
                    "score": sr.score,
                    "findings": sr.findings,
                    "warnings": sr.warnings,
                    "failures": sr.failures,
                    "duration_ms": sr.duration_ms,
                }
            )
        self._db.save_audit_package(audit.to_dict())

        logger.info(
            f"[CertAPI] ✓ run_id={run.run_id} status={decision.status.value} "
            f"score={scorecard.overall:.1f} grade={scorecard.grade}"
        )
        return run

    # ── API Endpoints ──────────────────────────────────────────────────────

    def get_certification_status(self, run_id: str) -> Dict:
        """Endpoint: GET /certification/{run_id}/status"""
        rec = self._db.get_run(run_id)
        if not rec:
            return {"error": f"run_id {run_id!r} not found"}
        return {
            "run_id": rec["run_id"],
            "model_id": rec["model_id"],
            "decision_status": rec.get("decision_status"),
            "overall_score": rec.get("overall_score"),
            "completed_at": rec.get("completed_at"),
        }

    def get_certification_report(self, run_id: str) -> Dict:
        """Endpoint: GET /certification/{run_id}/report"""
        rec = self._db.get_run(run_id)
        stages = self._db.get_stage_results(run_id)
        if not rec:
            return {"error": f"run_id {run_id!r} not found"}
        return {"run": rec, "stages": stages}

    def get_audit_package(self, run_id: str) -> Dict:
        """Endpoint: GET /certification/{run_id}/audit"""
        pkg = self._db.get_audit_package(run_id)
        if not pkg:
            return {"error": f"audit package for {run_id!r} not found"}
        return pkg

    def get_benchmark_results(self, run_id: str) -> Dict:
        """Endpoint: GET /certification/{run_id}/benchmarks"""
        bm = self._db.get_benchmarks(run_id)
        return {"run_id": run_id, "benchmarks": bm}

    def get_validation_results(self, run_id: str) -> Dict:
        """Endpoint: GET /certification/{run_id}/validations"""
        stages = self._db.get_stage_results(run_id)
        val_stages = [
            s for s in stages if s.get("stage") in ("VALIDATION", "WALK_FORWARD", "CALIBRATION")
        ]
        return {"run_id": run_id, "validation_stages": val_stages}

    # ── Internal helpers ───────────────────────────────────────────────────

    def _make_decision(
        self,
        scorecard: Any,
        checklist: Any,
    ) -> CertificationDecision:
        overall = scorecard.overall
        th = CERT_THRESHOLDS

        # Hard block: mandatory checklist failures
        if not checklist.can_proceed:
            mf = checklist.mandatory_failed
            return CertificationDecision(
                status=CertificationStatus.REJECTED,
                rationale=f"{mf} mandatory checklist item(s) failed. "
                "Production deployment blocked.",
            )

        # Minimum sub-category floor checks
        floors = [
            (scorecard.security, th["min_security"], "Security"),
            (scorecard.validation, th["min_validation"], "Validation"),
            (scorecard.operational, th["min_operational"], "Operational"),
        ]
        hard_failures = [
            f"{name} score {val:.1f} < required {req:.1f}" for val, req, name in floors if val < req
        ]
        if hard_failures:
            return CertificationDecision(
                status=CertificationStatus.REJECTED,
                rationale="Critical category floors not met: " + "; ".join(hard_failures),
                restrictions=hard_failures,
            )

        # Score-based decision
        if overall >= th["min_overall_score"]:
            conditions = []
            if scorecard.documentation < 60.0:
                conditions.append(
                    "Documentation score below 60 — complete docs before wider rollout."
                )
            if scorecard.reproducibility < th["min_reproducibility"]:
                conditions.append(
                    "Reproducibility checks incomplete — pin all versions before re-run."
                )
            if conditions:
                return CertificationDecision(
                    status=CertificationStatus.CERTIFIED_WITH_CONDITIONS,
                    rationale=f"Overall score {overall:.1f} meets certification threshold with conditions.",
                    conditions=conditions,
                )
            return CertificationDecision(
                status=CertificationStatus.CERTIFIED,
                rationale=f"All certification criteria met. Overall score: {overall:.1f}.",
            )

        if overall >= th["pilot_overall"]:
            return CertificationDecision(
                status=CertificationStatus.PILOT_ONLY,
                rationale=f"Overall score {overall:.1f} qualifies for pilot deployment only. "
                "Address warnings before full production.",
            )

        if overall >= th["research_overall"]:
            return CertificationDecision(
                status=CertificationStatus.RESEARCH_ONLY,
                rationale=f"Overall score {overall:.1f} insufficient for deployment. "
                "Model requires further development.",
            )

        return CertificationDecision(
            status=CertificationStatus.REJECTED,
            rationale=f"Overall score {overall:.1f} below minimum threshold {th['research_overall']}.",
        )

    def _build_audit(self, run: CertificationRun) -> AuditPackage:
        dec = run.decision
        sc = run.scorecard
        status_line = f"Status: {dec.status.value}" if dec else "PENDING"
        score_line = f"Overall Score: {sc.overall:.1f} ({sc.grade})" if sc else ""

        cert_report = (
            f"CERTIFICATION REPORT\n"
            f"Run ID:        {run.run_id}\n"
            f"Model:         {run.model_name} v{run.model_version}\n"
            f"{status_line}\n"
            f"{score_line}\n"
            f"Rationale:     {dec.rationale if dec else ''}\n"
        )
        val_report = f"VALIDATION REPORT\n" + "\n".join(
            f"  {v.check.value}: {v.status.value} (score={v.score:.1f})"
            for v in (run.validations or [])
        )
        bm_report = f"BENCHMARK REPORT\n" + "\n".join(
            f"  {b.metric.value}: {b.value:.2f}{b.unit} [{b.status.value}]"
            for b in (run.benchmarks or [])
        )
        st_report = f"STRESS TEST REPORT\n" + "\n".join(
            f"  {s.scenario.value}: degradation={s.degradation_pct:.1f}% [{s.status.value}]"
            for s in (run.stress_tests or [])
        )
        sec_report = f"SECURITY REPORT\n" + "\n".join(
            f"  {s.check_name}: {s.status.value} risk={s.risk_level.value}"
            for s in (run.security or [])
        )
        op_checklist = f"OPERATIONAL CHECKLIST\n" + "\n".join(
            f"  [{o.status.value}] {o.check_name}" for o in (run.operational or [])
        )
        summary = (
            f"EXECUTIVE SUMMARY\n"
            f"Model {run.model_name} v{run.model_version} — Certification {status_line}.\n"
            f"{score_line}\n"
            f"Stages run: {len(run.stages)}\n"
            f"Validations: {len(run.validations or [])}\n"
            f"Stress tests: {len(run.stress_tests or [])}\n"
            f"Monte Carlo: {len(run.monte_carlos or [])}\n"
            f"Benchmarks: {len(run.benchmarks or [])}\n"
        )
        pkg = AuditPackage(
            run_id=run.run_id,
            certification_report=cert_report,
            validation_report=val_report,
            benchmark_report=bm_report,
            stress_test_report=st_report,
            security_report=sec_report,
            operational_checklist=op_checklist,
            executive_summary=summary,
        )
        self._db.save_audit_package(pkg.to_dict())
        return pkg
