"""Phase 25 — Certification Dashboard Generator (5 dashboard types)."""

from __future__ import annotations

import time

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.cert.cert_models import (
    CertificationRun,
)
from trainer.cert.cert_types import (
    CheckStatus,
    DashboardType,
)


class CertDashboard:
    """
    Generates five dashboard dictionaries for each CertificationRun.

    Dashboards are plain dicts suitable for JSON serialization,
    API responses, or front-end rendering.
    """

    def generate_all(self, run: CertificationRun) -> dict[str, dict]:
        return {
            DashboardType.CERTIFICATION.value: self.certification(run),
            DashboardType.READINESS.value: self.readiness(run),
            DashboardType.VALIDATION.value: self.validation(run),
            DashboardType.BENCHMARK.value: self.benchmark(run),
            DashboardType.AUDIT.value: self.audit(run),
        }

    # ── 1. Certification Dashboard ─────────────────────────────────────────

    def certification(self, run: CertificationRun) -> dict:
        sc = run.scorecard
        dec = run.decision
        return {
            "type": DashboardType.CERTIFICATION.value,
            "run_id": run.run_id,
            "model_name": run.model_name,
            "model_version": run.model_version,
            "status": dec.status.value if dec else "PENDING",
            "overall_score": sc.overall if sc else 0.0,
            "grade": sc.grade if sc else "N/A",
            "rationale": dec.rationale if dec else "",
            "conditions": dec.conditions if dec else [],
            "stage_summary": [
                {"stage": s.stage.value, "status": s.status.value, "score": s.score}
                for s in run.stages
            ],
            "scorecard": sc.to_dict() if sc else {},
            "generated_at": time.time(),
        }

    # ── 2. Readiness Dashboard ────────────────────────────────────────────

    def readiness(self, run: CertificationRun) -> dict:
        cl = run.checklist
        op = run.operational
        return {
            "type": DashboardType.READINESS.value,
            "run_id": run.run_id,
            "can_proceed": cl.can_proceed if cl else False,
            "mandatory_failed": cl.mandatory_failed if cl else 0,
            "pass_count": cl.pass_count if cl else 0,
            "warning_count": cl.warning_count if cl else 0,
            "fail_count": cl.fail_count if cl else 0,
            "operational_items": [o.to_dict() for o in op] if op else [],
            "security_items": [s.to_dict() for s in run.security] if run.security else [],
            "generated_at": time.time(),
        }

    # ── 3. Validation Dashboard ───────────────────────────────────────────

    def validation(self, run: CertificationRun) -> dict:
        val = run.validations or []
        mc = run.monte_carlos or []
        return {
            "type": DashboardType.VALIDATION.value,
            "run_id": run.run_id,
            "validations": [v.to_dict() for v in val],
            "validation_summary": {
                "total": len(val),
                "pass": sum(1 for v in val if v.status == CheckStatus.PASS),
                "warning": sum(1 for v in val if v.status == CheckStatus.WARNING),
                "fail": sum(1 for v in val if v.status == CheckStatus.FAIL),
            },
            "monte_carlo": [m.to_dict() for m in mc],
            "mc_summary": {
                "total": len(mc),
                "pass": sum(1 for m in mc if m.status == CheckStatus.PASS),
                "avg_prob_profitable": (
                    round(sum(m.prob_profitable for m in mc) / len(mc), 4) if mc else 0.0
                ),
            },
            "generated_at": time.time(),
        }

    # ── 4. Benchmark Dashboard ────────────────────────────────────────────

    def benchmark(self, run: CertificationRun) -> dict:
        bm = run.benchmarks or []
        st = run.stress_tests or []
        return {
            "type": DashboardType.BENCHMARK.value,
            "run_id": run.run_id,
            "benchmarks": [b.to_dict() for b in bm],
            "benchmark_summary": {
                "total": len(bm),
                "pass": sum(1 for b in bm if b.status == CheckStatus.PASS),
                "warning": sum(1 for b in bm if b.status == CheckStatus.WARNING),
                "fail": sum(1 for b in bm if b.status == CheckStatus.FAIL),
            },
            "stress_tests": [s.to_dict() for s in st],
            "stress_summary": {
                "total": len(st),
                "pass": sum(1 for s in st if s.status == CheckStatus.PASS),
                "avg_degradation": (
                    round(sum(s.degradation_pct for s in st) / len(st), 2) if st else 0.0
                ),
            },
            "generated_at": time.time(),
        }

    # ── 5. Audit Dashboard ────────────────────────────────────────────────

    def audit(self, run: CertificationRun) -> dict:
        audit_pkg = run.audit
        repro = run.reproducibility or []
        return {
            "type": DashboardType.AUDIT.value,
            "run_id": run.run_id,
            "model_id": run.model_id,
            "model_version": run.model_version,
            "audit_package": audit_pkg.to_dict() if audit_pkg else {},
            "reproducibility": [r.to_dict() for r in repro],
            "repro_summary": {
                "total": len(repro),
                "pass": sum(1 for r in repro if r.status == CheckStatus.PASS),
                "fail": sum(1 for r in repro if r.status == CheckStatus.FAIL),
            },
            "stage_audit": [
                {
                    "stage": s.stage.value,
                    "status": s.status.value,
                    "score": s.score,
                    "findings": s.findings,
                    "warnings": s.warnings,
                    "failures": s.failures,
                    "duration_ms": s.duration_ms,
                }
                for s in run.stages
            ],
            "generated_at": time.time(),
        }
