"""Phase 25 — Certification Scorecard Engine (9 categories)."""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.cert.cert_models import (
    BenchmarkResult,
    CertificationInput,
    MonteCarloResult,
    OperationalResult,
    ReproducibilityResult,
    ScorecardResult,
    SecurityResult,
    StressTestResult,
    ValidationResult,
)
from trainer.cert.cert_types import (
    SCORECARD_WEIGHTS,
    CheckStatus,
    ScorecardCategory,
    clamp,
)


def _score_from_results(
    results: list,
    pass_w: float = 100.0,
    warn_w: float = 55.0,
    fail_w: float = 10.0,
    na_w: float = 65.0,
) -> float:
    if not results:
        return 50.0
    weights = {
        CheckStatus.PASS: pass_w,
        CheckStatus.WARNING: warn_w,
        CheckStatus.FAIL: fail_w,
        CheckStatus.NOT_APPLICABLE: na_w,
    }
    return clamp(sum(weights.get(r.status, 50.0) for r in results) / len(results))


def _grade(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    return "F"


class ScorecardEngine:
    """
    Computes the Certification Scorecard across 9 categories.

    Category scores are aggregated into an overall score using
    the SCORECARD_WEIGHTS defined in cert_types.
    """

    def compute(
        self,
        inp: CertificationInput,
        validations: list[ValidationResult],
        stress_tests: list[StressTestResult],
        monte_carlos: list[MonteCarloResult],
        benchmarks: list[BenchmarkResult],
        reproducibility: list[ReproducibilityResult],
        security: list[SecurityResult],
        operational: list[OperationalResult],
    ) -> ScorecardResult:

        research = self._research_score(inp)
        valid = _score_from_results(validations)
        perf = self._performance_score(inp, benchmarks)
        calib = self._calibration_score(inp)
        robust = self._robustness_score(stress_tests, monte_carlos)
        sec = self._security_score(security)
        ops = _score_from_results(operational, fail_w=0.0)
        docs = self._docs_score(inp)
        repro = _score_from_results(reproducibility, fail_w=5.0)

        weights = SCORECARD_WEIGHTS
        overall = clamp(
            research * weights[ScorecardCategory.RESEARCH_QUALITY]
            + valid * weights[ScorecardCategory.VALIDATION]
            + perf * weights[ScorecardCategory.PERFORMANCE]
            + calib * weights[ScorecardCategory.CALIBRATION]
            + robust * weights[ScorecardCategory.ROBUSTNESS]
            + sec * weights[ScorecardCategory.SECURITY]
            + ops * weights[ScorecardCategory.OPERATIONAL]
            + docs * weights[ScorecardCategory.DOCUMENTATION]
            + repro * weights[ScorecardCategory.REPRODUCIBILITY]
        )

        result = ScorecardResult(
            research_quality=round(research, 2),
            validation=round(valid, 2),
            performance=round(perf, 2),
            calibration=round(calib, 2),
            robustness=round(robust, 2),
            security=round(sec, 2),
            operational=round(ops, 2),
            documentation=round(docs, 2),
            reproducibility=round(repro, 2),
            overall=round(overall, 2),
            grade=_grade(overall),
        )
        logger.debug(
            f"[SCE] Scorecard: overall={overall:.1f} grade={result.grade} "
            f"research={research:.1f} valid={valid:.1f} robust={robust:.1f}"
        )
        return result

    # ── Category helpers ───────────────────────────────────────────────────

    def _research_score(self, inp: CertificationInput) -> float:
        rq = clamp(inp.research_quality_score)
        # Bonus for having git commit and proper model ID
        bonus = 0.0
        if len(inp.git_commit) >= 7:
            bonus += 5.0
        if inp.model_id:
            bonus += 5.0
        return clamp(rq + bonus)

    def _performance_score(
        self, inp: CertificationInput, benchmarks: list[BenchmarkResult]
    ) -> float:
        bench_s = _score_from_results(benchmarks)
        oos_s = clamp(inp.oos_sharpe * 25.0 + 50.0)  # map sharpe 0→50, 2→100
        return clamp((bench_s + oos_s) / 2.0)

    def _calibration_score(self, inp: CertificationInput) -> float:
        cs = clamp(inp.calibration_score * 100.0)
        brier_p = clamp((1.0 - inp.brier_score * 4.0) * 100.0)  # 0→100 at brier=0
        ece_p = clamp((1.0 - inp.ece * 5.0) * 100.0)
        return clamp((cs + brier_p + ece_p) / 3.0)

    def _robustness_score(
        self,
        stress_tests: list[StressTestResult],
        monte_carlos: list[MonteCarloResult],
    ) -> float:
        stress_s = _score_from_results(stress_tests, warn_w=50.0, fail_w=10.0)
        mc_s = _score_from_results(monte_carlos, warn_w=60.0, fail_w=20.0)
        return clamp(stress_s * 0.6 + mc_s * 0.4)

    def _security_score(self, security: list[SecurityResult]) -> float:
        if not security:
            return 50.0
        from trainer.cert.cert_types import RiskLevel

        weights = {
            CheckStatus.PASS: 100.0,
            CheckStatus.WARNING: 40.0,
            CheckStatus.FAIL: 0.0,
            CheckStatus.NOT_APPLICABLE: 70.0,
        }
        risk_mult = {
            RiskLevel.CRITICAL: 3.0,
            RiskLevel.HIGH: 2.0,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.LOW: 0.5,
            RiskLevel.INFO: 0.25,
        }
        total_w = sum(risk_mult.get(r.risk_level, 1.0) * 100.0 for r in security)
        earned = sum(
            risk_mult.get(r.risk_level, 1.0) * weights.get(r.status, 50.0) for r in security
        )
        return clamp(earned / total_w * 100.0 if total_w > 0 else 50.0)

    def _docs_score(self, inp: CertificationInput) -> float:
        score = 0.0
        if inp.notes:
            score += 20.0
        if inp.tags:
            score += 10.0
        if inp.training_config:
            score += 30.0
        if inp.package_versions:
            score += 20.0
        if inp.environment:
            score += 20.0
        return clamp(score)
