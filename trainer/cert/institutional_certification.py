"""Phase S27 — Institutional Validation, Governance and Certification.

This module implements the pieces of the S27 certification spec that the
existing Phase 25 certification pipeline (`cert_models.py`, the stage
runners, `ScorecardResult`) does not cover:

1. A fifth certification domain — **Market Certification** — alongside the
   four the spec requires (Performance, Risk, Operational, Research).
   `ScorecardResult` (Phase 25) has no market-regime dimension at all.
2. The specific institutional sub-metrics the spec names explicitly for the
   Performance domain (Sharpe, Sortino, Calmar, expectancy, max drawdown,
   recovery factor, consistency) — Phase 25's `ScorecardResult.performance`
   is a single 0-100 score with no breakdown.
3. A **permanent, immutable certification report** per QDE version — Phase
   25's `CertificationRun`/`AuditPackage` are in-memory pipeline artifacts;
   nothing persists a durable, queryable institutional certification record
   keyed by QDE version the way this spec requires.

This module is deliberately additive: it does not replace `CertificationRun`
or the Phase 25 stage pipeline. It *consumes* raw metrics (which may come
from a completed `CertificationRun`, a backtest, or live telemetry) and
produces the five-domain `InstitutionalCertificationReport`, which is then
persisted via `ResearchLabDatabase` / exposed via `ResearchLabAPI` (S27
integration layer) so it becomes the permanent record for that QDE version.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CertificationDomain(str, Enum):
    PERFORMANCE = "PERFORMANCE"
    RISK = "RISK"
    OPERATIONAL = "OPERATIONAL"
    RESEARCH = "RESEARCH"
    MARKET = "MARKET"


class DomainCertificationStatus(str, Enum):
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    FAIL = "FAIL"


# ── Domain thresholds ────────────────────────────────────────────────────────
#
# Institutional-grade defaults. Each is deliberately conservative — this is
# the gate that stands between a QDE version and real capital.

PERFORMANCE_THRESHOLDS: dict[str, float] = {
    "min_sharpe_ratio": 1.0,
    "min_sortino_ratio": 1.2,
    "min_calmar_ratio": 0.5,
    "min_expectancy": 0.0,  # expected value per trade must be positive
    "max_drawdown_pct": 20.0,  # max peak-to-trough drawdown, percent
    "min_recovery_factor": 1.5,  # net profit / max drawdown
    "min_consistency_score": 0.55,  # fraction of profitable sub-periods
}

RISK_THRESHOLDS: dict[str, float] = {
    "max_var_95_pct": 5.0,  # 95% Value-at-Risk, percent of capital
    "max_leverage": 5.0,
    "max_concentration_pct": 25.0,  # max single-instrument exposure
    "max_tail_risk_ratio": 3.0,  # CVaR / VaR
    "min_risk_adjusted_return": 0.5,
}

OPERATIONAL_THRESHOLDS: dict[str, float] = {
    "max_inference_latency_ms": 250.0,
    "min_uptime_pct": 99.5,
    "max_error_rate_pct": 1.0,
    "min_data_quality_score": 0.90,
    "min_failover_success_rate": 0.95,
}

RESEARCH_THRESHOLDS: dict[str, float] = {
    "min_research_quality_score": 60.0,
    "min_reproducibility_score": 0.90,
    "min_documentation_completeness": 0.80,
    "min_peer_review_score": 0.70,
}

MARKET_THRESHOLDS: dict[str, float] = {
    "min_regime_coverage": 0.75,  # fraction of known regimes tested
    "min_cross_instrument_consistency": 0.55,
    "max_regime_performance_variance": 0.40,
    "min_liquidity_stress_survival": 0.60,
    "min_crisis_scenario_score": 0.40,
}


def _pct(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


# ── Performance domain (with explicit institutional sub-metrics) ───────────


@dataclass
class PerformanceCertification:
    """Performance domain, broken out into the specific institutional
    sub-metrics the S27 spec names explicitly — not a single blended score.
    """

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0
    max_drawdown_pct: float = 0.0
    recovery_factor: float = 0.0
    consistency_score: float = 0.0

    status: DomainCertificationStatus = DomainCertificationStatus.FAIL
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def evaluate(
        cls, metrics: dict[str, float], thresholds: dict[str, float] | None = None
    ) -> PerformanceCertification:
        th = thresholds or PERFORMANCE_THRESHOLDS
        cert = cls(
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics.get("sortino_ratio", 0.0),
            calmar_ratio=metrics.get("calmar_ratio", 0.0),
            expectancy=metrics.get("expectancy", 0.0),
            max_drawdown_pct=metrics.get("max_drawdown_pct", 100.0),
            recovery_factor=metrics.get("recovery_factor", 0.0),
            consistency_score=metrics.get("consistency_score", 0.0),
        )
        checks = [
            ("sharpe_ratio", cert.sharpe_ratio >= th["min_sharpe_ratio"], False),
            ("sortino_ratio", cert.sortino_ratio >= th["min_sortino_ratio"], False),
            ("calmar_ratio", cert.calmar_ratio >= th["min_calmar_ratio"], True),
            ("expectancy", cert.expectancy >= th["min_expectancy"], False),
            ("max_drawdown_pct", cert.max_drawdown_pct <= th["max_drawdown_pct"], False),
            ("recovery_factor", cert.recovery_factor >= th["min_recovery_factor"], True),
            ("consistency_score", cert.consistency_score >= th["min_consistency_score"], True),
        ]
        cert.status = _resolve_status(checks, cert.failed_checks, cert.warnings)
        return cert

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": CertificationDomain.PERFORMANCE.value,
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "expectancy": round(self.expectancy, 6),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "recovery_factor": round(self.recovery_factor, 4),
            "consistency_score": round(self.consistency_score, 4),
            "status": self.status.value,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
        }


@dataclass
class RiskCertification:
    var_95_pct: float = 0.0
    leverage: float = 0.0
    concentration_pct: float = 0.0
    tail_risk_ratio: float = 0.0
    risk_adjusted_return: float = 0.0

    status: DomainCertificationStatus = DomainCertificationStatus.FAIL
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def evaluate(
        cls, metrics: dict[str, float], thresholds: dict[str, float] | None = None
    ) -> RiskCertification:
        th = thresholds or RISK_THRESHOLDS
        cert = cls(
            var_95_pct=metrics.get("var_95_pct", 100.0),
            leverage=metrics.get("leverage", 0.0),
            concentration_pct=metrics.get("concentration_pct", 100.0),
            tail_risk_ratio=metrics.get("tail_risk_ratio", 100.0),
            risk_adjusted_return=metrics.get("risk_adjusted_return", 0.0),
        )
        checks = [
            ("var_95_pct", cert.var_95_pct <= th["max_var_95_pct"], False),
            ("leverage", cert.leverage <= th["max_leverage"], False),
            ("concentration_pct", cert.concentration_pct <= th["max_concentration_pct"], True),
            ("tail_risk_ratio", cert.tail_risk_ratio <= th["max_tail_risk_ratio"], True),
            (
                "risk_adjusted_return",
                cert.risk_adjusted_return >= th["min_risk_adjusted_return"],
                False,
            ),
        ]
        cert.status = _resolve_status(checks, cert.failed_checks, cert.warnings)
        return cert

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": CertificationDomain.RISK.value,
            "var_95_pct": round(self.var_95_pct, 2),
            "leverage": round(self.leverage, 2),
            "concentration_pct": round(self.concentration_pct, 2),
            "tail_risk_ratio": round(self.tail_risk_ratio, 2),
            "risk_adjusted_return": round(self.risk_adjusted_return, 4),
            "status": self.status.value,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
        }


@dataclass
class OperationalCertification:
    inference_latency_ms: float = 0.0
    uptime_pct: float = 0.0
    error_rate_pct: float = 0.0
    data_quality_score: float = 0.0
    failover_success_rate: float = 0.0

    status: DomainCertificationStatus = DomainCertificationStatus.FAIL
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def evaluate(
        cls, metrics: dict[str, float], thresholds: dict[str, float] | None = None
    ) -> OperationalCertification:
        th = thresholds or OPERATIONAL_THRESHOLDS
        cert = cls(
            inference_latency_ms=metrics.get("inference_latency_ms", 10_000.0),
            uptime_pct=metrics.get("uptime_pct", 0.0),
            error_rate_pct=metrics.get("error_rate_pct", 100.0),
            data_quality_score=metrics.get("data_quality_score", 0.0),
            failover_success_rate=metrics.get("failover_success_rate", 0.0),
        )
        checks = [
            (
                "inference_latency_ms",
                cert.inference_latency_ms <= th["max_inference_latency_ms"],
                False,
            ),
            ("uptime_pct", cert.uptime_pct >= th["min_uptime_pct"], False),
            ("error_rate_pct", cert.error_rate_pct <= th["max_error_rate_pct"], False),
            (
                "data_quality_score",
                cert.data_quality_score >= th["min_data_quality_score"],
                False,
            ),
            (
                "failover_success_rate",
                cert.failover_success_rate >= th["min_failover_success_rate"],
                True,
            ),
        ]
        cert.status = _resolve_status(checks, cert.failed_checks, cert.warnings)
        return cert

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": CertificationDomain.OPERATIONAL.value,
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "uptime_pct": round(self.uptime_pct, 3),
            "error_rate_pct": round(self.error_rate_pct, 3),
            "data_quality_score": round(self.data_quality_score, 4),
            "failover_success_rate": round(self.failover_success_rate, 4),
            "status": self.status.value,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
        }


@dataclass
class ResearchCertification:
    research_quality_score: float = 0.0
    reproducibility_score: float = 0.0
    documentation_completeness: float = 0.0
    peer_review_score: float = 0.0

    status: DomainCertificationStatus = DomainCertificationStatus.FAIL
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def evaluate(
        cls, metrics: dict[str, float], thresholds: dict[str, float] | None = None
    ) -> ResearchCertification:
        th = thresholds or RESEARCH_THRESHOLDS
        cert = cls(
            research_quality_score=metrics.get("research_quality_score", 0.0),
            reproducibility_score=metrics.get("reproducibility_score", 0.0),
            documentation_completeness=metrics.get("documentation_completeness", 0.0),
            peer_review_score=metrics.get("peer_review_score", 0.0),
        )
        checks = [
            (
                "research_quality_score",
                cert.research_quality_score >= th["min_research_quality_score"],
                False,
            ),
            (
                "reproducibility_score",
                cert.reproducibility_score >= th["min_reproducibility_score"],
                False,
            ),
            (
                "documentation_completeness",
                cert.documentation_completeness >= th["min_documentation_completeness"],
                True,
            ),
            (
                "peer_review_score",
                cert.peer_review_score >= th["min_peer_review_score"],
                True,
            ),
        ]
        cert.status = _resolve_status(checks, cert.failed_checks, cert.warnings)
        return cert

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": CertificationDomain.RESEARCH.value,
            "research_quality_score": round(self.research_quality_score, 2),
            "reproducibility_score": round(self.reproducibility_score, 4),
            "documentation_completeness": round(self.documentation_completeness, 4),
            "peer_review_score": round(self.peer_review_score, 4),
            "status": self.status.value,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
        }


@dataclass
class MarketCertification:
    """S27's fifth domain — how the version performs *across market
    conditions*, not just in aggregate. Phase 25's scorecard has nothing
    equivalent to this; the closest existing signal (per-regime stress
    testing / regime coverage reports) never rolled up into a pass/fail
    certification gate.
    """

    regime_coverage: float = 0.0
    cross_instrument_consistency: float = 0.0
    regime_performance_variance: float = 1.0
    liquidity_stress_survival: float = 0.0
    crisis_scenario_score: float = 0.0

    status: DomainCertificationStatus = DomainCertificationStatus.FAIL
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def evaluate(
        cls, metrics: dict[str, float], thresholds: dict[str, float] | None = None
    ) -> MarketCertification:
        th = thresholds or MARKET_THRESHOLDS
        cert = cls(
            regime_coverage=metrics.get("regime_coverage", 0.0),
            cross_instrument_consistency=metrics.get("cross_instrument_consistency", 0.0),
            regime_performance_variance=metrics.get("regime_performance_variance", 1.0),
            liquidity_stress_survival=metrics.get("liquidity_stress_survival", 0.0),
            crisis_scenario_score=metrics.get("crisis_scenario_score", 0.0),
        )
        checks = [
            ("regime_coverage", cert.regime_coverage >= th["min_regime_coverage"], False),
            (
                "cross_instrument_consistency",
                cert.cross_instrument_consistency >= th["min_cross_instrument_consistency"],
                True,
            ),
            (
                "regime_performance_variance",
                cert.regime_performance_variance <= th["max_regime_performance_variance"],
                True,
            ),
            (
                "liquidity_stress_survival",
                cert.liquidity_stress_survival >= th["min_liquidity_stress_survival"],
                False,
            ),
            (
                "crisis_scenario_score",
                cert.crisis_scenario_score >= th["min_crisis_scenario_score"],
                False,
            ),
        ]
        cert.status = _resolve_status(checks, cert.failed_checks, cert.warnings)
        return cert

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": CertificationDomain.MARKET.value,
            "regime_coverage": round(self.regime_coverage, 4),
            "cross_instrument_consistency": round(self.cross_instrument_consistency, 4),
            "regime_performance_variance": round(self.regime_performance_variance, 4),
            "liquidity_stress_survival": round(self.liquidity_stress_survival, 4),
            "crisis_scenario_score": round(self.crisis_scenario_score, 4),
            "status": self.status.value,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
        }


def _resolve_status(
    checks: list[tuple[str, bool, bool]],
    failed_checks: list[str],
    warnings: list[str],
) -> DomainCertificationStatus:
    """Shared pass/conditional/fail resolution.

    Each check is (name, passed, is_soft). A failed hard check (is_soft=False)
    is an automatic domain FAIL. A failed soft check downgrades to
    CONDITIONAL at worst (recorded as a warning, not a hard failure) unless a
    hard check also failed.
    """
    hard_fail = False
    soft_fail = False
    for name, passed, is_soft in checks:
        if passed:
            continue
        if is_soft:
            soft_fail = True
            warnings.append(name)
        else:
            hard_fail = True
            failed_checks.append(name)
    if hard_fail:
        return DomainCertificationStatus.FAIL
    if soft_fail:
        return DomainCertificationStatus.CONDITIONAL
    return DomainCertificationStatus.PASS


# ── Overall institutional certification report (permanent record) ─────────


class InstitutionalCertificationDecision(str, Enum):
    CERTIFIED = "CERTIFIED"
    CERTIFIED_WITH_CONDITIONS = "CERTIFIED_WITH_CONDITIONS"
    REJECTED = "REJECTED"


@dataclass
class InstitutionalCertificationReport:
    """The permanent, immutable certification report for one QDE version.

    S27 requires a permanent record per QDE version across all five
    domains — this is that record. Once created it is never mutated in
    place (the persistence layer stores it write-once, keyed by
    ``report_id``); a re-certification produces a new report rather than
    overwriting this one, so the full certification history for a version
    is always reconstructable.
    """

    report_id: str
    qde_version: str
    generated_at: float
    performance: PerformanceCertification
    risk: RiskCertification
    operational: OperationalCertification
    research: ResearchCertification
    market: MarketCertification
    decision: InstitutionalCertificationDecision
    rationale: str
    certified_by: str = "AUTOMATED"
    notes: str = ""

    @classmethod
    def certify(
        cls,
        qde_version: str,
        metrics: dict[str, float],
        certified_by: str = "AUTOMATED",
        notes: str = "",
    ) -> InstitutionalCertificationReport:
        performance = PerformanceCertification.evaluate(metrics)
        risk = RiskCertification.evaluate(metrics)
        operational = OperationalCertification.evaluate(metrics)
        research = ResearchCertification.evaluate(metrics)
        market = MarketCertification.evaluate(metrics)

        domains = [performance, risk, operational, research, market]
        if any(d.status == DomainCertificationStatus.FAIL for d in domains):
            decision = InstitutionalCertificationDecision.REJECTED
            failing = [
                d.__class__.__name__ for d in domains if d.status == DomainCertificationStatus.FAIL
            ]
            rationale = f"Rejected: domain(s) failed hard checks: {', '.join(failing)}"
        elif any(d.status == DomainCertificationStatus.CONDITIONAL for d in domains):
            decision = InstitutionalCertificationDecision.CERTIFIED_WITH_CONDITIONS
            conditional = [
                d.__class__.__name__
                for d in domains
                if d.status == DomainCertificationStatus.CONDITIONAL
            ]
            rationale = (
                f"Certified with conditions: soft-check warnings in {', '.join(conditional)}"
            )
        else:
            decision = InstitutionalCertificationDecision.CERTIFIED
            rationale = "All five certification domains passed all checks."

        return cls(
            report_id=f"instcert-{str(uuid.uuid4())[:12]}",
            qde_version=qde_version,
            generated_at=time.time(),
            performance=performance,
            risk=risk,
            operational=operational,
            research=research,
            market=market,
            decision=decision,
            rationale=rationale,
            certified_by=certified_by,
            notes=notes,
        )

    def is_deployment_eligible(self) -> bool:
        """CERTIFIED_WITH_CONDITIONS still permits deployment (with the
        recorded conditions surfaced to approvers); only REJECTED blocks it.
        """
        return self.decision != InstitutionalCertificationDecision.REJECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "qde_version": self.qde_version,
            "generated_at": round(self.generated_at, 3),
            "performance": self.performance.to_dict(),
            "risk": self.risk.to_dict(),
            "operational": self.operational.to_dict(),
            "research": self.research.to_dict(),
            "market": self.market.to_dict(),
            "decision": self.decision.value,
            "rationale": self.rationale,
            "certified_by": self.certified_by,
            "notes": self.notes,
            "deployment_eligible": self.is_deployment_eligible(),
        }
