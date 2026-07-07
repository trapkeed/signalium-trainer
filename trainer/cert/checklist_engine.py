"""Phase 25 — Automated Certification Checklist Engine."""

from __future__ import annotations

from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.cert.cert_models import (
    CertificationInput,
    ChecklistItem,
    ChecklistReport,
)
from trainer.cert.cert_types import CheckStatus, RiskLevel


def _item(
    check_id: str,
    category: str,
    description: str,
    status: CheckStatus,
    value: Any = None,
    threshold: float | None = None,
    rationale: str = "",
    mandatory: bool = True,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
) -> ChecklistItem:
    return ChecklistItem(
        check_id=check_id,
        category=category,
        description=description,
        mandatory=mandatory,
        status=status,
        value=value,
        threshold=threshold,
        rationale=rationale,
        risk_level=risk_level,
    )


class ChecklistEngine:
    """
    Builds and evaluates a configurable certification checklist.

    Every requirement is marked: PASS | WARNING | FAIL | NOT_APPLICABLE.
    Certification may not proceed while mandatory checks FAIL.
    """

    def build(self, inp: CertificationInput) -> ChecklistReport:
        items: list[ChecklistItem] = []
        items.extend(self._research_checks(inp))
        items.extend(self._validation_checks(inp))
        items.extend(self._calibration_checks(inp))
        items.extend(self._stress_checks(inp))
        items.extend(self._reproducibility_checks(inp))
        items.extend(self._security_checks(inp))
        items.extend(self._operational_checks(inp))
        items.extend(self._benchmark_checks(inp))

        pass_count = sum(1 for i in items if i.status == CheckStatus.PASS)
        warn_count = sum(1 for i in items if i.status == CheckStatus.WARNING)
        fail_count = sum(1 for i in items if i.status == CheckStatus.FAIL)
        na_count = sum(1 for i in items if i.status == CheckStatus.NOT_APPLICABLE)
        mand_failed = sum(1 for i in items if i.mandatory and i.status == CheckStatus.FAIL)
        can_proceed = mand_failed == 0

        logger.debug(
            f"[CLE] Checklist: {pass_count}P {warn_count}W {fail_count}F {na_count}NA "
            f"mandatory_failed={mand_failed}"
        )
        return ChecklistReport(
            items=items,
            pass_count=pass_count,
            warning_count=warn_count,
            fail_count=fail_count,
            na_count=na_count,
            mandatory_failed=mand_failed,
            can_proceed=can_proceed,
        )

    # ── Check groups ───────────────────────────────────────────────────────

    def _research_checks(self, inp: CertificationInput) -> list[ChecklistItem]:
        rq = inp.research_quality_score
        return [
            _item(
                "RQ-001",
                "RESEARCH",
                "Research quality score ≥ 60",
                (
                    CheckStatus.PASS
                    if rq >= 60
                    else (CheckStatus.WARNING if rq >= 40 else CheckStatus.FAIL)
                ),
                value=rq,
                threshold=60.0,
                rationale="Minimum research quality required for production certification.",
                mandatory=True,
                risk_level=RiskLevel.HIGH,
            ),
            _item(
                "RQ-002",
                "RESEARCH",
                "Model ID provided",
                CheckStatus.PASS if bool(inp.model_id) else CheckStatus.FAIL,
                value=inp.model_id or "(empty)",
                rationale="Model ID is required for audit trail.",
                mandatory=True,
                risk_level=RiskLevel.HIGH,
            ),
            _item(
                "RQ-003",
                "RESEARCH",
                "Git commit recorded",
                CheckStatus.PASS if len(inp.git_commit) >= 7 else CheckStatus.FAIL,
                value=inp.git_commit or "(empty)",
                rationale="Git SHA links certification to a specific code state.",
                mandatory=True,
                risk_level=RiskLevel.HIGH,
            ),
        ]

    def _validation_checks(self, inp: CertificationInput) -> list[ChecklistItem]:
        sharpe = inp.oos_sharpe
        wr = inp.oos_win_rate
        dd = inp.oos_max_drawdown
        return [
            _item(
                "VL-001",
                "VALIDATION",
                "OOS Sharpe ratio > 0",
                CheckStatus.PASS if sharpe > 0 else CheckStatus.FAIL,
                value=sharpe,
                threshold=0.0,
                rationale="Positive Sharpe ratio required for production viability.",
                mandatory=True,
                risk_level=RiskLevel.CRITICAL,
            ),
            _item(
                "VL-002",
                "VALIDATION",
                "OOS win rate ≥ 50%",
                (
                    CheckStatus.PASS
                    if wr >= 0.5
                    else (CheckStatus.WARNING if wr >= 0.45 else CheckStatus.FAIL)
                ),
                value=wr,
                threshold=0.5,
                rationale="Minimum 50% win rate required.",
                mandatory=True,
                risk_level=RiskLevel.HIGH,
            ),
            _item(
                "VL-003",
                "VALIDATION",
                "OOS max drawdown ≤ 30%",
                (
                    CheckStatus.PASS
                    if dd <= 0.30
                    else (CheckStatus.WARNING if dd <= 0.50 else CheckStatus.FAIL)
                ),
                value=dd,
                threshold=0.30,
                rationale="Max drawdown cap protects capital.",
                mandatory=True,
                risk_level=RiskLevel.CRITICAL,
            ),
            _item(
                "VL-004",
                "VALIDATION",
                "Walk-forward results provided",
                CheckStatus.PASS if bool(inp.walk_forward_results) else CheckStatus.WARNING,
                value=len(inp.walk_forward_results),
                rationale="Walk-forward validates temporal generalization.",
                mandatory=False,
                risk_level=RiskLevel.HIGH,
            ),
        ]

    def _calibration_checks(self, inp: CertificationInput) -> list[ChecklistItem]:
        cs = inp.calibration_score
        bs = inp.brier_score
        ece = inp.ece
        return [
            _item(
                "CA-001",
                "CALIBRATION",
                "Calibration score ≥ 0.6",
                (
                    CheckStatus.PASS
                    if cs >= 0.6
                    else (CheckStatus.WARNING if cs >= 0.4 else CheckStatus.FAIL)
                ),
                value=cs,
                threshold=0.6,
                rationale="Well-calibrated probabilities are essential for risk management.",
                mandatory=True,
                risk_level=RiskLevel.HIGH,
            ),
            _item(
                "CA-002",
                "CALIBRATION",
                "Brier score < 0.25",
                (
                    CheckStatus.PASS
                    if bs < 0.25
                    else (CheckStatus.WARNING if bs < 0.30 else CheckStatus.FAIL)
                ),
                value=bs,
                threshold=0.25,
                rationale="Brier score measures probabilistic forecast quality.",
                mandatory=False,
                risk_level=RiskLevel.MEDIUM,
            ),
            _item(
                "CA-003",
                "CALIBRATION",
                "Expected Calibration Error < 0.15",
                (
                    CheckStatus.PASS
                    if ece < 0.15
                    else (CheckStatus.WARNING if ece < 0.25 else CheckStatus.FAIL)
                ),
                value=ece,
                threshold=0.15,
                rationale="Low ECE indicates probabilities match empirical frequencies.",
                mandatory=False,
                risk_level=RiskLevel.MEDIUM,
            ),
        ]

    def _stress_checks(self, inp: CertificationInput) -> list[ChecklistItem]:
        sr = inp.stress_results
        mandatory_scenarios = ["FLASH_CRASH", "HIGH_VOLATILITY", "LIQUIDITY_REDUCTION"]
        items = []
        for scenario in mandatory_scenarios:
            provided = scenario in sr
            if provided:
                deg = sr[scenario].get("degradation_pct", 0.0)
                status = (
                    CheckStatus.PASS
                    if deg < 40
                    else CheckStatus.WARNING if deg < 60 else CheckStatus.FAIL
                )
            else:
                status = CheckStatus.WARNING
            items.append(
                _item(
                    f"ST-{scenario[:3]}",
                    "STRESS",
                    f"{scenario} scenario result",
                    status,
                    value=sr[scenario].get("degradation_pct") if provided else None,
                    threshold=40.0,
                    rationale=f"Must withstand {scenario} stress.",
                    mandatory=True,
                    risk_level=RiskLevel.HIGH,
                )
            )
        return items

    def _reproducibility_checks(self, inp: CertificationInput) -> list[ChecklistItem]:
        checks = [
            ("RP-001", "Dataset version pinned", bool(inp.dataset_version), True, RiskLevel.HIGH),
            ("RP-002", "Feature version pinned", bool(inp.feature_version), True, RiskLevel.HIGH),
            ("RP-003", "Random seed fixed", inp.random_seed is not None, True, RiskLevel.HIGH),
            ("RP-004", "Model hash provided", len(inp.model_hash) >= 8, True, RiskLevel.CRITICAL),
            (
                "RP-005",
                "Checkpoint hash provided",
                len(inp.checkpoint_hash) >= 8,
                True,
                RiskLevel.CRITICAL,
            ),
        ]
        return [
            _item(
                cid,
                "REPRODUCIBILITY",
                desc,
                CheckStatus.PASS if ok else CheckStatus.FAIL,
                mandatory=mand,
                risk_level=rl,
            )
            for cid, desc, ok, mand, rl in checks
        ]

    def _security_checks(self, inp: CertificationInput) -> list[ChecklistItem]:
        sc = inp.security_checks
        critical = ["AUTHENTICATION", "AUTHORIZATION", "SECRETS_MANAGEMENT"]
        items = []
        for i, name in enumerate(
            ["AUTHENTICATION", "AUTHORIZATION", "SECRETS_MANAGEMENT", "API_SECURITY", "ENCRYPTION"],
            start=1,
        ):
            ok = sc.get(name, None)
            status = (
                CheckStatus.PASS
                if ok is True
                else CheckStatus.FAIL if ok is False else CheckStatus.WARNING
            )
            items.append(
                _item(
                    f"SC-{i:03d}",
                    "SECURITY",
                    f"{name} passed",
                    status,
                    value=ok,
                    mandatory=(name in critical),
                    risk_level=RiskLevel.CRITICAL if name in critical else RiskLevel.HIGH,
                )
            )
        return items

    def _operational_checks(self, inp: CertificationInput) -> list[ChecklistItem]:
        oc = inp.operational_checks
        mandatory = [
            "MONITORING_ENABLED",
            "HEALTH_CHECKS_PASSING",
            "ROLLBACK_AVAILABLE",
            "LOGGING_FUNCTIONAL",
        ]
        items = []
        for i, name in enumerate(
            [
                "MONITORING_ENABLED",
                "ALERTS_CONFIGURED",
                "HEALTH_CHECKS_PASSING",
                "ROLLBACK_AVAILABLE",
                "LOGGING_FUNCTIONAL",
            ],
            start=1,
        ):
            ok = oc.get(name, None)
            status = (
                CheckStatus.PASS
                if ok is True
                else CheckStatus.FAIL if ok is False else CheckStatus.WARNING
            )
            items.append(
                _item(
                    f"OP-{i:03d}",
                    "OPERATIONAL",
                    f"{name} confirmed",
                    status,
                    value=ok,
                    mandatory=(name in mandatory),
                    risk_level=RiskLevel.HIGH,
                )
            )
        return items

    def _benchmark_checks(self, inp: CertificationInput) -> list[ChecklistItem]:
        bv = inp.benchmark_values
        lat = bv.get("INFERENCE_LATENCY", None)
        status = (
            CheckStatus.NOT_APPLICABLE
            if lat is None
            else (
                CheckStatus.PASS
                if lat <= 50
                else CheckStatus.WARNING if lat <= 100 else CheckStatus.FAIL
            )
        )
        return [
            _item(
                "BM-001",
                "BENCHMARK",
                "Inference latency ≤ 50ms",
                status,
                value=lat,
                threshold=50.0,
                rationale="Low latency is required for live trading.",
                mandatory=False,
                risk_level=RiskLevel.MEDIUM,
            ),
        ]
