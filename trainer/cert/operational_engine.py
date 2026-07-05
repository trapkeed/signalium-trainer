"""Phase 25 — Operational Readiness Engine (8 checks)."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.cert.cert_types import CheckStatus, clamp
from trainer.cert.cert_models import CertificationInput, OperationalResult


_OPERATIONAL_CHECKS = [
    "MONITORING_ENABLED",
    "ALERTS_CONFIGURED",
    "HEALTH_CHECKS_PASSING",
    "ROLLBACK_AVAILABLE",
    "BACKUPS_VERIFIED",
    "LOGGING_FUNCTIONAL",
    "DOCUMENTATION_COMPLETE",
    "RECOVERY_PROCEDURES_VERIFIED",
]

# mandatory vs advisory
_MANDATORY: Dict[str, bool] = {
    "MONITORING_ENABLED":            True,
    "ALERTS_CONFIGURED":             True,
    "HEALTH_CHECKS_PASSING":         True,
    "ROLLBACK_AVAILABLE":            True,
    "BACKUPS_VERIFIED":              False,
    "LOGGING_FUNCTIONAL":            True,
    "DOCUMENTATION_COMPLETE":        False,
    "RECOVERY_PROCEDURES_VERIFIED":  False,
}


class OperationalEngine:
    """
    Confirms 8 operational readiness checks.

    Each check consults inp.operational_checks[check_name] (bool).
    Missing keys produce a WARNING; False produces FAIL or WARNING depending on
    whether the check is mandatory.
    """

    def run_all(self, inp: CertificationInput) -> List[OperationalResult]:
        results = [self._check(inp, name) for name in _OPERATIONAL_CHECKS]
        passed = sum(1 for r in results if r.status == CheckStatus.PASS)
        logger.debug(f"[OE] Operational: {passed}/{len(results)} checks confirmed")
        return results

    def overall_score(self, results: List[OperationalResult]) -> float:
        weights = {
            CheckStatus.PASS:           100.0,
            CheckStatus.WARNING:         55.0,
            CheckStatus.FAIL:            10.0,
            CheckStatus.NOT_APPLICABLE:  70.0,
        }
        if not results:
            return 50.0
        # Mandatory failures have extra weight
        score = 0.0
        for r in results:
            mandatory = _MANDATORY.get(r.check_name, False)
            w = weights.get(r.status, 50.0)
            if mandatory and r.status == CheckStatus.FAIL:
                w = 0.0
            score += w
        return clamp(score / len(results))

    def _check(self, inp: CertificationInput, name: str) -> OperationalResult:
        checks   = inp.operational_checks
        mandatory = _MANDATORY.get(name, False)
        if name not in checks:
            return OperationalResult(
                check_name=name,
                status=CheckStatus.WARNING,
                details=f"{name} not provided — unable to confirm.",
                action=f"Provide '{name}' in operational_checks.",
            )
        passed = bool(checks[name])
        if passed:
            return OperationalResult(
                check_name=name,
                status=CheckStatus.PASS,
                details=f"{name} confirmed.",
                action="",
            )
        severity = CheckStatus.FAIL if mandatory else CheckStatus.WARNING
        return OperationalResult(
            check_name=name,
            status=severity,
            details=f"{name} is {'MANDATORY and ' if mandatory else ''}not satisfied.",
            action=self._action(name),
        )

    @staticmethod
    def _action(name: str) -> str:
        return {
            "MONITORING_ENABLED":            "Deploy Prometheus/Grafana or equivalent.",
            "ALERTS_CONFIGURED":             "Configure PagerDuty / alerting rules for latency and error rate.",
            "HEALTH_CHECKS_PASSING":         "Fix failing health check endpoint before deployment.",
            "ROLLBACK_AVAILABLE":            "Tag previous model version and test rollback procedure.",
            "BACKUPS_VERIFIED":              "Run backup restoration drill and verify checksums.",
            "LOGGING_FUNCTIONAL":            "Ensure structured logging pipeline is active.",
            "DOCUMENTATION_COMPLETE":        "Complete model card, API docs, and runbook.",
            "RECOVERY_PROCEDURES_VERIFIED":  "Document and drill the disaster-recovery playbook.",
        }.get(name, "Resolve the operational gap before deployment.")
