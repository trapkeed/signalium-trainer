"""Phase 25 — Security Review Engine (8 checks)."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.cert.cert_types import CheckStatus, RiskLevel, clamp
from trainer.cert.cert_models import CertificationInput, SecurityResult

_SECURITY_CHECKS = [
    "AUTHENTICATION",
    "AUTHORIZATION",
    "AUDIT_LOGGING",
    "API_SECURITY",
    "SECRETS_MANAGEMENT",
    "INPUT_VALIDATION",
    "DEPENDENCY_REVIEW",
    "ENCRYPTION",
]

_RISK_MAP: Dict[str, RiskLevel] = {
    "AUTHENTICATION": RiskLevel.CRITICAL,
    "AUTHORIZATION": RiskLevel.CRITICAL,
    "AUDIT_LOGGING": RiskLevel.HIGH,
    "API_SECURITY": RiskLevel.HIGH,
    "SECRETS_MANAGEMENT": RiskLevel.CRITICAL,
    "INPUT_VALIDATION": RiskLevel.HIGH,
    "DEPENDENCY_REVIEW": RiskLevel.MEDIUM,
    "ENCRYPTION": RiskLevel.HIGH,
}


class SecurityEngine:
    """
    Runs 8 security checks.

    Each check consults inp.security_checks[check_name] (bool).
    If the key is absent, the check is marked WARNING.
    """

    def run_all(self, inp: CertificationInput) -> List[SecurityResult]:
        results = [self._check(inp, name) for name in _SECURITY_CHECKS]
        passed = sum(1 for r in results if r.status == CheckStatus.PASS)
        logger.debug(f"[SecE] Security: {passed}/{len(results)} checks passed")
        return results

    def overall_score(self, results: List[SecurityResult]) -> float:
        weights = {
            CheckStatus.PASS: 100.0,
            CheckStatus.WARNING: 50.0,
            CheckStatus.FAIL: 0.0,
            CheckStatus.NOT_APPLICABLE: 75.0,
        }
        # Critical failures are penalised more
        score = 0.0
        total_weight = 0.0
        for r in results:
            risk_mult = {
                RiskLevel.CRITICAL: 2.0,
                RiskLevel.HIGH: 1.5,
                RiskLevel.MEDIUM: 1.0,
                RiskLevel.LOW: 0.5,
                RiskLevel.INFO: 0.25,
            }.get(r.risk_level, 1.0)
            score += weights.get(r.status, 50.0) * risk_mult
            total_weight += 100.0 * risk_mult
        return clamp(score / total_weight * 100.0 if total_weight > 0 else 50.0)

    def _check(self, inp: CertificationInput, name: str) -> SecurityResult:
        checks = inp.security_checks
        risk = _RISK_MAP.get(name, RiskLevel.MEDIUM)
        if name not in checks:
            return SecurityResult(
                check_name=name,
                status=CheckStatus.WARNING,
                risk_level=risk,
                finding=f"{name} not verified — no result provided.",
                remediation=f"Provide a boolean result for '{name}' in security_checks.",
            )
        passed = bool(checks[name])
        return SecurityResult(
            check_name=name,
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            risk_level=risk,
            finding="" if passed else f"{name} check failed.",
            remediation="" if passed else self._remediation(name),
        )

    @staticmethod
    def _remediation(name: str) -> str:
        return {
            "AUTHENTICATION": "Implement JWT/OAuth2 authentication on all API endpoints.",
            "AUTHORIZATION": "Enforce role-based access control (RBAC) for model management.",
            "AUDIT_LOGGING": "Enable structured audit logging for all model inference calls.",
            "API_SECURITY": "Add rate limiting, input sanitisation, and TLS enforcement.",
            "SECRETS_MANAGEMENT": "Rotate secrets and use a vault (e.g. HashiCorp Vault).",
            "INPUT_VALIDATION": "Validate and sanitise all model input features at ingestion.",
            "DEPENDENCY_REVIEW": "Run dependency audit and resolve known CVEs.",
            "ENCRYPTION": "Encrypt model weights at rest and in transit.",
        }.get(name, "Investigate and remediate the security finding.")
