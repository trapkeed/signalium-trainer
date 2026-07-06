"""Phase 25 — Reproducibility Engine (10 checks)."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.cert.cert_types import CheckStatus, RiskLevel, clamp
from trainer.cert.cert_models import CertificationInput, ReproducibilityResult

_CHECKS = [
    "DATASET_VERSION",
    "FEATURE_VERSION",
    "LABEL_VERSION",
    "TRAINING_CONFIG",
    "RANDOM_SEED",
    "GIT_COMMIT",
    "ENVIRONMENT",
    "PACKAGE_VERSIONS",
    "CHECKPOINT_INTEGRITY",
    "MODEL_HASH",
]


class ReproducibilityEngine:
    """
    Verifies 10 reproducibility requirements.

    Each check inspects the CertificationInput for the corresponding field.
    """

    def run_all(self, inp: CertificationInput) -> List[ReproducibilityResult]:
        results = [
            self._dataset_version(inp),
            self._feature_version(inp),
            self._label_version(inp),
            self._training_config(inp),
            self._random_seed(inp),
            self._git_commit(inp),
            self._environment(inp),
            self._package_versions(inp),
            self._checkpoint_integrity(inp),
            self._model_hash(inp),
        ]
        passed = sum(1 for r in results if r.status == CheckStatus.PASS)
        logger.debug(f"[RE] Reproducibility: {passed}/{len(results)} checks passed")
        return results

    def overall_score(self, results: List[ReproducibilityResult]) -> float:
        weights = {
            CheckStatus.PASS: 100.0,
            CheckStatus.WARNING: 60.0,
            CheckStatus.FAIL: 10.0,
            CheckStatus.NOT_APPLICABLE: 70.0,
        }
        if not results:
            return 50.0
        return clamp(sum(weights.get(r.status, 50.0) for r in results) / len(results))

    # ── Individual checks ──────────────────────────────────────────────────

    def _dataset_version(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = bool(inp.dataset_version)
        return ReproducibilityResult(
            check_name="DATASET_VERSION",
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            expected="non-empty version string",
            actual=inp.dataset_version or "(empty)",
            details="Dataset version must be pinned for reproducibility.",
        )

    def _feature_version(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = bool(inp.feature_version)
        return ReproducibilityResult(
            check_name="FEATURE_VERSION",
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            expected="non-empty version string",
            actual=inp.feature_version or "(empty)",
            details="Feature engineering version must be pinned.",
        )

    def _label_version(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = bool(inp.label_version)
        return ReproducibilityResult(
            check_name="LABEL_VERSION",
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            expected="non-empty version string",
            actual=inp.label_version or "(empty)",
            details="Label generation version must be pinned.",
        )

    def _training_config(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = bool(inp.training_config)
        return ReproducibilityResult(
            check_name="TRAINING_CONFIG",
            status=CheckStatus.PASS if ok else CheckStatus.WARNING,
            expected="non-empty training configuration dict",
            actual=f"{len(inp.training_config)} keys" if inp.training_config else "(empty)",
            details="Training hyperparameters must be recorded.",
        )

    def _random_seed(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = inp.random_seed is not None
        return ReproducibilityResult(
            check_name="RANDOM_SEED",
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            expected="integer seed",
            actual=str(inp.random_seed) if ok else "(None)",
            details="Fixed random seed required for reproducibility.",
        )

    def _git_commit(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = len(inp.git_commit) >= 7
        return ReproducibilityResult(
            check_name="GIT_COMMIT",
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            expected="≥7-char git commit SHA",
            actual=inp.git_commit or "(empty)",
            details="Git commit SHA ensures code traceability.",
        )

    def _environment(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = bool(inp.environment)
        return ReproducibilityResult(
            check_name="ENVIRONMENT",
            status=CheckStatus.PASS if ok else CheckStatus.WARNING,
            expected="non-empty environment descriptor",
            actual=inp.environment or "(empty)",
            details="Environment string (e.g. 'cuda-11.8-py3.10') must be recorded.",
        )

    def _package_versions(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = len(inp.package_versions) >= 3
        return ReproducibilityResult(
            check_name="PACKAGE_VERSIONS",
            status=CheckStatus.PASS if ok else CheckStatus.WARNING,
            expected="≥3 pinned package versions",
            actual=f"{len(inp.package_versions)} packages",
            details="Key library versions must be locked.",
        )

    def _checkpoint_integrity(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = len(inp.checkpoint_hash) >= 8
        return ReproducibilityResult(
            check_name="CHECKPOINT_INTEGRITY",
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            expected="≥8-char checkpoint hash",
            actual=inp.checkpoint_hash or "(empty)",
            details="Checkpoint hash verifies model file integrity.",
        )

    def _model_hash(self, inp: CertificationInput) -> ReproducibilityResult:
        ok = len(inp.model_hash) >= 8
        return ReproducibilityResult(
            check_name="MODEL_HASH",
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            expected="≥8-char model hash",
            actual=inp.model_hash or "(empty)",
            details="Model hash ensures deployed weights match certified weights.",
        )
