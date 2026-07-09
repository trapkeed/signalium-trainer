"""S30 — Startup Self-Check.

A single integrated pre-serving health gate. Unlike ``trainer/cli/commands/
doctor.py`` (which diagnoses local environment/dependency issues before a
developer starts working), this runs at process-startup time, right before
the serving loop begins accepting inference traffic, and answers the
narrower question: "is it currently *safe* for this process to serve model
decisions?"

Every check below delegates to an already-existing verifier — this module
never re-derives model, artifact, or health logic.
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


@dataclass
class StartupCheckResult:
    name: str
    passed: bool
    details: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class StartupSelfCheckReport:
    generated_at: float
    checks: list[StartupCheckResult]
    ok_to_serve: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "ok_to_serve": self.ok_to_serve,
            "checks": [
                {"name": c.name, "passed": c.passed, "details": c.details, "evidence": c.evidence}
                for c in self.checks
            ],
        }


class StartupSelfCheck:
    """Runs a minimal set of mandatory pre-serving checks.

    A single mandatory-check failure sets ``ok_to_serve=False``. Checks
    with no configured dependency are treated as non-blocking (they can't
    be evaluated, so they are reported but do not fail the gate) — callers
    that require strict environments should pass every dependency in.
    """

    def __init__(
        self,
        *,
        production_verification_runner: Any | None = None,
        composite_health_scorer: Any | None = None,
        operational_safety: Any | None = None,
        health_threshold: float = 50.0,
    ) -> None:
        self._verification = production_verification_runner
        self._health_scorer = composite_health_scorer
        self._operational_safety = operational_safety
        self._health_threshold = health_threshold

    def _check_production_verification(self) -> StartupCheckResult:
        if self._verification is None:
            return StartupCheckResult("production_verification", True, "not configured — skipped")
        report = self._verification.run()
        passed = report.overall_status != "FAIL"
        return StartupCheckResult(
            "production_verification", passed, f"overall={report.overall_status}", report.to_dict()
        )

    def _check_model_health(self) -> StartupCheckResult:
        if self._health_scorer is None:
            return StartupCheckResult("model_health", True, "not configured — skipped")
        score = self._health_scorer.get_latest()
        if score is None:
            return StartupCheckResult("model_health", True, "no composite health score recorded yet")
        composite = score.get("composite", 100.0)
        passed = composite >= self._health_threshold
        return StartupCheckResult(
            "model_health", passed, f"composite={composite} threshold={self._health_threshold}", score
        )

    def _check_operational_safety(self) -> StartupCheckResult:
        if self._operational_safety is None:
            return StartupCheckResult("operational_safety", True, "not configured — skipped")
        state = self._operational_safety.get_state()
        blocking = state.get("emergency_disabled") or state.get("inference_killed")
        return StartupCheckResult(
            "operational_safety", not blocking, "safety switches clear" if not blocking else "blocking switch engaged", state
        )

    def run(self) -> StartupSelfCheckReport:
        checks = [
            self._check_production_verification(),
            self._check_model_health(),
            self._check_operational_safety(),
        ]
        ok_to_serve = all(c.passed for c in checks)
        report = StartupSelfCheckReport(generated_at=time.time(), checks=checks, ok_to_serve=ok_to_serve)
        logger.info(
            "StartupSelfCheck: " + ("OK_TO_SERVE" if ok_to_serve else "BLOCKED") +
            " " + ", ".join(f"{c.name}={'PASS' if c.passed else 'FAIL'}" for c in checks)
        )
        return report
