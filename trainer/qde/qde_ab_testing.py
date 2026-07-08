"""Phase 27.6 — QDE A/B Testing Engine.

Supports two observation-only deployment strategies:
  - SHADOW:       Challenger runs in parallel but decisions are not acted upon.
  - SIDE_BY_SIDE: Both versions receive identical inputs and results are compared.

CANARY mode (routing live traffic to a challenger) has been removed.
Only one model produces live decisions at any time.
"""

from __future__ import annotations

from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.qde_models import QDEABTest, QDEABTestType, QDEStage
from trainer.qde.qde_registry import QDERegistry


class QDEABTestingEngine:
    """Creates, manages, and evaluates A/B tests between QDE versions."""

    def __init__(self, registry: QDERegistry) -> None:
        self._registry = registry
        self._tests: dict[str, QDEABTest] = {}

    # ── Test Creation ─────────────────────────────────────────────────────────

    def create_shadow(
        self,
        challenger_version_id: str,
        control_version_id: str | None = None,
    ) -> QDEABTest:
        """Shadow deployment: challenger observes live traffic without affecting output.

        The challenger's decisions are logged but not executed. This is the safest
        first step before any live exposure.
        """
        control_id = control_version_id or self._require_active_id()
        self._validate_versions(control_id, challenger_version_id)

        test = QDEABTest.create(
            test_type=QDEABTestType.SHADOW,
            control_version_id=control_id,
            challenger_version_id=challenger_version_id,
        )
        self._tests[test.test_id] = test
        logger.info(
            f"QDEABTest SHADOW created: {test.test_id} "
            f"control={control_id} challenger={challenger_version_id}"
        )
        return test

    def create_side_by_side(
        self,
        challenger_version_id: str,
        control_version_id: str | None = None,
    ) -> QDEABTest:
        """Side-by-side evaluation: both versions receive identical inputs simultaneously.

        Neither version's output affects execution — this is purely analytical.
        Results are stored for statistical comparison.
        """
        control_id = control_version_id or self._require_active_id()
        self._validate_versions(control_id, challenger_version_id)

        test = QDEABTest.create(
            test_type=QDEABTestType.SIDE_BY_SIDE,
            control_version_id=control_id,
            challenger_version_id=challenger_version_id,
        )
        self._tests[test.test_id] = test
        logger.info(
            f"QDEABTest SIDE_BY_SIDE created: {test.test_id} "
            f"control={control_id} challenger={challenger_version_id}"
        )
        return test

    # ── Results & Conclusion ──────────────────────────────────────────────────

    def record_results(
        self,
        test_id: str,
        control_metrics: dict[str, float],
        challenger_metrics: dict[str, float],
    ) -> QDEABTest:
        """Record performance metrics for both versions in the test."""
        test = self._get_test(test_id)
        test.record_results(control_metrics, challenger_metrics)
        logger.debug(f"QDEABTest.record_results: {test_id}")
        return test

    def conclude(
        self,
        test_id: str,
        winner: str,
        conclusion: str,
    ) -> QDEABTest:
        """Conclude a test and record the winner.

        Args:
            test_id: The test to conclude.
            winner: version_id of the winning version (or "INCONCLUSIVE").
            conclusion: Human-readable summary of why this winner was chosen.
        """
        test = self._get_test(test_id)
        if test.status != "RUNNING":
            raise RuntimeError(f"Test {test_id} is already {test.status}")
        test.conclude(winner=winner, conclusion=conclusion)
        logger.info(f"QDEABTest.conclude: {test_id} winner={winner!r}")
        return test

    def compare(self, test_id: str) -> dict[str, Any]:
        """Return a statistical comparison between control and challenger metrics."""
        test = self._get_test(test_id)
        comparison: dict[str, Any] = {
            "test_id": test_id,
            "test_type": test.test_type.value,
            "control_version_id": test.control_version_id,
            "challenger_version_id": test.challenger_version_id,
            "status": test.status,
            "winner": test.winner,
            "metrics_comparison": {},
        }

        all_keys = set(test.control_metrics) | set(test.challenger_metrics)
        for key in sorted(all_keys):
            c_val = test.control_metrics.get(key)
            ch_val = test.challenger_metrics.get(key)
            if c_val is not None and ch_val is not None:
                delta = ch_val - c_val
                pct_change = (delta / abs(c_val) * 100) if c_val != 0 else None
                comparison["metrics_comparison"][key] = {
                    "control": round(c_val, 6),
                    "challenger": round(ch_val, 6),
                    "delta": round(delta, 6),
                    "pct_change": round(pct_change, 2) if pct_change is not None else None,
                    "challenger_wins": ch_val > c_val,
                }
            else:
                comparison["metrics_comparison"][key] = {
                    "control": c_val,
                    "challenger": ch_val,
                    "delta": None,
                    "pct_change": None,
                    "challenger_wins": None,
                }

        return comparison

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_test(self, test_id: str) -> QDEABTest | None:
        return self._tests.get(test_id)

    def list_tests(
        self,
        status: str | None = None,
        test_type: QDEABTestType | None = None,
    ) -> list[dict[str, Any]]:
        results = list(self._tests.values())
        if status:
            results = [t for t in results if t.status == status]
        if test_type:
            results = [t for t in results if t.test_type == test_type]
        return [t.to_dict() for t in sorted(results, key=lambda t: t.created_at)]

    def active_tests(self) -> list[dict[str, Any]]:
        return self.list_tests(status="RUNNING")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_test(self, test_id: str) -> QDEABTest:
        test = self._tests.get(test_id)
        if test is None:
            raise KeyError(f"A/B test {test_id!r} not found")
        return test

    def _require_active_id(self) -> str:
        active = self._registry.get_active()
        if active is None:
            raise RuntimeError("No active QDE version found; specify control_version_id explicitly")
        return active.version_id

    def _validate_versions(self, control_id: str, challenger_id: str) -> None:
        if control_id == challenger_id:
            raise ValueError("Control and challenger must be different versions")
        control = self._registry.get(control_id)
        if control is None:
            raise KeyError(f"Control version {control_id!r} not found")
        challenger = self._registry.get(challenger_id)
        if challenger is None:
            raise KeyError(f"Challenger version {challenger_id!r} not found")
        if challenger.stage not in (
            QDEStage.DEPLOYMENT_CANDIDATE,
            QDEStage.SHADOW_TESTING,
            QDEStage.VALIDATION,
            QDEStage.ACTIVE,
        ):
            raise ValueError(
                f"Challenger version {challenger_id!r} is in stage "
                f"{challenger.stage.value}; expected DEPLOYMENT_CANDIDATE, "
                "SHADOW_TESTING, VALIDATION, or ACTIVE"
            )
