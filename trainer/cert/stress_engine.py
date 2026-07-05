"""Phase 25 — Stress Testing Engine (10 scenarios)."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.cert.cert_types import CheckStatus, StressScenario, clamp
from trainer.cert.cert_models import CertificationInput, StressTestResult


# Max acceptable degradation per scenario (%)
_DEGRADATION_LIMITS: Dict[str, float] = {
    StressScenario.HIGH_VOLATILITY.value:     40.0,
    StressScenario.LOW_VOLATILITY.value:      30.0,
    StressScenario.FLASH_CRASH.value:         60.0,
    StressScenario.SPREAD_EXPANSION.value:    35.0,
    StressScenario.LIQUIDITY_REDUCTION.value: 45.0,
    StressScenario.DATA_GAPS.value:           25.0,
    StressScenario.DELAYED_DATA.value:        20.0,
    StressScenario.MODEL_LATENCY.value:       15.0,
    StressScenario.SERVER_RESTART.value:      10.0,
    StressScenario.MISSING_FEATURES.value:    30.0,
}

_DEGRADATION_WARN: Dict[str, float] = {k: v * 0.6 for k, v in _DEGRADATION_LIMITS.items()}


def _check_status(degradation: float, scenario: str) -> CheckStatus:
    warn_lim = _DEGRADATION_WARN.get(scenario, 20.0)
    fail_lim = _DEGRADATION_LIMITS.get(scenario, 40.0)
    if degradation <= warn_lim:
        return CheckStatus.PASS
    if degradation <= fail_lim:
        return CheckStatus.WARNING
    return CheckStatus.FAIL


class StressEngine:
    """
    Runs 10 stress scenarios against the provided CertificationInput.

    If explicit stress_results are provided in the input, they are used directly.
    Otherwise a deterministic simulation is computed from the baseline OOS metrics.
    """

    def run_all(self, inp: CertificationInput) -> List[StressTestResult]:
        results = [
            self._run_scenario(inp, s) for s in StressScenario
        ]
        passed = sum(1 for r in results if r.status == CheckStatus.PASS)
        logger.debug(f"[SE] Stress: {passed}/{len(results)} scenarios passed")
        return results

    def _run_scenario(
        self, inp: CertificationInput, scenario: StressScenario
    ) -> StressTestResult:
        provided = inp.stress_results.get(scenario.value, {})
        if provided:
            degradation  = float(provided.get("degradation_pct", 0.0))
            max_dd       = float(provided.get("max_drawdown", inp.oos_max_drawdown))
            sharpe       = float(provided.get("sharpe_under_stress", inp.oos_sharpe))
        else:
            degradation, max_dd, sharpe = self._simulate(inp, scenario)

        status = _check_status(degradation, scenario.value)
        return StressTestResult(
            scenario=scenario,
            status=status,
            degradation_pct=round(degradation, 2),
            max_drawdown=round(max_dd, 4),
            sharpe_under_stress=round(sharpe, 4),
            details={
                "scenario":    scenario.value,
                "limit_pct":   _DEGRADATION_LIMITS.get(scenario.value, 40.0),
                "simulated":   not bool(provided),
            },
        )

    def _simulate(
        self, inp: CertificationInput, scenario: StressScenario
    ) -> tuple:
        """Deterministic simulation based on baseline metrics."""
        base_sharpe = inp.oos_sharpe
        base_dd     = inp.oos_max_drawdown

        # Scenario-specific multipliers
        factors: Dict[str, float] = {
            StressScenario.HIGH_VOLATILITY.value:     (0.70, 1.40),
            StressScenario.LOW_VOLATILITY.value:      (0.80, 0.80),
            StressScenario.FLASH_CRASH.value:         (0.40, 2.50),
            StressScenario.SPREAD_EXPANSION.value:    (0.75, 1.30),
            StressScenario.LIQUIDITY_REDUCTION.value: (0.60, 1.60),
            StressScenario.DATA_GAPS.value:           (0.85, 1.15),
            StressScenario.DELAYED_DATA.value:        (0.90, 1.10),
            StressScenario.MODEL_LATENCY.value:       (0.95, 1.05),
            StressScenario.SERVER_RESTART.value:      (0.98, 1.02),
            StressScenario.MISSING_FEATURES.value:    (0.78, 1.25),
        }
        sharpe_mult, dd_mult = factors.get(scenario.value, (1.0, 1.0))
        stressed_sharpe = base_sharpe * sharpe_mult
        stressed_dd     = min(1.0, base_dd * dd_mult)
        degradation     = (1.0 - sharpe_mult) * 100.0
        return degradation, stressed_dd, stressed_sharpe
