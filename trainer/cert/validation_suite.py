"""Phase 25 — Validation Suite (8 checks)."""

from __future__ import annotations

import statistics

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.cert.cert_models import CertificationInput, ValidationResult
from trainer.cert.cert_types import CheckStatus, ValidationCheck, clamp


def _status(score: float, pass_thresh: float, warn_thresh: float) -> CheckStatus:
    if score >= pass_thresh:
        return CheckStatus.PASS
    if score >= warn_thresh:
        return CheckStatus.WARNING
    return CheckStatus.FAIL


class ValidationSuite:
    """
    Runs all 8 validation checks against the CertificationInput:

    1. Walk-Forward Performance
    2. Out-of-Sample Performance
    3. Cross-Regime Performance
    4. Cross-Instrument Performance
    5. Cross-Timeframe Performance
    6. Calibration
    7. Generalization
    8. Stability
    """

    # Thresholds: (pass, warning)
    _THRESHOLDS: dict[str, tuple] = {
        ValidationCheck.WALK_FORWARD.value: (60.0, 40.0),
        ValidationCheck.OUT_OF_SAMPLE.value: (65.0, 45.0),
        ValidationCheck.CROSS_REGIME.value: (55.0, 35.0),
        ValidationCheck.CROSS_INSTRUMENT.value: (50.0, 30.0),
        ValidationCheck.CROSS_TIMEFRAME.value: (50.0, 30.0),
        ValidationCheck.CALIBRATION.value: (60.0, 40.0),
        ValidationCheck.GENERALIZATION.value: (55.0, 35.0),
        ValidationCheck.STABILITY.value: (55.0, 35.0),
    }

    def run_all(self, inp: CertificationInput) -> list[ValidationResult]:
        results = [
            self._walk_forward(inp),
            self._out_of_sample(inp),
            self._cross_regime(inp),
            self._cross_instrument(inp),
            self._cross_timeframe(inp),
            self._calibration(inp),
            self._generalization(inp),
            self._stability(inp),
        ]
        logger.debug(f"[VS] Ran {len(results)} validation checks")
        return results

    # ── Individual checks ──────────────────────────────────────────────────

    def _walk_forward(self, inp: CertificationInput) -> ValidationResult:
        results = inp.walk_forward_results
        if not results:
            return ValidationResult(
                check=ValidationCheck.WALK_FORWARD,
                status=CheckStatus.WARNING,
                score=40.0,
                details={"note": "no walk-forward data provided"},
            )
        sharpes = [r.get("sharpe", 0.0) for r in results if "sharpe" in r]
        win_rates = [r.get("win_rate", 0.5) for r in results if "win_rate" in r]
        avg_sharpe = statistics.mean(sharpes) if sharpes else 0.0
        avg_wr = statistics.mean(win_rates) if win_rates else 0.5
        # Score: sharpe contrib (0-60) + win_rate contrib (0-40)
        sharpe_score = clamp(avg_sharpe * 30.0, 0.0, 60.0)
        wr_score = clamp((avg_wr - 0.4) / 0.2 * 40.0, 0.0, 40.0)
        score = clamp(sharpe_score + wr_score)
        p, w = self._THRESHOLDS[ValidationCheck.WALK_FORWARD.value]
        return ValidationResult(
            check=ValidationCheck.WALK_FORWARD,
            status=_status(score, p, w),
            score=score,
            details={
                "avg_sharpe": round(avg_sharpe, 4),
                "avg_win_rate": round(avg_wr, 4),
                "folds": len(results),
            },
        )

    def _out_of_sample(self, inp: CertificationInput) -> ValidationResult:
        sharpe = inp.oos_sharpe
        wr = inp.oos_win_rate
        dd = inp.oos_max_drawdown
        # Score: sharpe (0-50) + win_rate (0-30) + drawdown penalty (0-20)
        sharpe_s = clamp(sharpe * 25.0, 0.0, 50.0)
        wr_s = clamp((wr - 0.4) / 0.2 * 30.0, 0.0, 30.0)
        dd_s = clamp((1.0 - dd / 0.5) * 20.0, 0.0, 20.0)
        score = clamp(sharpe_s + wr_s + dd_s)
        p, w = self._THRESHOLDS[ValidationCheck.OUT_OF_SAMPLE.value]
        return ValidationResult(
            check=ValidationCheck.OUT_OF_SAMPLE,
            status=_status(score, p, w),
            score=score,
            details={
                "sharpe": round(sharpe, 4),
                "win_rate": round(wr, 4),
                "max_drawdown": round(dd, 4),
            },
        )

    def _cross_regime(self, inp: CertificationInput) -> ValidationResult:
        results = inp.cross_regime_results
        if not results:
            return ValidationResult(
                check=ValidationCheck.CROSS_REGIME,
                status=CheckStatus.WARNING,
                score=35.0,
                details={"note": "no cross-regime data"},
            )
        scores = [r.get("score", 50.0) for r in results]
        avg = statistics.mean(scores)
        # Penalise high variance across regimes
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        score = clamp(avg - stdev * 0.5)
        p, w = self._THRESHOLDS[ValidationCheck.CROSS_REGIME.value]
        return ValidationResult(
            check=ValidationCheck.CROSS_REGIME,
            status=_status(score, p, w),
            score=score,
            details={
                "avg_score": round(avg, 2),
                "stdev": round(stdev, 2),
                "regimes_tested": len(results),
            },
        )

    def _cross_instrument(self, inp: CertificationInput) -> ValidationResult:
        results = inp.cross_instrument_results
        if not results:
            return ValidationResult(
                check=ValidationCheck.CROSS_INSTRUMENT,
                status=CheckStatus.NOT_APPLICABLE,
                score=50.0,
                details={"note": "no cross-instrument data"},
            )
        scores = [r.get("score", 50.0) for r in results]
        score = clamp(statistics.mean(scores))
        p, w = self._THRESHOLDS[ValidationCheck.CROSS_INSTRUMENT.value]
        return ValidationResult(
            check=ValidationCheck.CROSS_INSTRUMENT,
            status=_status(score, p, w),
            score=score,
            details={"avg_score": round(score, 2), "instruments": len(results)},
        )

    def _cross_timeframe(self, inp: CertificationInput) -> ValidationResult:
        results = inp.cross_timeframe_results
        if not results:
            return ValidationResult(
                check=ValidationCheck.CROSS_TIMEFRAME,
                status=CheckStatus.NOT_APPLICABLE,
                score=50.0,
                details={"note": "no cross-timeframe data"},
            )
        scores = [r.get("score", 50.0) for r in results]
        score = clamp(statistics.mean(scores))
        p, w = self._THRESHOLDS[ValidationCheck.CROSS_TIMEFRAME.value]
        return ValidationResult(
            check=ValidationCheck.CROSS_TIMEFRAME,
            status=_status(score, p, w),
            score=score,
            details={"avg_score": round(score, 2), "timeframes": len(results)},
        )

    def _calibration(self, inp: CertificationInput) -> ValidationResult:
        # calibration_score is 0-1; convert to 0-100
        calib = clamp(inp.calibration_score * 100.0)
        brier = inp.brier_score  # lower better: 0.25=random, 0=perfect
        ece = inp.ece  # lower better
        # brier penalty
        brier_pen = clamp(brier * 100.0, 0.0, 25.0)
        ece_pen = clamp(ece * 100.0, 0.0, 20.0)
        score = clamp(calib - brier_pen - ece_pen)
        p, w = self._THRESHOLDS[ValidationCheck.CALIBRATION.value]
        return ValidationResult(
            check=ValidationCheck.CALIBRATION,
            status=_status(score, p, w),
            score=score,
            details={
                "calibration_score": round(inp.calibration_score, 4),
                "brier_score": round(brier, 4),
                "ece": round(ece, 4),
            },
        )

    def _generalization(self, inp: CertificationInput) -> ValidationResult:
        score = clamp(inp.generalization_score)
        p, w = self._THRESHOLDS[ValidationCheck.GENERALIZATION.value]
        status = CheckStatus.NOT_APPLICABLE if score == 0.0 else _status(score, p, w)
        return ValidationResult(
            check=ValidationCheck.GENERALIZATION,
            status=status,
            score=score,
            details={"generalization_score": round(score, 2)},
        )

    def _stability(self, inp: CertificationInput) -> ValidationResult:
        score = clamp(inp.stability_score)
        p, w = self._THRESHOLDS[ValidationCheck.STABILITY.value]
        status = CheckStatus.NOT_APPLICABLE if score == 0.0 else _status(score, p, w)
        return ValidationResult(
            check=ValidationCheck.STABILITY,
            status=status,
            score=score,
            details={"stability_score": round(score, 2)},
        )
