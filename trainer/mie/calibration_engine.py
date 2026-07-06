"""Phase 24.9 — Calibration Engine."""

from __future__ import annotations

import math
import statistics
from typing import List, Tuple

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import CalibrationResult, MarketInput
from trainer.mie.mie_types import clamp


def _brier_score(predictions: List[float], outcomes: List[float]) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    if not predictions or len(predictions) != len(outcomes):
        return 0.25  # default (uninformative prior)
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)


def _expected_calibration_error(
    predictions: List[float],
    outcomes: List[float],
    n_bins: int = 5,
) -> float:
    """
    Simplified ECE: partition predictions into `n_bins` equal-width bins,
    compute |mean_pred - mean_outcome| weighted by bin size.
    """
    if len(predictions) < n_bins or len(predictions) != len(outcomes):
        return 0.10  # default

    bins: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, o in zip(predictions, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))

    n_total = len(predictions)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        mean_p = statistics.mean(x[0] for x in b)
        mean_o = statistics.mean(x[1] for x in b)
        ece += (len(b) / n_total) * abs(mean_p - mean_o)
    return ece


class CalibrationEngine:
    """
    Estimates model calibration from a series of (prediction, outcome) pairs.

    If no historical pairs are available, falls back to estimating calibration
    from the raw calibration_score in MarketInput.

    Metrics produced:
    - calibration_score  : 0-100 (100 = perfectly calibrated)
    - brier_score        : 0-1   (lower is better; 0.25 = random)
    - ece                : 0-1   (expected calibration error)
    - overconfidence     : signed float (positive = overconfident)
    - reliability        : 0-1
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        inp: MarketInput,
        pred_hist: List[float] = (),
        out_hist: List[float] = (),
    ) -> CalibrationResult:
        pred_hist = list(pred_hist)
        out_hist = list(out_hist)

        if pred_hist and len(pred_hist) == len(out_hist):
            brier = _brier_score(pred_hist, out_hist)
            ece = _expected_calibration_error(pred_hist, out_hist)
            overconf = self._overconfidence(pred_hist, out_hist)
            # Accuracy: fraction correctly classified at threshold 0.5
            correct = sum(1 for p, o in zip(pred_hist, out_hist) if (p >= 0.5) == (o >= 0.5))
            accuracy = correct / len(pred_hist)
        else:
            # Use calibration_score from input as direct proxy
            cs = clamp(inp.calibration_score, 0.0, 1.0)
            accuracy = cs
            brier = 0.25 - (cs - 0.5) * 0.25
            ece = clamp(1.0 - cs, 0.0, 1.0) * 0.20
            overconf = (inp.prediction_confidence - cs) * 0.5

        # calibration_score: accuracy × (1 - 2×ECE), scaled to 0-100.
        # This gives high scores for well-calibrated, accurate predictions and
        # correctly penalises high ECE (miscalibration) regardless of raw accuracy.
        calib_score = clamp(accuracy * 100.0 * (1.0 - ece * 2.0), 0.0, 100.0)
        reliability = clamp(1.0 - brier * 2.0, 0.0, 1.0)

        logger.debug(
            f"[CE] brier={brier:.4f} ece={ece:.4f} overconf={overconf:.4f} "
            f"→ calib={calib_score:.1f}"
        )
        return CalibrationResult(
            calibration_score=calib_score,
            brier_score=brier,
            ece=ece,
            overconfidence=overconf,
            reliability=reliability,
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _overconfidence(
        self,
        predictions: List[float],
        outcomes: List[float],
    ) -> float:
        """
        Positive = model predicts higher probabilities than observed success rate.
        """
        mean_pred = statistics.mean(predictions)
        mean_out = statistics.mean(outcomes)
        return mean_pred - mean_out
