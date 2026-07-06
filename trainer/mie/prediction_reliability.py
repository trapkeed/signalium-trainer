"""Phase 24.9 — Prediction Reliability Engine."""

from __future__ import annotations

from typing import List

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import MarketInput, PredictionReliabilityResult
from trainer.mie.mie_types import clamp


class PredictionReliabilityEngine:
    """
    Scores prediction reliability 0-100 from four independent signals:

    1. Confidence quality    (0-25): raw model confidence, penalised at extremes
    2. Consistency quality   (0-25): calibration score alignment with confidence
    3. Model health          (0-25): model health indicator
    4. Historical win rate   (0-25): adjusted win-rate score

    Total reliability = sum of all four components.
    """

    def __init__(
        self,
        min_confidence: float = 0.30,
        max_confidence: float = 0.95,
    ) -> None:
        self._min_conf = min_confidence
        self._max_conf = max_confidence

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(self, inp: MarketInput) -> PredictionReliabilityResult:
        conf = clamp(inp.prediction_confidence, 0.0, 1.0)
        calib = clamp(inp.calibration_score, 0.0, 1.0)
        health = clamp(inp.model_health, 0.0, 1.0)
        win_rate = clamp(inp.historical_win_rate, 0.0, 1.0)

        conf_score = self._score_confidence(conf)
        cons_score = self._score_consistency(conf, calib)
        health_score = health * 25.0
        wr_score = self._score_win_rate(win_rate)

        reliability = conf_score + cons_score + health_score + wr_score
        reasons = self._generate_reasons(conf, calib, health, win_rate)

        logger.debug(
            f"[PR] conf={conf:.3f} calib={calib:.3f} health={health:.3f} "
            f"wr={win_rate:.3f} → reliability={reliability:.1f}"
        )
        return PredictionReliabilityResult(
            reliability_score=clamp(reliability, 0.0, 100.0),
            confidence_score=clamp(conf_score * 4.0, 0.0, 100.0),
            consistency_score=clamp(cons_score * 4.0, 0.0, 100.0),
            model_health=clamp(health * 100.0, 0.0, 100.0),
            win_rate_adj=clamp(wr_score * 4.0, 0.0, 100.0),
            reasons=reasons,
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _score_confidence(self, conf: float) -> float:
        """Peak quality in the [min_conf, max_conf] sweet spot (0-25)."""
        if conf < self._min_conf:
            return clamp(conf / self._min_conf * 10.0, 0.0, 10.0)
        if conf > self._max_conf:
            return clamp((1.0 - conf) / (1.0 - self._max_conf) * 15.0, 0.0, 15.0)
        # Smooth bell-peak at 0.70
        peak = 0.70
        spread = self._max_conf - self._min_conf
        dist = abs(conf - peak) / max(spread / 2.0, 1e-9)
        return clamp(25.0 * (1.0 - dist * 0.5), 0.0, 25.0)

    def _score_consistency(self, conf: float, calib: float) -> float:
        """Rewards calibration and agreement between confidence and calib (0-25)."""
        calib_bonus = calib * 15.0
        agreement = 1.0 - abs(conf - calib)
        agree_bonus = agreement * 10.0
        return clamp(calib_bonus + agree_bonus, 0.0, 25.0)

    def _score_win_rate(self, win_rate: float) -> float:
        """Win rate 0-1 → score 0-25 with penalty below 40% (0-25)."""
        if win_rate < 0.40:
            return clamp(win_rate / 0.40 * 10.0, 0.0, 10.0)
        return clamp((win_rate - 0.40) / 0.60 * 25.0, 0.0, 25.0)

    def _generate_reasons(
        self,
        conf: float,
        calib: float,
        health: float,
        win_rate: float,
    ) -> List[str]:
        reasons: List[str] = []
        if conf < self._min_conf:
            reasons.append(f"Low prediction confidence ({conf:.0%})")
        if conf > self._max_conf:
            reasons.append(f"Overconfident prediction ({conf:.0%})")
        if calib < 0.40:
            reasons.append(f"Poor calibration ({calib:.0%})")
        if health < 0.60:
            reasons.append(f"Degraded model health ({health:.0%})")
        if win_rate < 0.45:
            reasons.append(f"Below-average win rate ({win_rate:.0%})")
        return reasons
