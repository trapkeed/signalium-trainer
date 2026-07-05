"""Phase 21 — Entry Quality Engine.

Evaluates and classifies the quality of a proposed trade entry.

Grades: EXCELLENT / GOOD / ACCEPTABLE / POOR / REJECT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.execution.execution_types import (
    ExecutionForecast,
    ExecutionGrade,
    SESSION_QUALITY,
    _clamp,
    _grade_from_score,
)
from trainer.execution.entry_zones import EntryZones


@dataclass
class EntryQualityResult:
    """Output of entry quality evaluation."""

    instrument:   str
    grade:        ExecutionGrade
    score:        float          # 0-1
    reasons:      List[str]      = field(default_factory=list)
    risk_factors: List[str]      = field(default_factory=list)

    # Sub-scores ─────────────────────────────────────────────────────────────
    spread_score:    float = 0.0
    session_score:   float = 0.0
    momentum_score:  float = 0.0
    volatility_score: float = 0.0
    zone_score:      float = 0.0

    def is_tradeable(self) -> bool:
        return self.grade not in (ExecutionGrade.REJECT, ExecutionGrade.POOR)

    def to_dict(self) -> Dict:
        return {
            "instrument":      self.instrument,
            "grade":           self.grade.value,
            "score":           round(self.score, 4),
            "tradeable":       self.is_tradeable(),
            "reasons":         self.reasons,
            "risk_factors":    self.risk_factors,
            "sub_scores": {
                "spread":    round(self.spread_score, 4),
                "session":   round(self.session_score, 4),
                "momentum":  round(self.momentum_score, 4),
                "volatility": round(self.volatility_score, 4),
                "zone":      round(self.zone_score, 4),
            },
        }


class EntryQualityEngine:
    """Scores and grades entry quality across five dimensions.

    Weights:
      spread    20%
      session   20%
      momentum  20%
      volatility 20%
      zone      20%
    """

    _WEIGHTS = {
        "spread":     0.20,
        "session":    0.20,
        "momentum":   0.20,
        "volatility": 0.20,
        "zone":       0.20,
    }

    def evaluate(
        self,
        forecast: ExecutionForecast,
        zones:    Optional[EntryZones] = None,
    ) -> EntryQualityResult:

        reasons: List[str] = []
        risks:   List[str] = []

        # 1. Spread score ─────────────────────────────────────────────────────
        spread_ratio = forecast.spread / max(forecast.atr, 1e-9)
        spread_score = _clamp(1.0 - spread_ratio * 5)
        if spread_ratio > 0.30:
            risks.append(f"High spread/ATR ratio ({spread_ratio:.2f})")
        elif spread_ratio < 0.10:
            reasons.append("Tight spread — excellent fill conditions")

        # 2. Session score ────────────────────────────────────────────────────
        session_score = SESSION_QUALITY.get(forecast.session, 0.5)
        if forecast.session == forecast.session.OVERLAP:
            reasons.append("London-NY overlap — maximum liquidity")
        elif forecast.session == forecast.session.CLOSED:
            risks.append("Market session closed — low liquidity")

        # 3. Momentum score ───────────────────────────────────────────────────
        # momentum -1..+1; BUY wants positive, SELL wants negative
        aligned_mom = forecast.momentum if forecast.is_buy else -forecast.momentum
        momentum_score = _clamp(0.5 + aligned_mom * 0.5)
        if aligned_mom > 0.3:
            reasons.append("Momentum aligned with trade direction")
        elif aligned_mom < -0.3:
            risks.append("Momentum opposing trade direction")

        # 4. Volatility score ─────────────────────────────────────────────────
        # Prefer moderate volatility; penalise very high or very low
        vol = forecast.volatility
        if vol < 0.003:
            volatility_score = 0.4
            risks.append("Very low volatility — poor momentum environment")
        elif vol < 0.008:
            volatility_score = 0.75
        elif vol < 0.015:
            volatility_score = 1.0
            reasons.append("Optimal volatility range")
        elif vol < 0.025:
            volatility_score = 0.70
        else:
            volatility_score = _clamp(1.0 - (vol - 0.025) * 10)
            risks.append("Elevated volatility — wider stops required")

        # 5. Zone score ───────────────────────────────────────────────────────
        if zones is not None:
            # How close is current price to primary entry?
            dist = abs(forecast.current_price - zones.primary_entry)
            zone_score = _clamp(1.0 - dist / max(zones.atr_reference, 1e-9))
            if forecast.current_price == zones.primary_entry:
                reasons.append("Price at primary entry zone")
            elif zone_score > 0.70:
                reasons.append("Price near primary entry zone")
        else:
            zone_score = 0.5  # neutral if no zone info

        # 6. News risk penalty ────────────────────────────────────────────────
        news_penalty = forecast.news_risk * 0.20
        if forecast.news_risk > 0.5:
            risks.append(f"Significant news risk ({forecast.news_risk:.1%})")

        # Weighted composite
        raw = (
            self._WEIGHTS["spread"]     * spread_score +
            self._WEIGHTS["session"]    * session_score +
            self._WEIGHTS["momentum"]   * momentum_score +
            self._WEIGHTS["volatility"] * volatility_score +
            self._WEIGHTS["zone"]       * zone_score
        )
        final_score = _clamp(raw - news_penalty)
        grade = _grade_from_score(final_score)

        if grade == ExecutionGrade.REJECT:
            risks.append("Entry rejected — composite quality too low")
        elif grade == ExecutionGrade.EXCELLENT:
            reasons.append("All quality dimensions passing — excellent setup")

        logger.debug(
            f"EntryQuality {forecast.instrument}: score={final_score:.3f} grade={grade.value}"
        )

        return EntryQualityResult(
            instrument      = forecast.instrument,
            grade           = grade,
            score           = round(final_score, 4),
            reasons         = reasons,
            risk_factors    = risks,
            spread_score    = round(spread_score, 4),
            session_score   = round(session_score, 4),
            momentum_score  = round(momentum_score, 4),
            volatility_score = round(volatility_score, 4),
            zone_score      = round(zone_score, 4),
        )
