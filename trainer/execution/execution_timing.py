"""Phase 21 — Execution Timing Engine.

Evaluates market conditions to recommend:
  EXECUTE_NOW / WAIT / AVOID

Factors: Spread, Session, Volatility, Liquidity, News Risk,
         Market Open / Market Close proximity.
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
    BrokerConstraints,
    Session,
    TimingRecommendation,
    SESSION_QUALITY,
    _clamp,
    default_broker,
)


@dataclass
class TimingFactor:
    """One individual timing dimension."""
    name:   str
    score:  float   # 0-1
    detail: str

    def to_dict(self) -> Dict:
        return {"name": self.name, "score": round(self.score, 4), "detail": self.detail}


@dataclass
class TimingAssessment:
    """Complete timing evaluation."""

    instrument:     str
    recommendation: TimingRecommendation
    overall_score:  float

    factors:        List[TimingFactor] = field(default_factory=list)

    # Individual flags
    spread_ok:      bool  = True
    session_ok:     bool  = True
    volatility_ok:  bool  = True
    news_risk_ok:   bool  = True
    liquidity_ok:   bool  = True

    session:        str   = Session.LONDON.value
    expected_slippage: float = 0.0   # price units
    expected_spread:   float = 0.0

    def to_dict(self) -> Dict:
        return {
            "instrument":       self.instrument,
            "recommendation":   self.recommendation.value,
            "overall_score":    round(self.overall_score, 4),
            "factors":          [f.to_dict() for f in self.factors],
            "flags": {
                "spread_ok":    self.spread_ok,
                "session_ok":   self.session_ok,
                "volatility_ok": self.volatility_ok,
                "news_risk_ok": self.news_risk_ok,
                "liquidity_ok": self.liquidity_ok,
            },
            "session":           self.session,
            "expected_slippage": round(self.expected_slippage, 5),
            "expected_spread":   round(self.expected_spread, 5),
        }


class ExecutionTiming:
    """Evaluates execution timing across five dimensions.

    Weights:
      spread    25%
      session   25%
      volatility 20%
      liquidity  15%
      news       15%
    """

    _WEIGHTS = {
        "spread":     0.25,
        "session":    0.25,
        "volatility": 0.20,
        "liquidity":  0.15,
        "news":       0.15,
    }

    def __init__(self, broker: Optional[BrokerConstraints] = None):
        self._broker = broker or default_broker()

    def evaluate(self, forecast: ExecutionForecast) -> TimingAssessment:
        factors: List[TimingFactor] = []

        # 1. Spread ───────────────────────────────────────────────────────────
        spread_ok    = self._broker.spread_ok(forecast.spread)
        spread_ratio = forecast.spread / max(self._broker.max_spread_to_trade, 1e-9)
        spread_score = _clamp(1.0 - spread_ratio)
        factors.append(TimingFactor(
            "spread",
            round(spread_score, 4),
            f"spread={forecast.spread:.5f} limit={self._broker.max_spread_to_trade:.5f} "
            f"{'OK' if spread_ok else 'WIDE'}",
        ))

        # 2. Session ──────────────────────────────────────────────────────────
        session_score = SESSION_QUALITY.get(forecast.session, 0.5)
        session_ok    = forecast.session not in (Session.CLOSED, Session.SYDNEY)
        factors.append(TimingFactor(
            "session",
            round(session_score, 4),
            f"{forecast.session.value} — quality={session_score:.0%}",
        ))

        # 3. Volatility ───────────────────────────────────────────────────────
        vol = forecast.volatility
        if vol < 0.003:
            vol_score = 0.35
            vol_detail = "Very low vol — poor momentum"
            vol_ok = False
        elif vol < 0.020:
            vol_score = 0.90
            vol_detail = "Optimal volatility"
            vol_ok = True
        elif vol < 0.035:
            vol_score = 0.60
            vol_detail = "Elevated volatility — widen stops"
            vol_ok = True
        else:
            vol_score = 0.25
            vol_detail = "Extreme volatility — avoid or reduce size"
            vol_ok = False
        factors.append(TimingFactor("volatility", round(vol_score, 4), vol_detail))

        # 4. Liquidity (proxy: session + spread combined) ─────────────────────
        liq_score = _clamp((session_score + spread_score) / 2)
        liq_ok    = liq_score >= 0.50
        factors.append(TimingFactor(
            "liquidity",
            round(liq_score, 4),
            f"Estimated liquidity index={liq_score:.0%}",
        ))

        # 5. News risk ────────────────────────────────────────────────────────
        news_score = _clamp(1.0 - forecast.news_risk)
        news_ok    = forecast.news_risk < 0.50
        factors.append(TimingFactor(
            "news",
            round(news_score, 4),
            f"news_risk={forecast.news_risk:.0%} {'— CAUTION' if not news_ok else ''}",
        ))

        # Composite
        composite = (
            self._WEIGHTS["spread"]     * spread_score +
            self._WEIGHTS["session"]    * session_score +
            self._WEIGHTS["volatility"] * vol_score +
            self._WEIGHTS["liquidity"]  * liq_score +
            self._WEIGHTS["news"]       * news_score
        )

        # Recommendation
        if not spread_ok or not news_ok or not vol_ok:
            if not spread_ok or forecast.news_risk > 0.75:
                rec = TimingRecommendation.AVOID
            else:
                rec = TimingRecommendation.WAIT
        elif composite >= 0.65:
            rec = TimingRecommendation.EXECUTE_NOW
        elif composite >= 0.45:
            rec = TimingRecommendation.WAIT
        else:
            rec = TimingRecommendation.AVOID

        # Slippage estimate: 0.5–2× spread depending on session quality
        slippage = round(forecast.spread * (1.0 + (1.0 - session_score)), 5)

        logger.debug(
            f"ExecutionTiming {forecast.instrument}: {rec.value} score={composite:.3f}"
        )

        return TimingAssessment(
            instrument       = forecast.instrument,
            recommendation   = rec,
            overall_score    = round(composite, 4),
            factors          = factors,
            spread_ok        = spread_ok,
            session_ok       = session_ok,
            volatility_ok    = vol_ok,
            news_risk_ok     = news_ok,
            liquidity_ok     = liq_ok,
            session          = forecast.session.value,
            expected_slippage = slippage,
            expected_spread   = forecast.spread,
        )
