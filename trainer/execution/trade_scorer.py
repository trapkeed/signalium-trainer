"""Phase 21 — Trade Scorer.

Produces an Execution Quality Score and Grade for each execution plan.

Factors: Forecast Quality, Entry Quality, Risk, Liquidity,
         Spread, Volatility, Expected Reward, Portfolio Impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.entry_quality import EntryQualityResult
from trainer.execution.execution_timing import TimingAssessment
from trainer.execution.execution_types import (
    ExecutionForecast,
    ExecutionGrade,
    _clamp,
    _grade_from_score,
)
from trainer.execution.risk_engine import RiskProfile


@dataclass
class TradeScore:
    """Comprehensive execution quality score."""

    instrument: str

    # Sub-scores (all 0-1)
    forecast_quality: float
    entry_quality: float
    risk_score: float
    liquidity_score: float
    spread_score: float
    volatility_score: float
    reward_score: float
    portfolio_impact: float

    # Overall
    overall_score: float
    grade: ExecutionGrade
    rank_percentile: float = 50.0  # placeholder; relative ranking

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "instrument": self.instrument,
            "overall_score": round(self.overall_score, 4),
            "grade": self.grade.value,
            "rank_percentile": round(self.rank_percentile, 1),
            "sub_scores": {
                "forecast_quality": round(self.forecast_quality, 4),
                "entry_quality": round(self.entry_quality, 4),
                "risk": round(self.risk_score, 4),
                "liquidity": round(self.liquidity_score, 4),
                "spread": round(self.spread_score, 4),
                "volatility": round(self.volatility_score, 4),
                "reward": round(self.reward_score, 4),
                "portfolio_impact": round(self.portfolio_impact, 4),
            },
            "notes": self.notes,
        }


class TradeScorer:
    """Weights and combines sub-scores into a final execution grade.

    Weights:
      forecast_quality  20%
      entry_quality     20%
      risk              15%
      liquidity         10%
      spread            10%
      volatility        10%
      reward            10%
      portfolio_impact   5%
    """

    _WEIGHTS = {
        "forecast_quality": 0.20,
        "entry_quality": 0.20,
        "risk": 0.15,
        "liquidity": 0.10,
        "spread": 0.10,
        "volatility": 0.10,
        "reward": 0.10,
        "portfolio_impact": 0.05,
    }
    # remaining 0% is the portfolio_impact weight above (total=1.0)

    def score(
        self,
        forecast: ExecutionForecast,
        entry_quality: Optional[EntryQualityResult] = None,
        risk_profile: Optional[RiskProfile] = None,
        timing: Optional[TimingAssessment] = None,
        portfolio_impact_score: float = 0.70,
    ) -> TradeScore:

        notes: List[str] = []

        # 1. Forecast quality: confidence + expected return scaled
        forecast_q = _clamp(
            forecast.confidence * 0.70 + _clamp(forecast.expected_return / 0.20) * 0.30
        )
        if forecast.confidence > 0.80:
            notes.append("High-confidence forecast")

        # 2. Entry quality
        if entry_quality:
            entry_q = entry_quality.score
        else:
            entry_q = _clamp(
                forecast.confidence * 0.5 + (1 - forecast.spread / max(forecast.atr, 1e-9)) * 0.5
            )

        # 3. Risk score: positive EV + good RR
        if risk_profile:
            rr_score = _clamp(risk_profile.risk_reward_ratio / 3.0)
            ev_score = 1.0 if risk_profile.is_positive_ev() else 0.3
            risk_s = _clamp(rr_score * 0.6 + ev_score * 0.4)
            if not risk_profile.is_positive_ev():
                notes.append("Negative expected value — review trade logic")
        else:
            rr = forecast.risk_reward_raw
            risk_s = _clamp(rr / 3.0)

        # 4. Liquidity score (from timing)
        if timing:
            liq_factor = next((f for f in timing.factors if f.name == "liquidity"), None)
            liq_s = liq_factor.score if liq_factor else 0.65
        else:
            liq_s = 0.65

        # 5. Spread score
        spread_ratio = forecast.spread / max(forecast.atr, 1e-9)
        spread_s = _clamp(1.0 - spread_ratio * 4)
        if spread_ratio > 0.25:
            notes.append(f"Wide spread ({spread_ratio:.0%} of ATR) — reduces edge")

        # 6. Volatility score (prefer moderate)
        vol = forecast.volatility
        if 0.005 <= vol <= 0.020:
            vol_s = 0.90
        elif vol < 0.003:
            vol_s = 0.35
        elif vol < 0.030:
            vol_s = _clamp(1.0 - (vol - 0.020) * 10)
        else:
            vol_s = 0.20
            notes.append("Extreme volatility — execution risk elevated")

        # 7. Reward score: expected return / risk ratio
        reward_s = _clamp(forecast.expected_return / max(forecast.expected_risk, 1e-9) / 3.0)

        # 8. Portfolio impact (external input or default)
        port_s = _clamp(portfolio_impact_score)

        # Composite
        overall = (
            self._WEIGHTS["forecast_quality"] * forecast_q
            + self._WEIGHTS["entry_quality"] * entry_q
            + self._WEIGHTS["risk"] * risk_s
            + self._WEIGHTS["liquidity"] * liq_s
            + self._WEIGHTS["spread"] * spread_s
            + self._WEIGHTS["volatility"] * vol_s
            + self._WEIGHTS["reward"] * reward_s
            + self._WEIGHTS["portfolio_impact"] * port_s
        )
        overall = _clamp(overall)
        grade = _grade_from_score(overall)

        if grade in (ExecutionGrade.EXCELLENT, ExecutionGrade.GOOD):
            notes.append(f"Trade meets institutional execution standards ({grade.value})")

        logger.debug(
            f"TradeScorer {forecast.instrument}: overall={overall:.3f} grade={grade.value}"
        )

        return TradeScore(
            instrument=forecast.instrument,
            forecast_quality=round(forecast_q, 4),
            entry_quality=round(entry_q, 4),
            risk_score=round(risk_s, 4),
            liquidity_score=round(liq_s, 4),
            spread_score=round(spread_s, 4),
            volatility_score=round(vol_s, 4),
            reward_score=round(reward_s, 4),
            portfolio_impact=round(port_s, 4),
            overall_score=round(overall, 4),
            grade=grade,
            notes=notes,
        )
