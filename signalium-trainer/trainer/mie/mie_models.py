"""Phase 24.9 — MIE data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trainer.mie.mie_types import (
    MarketRegime, RiskRegime, TradeabilityGrade, ExecutionMode,
    SessionType, NewsImpact, MomentumState, TrendState,
    VolatilityState, LiquidityState, WorkerTrigger,
)


# ── Primary Input ──────────────────────────────────────────────────────────────

@dataclass
class MarketInput:
    """Canonical market input flowing through all MIE engines."""
    symbol:                str
    timestamp:             float           = field(default_factory=time.time)
    # OHLCV
    open:                  float           = 0.0
    high:                  float           = 0.0
    low:                   float           = 0.0
    close:                 float           = 1.0
    volume:                float           = 1_000.0
    # Microstructure
    bid:                   float           = 0.0
    ask:                   float           = 0.0
    spread:                float           = 0.0001
    # Technical indicators (pre-computed)
    atr:                   float           = 0.001
    adx:                   float           = 25.0
    rsi:                   float           = 50.0
    # Historical series (most-recent-last)
    closes:                List[float]     = field(default_factory=list)
    highs:                 List[float]     = field(default_factory=list)
    lows:                  List[float]     = field(default_factory=list)
    volumes:               List[float]     = field(default_factory=list)
    returns:               List[float]     = field(default_factory=list)
    # Prediction context
    prediction_confidence: float           = 0.5
    expected_rr:           float           = 1.5
    expected_move:         float           = 0.001
    expected_holding_bars: int             = 10
    # Portfolio context
    portfolio_exposure:    float           = 0.0
    correlation_avg:       float           = 0.0
    capital:               float           = 100_000.0
    open_trades:           int             = 0
    # News & events
    news_events:           List[Dict]      = field(default_factory=list)
    session_hour:          int             = 12   # UTC hour 0-23
    # Historical similarity context
    historical_similar:    List[Dict]      = field(default_factory=list)
    # Calibration / model quality
    calibration_score:     float           = 0.5
    historical_win_rate:   float           = 0.5
    model_health:          float           = 0.8

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()
                if not isinstance(v, list)}


# ── Engine Results ─────────────────────────────────────────────────────────────

@dataclass
class MarketRegimeResult:
    regime:      MarketRegime    = MarketRegime.UNKNOWN
    confidence:  float           = 0.5
    sub_regime:  Optional[str]   = None
    signals:     Dict[str, Any]  = field(default_factory=dict)
    timestamp:   float           = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "regime":     self.regime.value,
            "confidence": round(self.confidence, 4),
            "sub_regime": self.sub_regime,
            "signals":    self.signals,
        }


@dataclass
class RiskIntelligenceResult:
    regime:      RiskRegime    = RiskRegime.NORMAL
    risk_score:  float         = 50.0   # 0=safe, 100=extreme risk
    multiplier:  float         = 1.0
    factors:     Dict[str, float] = field(default_factory=dict)
    warnings:    List[str]     = field(default_factory=list)
    timestamp:   float         = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "regime":     self.regime.value,
            "risk_score": round(self.risk_score, 2),
            "multiplier": round(self.multiplier, 4),
            "factors":    {k: round(v, 4) for k, v in self.factors.items()},
            "warnings":   self.warnings,
        }


@dataclass
class VolatilityResult:
    state:           VolatilityState = VolatilityState.NORMAL
    quality:         float           = 50.0   # 0-100 (higher = better for trading)
    current_vol:     float           = 0.0
    baseline_vol:    float           = 0.0
    vol_ratio:       float           = 1.0
    atr_percentile:  float           = 50.0
    timestamp:       float           = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "state":          self.state.value,
            "quality":        round(self.quality, 2),
            "current_vol":    round(self.current_vol, 6),
            "baseline_vol":   round(self.baseline_vol, 6),
            "vol_ratio":      round(self.vol_ratio, 4),
            "atr_percentile": round(self.atr_percentile, 2),
        }


@dataclass
class MomentumResult:
    state:       MomentumState  = MomentumState.FLAT
    quality:     float          = 50.0
    strength:    float          = 0.0    # signed: positive=bullish
    acceleration: float         = 0.0
    rsi_signal:  float          = 50.0
    timestamp:   float          = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "state":        self.state.value,
            "quality":      round(self.quality, 2),
            "strength":     round(self.strength, 4),
            "acceleration": round(self.acceleration, 4),
            "rsi_signal":   round(self.rsi_signal, 2),
        }


@dataclass
class TrendResult:
    state:       TrendState   = TrendState.SIDEWAYS
    quality:     float        = 50.0
    direction:   str          = "NEUTRAL"   # UP | DOWN | NEUTRAL
    strength:    float        = 0.0         # 0-100
    adx_value:   float        = 0.0
    timestamp:   float        = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "state":     self.state.value,
            "quality":   round(self.quality, 2),
            "direction": self.direction,
            "strength":  round(self.strength, 2),
            "adx_value": round(self.adx_value, 2),
        }


@dataclass
class LiquidityResult:
    state:          LiquidityState = LiquidityState.GOOD
    quality:        float          = 50.0
    volume_score:   float          = 50.0
    spread_score:   float          = 50.0
    depth_estimate: float          = 50.0
    timestamp:      float          = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "state":          self.state.value,
            "quality":        round(self.quality, 2),
            "volume_score":   round(self.volume_score, 2),
            "spread_score":   round(self.spread_score, 2),
            "depth_estimate": round(self.depth_estimate, 2),
        }


@dataclass
class SessionResult:
    session:       SessionType = SessionType.OFF_HOURS
    quality:       float       = 25.0
    is_overlap:    bool        = False
    is_high_volume: bool       = False
    timestamp:     float       = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "session":        self.session.value,
            "quality":        round(self.quality, 2),
            "is_overlap":     self.is_overlap,
            "is_high_volume": self.is_high_volume,
        }


@dataclass
class NewsIntelligenceResult:
    impact:        NewsImpact  = NewsImpact.NONE
    impact_score:  float       = 0.0    # 0-100 (higher = more impact)
    event_count:   int         = 0
    upcoming_high: int         = 0
    recent_high:   int         = 0
    details:       List[str]   = field(default_factory=list)
    timestamp:     float       = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "impact":        self.impact.value,
            "impact_score":  round(self.impact_score, 2),
            "event_count":   self.event_count,
            "upcoming_high": self.upcoming_high,
            "recent_high":   self.recent_high,
        }


@dataclass
class BehaviorSimilarityResult:
    similarity_score:   float      = 0.0    # 0-100
    historical_success: float      = 0.5    # 0-1 historical win rate on similar
    match_count:        int        = 0
    best_match_score:   float      = 0.0
    features_used:      List[str]  = field(default_factory=list)
    timestamp:          float      = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "similarity_score":   round(self.similarity_score, 2),
            "historical_success": round(self.historical_success, 4),
            "match_count":        self.match_count,
            "best_match_score":   round(self.best_match_score, 2),
            "features_used":      self.features_used,
        }


@dataclass
class PredictionReliabilityResult:
    reliability_score: float     = 50.0   # 0-100
    confidence_score:  float     = 50.0
    consistency_score: float     = 50.0
    model_health:      float     = 80.0
    win_rate_adj:      float     = 50.0
    reasons:           List[str] = field(default_factory=list)
    timestamp:         float     = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "reliability_score": round(self.reliability_score, 2),
            "confidence_score":  round(self.confidence_score, 2),
            "consistency_score": round(self.consistency_score, 2),
            "model_health":      round(self.model_health, 2),
            "win_rate_adj":      round(self.win_rate_adj, 2),
            "reasons":           self.reasons,
        }


@dataclass
class CalibrationResult:
    calibration_score: float     = 50.0   # 0-100 (100=perfectly calibrated)
    brier_score:       float     = 0.25   # lower is better
    ece:               float     = 0.10   # Expected Calibration Error
    overconfidence:    float     = 0.0    # positive = overconfident
    reliability:       float     = 0.5
    timestamp:         float     = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "calibration_score": round(self.calibration_score, 2),
            "brier_score":       round(self.brier_score, 4),
            "ece":               round(self.ece, 4),
            "overconfidence":    round(self.overconfidence, 4),
            "reliability":       round(self.reliability, 4),
        }


@dataclass
class TradeabilityResult:
    score:       float               = 0.0
    grade:       TradeabilityGrade   = TradeabilityGrade.AVOID
    factors:     Dict[str, float]    = field(default_factory=dict)
    reasons:     List[str]           = field(default_factory=list)
    suggestions: List[str]           = field(default_factory=list)
    timestamp:   float               = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "score":       round(self.score, 2),
            "grade":       self.grade.value,
            "factors":     {k: round(v, 4) for k, v in self.factors.items()},
            "reasons":     self.reasons,
            "suggestions": self.suggestions,
        }


@dataclass
class ExecutionResult:
    mode:              ExecutionMode = ExecutionMode.WAIT
    entry_zone_low:    float         = 0.0
    entry_zone_high:   float         = 0.0
    exit_zone_low:     float         = 0.0
    exit_zone_high:    float         = 0.0
    max_holding_bars:  int           = 20
    expected_drawdown: float         = 0.05
    optimal_stop:      float         = 0.0
    optimal_target:    float         = 0.0
    rationale:         str           = ""
    timestamp:         float         = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "mode":              self.mode.value,
            "entry_zone_low":    round(self.entry_zone_low, 6),
            "entry_zone_high":   round(self.entry_zone_high, 6),
            "exit_zone_low":     round(self.exit_zone_low, 6),
            "exit_zone_high":    round(self.exit_zone_high, 6),
            "max_holding_bars":  self.max_holding_bars,
            "expected_drawdown": round(self.expected_drawdown, 4),
            "optimal_stop":      round(self.optimal_stop, 6),
            "optimal_target":    round(self.optimal_target, 6),
            "rationale":         self.rationale,
        }


@dataclass
class MarketQualityResult:
    market_quality:       float = 50.0
    execution_quality:    float = 50.0
    prediction_quality:   float = 50.0
    liquidity_quality:    float = 50.0
    volatility_quality:   float = 50.0
    momentum_quality:     float = 50.0
    trend_quality:        float = 50.0
    session_quality:      float = 50.0
    news_impact_score:    float = 0.0
    timestamp:            float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "market_quality":     round(self.market_quality, 2),
            "execution_quality":  round(self.execution_quality, 2),
            "prediction_quality": round(self.prediction_quality, 2),
            "liquidity_quality":  round(self.liquidity_quality, 2),
            "volatility_quality": round(self.volatility_quality, 2),
            "momentum_quality":   round(self.momentum_quality, 2),
            "trend_quality":      round(self.trend_quality, 2),
            "session_quality":    round(self.session_quality, 2),
            "news_impact_score":  round(self.news_impact_score, 2),
        }


@dataclass
class MIEPortfolioRisk:
    risk_budget_used:    float     = 0.0    # 0-1
    max_exposure:        float     = 0.25   # fraction of capital
    correlation_limit:   float     = 0.7
    diversification:     float     = 0.5    # 0-1
    max_open_trades:     int       = 5
    position_size_frac:  float     = 0.02
    capital_allocation:  float     = 0.0
    warnings:            List[str] = field(default_factory=list)
    timestamp:           float     = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "risk_budget_used":   round(self.risk_budget_used, 4),
            "max_exposure":       round(self.max_exposure, 4),
            "correlation_limit":  round(self.correlation_limit, 4),
            "diversification":    round(self.diversification, 4),
            "max_open_trades":    self.max_open_trades,
            "position_size_frac": round(self.position_size_frac, 4),
            "capital_allocation": round(self.capital_allocation, 2),
            "warnings":           self.warnings,
        }


@dataclass
class AICommentaryResult:
    summary:        str       = ""
    regime_note:    str       = ""
    risk_note:      str       = ""
    trade_note:     str       = ""
    recommendation: str       = ""
    confidence:     float     = 0.5
    tags:           List[str] = field(default_factory=list)
    timestamp:      float     = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "summary":        self.summary,
            "regime_note":    self.regime_note,
            "risk_note":      self.risk_note,
            "trade_note":     self.trade_note,
            "recommendation": self.recommendation,
            "confidence":     round(self.confidence, 4),
            "tags":           self.tags,
        }


@dataclass
class FeatureStoreExport:
    """12 engineered features for the Deep Learning model."""
    symbol:               str   = ""
    timestamp:            float = field(default_factory=time.time)
    market_regime_enc:    float = 0.0    # encoded MarketRegime ordinal 0-1
    risk_regime_enc:      float = 0.0    # encoded RiskRegime ordinal 0-1
    tradeability:         float = 0.0    # 0-100
    execution_quality:    float = 0.0    # 0-100
    market_quality:       float = 0.0    # 0-100
    prediction_reliability: float = 0.0 # 0-100
    behavior_similarity:  float = 0.0   # 0-100
    calibration:          float = 0.0   # 0-100
    historical_success:   float = 0.0   # 0-1
    trend_quality:        float = 0.0   # 0-100
    momentum_quality:     float = 0.0   # 0-100
    liquidity_quality:    float = 0.0   # 0-100

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    def to_feature_vector(self) -> List[float]:
        return [
            self.market_regime_enc / 15.0,
            self.risk_regime_enc / 11.0,
            self.tradeability / 100.0,
            self.execution_quality / 100.0,
            self.market_quality / 100.0,
            self.prediction_reliability / 100.0,
            self.behavior_similarity / 100.0,
            self.calibration / 100.0,
            self.historical_success,
            self.trend_quality / 100.0,
            self.momentum_quality / 100.0,
            self.liquidity_quality / 100.0,
        ]


@dataclass
class MarketIntelligenceReport:
    """Unified output of a full MIE run."""
    report_id:            str                    = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol:               str                    = ""
    timestamp:            float                  = field(default_factory=time.time)
    regime:               Optional[MarketRegimeResult]        = None
    risk:                 Optional[RiskIntelligenceResult]    = None
    volatility:           Optional[VolatilityResult]          = None
    momentum:             Optional[MomentumResult]            = None
    trend:                Optional[TrendResult]               = None
    liquidity:            Optional[LiquidityResult]           = None
    session:              Optional[SessionResult]             = None
    news:                 Optional[NewsIntelligenceResult]    = None
    similarity:           Optional[BehaviorSimilarityResult]  = None
    reliability:          Optional[PredictionReliabilityResult] = None
    calibration:          Optional[CalibrationResult]         = None
    tradeability:         Optional[TradeabilityResult]        = None
    execution:            Optional[ExecutionResult]           = None
    quality:              Optional[MarketQualityResult]       = None
    portfolio_risk:       Optional[MIEPortfolioRisk]          = None
    commentary:           Optional[AICommentaryResult]        = None
    features:             Optional[FeatureStoreExport]        = None

    def to_dict(self) -> Dict:
        out: Dict = {
            "report_id": self.report_id,
            "symbol":    self.symbol,
            "timestamp": self.timestamp,
        }
        for attr in [
            "regime", "risk", "volatility", "momentum", "trend",
            "liquidity", "session", "news", "similarity", "reliability",
            "calibration", "tradeability", "execution", "quality",
            "portfolio_risk", "commentary",
        ]:
            val = getattr(self, attr)
            if val is not None:
                out[attr] = val.to_dict()
        return out


@dataclass
class WorkerJob:
    job_id:   str          = field(default_factory=lambda: str(uuid.uuid4())[:12])
    trigger:  WorkerTrigger = WorkerTrigger.MANUAL_REFRESH
    symbol:   str          = ""
    payload:  Dict         = field(default_factory=dict)
    created:  float        = field(default_factory=time.time)
    status:   str          = "PENDING"   # PENDING | RUNNING | DONE | FAILED
    result:   Optional[MarketIntelligenceReport] = None
