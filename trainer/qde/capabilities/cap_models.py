"""Phase 28.3 — QDE capability shared data models.

Defines every input and output type used across the 11 capability modules
and the QDE intelligence brain.

DECISION BOUNDARY
-----------------
QDEDecision is the ONLY type that encodes an actionable trade instruction.
It may ONLY be instantiated inside qde_intelligence.py.
No capability module may import or construct a QDEDecision.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# ── Enumerations ──────────────────────────────────────────────────────────────


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class RegimeType(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    CRISIS = "CRISIS"
    RECOVERY = "RECOVERY"
    UNKNOWN = "UNKNOWN"


class RegimeTransition(str, Enum):
    STABLE = "STABLE"
    ACCELERATING = "ACCELERATING"
    DECELERATING = "DECELERATING"
    REVERSING = "REVERSING"


class DecisionType(str, Enum):
    """The type of decision the QDE has produced."""

    ENTER = "ENTER"
    EXIT = "EXIT"
    SCALE = "SCALE"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    BREAK_EVEN = "BREAK_EVEN"
    HOLD = "HOLD"
    PRESERVE = "PRESERVE"  # capital preservation circuit fired — no new entries


class TrendStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


class LiquidityLevel(str, Enum):
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"
    ILLIQUID = "ILLIQUID"


class CircuitBreakerStatus(str, Enum):
    CLEAR = "CLEAR"
    WARNING = "WARNING"
    TRIGGERED = "TRIGGERED"


# ── Market Snapshot (shared input to all capabilities) ────────────────────────


@dataclass
class OHLCV:
    """Single bar of market data."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float  # Unix seconds


@dataclass
class OpenTrade:
    """A currently open trade position."""

    trade_id: str
    instrument: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float | None
    lot_size: float
    opened_at: float
    current_price: float
    unrealised_pnl: float
    swap: float = 0.0
    commission: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return time.time() - self.opened_at

    @property
    def risk_at_open(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.lot_size

    @property
    def r_multiple(self) -> float:
        if self.risk_at_open == 0:
            return 0.0
        return self.unrealised_pnl / self.risk_at_open


@dataclass
class AccountState:
    """Live account state at the moment of reasoning."""

    balance: float
    equity: float
    free_margin: float
    margin_used: float
    drawdown_from_peak: float  # 0.0 – 1.0
    daily_pnl: float
    daily_pnl_pct: float
    session_pnl: float
    consecutive_losses: int
    consecutive_wins: int
    open_trade_count: int


@dataclass
class MarketSnapshot:
    """Complete market snapshot — the sole input to the QDE reasoning cycle.

    All 11 capabilities receive this object. It is immutable during a
    reasoning cycle; capabilities must not modify it.
    """

    # Identity
    instrument: str
    snapshot_id: str = field(default_factory=lambda: f"snap-{uuid.uuid4().hex[:10]}")
    timestamp: float = field(default_factory=time.time)

    # Price and bars
    bid: float = 0.0
    ask: float = 0.0
    spread_pips: float = 0.0
    recent_bars: list[OHLCV] = field(default_factory=list)  # newest-last

    # Computed market metrics
    atr_14: float = 0.0  # Average True Range (14-period)
    atr_50: float = 0.0
    rsi_14: float = 50.0
    adx_14: float = 0.0  # Trend strength
    bb_width: float = 0.0  # Bollinger Band width (normalised)
    ema_fast: float = 0.0  # e.g. 20-period
    ema_slow: float = 0.0  # e.g. 50-period
    volume_ratio: float = 1.0  # current / average volume

    # Market session
    session: str = ""  # LONDON | NEW_YORK | TOKYO | SYDNEY | OVERLAP
    day_of_week: int = 0  # 0=Mon … 4=Fri

    # Liquidity
    liquidity: LiquidityLevel = LiquidityLevel.NORMAL

    # Account
    account: AccountState | None = None

    # Open trades for this instrument
    open_trades: list[OpenTrade] = field(default_factory=list)

    # All open trades across all instruments (for exposure reasoning)
    all_open_trades: list[OpenTrade] = field(default_factory=list)

    # Arbitrary feature dict for model-computed signals
    features: dict[str, float] = field(default_factory=dict)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def latest_bar(self) -> OHLCV | None:
        return self.recent_bars[-1] if self.recent_bars else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "instrument": self.instrument,
            "timestamp": round(self.timestamp, 3),
            "bid": self.bid,
            "ask": self.ask,
            "spread_pips": round(self.spread_pips, 1),
            "atr_14": round(self.atr_14, 6),
            "rsi_14": round(self.rsi_14, 2),
            "adx_14": round(self.adx_14, 2),
            "session": self.session,
            "liquidity": self.liquidity.value,
            "open_trade_count": len(self.open_trades),
        }


# ── Capability Output Base ────────────────────────────────────────────────────


@dataclass
class CapabilityOutput:
    """Base class for all capability outputs.

    All outputs are advisory — they describe market state or provide scored
    candidates. They do NOT instruct trades or produce decisions.
    """

    capability: str
    snapshot_id: str
    produced_at: float = field(default_factory=time.time)
    confidence: float = 0.5  # 0.0 – 1.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "snapshot_id": self.snapshot_id,
            "produced_at": round(self.produced_at, 3),
            "confidence": round(self.confidence, 4),
            "warnings": self.warnings,
        }


# ── Market Understanding Output ───────────────────────────────────────────────


@dataclass
class MarketContext(CapabilityOutput):
    """Output of the Market Understanding capability.

    Synthesises raw market data into a structured context object.
    Does not advise trades.
    """

    regime: RegimeType = RegimeType.UNKNOWN
    regime_transition: RegimeTransition = RegimeTransition.STABLE
    regime_confidence: float = 0.5
    trend_direction: Direction = Direction.FLAT
    trend_strength: TrendStrength = TrendStrength.NONE
    volatility_state: str = "NORMAL"  # LOW | NORMAL | HIGH | EXTREME
    volatility_percentile: float = 50.0  # 0-100 vs. recent history
    spread_state: str = "NORMAL"  # NORMAL | WIDE | VERY_WIDE
    session_quality: float = 0.5  # 0-1; higher = better for trading
    structural_level_above: float | None = None
    structural_level_below: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "regime": self.regime.value,
                "regime_transition": self.regime_transition.value,
                "regime_confidence": round(self.regime_confidence, 4),
                "trend_direction": self.trend_direction.value,
                "trend_strength": self.trend_strength.value,
                "volatility_state": self.volatility_state,
                "volatility_percentile": round(self.volatility_percentile, 2),
                "spread_state": self.spread_state,
                "session_quality": round(self.session_quality, 4),
                "structural_level_above": (
                    round(self.structural_level_above, 6) if self.structural_level_above else None
                ),
                "structural_level_below": (
                    round(self.structural_level_below, 6) if self.structural_level_below else None
                ),
                "notes": self.notes,
            }
        )
        return d


# ── Opportunity Discovery Output ──────────────────────────────────────────────


@dataclass
class SetupCandidate:
    """A scored setup candidate. Not a trade decision."""

    setup_id: str = field(default_factory=lambda: f"setup-{uuid.uuid4().hex[:8]}")
    direction: Direction = Direction.FLAT
    setup_type: str = ""  # e.g. TREND_CONTINUATION | REVERSAL | BREAKOUT
    confluence_score: float = 0.0  # 0-1; how many signals align
    signal_count: int = 0
    signals: list[str] = field(default_factory=list)
    ideal_entry_zone: tuple[float, float] = (0.0, 0.0)
    initial_stop_estimate: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "direction": self.direction.value,
            "setup_type": self.setup_type,
            "confluence_score": round(self.confluence_score, 4),
            "signal_count": self.signal_count,
            "signals": self.signals,
            "ideal_entry_zone": [
                round(self.ideal_entry_zone[0], 6),
                round(self.ideal_entry_zone[1], 6),
            ],
            "initial_stop_estimate": round(self.initial_stop_estimate, 6),
            "notes": self.notes,
        }


@dataclass
class OpportunityReport(CapabilityOutput):
    """Output of Opportunity Discovery.  Advisory — no trade instructions."""

    candidates: list[SetupCandidate] = field(default_factory=list)
    best_candidate: SetupCandidate | None = None
    opportunity_quality: float = 0.0  # 0-1
    market_is_tradeable: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "opportunity_quality": round(self.opportunity_quality, 4),
                "market_is_tradeable": self.market_is_tradeable,
                "candidate_count": len(self.candidates),
                "candidates": [c.to_dict() for c in self.candidates],
                "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            }
        )
        return d


# ── Expected Value Reasoning Output ──────────────────────────────────────────


@dataclass
class EVScore(CapabilityOutput):
    """Output of Expected Value Reasoning.  Advisory — no trade instructions."""

    ev: float = 0.0  # Expected value in account currency
    ev_per_unit: float = 0.0  # EV per pip or per lot
    win_probability: float = 0.0  # 0-1
    loss_probability: float = 0.0  # 0-1 = 1 - win_probability
    expected_gain: float = 0.0  # if win
    expected_loss: float = 0.0  # if lose (positive magnitude)
    risk_reward_ratio: float = 0.0  # reward / risk
    ev_is_positive: bool = False
    ev_threshold_met: bool = False  # True if EV exceeds minimum threshold
    setup_candidate_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "ev": round(self.ev, 6),
                "ev_per_unit": round(self.ev_per_unit, 6),
                "win_probability": round(self.win_probability, 4),
                "loss_probability": round(self.loss_probability, 4),
                "expected_gain": round(self.expected_gain, 6),
                "expected_loss": round(self.expected_loss, 6),
                "risk_reward_ratio": round(self.risk_reward_ratio, 4),
                "ev_is_positive": self.ev_is_positive,
                "ev_threshold_met": self.ev_threshold_met,
                "setup_candidate_id": self.setup_candidate_id,
            }
        )
        return d


# ── Risk Reasoning Output ─────────────────────────────────────────────────────


@dataclass
class RiskAssessment(CapabilityOutput):
    """Output of Risk Reasoning.  Advisory — no trade instructions."""

    stop_loss_price: float = 0.0
    take_profit_price: float | None = None
    stop_distance_pips: float = 0.0
    tp_distance_pips: float | None = None
    atr_stop_multiple: float = 1.5  # stop = entry ± N × ATR
    risk_per_trade_pct: float = 0.01  # fraction of equity at risk
    risk_environment: str = "NORMAL"  # LOW | NORMAL | HIGH | EXTREME
    volatility_adjusted: bool = True
    structural_stop_valid: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "stop_loss_price": round(self.stop_loss_price, 6),
                "take_profit_price": (
                    round(self.take_profit_price, 6) if self.take_profit_price else None
                ),
                "stop_distance_pips": round(self.stop_distance_pips, 2),
                "tp_distance_pips": (
                    round(self.tp_distance_pips, 2) if self.tp_distance_pips else None
                ),
                "atr_stop_multiple": round(self.atr_stop_multiple, 3),
                "risk_per_trade_pct": round(self.risk_per_trade_pct, 4),
                "risk_environment": self.risk_environment,
                "volatility_adjusted": self.volatility_adjusted,
                "structural_stop_valid": self.structural_stop_valid,
                "notes": self.notes,
            }
        )
        return d


# ── Exposure Reasoning Output ─────────────────────────────────────────────────


@dataclass
class ExposureReport(CapabilityOutput):
    """Output of Exposure Reasoning.  Advisory — no trade instructions."""

    gross_exposure_pct: float = 0.0  # total open risk / equity
    net_directional_exposure: float = 0.0  # net long − short exposure
    instrument_exposures: dict[str, float] = field(default_factory=dict)
    correlation_risk: float = 0.0  # 0-1; how correlated open positions are
    concentration_risk: float = 0.0  # 0-1; single-instrument concentration
    exposure_headroom_pct: float = 1.0  # fraction of max exposure still available
    adding_exposure_is_safe: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "gross_exposure_pct": round(self.gross_exposure_pct, 4),
                "net_directional_exposure": round(self.net_directional_exposure, 4),
                "instrument_exposures": {
                    k: round(v, 4) for k, v in self.instrument_exposures.items()
                },
                "correlation_risk": round(self.correlation_risk, 4),
                "concentration_risk": round(self.concentration_risk, 4),
                "exposure_headroom_pct": round(self.exposure_headroom_pct, 4),
                "adding_exposure_is_safe": self.adding_exposure_is_safe,
                "notes": self.notes,
            }
        )
        return d


# ── Trade Management Output ───────────────────────────────────────────────────


@dataclass
class TradeManagementSignals(CapabilityOutput):
    """Output of Trade Management.  Advisory — no trade instructions."""

    trades_needing_exit: list[str] = field(default_factory=list)  # trade_ids
    trades_needing_break_even: list[str] = field(default_factory=list)
    trades_needing_partial_close: list[str] = field(default_factory=list)
    trades_eligible_for_scaling: list[str] = field(default_factory=list)
    trades_exceeding_duration: list[str] = field(default_factory=list)
    reversal_detected: bool = False
    continuation_detected: bool = False
    reversal_confidence: float = 0.0
    continuation_confidence: float = 0.0
    per_trade_notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "trades_needing_exit": self.trades_needing_exit,
                "trades_needing_break_even": self.trades_needing_break_even,
                "trades_needing_partial_close": self.trades_needing_partial_close,
                "trades_eligible_for_scaling": self.trades_eligible_for_scaling,
                "trades_exceeding_duration": self.trades_exceeding_duration,
                "reversal_detected": self.reversal_detected,
                "continuation_detected": self.continuation_detected,
                "reversal_confidence": round(self.reversal_confidence, 4),
                "continuation_confidence": round(self.continuation_confidence, 4),
                "per_trade_notes": self.per_trade_notes,
            }
        )
        return d


# ── Position Management Output ────────────────────────────────────────────────


@dataclass
class PositionSizeRecommendation(CapabilityOutput):
    """Output of Position Management.  Advisory — no trade instructions."""

    lot_size: float = 0.0  # Recommended lot size
    risk_amount: float = 0.0  # In account currency
    risk_pct: float = 0.0  # Fraction of equity
    max_lot_size: float = 0.0  # Hard cap
    volatility_adjustment_factor: float = 1.0
    kelly_fraction: float = 0.0  # Raw Kelly fraction
    kelly_adjusted: float = 0.0  # Reduced Kelly (e.g. half-Kelly)
    sizing_method: str = "FIXED_RISK"  # FIXED_RISK | KELLY | ATR_SCALED
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "lot_size": round(self.lot_size, 4),
                "risk_amount": round(self.risk_amount, 2),
                "risk_pct": round(self.risk_pct, 4),
                "max_lot_size": round(self.max_lot_size, 4),
                "volatility_adjustment_factor": round(self.volatility_adjustment_factor, 4),
                "kelly_fraction": round(self.kelly_fraction, 4),
                "kelly_adjusted": round(self.kelly_adjusted, 4),
                "sizing_method": self.sizing_method,
                "notes": self.notes,
            }
        )
        return d


# ── Capital Preservation Output ───────────────────────────────────────────────


@dataclass
class PreservationStatus(CapabilityOutput):
    """Output of Capital Preservation.  Advisory — no trade instructions."""

    circuit_breaker: CircuitBreakerStatus = CircuitBreakerStatus.CLEAR
    daily_loss_limit_pct: float = 0.03  # configured limit
    daily_loss_current_pct: float = 0.0
    daily_loss_headroom_pct: float = 0.03
    drawdown_limit_pct: float = 0.10
    drawdown_current_pct: float = 0.0
    drawdown_headroom_pct: float = 0.10
    consecutive_loss_limit: int = 5
    consecutive_losses_current: int = 0
    new_entries_allowed: bool = True
    scaling_allowed: bool = True
    forced_exit_required: bool = False
    forced_exit_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "circuit_breaker": self.circuit_breaker.value,
                "daily_loss_limit_pct": round(self.daily_loss_limit_pct, 4),
                "daily_loss_current_pct": round(self.daily_loss_current_pct, 4),
                "daily_loss_headroom_pct": round(self.daily_loss_headroom_pct, 4),
                "drawdown_limit_pct": round(self.drawdown_limit_pct, 4),
                "drawdown_current_pct": round(self.drawdown_current_pct, 4),
                "drawdown_headroom_pct": round(self.drawdown_headroom_pct, 4),
                "consecutive_loss_limit": self.consecutive_loss_limit,
                "consecutive_losses_current": self.consecutive_losses_current,
                "new_entries_allowed": self.new_entries_allowed,
                "scaling_allowed": self.scaling_allowed,
                "forced_exit_required": self.forced_exit_required,
                "forced_exit_reason": self.forced_exit_reason,
            }
        )
        return d


# ── Market Memory Output ──────────────────────────────────────────────────────


@dataclass
class MemoryPattern:
    """A historically matched pattern."""

    pattern_id: str
    similarity_score: float  # 0-1
    pattern_type: str
    historical_outcome: str  # WIN | LOSS | NEUTRAL
    historical_pnl_r: float  # R-multiple of historical outcome
    occurred_at: float
    regime_at_occurrence: str


@dataclass
class MemoryContext(CapabilityOutput):
    """Output of Market Memory.  Advisory — no trade instructions."""

    matched_patterns: list[MemoryPattern] = field(default_factory=list)
    pattern_match_quality: float = 0.0  # 0-1
    historical_win_rate_this_setup: float | None = None
    historical_ev_this_setup: float | None = None
    regime_has_been_seen_before: bool = False
    regime_historical_performance: dict[str, float] = field(default_factory=dict)
    anomaly_detected: bool = False
    anomaly_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "pattern_count": len(self.matched_patterns),
                "pattern_match_quality": round(self.pattern_match_quality, 4),
                "historical_win_rate_this_setup": (
                    round(self.historical_win_rate_this_setup, 4)
                    if self.historical_win_rate_this_setup is not None
                    else None
                ),
                "historical_ev_this_setup": (
                    round(self.historical_ev_this_setup, 6)
                    if self.historical_ev_this_setup is not None
                    else None
                ),
                "regime_has_been_seen_before": self.regime_has_been_seen_before,
                "regime_historical_performance": {
                    k: round(v, 6) for k, v in self.regime_historical_performance.items()
                },
                "anomaly_detected": self.anomaly_detected,
                "anomaly_description": self.anomaly_description,
            }
        )
        return d


# ── Reflection Output ─────────────────────────────────────────────────────────


@dataclass
class TradeOutcome:
    """A completed trade outcome provided to the Reflection capability."""

    trade_id: str
    instrument: str
    direction: Direction
    entry_price: float
    exit_price: float
    stop_loss: float
    lot_size: float
    pnl: float
    pnl_r: float  # In R-multiples
    opened_at: float
    closed_at: float
    exit_reason: str
    setup_type: str
    regime_at_entry: str
    capability_scores_at_entry: dict[str, float] = field(default_factory=dict)


@dataclass
class ReflectionInsight(CapabilityOutput):
    """Output of the Reflection capability.  Advisory — no trade instructions."""

    trade_id: str = ""
    reasoning_was_correct: bool = False
    capability_accuracy: dict[str, float] = field(default_factory=dict)  # capability → accuracy 0-1
    setup_quality_actual: float = 0.0  # what confluence score should have been
    ev_prediction_error: float = 0.0  # predicted EV − actual EV
    stop_placement_quality: float = 0.5  # 0-1
    regime_reading_quality: float = 0.5
    key_lesson: str = ""
    suggested_weight_adjustments: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "trade_id": self.trade_id,
                "reasoning_was_correct": self.reasoning_was_correct,
                "capability_accuracy": {
                    k: round(v, 4) for k, v in self.capability_accuracy.items()
                },
                "setup_quality_actual": round(self.setup_quality_actual, 4),
                "ev_prediction_error": round(self.ev_prediction_error, 6),
                "stop_placement_quality": round(self.stop_placement_quality, 4),
                "regime_reading_quality": round(self.regime_reading_quality, 4),
                "key_lesson": self.key_lesson,
                "suggested_weight_adjustments": {
                    k: round(v, 6) for k, v in self.suggested_weight_adjustments.items()
                },
            }
        )
        return d


# ── Learning Output ───────────────────────────────────────────────────────────


@dataclass
class LearningUpdate(CapabilityOutput):
    """Output of the Learning capability.  Advisory — no trade instructions."""

    updated_weights: dict[str, float] = field(default_factory=dict)
    prior_updates: dict[str, Any] = field(default_factory=dict)
    capability_trust_scores: dict[str, float] = field(default_factory=dict)  # 0-1 per capability
    regime_performance_priors: dict[str, dict[str, float]] = field(default_factory=dict)
    learning_rate_used: float = 0.01
    n_samples_used: int = 0
    convergence_delta: float = 0.0  # |new_weights − old_weights|
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "updated_weights": {k: round(v, 6) for k, v in self.updated_weights.items()},
                "capability_trust_scores": {
                    k: round(v, 4) for k, v in self.capability_trust_scores.items()
                },
                "regime_performance_priors": {
                    r: {k: round(v, 6) for k, v in m.items()}
                    for r, m in self.regime_performance_priors.items()
                },
                "learning_rate_used": self.learning_rate_used,
                "n_samples_used": self.n_samples_used,
                "convergence_delta": round(self.convergence_delta, 8),
                "notes": self.notes,
            }
        )
        return d


# ── QDE Decision (ONLY produced by QDEIntelligence) ──────────────────────────


@dataclass
class EntrySignal:
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float | None
    lot_size: float
    setup_type: str
    confluence_score: float
    ev_score: float
    risk_pct: float


@dataclass
class ExitSignal:
    trade_id: str
    exit_price: float
    reason: str
    urgency: str = "NORMAL"  # NORMAL | URGENT | FORCED


@dataclass
class ScalingSignal:
    trade_id: str
    additional_lot_size: float
    new_stop_loss: float
    reason: str


@dataclass
class PartialCloseSignal:
    trade_id: str
    close_fraction: float  # 0-1; fraction of current position to close
    reason: str


@dataclass
class BreakEvenSignal:
    trade_id: str
    new_stop_loss: float
    reason: str


@dataclass
class QDEDecision:
    """The sole output of the QDE reasoning cycle.

    THIS IS THE ONLY TYPE THAT ENCODES ACTIONABLE TRADE INSTRUCTIONS.
    It may only be constructed inside qde_intelligence.py.

    All 13 QDE responsibilities are encoded here — none live anywhere else.
    """

    decision_id: str
    snapshot_id: str
    instrument: str
    decision_type: DecisionType
    decided_at: str

    # 13 Responsibilities — all populated by QDEIntelligence
    # Entry / Exit
    entry: EntrySignal | None = None
    exits: list[ExitSignal] = field(default_factory=list)

    # Position lifecycle
    scaling: list[ScalingSignal] = field(default_factory=list)
    partial_closes: list[PartialCloseSignal] = field(default_factory=list)
    break_evens: list[BreakEvenSignal] = field(default_factory=list)

    # Market reading (informational fields — no instruction)
    regime: RegimeType = RegimeType.UNKNOWN
    regime_transition: RegimeTransition = RegimeTransition.STABLE
    trend_direction: Direction = Direction.FLAT
    trend_strength: TrendStrength = TrendStrength.NONE
    reversal_detected: bool = False
    continuation_detected: bool = False

    # Capital protection
    circuit_breaker: CircuitBreakerStatus = CircuitBreakerStatus.CLEAR
    preservation_active: bool = False

    # Reasoning trace — capability scores feeding this decision
    capability_contributions: dict[str, float] = field(default_factory=dict)
    reasoning_summary: str = ""
    overall_confidence: float = 0.0

    @classmethod
    def hold(
        cls,
        snapshot: MarketSnapshot,
        reasoning: str,
        capability_contributions: dict[str, float] | None = None,
        regime: RegimeType = RegimeType.UNKNOWN,
    ) -> QDEDecision:
        return cls(
            decision_id=f"dec-{uuid.uuid4().hex[:10]}",
            snapshot_id=snapshot.snapshot_id,
            instrument=snapshot.instrument,
            decision_type=DecisionType.HOLD,
            decided_at=datetime.now(UTC).isoformat(),
            regime=regime,
            reasoning_summary=reasoning,
            capability_contributions=capability_contributions or {},
            overall_confidence=0.0,
        )

    @classmethod
    def preserve(
        cls,
        snapshot: MarketSnapshot,
        reason: str,
        circuit_breaker: CircuitBreakerStatus = CircuitBreakerStatus.TRIGGERED,
        forced_exits: list[ExitSignal] | None = None,
    ) -> QDEDecision:
        return cls(
            decision_id=f"dec-{uuid.uuid4().hex[:10]}",
            snapshot_id=snapshot.snapshot_id,
            instrument=snapshot.instrument,
            decision_type=DecisionType.PRESERVE,
            decided_at=datetime.now(UTC).isoformat(),
            circuit_breaker=circuit_breaker,
            preservation_active=True,
            exits=forced_exits or [],
            reasoning_summary=reason,
            overall_confidence=1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "snapshot_id": self.snapshot_id,
            "instrument": self.instrument,
            "decision_type": self.decision_type.value,
            "decided_at": self.decided_at,
            "entry": (
                {
                    "direction": self.entry.direction.value,
                    "entry_price": round(self.entry.entry_price, 6),
                    "stop_loss": round(self.entry.stop_loss, 6),
                    "take_profit": (
                        round(self.entry.take_profit, 6) if self.entry.take_profit else None
                    ),
                    "lot_size": round(self.entry.lot_size, 4),
                    "setup_type": self.entry.setup_type,
                    "confluence_score": round(self.entry.confluence_score, 4),
                    "ev_score": round(self.entry.ev_score, 6),
                    "risk_pct": round(self.entry.risk_pct, 4),
                }
                if self.entry
                else None
            ),
            "exits": [
                {
                    "trade_id": e.trade_id,
                    "exit_price": round(e.exit_price, 6),
                    "reason": e.reason,
                    "urgency": e.urgency,
                }
                for e in self.exits
            ],
            "scaling": [
                {
                    "trade_id": s.trade_id,
                    "additional_lot_size": round(s.additional_lot_size, 4),
                    "new_stop_loss": round(s.new_stop_loss, 6),
                    "reason": s.reason,
                }
                for s in self.scaling
            ],
            "partial_closes": [
                {
                    "trade_id": p.trade_id,
                    "close_fraction": round(p.close_fraction, 4),
                    "reason": p.reason,
                }
                for p in self.partial_closes
            ],
            "break_evens": [
                {
                    "trade_id": b.trade_id,
                    "new_stop_loss": round(b.new_stop_loss, 6),
                    "reason": b.reason,
                }
                for b in self.break_evens
            ],
            "regime": self.regime.value,
            "regime_transition": self.regime_transition.value,
            "trend_direction": self.trend_direction.value,
            "trend_strength": self.trend_strength.value,
            "reversal_detected": self.reversal_detected,
            "continuation_detected": self.continuation_detected,
            "circuit_breaker": self.circuit_breaker.value,
            "preservation_active": self.preservation_active,
            "capability_contributions": {
                k: round(v, 4) for k, v in self.capability_contributions.items()
            },
            "reasoning_summary": self.reasoning_summary,
            "overall_confidence": round(self.overall_confidence, 4),
        }
