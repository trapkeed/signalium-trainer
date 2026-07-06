"""Phase 22 — Microstructure data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trainer.microstructure.micro_types import (
    EMBEDDING_TOTAL,
    DataGranularity,
    LiquidityState,
    MarketState,
    MomentumState,
    TrendStrength,
    VolatilityState,
)

# ── OHLCBar ────────────────────────────────────────────────────────────────────


@dataclass
class OHLCBar:
    """Single OHLC price bar."""

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timeframe: str = "H1"
    instrument: str = ""

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "open": round(self.open, 5),
            "high": round(self.high, 5),
            "low": round(self.low, 5),
            "close": round(self.close, 5),
            "volume": round(self.volume, 2),
            "timeframe": self.timeframe,
            "instrument": self.instrument,
        }


# ── MicrostructureFeatures ────────────────────────────────────────────────────


@dataclass
class MicrostructureFeatures:
    """Complete microstructure feature set derived from OHLC data."""

    bar_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)
    instrument: str = ""
    timeframe: str = "H1"
    granularity: str = DataGranularity.OHLC.value

    # ── Price action ─────────────────────────────────────────────────────────
    body_ratio: float = 0.0  # 0–1
    upper_wick_ratio: float = 0.0
    lower_wick_ratio: float = 0.0
    close_position: float = 0.5  # 0=at low, 1=at high
    open_position: float = 0.5
    candle_efficiency: float = 0.0
    directional_efficiency: float = 0.0  # -1 to +1
    compression: float = 0.0  # 0–1 (how compressed the bar is)
    expansion: float = 0.0  # 0–1 (how expanded)
    impulse_strength: float = 0.0  # 0–1
    retracement_quality: float = 0.0  # 0–1 (quality of pullback)
    continuation_probability: float = 0.5  # 0–1

    # ── Volatility ───────────────────────────────────────────────────────────
    atr_expansion: float = 0.0
    atr_compression: float = 0.0
    volatility_clustering: float = 0.0
    volatility_persistence: float = 0.0
    volatility_shock: float = 0.0  # 0 or 1
    noise_ratio: float = 0.0
    range_compression: float = 0.0
    range_expansion: float = 0.0

    # ── Momentum ─────────────────────────────────────────────────────────────
    momentum_acceleration: float = 0.0
    momentum_deceleration: float = 0.0
    momentum_exhaustion: float = 0.0
    directional_persistence: float = 0.0  # fraction of same-direction bars
    trend_energy: float = 0.0
    impulse_decay: float = 0.0

    # ── Liquidity proxies ─────────────────────────────────────────────────────
    spread_proxy: float = 0.0
    gap_frequency: float = 0.0
    large_candle_frequency: float = 0.0
    compression_zone_score: float = 0.0
    reaction_zone_score: float = 0.0
    breakout_strength: float = 0.0
    reversal_strength: float = 0.0
    liquidity_vacuum_prob: float = 0.0

    # ── Market efficiency ─────────────────────────────────────────────────────
    trend_efficiency: float = 0.0
    mean_reversion_probability: float = 0.5
    randomness_index: float = 0.5
    fractal_dimension: float = 1.5  # 1=trend, 2=random walk
    directional_stability: float = 0.0
    noise_level: float = 0.5

    # ── Tick features (optional, always None if not available) ────────────────
    tick_velocity: Optional[float] = None
    tick_density: Optional[float] = None
    tick_imbalance: Optional[float] = None
    tick_momentum: Optional[float] = None
    tick_volatility: Optional[float] = None
    tick_acceleration: Optional[float] = None
    tick_entropy: Optional[float] = None

    @property
    def has_tick_features(self) -> bool:
        return self.tick_velocity is not None

    def to_dict(self) -> Dict:
        d = {
            "bar_id": self.bar_id,
            "timestamp": round(self.timestamp, 3),
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "granularity": self.granularity,
            # Price action
            "body_ratio": round(self.body_ratio, 4),
            "upper_wick_ratio": round(self.upper_wick_ratio, 4),
            "lower_wick_ratio": round(self.lower_wick_ratio, 4),
            "close_position": round(self.close_position, 4),
            "open_position": round(self.open_position, 4),
            "candle_efficiency": round(self.candle_efficiency, 4),
            "directional_efficiency": round(self.directional_efficiency, 4),
            "compression": round(self.compression, 4),
            "expansion": round(self.expansion, 4),
            "impulse_strength": round(self.impulse_strength, 4),
            "retracement_quality": round(self.retracement_quality, 4),
            "continuation_probability": round(self.continuation_probability, 4),
            # Volatility
            "atr_expansion": round(self.atr_expansion, 4),
            "atr_compression": round(self.atr_compression, 4),
            "volatility_clustering": round(self.volatility_clustering, 4),
            "volatility_persistence": round(self.volatility_persistence, 4),
            "volatility_shock": round(self.volatility_shock, 4),
            "noise_ratio": round(self.noise_ratio, 4),
            "range_compression": round(self.range_compression, 4),
            "range_expansion": round(self.range_expansion, 4),
            # Momentum
            "momentum_acceleration": round(self.momentum_acceleration, 4),
            "momentum_deceleration": round(self.momentum_deceleration, 4),
            "momentum_exhaustion": round(self.momentum_exhaustion, 4),
            "directional_persistence": round(self.directional_persistence, 4),
            "trend_energy": round(self.trend_energy, 4),
            "impulse_decay": round(self.impulse_decay, 4),
            # Liquidity
            "spread_proxy": round(self.spread_proxy, 4),
            "gap_frequency": round(self.gap_frequency, 4),
            "large_candle_frequency": round(self.large_candle_frequency, 4),
            "compression_zone_score": round(self.compression_zone_score, 4),
            "reaction_zone_score": round(self.reaction_zone_score, 4),
            "breakout_strength": round(self.breakout_strength, 4),
            "reversal_strength": round(self.reversal_strength, 4),
            "liquidity_vacuum_prob": round(self.liquidity_vacuum_prob, 4),
            # Efficiency
            "trend_efficiency": round(self.trend_efficiency, 4),
            "mean_reversion_probability": round(self.mean_reversion_probability, 4),
            "randomness_index": round(self.randomness_index, 4),
            "fractal_dimension": round(self.fractal_dimension, 4),
            "directional_stability": round(self.directional_stability, 4),
            "noise_level": round(self.noise_level, 4),
        }
        if self.has_tick_features:
            d.update(
                {
                    "tick_velocity": self.tick_velocity,
                    "tick_density": self.tick_density,
                    "tick_imbalance": self.tick_imbalance,
                    "tick_momentum": self.tick_momentum,
                    "tick_volatility": self.tick_volatility,
                    "tick_acceleration": self.tick_acceleration,
                    "tick_entropy": self.tick_entropy,
                }
            )
        return d

    def price_action_vector(self) -> List[float]:
        return [
            self.body_ratio,
            self.upper_wick_ratio,
            self.lower_wick_ratio,
            self.close_position,
            self.open_position,
            self.candle_efficiency,
            self.impulse_strength,
            self.continuation_probability,
        ]

    def liquidity_vector(self) -> List[float]:
        return [
            self.spread_proxy,
            self.gap_frequency,
            self.breakout_strength,
            self.reversal_strength,
            self.liquidity_vacuum_prob,
            self.large_candle_frequency,
        ]

    def momentum_vector(self) -> List[float]:
        return [
            self.trend_energy,
            self.momentum_acceleration,
            self.directional_persistence,
            self.impulse_decay,
            self.momentum_exhaustion,
            self.directional_efficiency,
        ]

    def volatility_vector(self) -> List[float]:
        return [
            self.atr_expansion,
            self.atr_compression,
            self.volatility_clustering,
            self.volatility_persistence,
            self.noise_ratio,
            self.volatility_shock,
        ]

    def efficiency_vector(self) -> List[float]:
        return [
            self.trend_efficiency,
            self.mean_reversion_probability,
            self.randomness_index,
            self.directional_stability,
            self.noise_level,
            1.0 - self.fractal_dimension / 2.0,
        ]


# ── MicrostructureScore ───────────────────────────────────────────────────────


@dataclass
class MicrostructureScore:
    """Aggregate scores across microstructure dimensions."""

    trend_score: float = 0.0
    liquidity_score: float = 0.0
    momentum_score: float = 0.0
    efficiency_score: float = 0.0
    volatility_score: float = 0.0
    compression_score: float = 0.0
    expansion_score: float = 0.0
    overall_score: float = 0.0

    def to_dict(self) -> Dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


# ── MarketStateRecord ─────────────────────────────────────────────────────────


@dataclass
class MarketStateRecord:
    state: MarketState
    confidence: float
    momentum_state: MomentumState
    volatility_state: VolatilityState
    liquidity_state: LiquidityState
    trend_strength: TrendStrength
    score: MicrostructureScore
    timestamp: float = field(default_factory=time.time)
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "momentum_state": self.momentum_state.value,
            "volatility_state": self.volatility_state.value,
            "liquidity_state": self.liquidity_state.value,
            "trend_strength": self.trend_strength.value,
            "score": self.score.to_dict(),
            "timestamp": round(self.timestamp, 3),
            "reasoning": self.reasoning,
        }


# ── MicrostructureEmbedding ───────────────────────────────────────────────────


@dataclass
class MicrostructureEmbedding:
    """Dense embedding vector (32 dims) for microstructure state."""

    timestamp: float
    instrument: str = ""
    timeframe: str = "H1"
    embedding: List[float] = field(default_factory=list)
    price_action_state: List[float] = field(default_factory=list)  # 8 dims
    liquidity_state: List[float] = field(default_factory=list)  # 6 dims
    momentum_state: List[float] = field(default_factory=list)  # 6 dims
    volatility_state: List[float] = field(default_factory=list)  # 6 dims
    efficiency_state: List[float] = field(default_factory=list)  # 6 dims

    @property
    def dim(self) -> int:
        return len(self.embedding)

    def to_dict(self) -> Dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "embedding": [round(v, 5) for v in self.embedding],
            "dim": self.dim,
            "price_action_state": [round(v, 5) for v in self.price_action_state],
            "liquidity_state": [round(v, 5) for v in self.liquidity_state],
            "momentum_state": [round(v, 5) for v in self.momentum_state],
            "volatility_state": [round(v, 5) for v in self.volatility_state],
            "efficiency_state": [round(v, 5) for v in self.efficiency_state],
        }
