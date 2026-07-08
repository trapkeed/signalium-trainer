"""Phase S9 — Hierarchical Timeframes: feature contracts and model I/O.

`TimeframeBar` is the raw OHLCV candle schema for one timeframe.

`TimeframeFeatureVector` is the per-timeframe *feature contract* — the
normalized, descriptive quantities computed from recent bars on one
timeframe. These are deliberately observational (returns, volatility,
where price sits in its recent range, momentum, session) rather than
interpretive (no "trend=BULLISH" label, no "structure=PULLBACK" label).
Interpretation is left entirely to the learned attention architecture; a
feature contract that already encoded "this is a pullback" would be a
hard-coded rule wearing a data-schema costume.

`MultiTimeframeSnapshot` composes six `TimeframeFeatureVector`s (one per
`Timeframe`) aligned to a single M1 timestamp — this is the object the
synchronization pipeline produces and the temporal attention architecture
consumes.

`TemporalAssessment` is the model's output: whether cross-timeframe
alignment matters right now, whether a countertrend or continuation
opportunity exists, and the relative probability of reversal vs.
continuation — plus the attention weights the model placed on each
timeframe, kept for interpretability/telemetry, not as an input to any
further rule.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trainer.qde.qde_timeframe_types import (
    ALL_TIMEFRAMES,
    QDE_TIMEFRAME_SCHEMA_VERSION,
    Timeframe,
)


# ── TimeframeBar (raw OHLCV) ──────────────────────────────────────────────────


@dataclass
class TimeframeBar:
    """One closed OHLCV candle for a single timeframe."""

    instrument: str
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    open_time: float = 0.0
    close_time: float = 0.0  # authoritative "as-of" timestamp for sync/no-lookahead

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "timeframe": self.timeframe.value,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "open_time": round(self.open_time, 3),
            "close_time": round(self.close_time, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TimeframeBar:
        return cls(
            instrument=data["instrument"],
            timeframe=Timeframe(data["timeframe"]),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data.get("volume", 0.0),
            open_time=data.get("open_time", 0.0),
            close_time=data.get("close_time", 0.0),
        )


# ── TimeframeFeatureVector (per-timeframe feature contract) ──────────────────


@dataclass
class TimeframeFeatureVector:
    """Normalized, descriptive features for one timeframe as-of one bar.

    Every field is a plain observation of price/volume behavior on this
    timeframe alone; nothing here is computed *relative to* another
    timeframe (that composition happens only inside the attention model).
    """

    instrument: str
    timeframe: Timeframe
    as_of: float = 0.0  # close_time of the bar this vector describes

    # ── Returns / momentum ───────────────────────────────────────────────
    log_return_1: float = 0.0  # log(close / prev_close) for this bar
    momentum_n: float = 0.0  # log(close / close_n_bars_ago), normalized by realized_vol
    realized_vol: float = 0.0  # stdev of log returns over the lookback window

    # ── Structural context (position within recent range, not a label) ───
    range_position: float = 0.5  # (close - low_n) / (high_n - low_n), 0-1
    distance_to_swing_high: float = 0.0  # (swing_high - close) / close
    distance_to_swing_low: float = 0.0  # (close - swing_low) / close

    # ── Candle shape ───────────────────────────────────────────────────────
    body_ratio: float = 0.0  # |close-open| / (high-low), 0-1
    upper_wick_ratio: float = 0.0
    lower_wick_ratio: float = 0.0

    # ── Volume / session context ────────────────────────────────────────────
    volume_zscore: float = 0.0  # vs rolling average volume on this timeframe
    session: str = "UNKNOWN"

    bars_available: int = 0  # how many historical bars fed this vector (0 = cold start)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "timeframe": self.timeframe.value,
            "as_of": round(self.as_of, 3),
            "log_return_1": round(self.log_return_1, 6),
            "momentum_n": round(self.momentum_n, 6),
            "realized_vol": round(self.realized_vol, 6),
            "range_position": round(self.range_position, 4),
            "distance_to_swing_high": round(self.distance_to_swing_high, 6),
            "distance_to_swing_low": round(self.distance_to_swing_low, 6),
            "body_ratio": round(self.body_ratio, 4),
            "upper_wick_ratio": round(self.upper_wick_ratio, 4),
            "lower_wick_ratio": round(self.lower_wick_ratio, 4),
            "volume_zscore": round(self.volume_zscore, 4),
            "session": self.session,
            "bars_available": self.bars_available,
        }

    def as_vector(self) -> list[float]:
        """Fixed-order numeric feature vector, for feeding the attention model."""
        return [
            self.log_return_1,
            self.momentum_n,
            self.realized_vol,
            self.range_position,
            self.distance_to_swing_high,
            self.distance_to_swing_low,
            self.body_ratio,
            self.upper_wick_ratio,
            self.lower_wick_ratio,
            self.volume_zscore,
        ]


TIMEFRAME_FEATURE_DIM = len(
    TimeframeFeatureVector(instrument="_", timeframe=Timeframe.M1).as_vector()
)


# ── MultiTimeframeSnapshot (the feature contract handed to the model) ────────


@dataclass
class MultiTimeframeSnapshot:
    """Six `TimeframeFeatureVector`s aligned to one M1 timestamp.

    This is produced by `qde_timeframe_sync.TimeframeSynchronizer` and is
    the single object the temporal attention architecture consumes — the
    contract boundary between "data plumbing" and "learned reasoning".
    """

    instrument: str
    as_of: float = field(default_factory=time.time)
    features: dict[Timeframe, TimeframeFeatureVector] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return all(tf in self.features for tf in ALL_TIMEFRAMES)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "as_of": round(self.as_of, 3),
            "features": {tf.value: v.to_dict() for tf, v in self.features.items()},
            "complete": self.is_complete(),
        }

    def as_matrix(self) -> list[list[float]]:
        """Ordered (coarsest->finest) list of feature vectors, missing
        timeframes filled with zeros — the raw shape the attention encoder
        expects: `[num_timeframes][feature_dim]`.
        """
        zero = [0.0] * TIMEFRAME_FEATURE_DIM
        return [
            self.features[tf].as_vector() if tf in self.features else zero
            for tf in ALL_TIMEFRAMES
        ]


# ── TemporalAssessment (model output) ─────────────────────────────────────────


@dataclass
class TemporalAssessment:
    """The model's hierarchical-timeframe judgement for one M1 tick.

    All four scores are learned outputs (see `qde_temporal_attention.py`)
    — this dataclass is just their typed container plus interpretability
    metadata (`attention_weights`). Nothing here re-derives or overrides
    these scores with a fixed rule.
    """

    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    as_of: float = field(default_factory=time.time)

    alignment_score: float = 0.5  # 0-1: how much cross-timeframe alignment matters right now
    countertrend_opportunity_score: float = 0.0  # 0-1
    continuation_opportunity_score: float = 0.0  # 0-1
    reversal_probability: float = 0.5  # reversal_probability + continuation_probability == 1
    continuation_probability: float = 0.5

    attention_weights: dict[str, float] = field(default_factory=dict)  # timeframe -> weight, sums to 1

    schema_version: str = QDE_TIMEFRAME_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "assessment_id": self.assessment_id,
            "instrument": self.instrument,
            "as_of": round(self.as_of, 3),
            "alignment_score": round(self.alignment_score, 4),
            "countertrend_opportunity_score": round(self.countertrend_opportunity_score, 4),
            "continuation_opportunity_score": round(self.continuation_opportunity_score, 4),
            "reversal_probability": round(self.reversal_probability, 4),
            "continuation_probability": round(self.continuation_probability, 4),
            "attention_weights": {k: round(v, 4) for k, v in self.attention_weights.items()},
            "schema_version": self.schema_version,
        }
