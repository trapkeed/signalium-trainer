"""QDE Brain — projection: data models and rolling memory.

Stores ``(conditions → outcome)`` observation pairs that accumulate as
trades complete, forming the empirical distribution from which expected
move/duration/volatility/reward/MAE are later estimated.

Key types
---------
``ProjectionConditions``
    The market-conditions fingerprint captured at opportunity time.
    Nine numeric features that characterise *how the market looks* when
    this kind of setup arises — not a label, not a regime name, just
    continuous measurements that the similarity function can compare.

``ProjectionOutcome``
    What actually happened after entry: realized return, max favorable
    excursion, max adverse excursion, duration, and realized vol — all
    expressed in ATR multiples so they are instrument- and
    timeframe-agnostic.

    The bridge constructor ``from_outcome_path`` accepts a
    ``TradeOutcomePath`` (trainer/training_signals/outcome_path.py) and
    the ATR at entry, so the outcome labeling pipeline can feed directly
    into the learning loop without conversion on the caller side.

``ProjectionRecord``
    One completed (conditions, outcome) pair plus metadata (direction,
    timestamp, record_id).

``ProjectionMemory``
    Per-symbol rolling store: bounded FIFO of records with optional
    regime-bucket filtering, ``to_snapshot`` / ``load_snapshot``
    persistence, and no business logic of its own.

Design rules
------------
* Does not import ``QDEDecision`` or any execution-layer type.
* No hardcoded pip / ATR targets — all magnitudes are relative to the
  ATR at the time of the trade.
* Does not call ``TradeOutcomePath`` internally; callers build the
  outcome and pass it in.
"""

from __future__ import annotations

import math
import statistics
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Soft import — only for type hints so the liquidity package boundary
    # is respected.  Callers construct ``ProjectionOutcome`` themselves via
    # ``from_outcome_path`` or directly.
    from trainer.training_signals.outcome_path import TradeOutcomePath

# ── Numeric feature dimension ─────────────────────────────────────────────────

_N_FEATURES = 9   # length of ProjectionConditions.as_vector()

# Rolling-store limit (per symbol).  Old records are evicted FIFO once
# this limit is exceeded so memory is bounded.
_DEFAULT_MAX_RECORDS = 1000

# Recency decay time constant in days.  Records older than ~2 weeks
# contribute exponentially less to weighted estimates.
DEFAULT_RECENCY_TAU_DAYS = 14.0


# ── ProjectionConditions ──────────────────────────────────────────────────────


@dataclass
class ProjectionConditions:
    """Nine-dimensional market fingerprint at opportunity time.

    All components are instrument-agnostic continuous values; the vector
    is intended for cosine-similarity comparison, so absolute scale
    matters less than relative pattern across components.

    Components
    ----------
    atr_pct               — ATR / price (relative volatility measure)
    spread_atr_ratio      — spread / ATR (execution friction relative to range)
    volatility            — MarketContext.volatility (raw rolling vol)
    momentum              — MarketContext.momentum (signed −1..+1)
    vol_state_rank_norm   — ordinal vol-state rank (0..4) normalised to 0..1
    volatility_quality    — 0-1 (how tradeable is the current vol environment)
    microstructure_quality — 0-1 (order-flow / spread cleanliness)
    session_quality        — 0-1 (session time quality)
    trend_valid_f          — 1.0 if trend_valid else 0.0
    """

    instrument:            str   = ""
    atr_pct:               float = 0.0
    spread_atr_ratio:      float = 0.0
    volatility:            float = 0.0
    momentum:              float = 0.0
    vol_state_rank_norm:   float = 0.25   # NORMAL = rank 1/4
    volatility_quality:    float = 0.5
    microstructure_quality: float = 0.5
    session_quality:       float = 0.5
    trend_valid_f:         float = 1.0
    # Non-feature context retained for filtering / display
    regime:                str   = ""
    timestamp:             float = field(default_factory=time.time)

    def as_vector(self) -> list[float]:
        """Return the 9-dimensional numeric feature vector."""
        return [
            self.atr_pct,
            self.spread_atr_ratio,
            self.volatility,
            self.momentum,
            self.vol_state_rank_norm,
            self.volatility_quality,
            self.microstructure_quality,
            self.session_quality,
            self.trend_valid_f,
        ]

    @classmethod
    def from_observation(
        cls,
        instrument: str,
        inp: object,          # SymbolObservationInput — duck-typed
        now: float | None = None,
    ) -> ProjectionConditions:
        """Construct from a live ``SymbolObservationInput`` tick."""
        mkt   = inp.market
        price = float(getattr(mkt, "price", 0.0)) or 1.0
        atr   = float(getattr(mkt, "atr",   0.0))
        spread = float(getattr(mkt, "spread", 0.0))

        vol_state = str(getattr(inp, "volatility_state", "NORMAL")).upper()
        _RANK = {"LOW": 0, "QUIET": 0, "NORMAL": 1, "MODERATE": 1,
                 "ELEVATED": 2, "HIGH": 3, "VOLATILE": 3, "EXTREME": 4}
        rank_norm = _RANK.get(vol_state, 1) / 4.0

        return cls(
            instrument=instrument,
            atr_pct=atr / max(price, 1e-9),
            spread_atr_ratio=spread / max(atr, 1e-9),
            volatility=float(getattr(mkt, "volatility", 0.0)),
            momentum=float(getattr(mkt, "momentum", 0.0)),
            vol_state_rank_norm=rank_norm,
            volatility_quality=float(getattr(inp, "volatility_quality", 0.5)),
            microstructure_quality=float(getattr(inp, "microstructure_quality", 0.5)),
            session_quality=float(getattr(inp, "session_quality", 0.5)),
            trend_valid_f=1.0 if getattr(mkt, "trend_valid", True) else 0.0,
            regime=str(getattr(inp, "market_regime", "")),
            timestamp=now if now is not None else time.time(),
        )

    def to_dict(self) -> dict:
        return {
            "instrument":             self.instrument,
            "atr_pct":                round(self.atr_pct, 8),
            "spread_atr_ratio":       round(self.spread_atr_ratio, 6),
            "volatility":             round(self.volatility, 8),
            "momentum":               round(self.momentum, 4),
            "vol_state_rank_norm":    round(self.vol_state_rank_norm, 4),
            "volatility_quality":     round(self.volatility_quality, 4),
            "microstructure_quality": round(self.microstructure_quality, 4),
            "session_quality":        round(self.session_quality, 4),
            "trend_valid_f":          round(self.trend_valid_f, 1),
            "regime":                 self.regime,
            "timestamp":              round(self.timestamp, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProjectionConditions:
        return cls(
            instrument=d.get("instrument", ""),
            atr_pct=d.get("atr_pct", 0.0),
            spread_atr_ratio=d.get("spread_atr_ratio", 0.0),
            volatility=d.get("volatility", 0.0),
            momentum=d.get("momentum", 0.0),
            vol_state_rank_norm=d.get("vol_state_rank_norm", 0.25),
            volatility_quality=d.get("volatility_quality", 0.5),
            microstructure_quality=d.get("microstructure_quality", 0.5),
            session_quality=d.get("session_quality", 0.5),
            trend_valid_f=d.get("trend_valid_f", 1.0),
            regime=d.get("regime", ""),
            timestamp=d.get("timestamp", 0.0),
        )


# ── ProjectionOutcome ─────────────────────────────────────────────────────────


@dataclass
class ProjectionOutcome:
    """What actually happened after entry — all magnitudes in ATR multiples
    so they are instrument- and timeframe-agnostic.

    ``reward_atr`` is signed: positive means the trade moved in the
    intended direction; negative means it moved against it.

    ``mae_atr`` and ``mfe_atr`` are always non-negative (they measure
    excursion magnitude, not direction).
    """

    realized_move_pct:   float = 0.0   # |realized_return| as fraction of price
    realized_return_pct: float = 0.0   # signed fraction (positive = trade won)
    reward_atr:          float = 0.0   # signed: realized move in ATR multiples
    mae_atr:             float = 0.0   # max adverse excursion in ATR multiples
    mfe_atr:             float = 0.0   # max favorable excursion in ATR multiples
    duration_seconds:    float = 0.0   # trade holding time
    realized_vol:        float = 0.0   # rolling vol during trade (if available)

    @classmethod
    def from_outcome_path(
        cls,
        path: object,           # TradeOutcomePath — duck-typed
        atr_at_entry: float,
        bar_duration_seconds: float = 3600.0,
    ) -> ProjectionOutcome:
        """Build from a realized ``TradeOutcomePath``.

        ``path.direction`` (+1 / −1) is used to sign the reward.
        ``atr_at_entry`` normalises all excursion values.
        ``bar_duration_seconds`` converts bar count → calendar time.
        """
        atr = max(atr_at_entry, 1e-9)
        entry_price = float(getattr(path, "entry_price", 0.0)) or 1.0

        ret_pct: float = path.realized_return_pct()
        mfe_raw, _    = path.max_favorable_excursion()
        mae_raw: float = path.max_adverse_excursion()
        n_bars: int    = len(getattr(path, "future_bars", []))

        # Realized vol from bar returns if enough bars
        bars = getattr(path, "future_bars", [])
        closes = [getattr(b, "close", entry_price) for b in bars]
        if len(closes) >= 3:
            returns = [(closes[i] - closes[i - 1]) / max(closes[i - 1], 1e-9)
                       for i in range(1, len(closes))]
            realized_vol = statistics.stdev(returns) if len(returns) >= 2 else 0.0
        else:
            realized_vol = 0.0

        return cls(
            realized_move_pct=abs(ret_pct),
            realized_return_pct=ret_pct,
            reward_atr=(ret_pct * entry_price) / atr,
            mae_atr=mae_raw / atr,
            mfe_atr=mfe_raw / atr,
            duration_seconds=n_bars * bar_duration_seconds,
            realized_vol=realized_vol,
        )

    def to_dict(self) -> dict:
        return {
            "realized_move_pct":   round(self.realized_move_pct, 6),
            "realized_return_pct": round(self.realized_return_pct, 6),
            "reward_atr":          round(self.reward_atr, 4),
            "mae_atr":             round(self.mae_atr, 4),
            "mfe_atr":             round(self.mfe_atr, 4),
            "duration_seconds":    round(self.duration_seconds, 1),
            "realized_vol":        round(self.realized_vol, 8),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProjectionOutcome:
        return cls(
            realized_move_pct=d.get("realized_move_pct", 0.0),
            realized_return_pct=d.get("realized_return_pct", 0.0),
            reward_atr=d.get("reward_atr", 0.0),
            mae_atr=d.get("mae_atr", 0.0),
            mfe_atr=d.get("mfe_atr", 0.0),
            duration_seconds=d.get("duration_seconds", 0.0),
            realized_vol=d.get("realized_vol", 0.0),
        )


# ── ProjectionRecord ──────────────────────────────────────────────────────────


@dataclass
class ProjectionRecord:
    """One completed (conditions → outcome) observation.

    ``direction`` mirrors the ``TradeOutcomePath`` / ``BeliefDirection``
    convention: +1 = long, −1 = short.
    """

    record_id:   str                = field(
        default_factory=lambda: str(uuid.uuid4())[:12]
    )
    instrument:  str                = ""
    direction:   int                = 1       # +1 or -1
    conditions:  ProjectionConditions = field(default_factory=ProjectionConditions)
    outcome:     ProjectionOutcome  = field(default_factory=ProjectionOutcome)
    recorded_at: float              = field(default_factory=time.time)

    def age_days(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, (now - self.recorded_at) / 86400.0)

    def recency_weight(
        self,
        now: float | None = None,
        tau_days: float = DEFAULT_RECENCY_TAU_DAYS,
    ) -> float:
        return math.exp(-self.age_days(now) / max(tau_days, 1e-3))

    def to_dict(self) -> dict:
        return {
            "record_id":   self.record_id,
            "instrument":  self.instrument,
            "direction":   self.direction,
            "conditions":  self.conditions.to_dict(),
            "outcome":     self.outcome.to_dict(),
            "recorded_at": round(self.recorded_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProjectionRecord:
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())[:12]),
            instrument=d.get("instrument", ""),
            direction=d.get("direction", 1),
            conditions=ProjectionConditions.from_dict(d.get("conditions", {})),
            outcome=ProjectionOutcome.from_dict(d.get("outcome", {})),
            recorded_at=d.get("recorded_at", 0.0),
        )


# ── ProjectionMemory ──────────────────────────────────────────────────────────


class ProjectionMemory:
    """Per-symbol rolling store of ``ProjectionRecord`` observations.

    Records are appended and the oldest are evicted once ``max_records``
    is reached.  ``records()`` returns a live copy; callers must not
    mutate the returned list.

    No business logic lives here — all estimation happens in the engine
    layer.
    """

    def __init__(self, max_records: int = _DEFAULT_MAX_RECORDS) -> None:
        self._max = max_records
        # instrument → deque[ProjectionRecord]
        self._store: dict[str, deque[ProjectionRecord]] = {}

    # ── Write ──────────────────────────────────────────────────────────────

    def record(
        self,
        instrument: str,
        direction:  int,
        conditions: ProjectionConditions,
        outcome:    ProjectionOutcome,
        now:        float | None = None,
    ) -> ProjectionRecord:
        """Append one observation and return the new record."""
        now = now if now is not None else time.time()
        rec = ProjectionRecord(
            instrument=instrument,
            direction=direction,
            conditions=conditions,
            outcome=outcome,
            recorded_at=now,
        )
        q = self._store.setdefault(instrument, deque(maxlen=self._max))
        q.append(rec)
        return rec

    # ── Read ───────────────────────────────────────────────────────────────

    def records(
        self,
        instrument: str,
        regime:     str | None = None,
        direction:  int | None = None,
    ) -> list[ProjectionRecord]:
        """Return matching records, oldest → newest.

        If ``regime`` is given, only records whose ``conditions.regime``
        matches are returned — useful for regime-conditional estimation.
        If ``direction`` is given (+1 or −1), only same-direction records
        are returned.
        """
        all_recs = list(self._store.get(instrument, []))
        if regime is not None:
            all_recs = [r for r in all_recs if r.conditions.regime == regime]
        if direction is not None:
            all_recs = [r for r in all_recs if r.direction == direction]
        return all_recs

    def record_count(self, instrument: str) -> int:
        return len(self._store.get(instrument, []))

    # ── Persistence ─────────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        return {"records": [r.to_dict() for r in self.records(instrument)]}

    def load_snapshot(self, instrument: str, data: dict) -> None:
        q: deque[ProjectionRecord] = deque(maxlen=self._max)
        for d in data.get("records", []):
            q.append(ProjectionRecord.from_dict(d))
        self._store[instrument] = q
