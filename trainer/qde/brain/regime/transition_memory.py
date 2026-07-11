"""QDE Brain — regime transition: data models and rolling memory.

This module records what the market looked like in the ticks *before* a
regime transition occurred, so the predictor can later ask: "do the
current conditions resemble patterns that historically preceded a
particular type of transition?"

Key types
---------
``TransitionTarget`` — the six regime-change categories the system
    forecasts.  Distinct from the 16 ``MarketRegime`` labels from the MIE
    (which describe current state).  ``TransitionTarget`` describes
    *direction of change*.

``RegimeSnapshot`` — one tick's observable market features extracted from
    ``SymbolObservationInput``.  No regime label or transition direction
    is stored here: the snapshot is purely observational.

``TransitionRecord`` — a confirmed regime change with the N snapshots that
    preceded it and a cached ``list[float]`` feature vector derived from
    those snapshots (stored for fast similarity look-up).

``TransitionMemory`` — long-lived per-symbol store of rolling snapshots
    and confirmed transition records.

Design rules
------------
* No hardcoded thresholds — no "if ATR slope > X then predict Y".
* Does not call ``MarketRegimeEngine`` itself — callers pass the already-
  classified regime string and notify via ``record_transition`` when a
  change is detected.
* Does not import ``QDEDecision`` or any execution-layer type.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# ── Ordinal rank for volatility state ────────────────────────────────────────

_VOL_STATE_RANK: dict[str, int] = {
    "LOW": 0, "QUIET": 0,
    "NORMAL": 1, "MODERATE": 1,
    "ELEVATED": 2,
    "HIGH": 3, "VOLATILE": 3,
    "EXTREME": 4,
}


def vol_state_rank(state: str) -> int:
    return _VOL_STATE_RANK.get(state.upper(), 1)


# ── Regime family groupings (used for target inference) ───────────────────────

_TRENDING_REGIMES: frozenset[str] = frozenset({
    "STRONG_TREND", "TRENDING", "EXPANSION",
    "TREND_UP", "TREND_DOWN",          # aliases some callers use
})
_RANGING_REGIMES: frozenset[str] = frozenset({
    "RANGE", "COMPRESSION", "ACCUMULATION", "DISTRIBUTION", "CHOP", "CHOPPY",
})
_VOLATILE_REGIMES: frozenset[str] = frozenset({
    "VOLATILE", "BLACK_SWAN", "NEWS_DRIVEN",
})
_BREAKOUT_REGIMES: frozenset[str] = frozenset({
    "BREAKOUT", "FALSE_BREAKOUT",
})
_EXHAUSTION_REGIMES: frozenset[str] = frozenset({
    "REVERSAL", "WEAK_TREND",
})


# ── TransitionTarget ──────────────────────────────────────────────────────────


class TransitionTarget(str, Enum):
    """The six regime-change directions the system learns to anticipate.

    Each target describes *what kind of shift* is coming, not which
    specific ``MarketRegime`` label will follow.  The predictor produces
    a probability over this 6-way vocabulary; callers use that distribution
    to adjust attention, conviction, and risk appetite.
    """

    TREND_FORMATION  = "TREND_FORMATION"   # consolidation → directional move
    TREND_EXHAUSTION = "TREND_EXHAUSTION"  # trend → reversal / range / weakness
    VOL_EXPANSION    = "VOL_EXPANSION"     # low/normal vol → elevated/high/extreme
    VOL_CONTRACTION  = "VOL_CONTRACTION"   # elevated vol → low/normal
    BREAKOUT         = "BREAKOUT"          # range/compression → breakout
    RANGE_FORMATION  = "RANGE_FORMATION"   # trend/volatile → range/compression


def infer_target(from_regime: str, to_regime: str) -> TransitionTarget:
    """Infer the transition target from a regime-label pair.

    Priority order matters — several pairs are ambiguous:
    e.g. ``TRENDING → RANGE`` is both a trend exit and a range entry;
    ``VOLATILE → RANGE`` is both a vol contraction and a range entry.
    The ordering below resolves those conflicts by semantic precedence.

    Used when the caller has not provided an explicit target (the common
    case when the transition is detected automatically from two consecutive
    ``market_regime`` values).
    """
    fr = from_regime.upper()
    to = to_regime.upper()

    # 1. Entering a breakout regime overrides everything else
    if to in _BREAKOUT_REGIMES:
        return TransitionTarget.BREAKOUT

    # 2. Exiting a volatile/news regime is vol contraction first, even if
    #    the destination is also a ranging regime
    if fr in _VOLATILE_REGIMES and to not in _VOLATILE_REGIMES:
        return TransitionTarget.VOL_CONTRACTION

    # 3. Entering a ranging regime from a non-ranging one (e.g. TRENDING→RANGE)
    if to in _RANGING_REGIMES and fr not in _RANGING_REGIMES:
        return TransitionTarget.RANGE_FORMATION

    # 4. Entering a trending regime from a non-trending one
    if fr not in _TRENDING_REGIMES and to in _TRENDING_REGIMES:
        return TransitionTarget.TREND_FORMATION

    # 5. From trending into an explicit exhaustion regime (REVERSAL, WEAK_TREND)
    if fr in _TRENDING_REGIMES and to in _EXHAUSTION_REGIMES:
        return TransitionTarget.TREND_EXHAUSTION

    # 6. Entering a volatile regime from a non-volatile one
    if fr not in _VOLATILE_REGIMES and to in _VOLATILE_REGIMES:
        return TransitionTarget.VOL_EXPANSION

    # 7. Any remaining exit from trending → trend exhaustion
    if fr in _TRENDING_REGIMES and to not in _TRENDING_REGIMES:
        return TransitionTarget.TREND_EXHAUSTION

    # 8. Fallback
    if to in _TRENDING_REGIMES:
        return TransitionTarget.TREND_FORMATION
    return TransitionTarget.RANGE_FORMATION


# ── RegimeSnapshot ────────────────────────────────────────────────────────────


@dataclass
class RegimeSnapshot:
    """One tick's observable market features for one instrument.

    Extracted from ``SymbolObservationInput``; stores only the numeric
    and categorical signals relevant to transition forecasting.  The
    current ``regime`` label is recorded as a string for reference but
    is never used in similarity computation — the feature vector is
    built from the continuous/ordinal fields only.
    """

    instrument:           str   = ""
    timestamp:            float = 0.0
    atr:                  float = 0.0
    spread:               float = 0.0
    volatility:           float = 0.0
    momentum:             float = 0.0   # -1..+1
    trend_valid:          bool  = True
    vol_state_rank:       int   = 1     # 0..4 ordinal
    volatility_quality:   float = 0.5
    microstructure_quality: float = 0.5
    regime:               str   = ""   # current classified regime (label only)

    @classmethod
    def from_observation(
        cls,
        instrument: str,
        inp: object,          # SymbolObservationInput
        now: float | None = None,
    ) -> RegimeSnapshot:
        """Construct from a ``SymbolObservationInput`` tick."""
        mkt = inp.market
        return cls(
            instrument=instrument,
            timestamp=now if now is not None else time.time(),
            atr=float(getattr(mkt, "atr", 0.0)),
            spread=float(getattr(mkt, "spread", 0.0)),
            volatility=float(getattr(mkt, "volatility", 0.0)),
            momentum=float(getattr(mkt, "momentum", 0.0)),
            trend_valid=bool(getattr(mkt, "trend_valid", True)),
            vol_state_rank=vol_state_rank(
                getattr(inp, "volatility_state", "NORMAL")
            ),
            volatility_quality=float(getattr(inp, "volatility_quality", 0.5)),
            microstructure_quality=float(
                getattr(inp, "microstructure_quality", 0.5)
            ),
            regime=str(getattr(inp, "market_regime", "")),
        )

    def to_dict(self) -> dict:
        return {
            "instrument":             self.instrument,
            "timestamp":              round(self.timestamp, 3),
            "atr":                    round(self.atr, 6),
            "spread":                 round(self.spread, 6),
            "volatility":             round(self.volatility, 6),
            "momentum":               round(self.momentum, 4),
            "trend_valid":            self.trend_valid,
            "vol_state_rank":         self.vol_state_rank,
            "volatility_quality":     round(self.volatility_quality, 4),
            "microstructure_quality": round(self.microstructure_quality, 4),
            "regime":                 self.regime,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegimeSnapshot:
        return cls(
            instrument=d.get("instrument", ""),
            timestamp=d.get("timestamp", 0.0),
            atr=d.get("atr", 0.0),
            spread=d.get("spread", 0.0),
            volatility=d.get("volatility", 0.0),
            momentum=d.get("momentum", 0.0),
            trend_valid=d.get("trend_valid", True),
            vol_state_rank=d.get("vol_state_rank", 1),
            volatility_quality=d.get("volatility_quality", 0.5),
            microstructure_quality=d.get("microstructure_quality", 0.5),
            regime=d.get("regime", ""),
        )


# ── TransitionRecord ──────────────────────────────────────────────────────────

_PRE_WINDOW = 10   # number of snapshots stored before each transition


@dataclass
class TransitionRecord:
    """A confirmed regime change with its pre-transition snapshot window.

    ``feature_vector`` is a ``list[float]`` computed by
    ``TransitionPredictor.extract_feature_vector`` from ``pre_snapshots``
    and cached here to avoid recomputing it on every prediction call.
    It is populated by the engine when the record is first created.
    """

    record_id:       str                 = field(
        default_factory=lambda: str(uuid.uuid4())[:12]
    )
    instrument:      str                 = ""
    from_regime:     str                 = ""
    to_regime:       str                 = ""
    target:          TransitionTarget    = TransitionTarget.TREND_FORMATION
    transition_at:   float               = field(default_factory=time.time)
    pre_snapshots:   list[RegimeSnapshot] = field(default_factory=list)
    feature_vector:  list[float]         = field(default_factory=list)

    def age_hours(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, (now - self.transition_at) / 3600.0)

    def to_dict(self) -> dict:
        return {
            "record_id":      self.record_id,
            "instrument":     self.instrument,
            "from_regime":    self.from_regime,
            "to_regime":      self.to_regime,
            "target":         self.target.value,
            "transition_at":  round(self.transition_at, 3),
            "pre_snapshots":  [s.to_dict() for s in self.pre_snapshots],
            "feature_vector": [round(x, 6) for x in self.feature_vector],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TransitionRecord:
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())[:12]),
            instrument=d.get("instrument", ""),
            from_regime=d.get("from_regime", ""),
            to_regime=d.get("to_regime", ""),
            target=TransitionTarget(d["target"]),
            transition_at=d.get("transition_at", 0.0),
            pre_snapshots=[RegimeSnapshot.from_dict(s) for s in d.get("pre_snapshots", [])],
            feature_vector=d.get("feature_vector", []),
        )


# ── TransitionMemory ──────────────────────────────────────────────────────────

_DEFAULT_SNAPSHOT_DEPTH     = 200   # rolling snapshots per symbol
_DEFAULT_MAX_TRANSITIONS    = 500   # maximum stored transition records per symbol


class TransitionMemory:
    """Per-symbol rolling store of regime snapshots and confirmed transitions.

    ``push_snapshot`` is called on every tick.
    ``record_transition`` is called when the upstream regime classifier
    detects a change (never called by this class itself).

    ``snapshots`` and ``transitions`` return read-only views; callers
    that need to mutate records should do so via the provided helpers.
    """

    def __init__(
        self,
        snapshot_depth:   int = _DEFAULT_SNAPSHOT_DEPTH,
        max_transitions:  int = _DEFAULT_MAX_TRANSITIONS,
    ) -> None:
        self._depth          = snapshot_depth
        self._max_transitions = max_transitions
        # instrument → deque[RegimeSnapshot]
        self._snapshots: dict[str, deque[RegimeSnapshot]] = {}
        # instrument → list[TransitionRecord]  (oldest → newest)
        self._transitions: dict[str, list[TransitionRecord]] = {}

    # ── Ingestion ─────────────────────────────────────────────────────────

    def push_snapshot(self, instrument: str, snap: RegimeSnapshot) -> None:
        q = self._snapshots.setdefault(
            instrument, deque(maxlen=self._depth)
        )
        q.append(snap)

    def record_transition(
        self,
        instrument:  str,
        from_regime: str,
        to_regime:   str,
        target:      TransitionTarget | None = None,
        now:         float | None = None,
    ) -> TransitionRecord:
        """Record a confirmed regime change.

        Captures the last ``_PRE_WINDOW`` snapshots as the pre-transition
        context window; the feature vector is populated later by the engine
        (after the predictor has been initialised).
        """
        now = now if now is not None else time.time()
        if target is None:
            target = infer_target(from_regime, to_regime)

        pre = list(self._snapshots.get(instrument, []))[-_PRE_WINDOW:]
        record = TransitionRecord(
            instrument=instrument,
            from_regime=from_regime,
            to_regime=to_regime,
            target=target,
            transition_at=now,
            pre_snapshots=pre,
        )

        records = self._transitions.setdefault(instrument, [])
        records.append(record)
        # Prune oldest if limit exceeded
        if len(records) > self._max_transitions:
            del records[: len(records) - self._max_transitions]

        return record

    # ── Read accessors ─────────────────────────────────────────────────────

    def snapshots(self, instrument: str) -> list[RegimeSnapshot]:
        """Recent snapshots, oldest → newest."""
        return list(self._snapshots.get(instrument, []))

    def transitions(self, instrument: str) -> list[TransitionRecord]:
        """All transition records, oldest → newest."""
        return list(self._transitions.get(instrument, []))

    def last_transition(self, instrument: str) -> TransitionRecord | None:
        records = self._transitions.get(instrument)
        return records[-1] if records else None

    def transition_count(self, instrument: str) -> int:
        return len(self._transitions.get(instrument, []))

    def current_regime(self, instrument: str) -> str:
        """Most recently observed regime label, or empty string."""
        snaps = self._snapshots.get(instrument)
        if snaps:
            return snaps[-1].regime
        return ""

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        return {
            "snapshots":   [s.to_dict() for s in self.snapshots(instrument)],
            "transitions": [r.to_dict() for r in self.transitions(instrument)],
        }

    def load_snapshot(self, instrument: str, data: dict) -> None:
        q = deque(maxlen=self._depth)
        for s in data.get("snapshots", []):
            q.append(RegimeSnapshot.from_dict(s))
        self._snapshots[instrument] = q
        self._transitions[instrument] = [
            TransitionRecord.from_dict(r) for r in data.get("transitions", [])
        ]

    # ── Transition statistics ──────────────────────────────────────────────

    def transition_statistics(
        self,
        instrument:      str,
        recency_lambda:  float = 1.0 / 30.0,   # decay rate: e^(-λ×age_days)
        now:             float | None = None,
    ) -> dict:
        """Return transition probability estimates for ``instrument``.

        Computes three complementary views over the stored
        ``TransitionRecord`` history:

        ``target_counts``
            Raw count of confirmed transitions per ``TransitionTarget``.

        ``target_probabilities``
            Uniform-weighted probability over the six targets: each
            target's count divided by the total number of transitions.
            Uniform prior (1/6) when no data.

        ``target_probabilities_weighted``
            Recency-weighted probability.  Each record contributes weight
            ``e^(−λ × age_days)`` rather than 1, so recent transitions
            influence the distribution more than old ones.  λ is set by
            ``recency_lambda`` (default: e-fold every 30 days).

        ``from_regime_probs``
            Conditional probability table: ``from_regime → {target: prob}``.
            "Given the market is currently in regime X, what transition is
            most likely?"  Each row is normalized independently.

        ``total_transitions``
            Total confirmed transition records for ``instrument``.

        ``instruments_available``
            List of instruments for which transition data exists.
        """
        import math as _math
        now = now if now is not None else __import__("time").time()

        records = self.transitions(instrument)

        # ── Raw counts ────────────────────────────────────────────────────
        target_counts: dict[str, int] = {t.value: 0 for t in TransitionTarget}
        weighted_sums: dict[str, float] = {t.value: 0.0 for t in TransitionTarget}
        total_weight = 0.0

        # from_regime → target → count
        from_regime_counts: dict[str, dict[str, int]] = {}

        for rec in records:
            tv  = rec.target.value
            age = max(0.0, now - rec.transition_at) / 86400.0   # age in days
            w   = _math.exp(-recency_lambda * age)

            target_counts[tv] += 1
            weighted_sums[tv] += w
            total_weight       += w

            fr = rec.from_regime or "UNKNOWN"
            if fr not in from_regime_counts:
                from_regime_counts[fr] = {t.value: 0 for t in TransitionTarget}
            from_regime_counts[fr][tv] += 1

        total = len(records)

        # ── Uniform probabilities ─────────────────────────────────────────
        n_targets = len(TransitionTarget)
        if total > 0:
            target_probs = {
                tv: round(count / total, 4)
                for tv, count in target_counts.items()
            }
        else:
            target_probs = {t.value: round(1.0 / n_targets, 4) for t in TransitionTarget}

        # ── Recency-weighted probabilities ────────────────────────────────
        if total_weight > 1e-12:
            target_probs_weighted = {
                tv: round(ws / total_weight, 4)
                for tv, ws in weighted_sums.items()
            }
        else:
            target_probs_weighted = {t.value: round(1.0 / n_targets, 4) for t in TransitionTarget}

        # ── Conditional from-regime probabilities ─────────────────────────
        from_regime_probs: dict[str, dict[str, float]] = {}
        for fr, counts in from_regime_counts.items():
            row_total = sum(counts.values())
            if row_total > 0:
                from_regime_probs[fr] = {
                    tv: round(c / row_total, 4) for tv, c in counts.items()
                }

        return {
            "instrument":              instrument,
            "total_transitions":       total,
            "target_counts":           target_counts,
            "target_probabilities":    target_probs,
            "target_probabilities_weighted": target_probs_weighted,
            "from_regime_probs":       from_regime_probs,
            "instruments_available":   list(self._transitions.keys()),
        }
