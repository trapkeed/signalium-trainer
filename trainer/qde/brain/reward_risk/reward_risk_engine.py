"""QDE Brain — reward_risk: learned reward/risk geometry engine.

``RewardRiskEngine`` is the learning layer that answers four questions
that the projection package cannot:

1. **Distribution shape**: where do the 50th, 75th, and 90th percentiles
   of MAE and MFE sit for setups like this one?  The projection engine
   reports the *mean*; stop and target placement are *tail* problems.

2. **Stop-hit probability**: given a specific proposed stop distance,
   what fraction of similar historical setups would have triggered that
   stop before resolving?  Requires explicit stop_distance_atr input.

3. **Win rate and capture rate**: what fraction of similar setups were
   profitable?  And of the profitable ones, how much of the maximum
   favorable excursion was captured as realized reward?

4. **Principled position sizing**: Kelly-inspired sizing uses the
   empirical win rate and expected reward/risk ratio — a statistically
   grounded alternative to the projection engine's linear-interpolation
   approach.

Relationship to the projection package
---------------------------------------
``ExpectedMoveEngine`` (projection) and ``RewardRiskEngine`` are
**complementary, not competing**:

  * Projection uses cosine similarity on a 9-dim *market conditions*
    fingerprint to estimate *average* trajectory.
  * Reward/risk uses cosine similarity on a 7-dim fingerprint (same
    market conditions, plus directed momentum) to estimate the
    *distribution shape* and computes stop/target hit probabilities by
    querying that distribution.

The two engines produce distinct ``proj_*`` and ``rr_*`` key sets that
coexist in ``QDEInferenceInput.brain_context``.

Learning loop
-------------
Call sequence for one completed trade:

1. At opportunity identification time::

       conditions = RewardRiskConditions(
           instrument="EURUSD",
           spread_atr_ratio=...,
           atr_pct=...,
           volatility=...,
           session_quality=...,
           momentum=...,
           direction=+1,
           microstructure_quality=...,
       )
       snapshot = engine.snapshot_conditions(conditions)
       # store snapshot alongside the trade record

2. After the trade closes::

       outcome = TradeGeometryOutcome(
           mae_atr=realized_mae / atr,
           mfe_atr=realized_mfe / atr,
           reward_atr=realized_pnl / atr,   # signed
           was_stopped=stop_triggered,
           hit_target=target_triggered,
       )
       engine.record_outcome(snapshot, outcome)

Algorithm
---------
For each metric (MAE mean, MFE mean, win rate, capture rate, RR):

1. Retrieve all historical ``RewardRiskRecord`` for the instrument.
2. Score each: ``weight = cosine_sim(current_vec, hist_vec) × recency``.
3. Take the top-K highest-weight records.
4. Compute weighted estimates of the distribution (mean, quantiles,
   win rate, capture rate) from those K records.
5. Blend structural baseline (from ``AdverseExcursionPredictor`` and
   ``FavorableExcursionPredictor``) with historical estimate by
   confidence: ``result = structural × (1 − conf) + historical × conf``.

Quantile estimation from samples
---------------------------------
Given K weighted samples of MAE values, the Pth quantile is estimated
via the weighted empirical distribution: sort samples ascending, compute
cumulative normalized weights, interpolate at the Pth percentile.  This
avoids assuming log-normality for historical data (the structural model
makes that assumption; the historical model is non-parametric).

Position sizing (Kelly-inspired)
---------------------------------
::

    rr = expected_reward_atr / max(expected_mae_atr, 1e-6)
    kelly_f = win_rate - (1 - win_rate) / rr
    half_kelly = kelly_f * 0.5   # half-Kelly for variance control
    size_mod = clamp(1.0 + half_kelly, 0.5, 2.0)

When history is thin, ``win_rate`` and ``rr`` fall back to structural
estimates (MFE / (MFE + MAE) and structural_MFE / structural_MAE).

Design rules
------------
* Does not import ``QDEDecision`` or any execution-layer type.
* Does not import from ``trainer.qde.brain.projection`` (no cross-package
  dependency; conditions are distinct).
* Multi-symbol: one engine instance manages all watched instruments via
  a keyed memory store (matching ``ExpectedMoveEngine`` convention).
* All magnitude outputs in ATR units (instrument-agnostic).
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.reward_risk.adverse_excursion_predictor import (
    AdverseExcursionInput,
    AdverseExcursionPredictor,
)
from trainer.qde.brain.reward_risk.favorable_excursion_predictor import (
    FavorableExcursionInput,
    FavorableExcursionPredictor,
)

# ── Hyper-parameters ──────────────────────────────────────────────────────────

_DEFAULT_MAX_RECORDS = 2000  # per-symbol bounded FIFO capacity
_DEFAULT_K = 30  # top-K for weighted estimates
_MIN_INFORMATIVE = 5  # confidence floor sample count
_RECENCY_TAU_DAYS = 21.0  # exponential recency decay (days)
_COSINE_FALLBACK_W = 1e-3  # weight floor for recency-only fallback

# Conditions vector dimension
_N_DIMS = 7

# Integration signal bounds
_SIZE_MOD_MIN = 0.50
_SIZE_MOD_MAX = 2.00
_RISK_DRAG_MAX = -0.50
_RISK_BOOST_MAX = 0.25

# Kelly half-Kelly clamp bounds (before mapping to size modifier)
_KELLY_LO = -0.50
_KELLY_HI = 1.00


# ── RewardRiskConditions ──────────────────────────────────────────────────────


@dataclass
class RewardRiskConditions:
    """Market conditions snapshot for reward/risk estimation.

    Seven numeric features characterise the market and trade direction at
    the time an opportunity is identified.  The 7-dim ``as_vector()`` is
    used for cosine-similarity lookup against historical records.

    ``directed_momentum``
        Computed property: ``momentum × direction``.  Positive means
        market momentum is already in the trade direction (favorable
        tailwind); negative means adverse momentum.

    Conditions are direction-aware so that a long setup in a bull trend
    and a short setup in the same bull trend correctly produce different
    similarity scores against historical records.
    """

    instrument: str = ""
    spread_atr_ratio: float = 0.10
    atr_pct: float = 0.005
    volatility: float = 0.010
    session_quality: float = 0.80
    momentum: float = 0.0
    direction: int = 1
    microstructure_quality: float = 0.80
    timestamp: float = field(default_factory=time.time)

    @property
    def directed_momentum(self) -> float:
        """Positive when momentum is in trade direction; negative otherwise."""
        return self.momentum * self.direction

    def as_vector(self) -> list[float]:
        """7-dim cosine-similarity input vector.

        Dimensions:
            [spread_atr_ratio, atr_pct, volatility, session_quality,
             directed_momentum, microstructure_quality, direction_f]

        ``direction_f`` encodes long (+0.5) vs short (−0.5) so that
        otherwise identical conditions for opposite directions are
        separated in the feature space.
        """
        return [
            self.spread_atr_ratio,
            self.atr_pct,
            self.volatility,
            self.session_quality,
            self.directed_momentum,
            self.microstructure_quality,
            float(self.direction) * 0.5,  # ±0.5 direction flag
        ]

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "spread_atr_ratio": round(self.spread_atr_ratio, 4),
            "atr_pct": round(self.atr_pct, 8),
            "volatility": round(self.volatility, 8),
            "session_quality": round(self.session_quality, 4),
            "momentum": round(self.momentum, 4),
            "direction": self.direction,
            "microstructure_quality": round(self.microstructure_quality, 4),
            "timestamp": round(self.timestamp, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RewardRiskConditions:
        return cls(
            instrument=d.get("instrument", ""),
            spread_atr_ratio=d.get("spread_atr_ratio", 0.10),
            atr_pct=d.get("atr_pct", 0.005),
            volatility=d.get("volatility", 0.010),
            session_quality=d.get("session_quality", 0.80),
            momentum=d.get("momentum", 0.0),
            direction=d.get("direction", 1),
            microstructure_quality=d.get("microstructure_quality", 0.80),
            timestamp=d.get("timestamp", 0.0),
        )


# ── TradeGeometryOutcome ──────────────────────────────────────────────────────


@dataclass
class TradeGeometryOutcome:
    """Realized trade geometry outcome recorded after a trade closes.

    ``mae_atr``
        Realized maximum adverse excursion in ATR units.  Always ≥ 0.

    ``mfe_atr``
        Realized maximum favorable excursion in ATR units.  Always ≥ 0.

    ``reward_atr``
        Realized reward (= exit price − entry price, signed by direction)
        in ATR units.  Positive = trade was profitable.

    ``was_stopped``
        True if the stop-loss was triggered before the trade reached its
        natural exit.

    ``hit_target``
        True if the take-profit target was reached before any stop.

    ``capture_rate``
        ``reward_atr / mfe_atr`` when the trade was profitable; 0.0
        otherwise.  Measures how efficiently the favorable excursion was
        converted to realized profit.
    """

    mae_atr: float = 0.0
    mfe_atr: float = 0.0
    reward_atr: float = 0.0  # signed
    was_stopped: bool = False
    hit_target: bool = False
    capture_rate: float = 0.0  # reward / mfe; 0.0 if not profitable

    @staticmethod
    def compute(
        mae_atr: float,
        mfe_atr: float,
        reward_atr: float,
        was_stopped: bool = False,
        hit_target: bool = False,
    ) -> "TradeGeometryOutcome":
        """Factory with automatic capture-rate computation.

        ``capture_rate`` is set to ``reward_atr / mfe_atr`` when the
        trade was profitable and ``mfe_atr > 0``; otherwise 0.0.
        """
        if reward_atr > 0.0 and mfe_atr > 1e-9:
            capture_rate = min(1.0, reward_atr / mfe_atr)
        else:
            capture_rate = 0.0
        return TradeGeometryOutcome(
            mae_atr=max(0.0, mae_atr),
            mfe_atr=max(0.0, mfe_atr),
            reward_atr=reward_atr,
            was_stopped=was_stopped,
            hit_target=hit_target,
            capture_rate=capture_rate,
        )

    def to_dict(self) -> dict:
        return {
            "mae_atr": round(self.mae_atr, 6),
            "mfe_atr": round(self.mfe_atr, 6),
            "reward_atr": round(self.reward_atr, 6),
            "was_stopped": self.was_stopped,
            "hit_target": self.hit_target,
            "capture_rate": round(self.capture_rate, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TradeGeometryOutcome:
        return cls(
            mae_atr=d.get("mae_atr", 0.0),
            mfe_atr=d.get("mfe_atr", 0.0),
            reward_atr=d.get("reward_atr", 0.0),
            was_stopped=d.get("was_stopped", False),
            hit_target=d.get("hit_target", False),
            capture_rate=d.get("capture_rate", 0.0),
        )


# ── RewardRiskRecord ──────────────────────────────────────────────────────────


@dataclass
class RewardRiskRecord:
    """One completed (conditions → geometry outcome) learning observation."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    conditions: RewardRiskConditions = field(default_factory=RewardRiskConditions)
    outcome: TradeGeometryOutcome = field(default_factory=TradeGeometryOutcome)
    recorded_at: float = field(default_factory=time.time)

    def recency_weight(self, now: float, tau_days: float = _RECENCY_TAU_DAYS) -> float:
        """Exponential recency weight.  1.0 when fresh, decays over tau_days."""
        elapsed_days = max(0.0, now - self.recorded_at) / 86400.0
        return math.exp(-elapsed_days / max(tau_days, 1e-9))

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "conditions": self.conditions.to_dict(),
            "outcome": self.outcome.to_dict(),
            "recorded_at": round(self.recorded_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RewardRiskRecord:
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())[:12]),
            conditions=RewardRiskConditions.from_dict(d.get("conditions", {})),
            outcome=TradeGeometryOutcome.from_dict(d.get("outcome", {})),
            recorded_at=d.get("recorded_at", 0.0),
        )


# ── RewardRiskMemory ──────────────────────────────────────────────────────────


class RewardRiskMemory:
    """Per-symbol rolling store of trade geometry observations.

    One store per instrument, keyed by instrument name.  Each store is a
    bounded FIFO — oldest records are evicted when the capacity limit is
    reached.  No business logic lives here; all computation happens in
    ``RewardRiskEngine``.
    """

    def __init__(self, max_records: int = _DEFAULT_MAX_RECORDS) -> None:
        self._max = max(1, max_records)
        self._store: dict[str, deque[RewardRiskRecord]] = {}

    def record(
        self,
        instrument: str,
        conditions: RewardRiskConditions,
        outcome: TradeGeometryOutcome,
        now: float | None = None,
    ) -> RewardRiskRecord:
        """Append one observation.  Evicts oldest when at capacity."""
        now = now if now is not None else time.time()
        rec = RewardRiskRecord(conditions=conditions, outcome=outcome, recorded_at=now)
        q = self._store.setdefault(instrument, deque(maxlen=self._max))
        q.append(rec)
        return rec

    def records(self, instrument: str) -> list[RewardRiskRecord]:
        """All records for instrument, oldest first."""
        return list(self._store.get(instrument, []))

    def record_count(self, instrument: str) -> int:
        return len(self._store.get(instrument, []))

    def instruments(self) -> list[str]:
        return list(self._store.keys())

    def to_snapshot(self, instrument: str) -> dict:
        return {
            "max_records": self._max,
            "records": [r.to_dict() for r in self.records(instrument)],
        }

    def load_snapshot(self, instrument: str, data: dict) -> None:
        q: deque[RewardRiskRecord] = deque(maxlen=self._max)
        for d in data.get("records", []):
            try:
                q.append(RewardRiskRecord.from_dict(d))
            except Exception:
                pass
        self._store[instrument] = q


# ── RewardRiskSignal ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RewardRiskSignal:
    """Complete reward/risk geometry estimate for one opportunity.

    Produced by ``RewardRiskEngine.score()``.  All magnitude fields are
    in ATR units.

    Distribution fields (MAE)
    --------------------------
    ``expected_mae_atr``     — mean MAE (structural + historical blend)
    ``mae_p75_atr``          — 75th percentile MAE (sizing-aware stop floor)
    ``mae_p90_atr``          — 90th percentile MAE (stress-scenario bound)

    Distribution fields (MFE)
    --------------------------
    ``expected_mfe_atr``     — mean MFE
    ``mfe_p50_atr``          — median MFE (conservative take-profit target)
    ``mfe_p75_atr``          — 75th percentile MFE

    Ratio and probability fields
    ----------------------------
    ``expected_rr``           — expected_reward_atr / expected_mae_atr
    ``win_rate``              — fraction of similar trades that were profitable
    ``win_rate_at_rr2``       — fraction achieving ≥ 2:1 reward/risk
    ``stop_hit_probability``  — P(MAE > stop_distance_atr); 0 if not specified
    ``target_hit_probability`` — P(MFE > target_distance_atr); 0 if not specified
    ``expected_capture_rate`` — historically captured fraction of MFE

    Integration signals
    -------------------
    ``position_size_modifier``  — Kelly-inspired 0.5..2.0
    ``risk_appetite_modifier``  — −0.50..+0.25
    ``opportunity_rank_signal`` — 0-1 for trade selection / opportunity ranking

    Metadata
    --------
    ``confidence``    — 0-1 based on similar-record count
    ``sample_count``  — K records used in estimate
    ``is_structural`` — True when using pure structural baseline (< min_informative records)
    """

    # MAE distribution
    expected_mae_atr: float = 0.0
    mae_p75_atr: float = 0.0
    mae_p90_atr: float = 0.0

    # MFE distribution
    expected_mfe_atr: float = 0.0
    mfe_p50_atr: float = 0.0
    mfe_p75_atr: float = 0.0

    # Ratio and probabilities
    expected_rr: float = 0.0
    win_rate: float = 0.5
    win_rate_at_rr2: float = 0.0
    stop_hit_probability: float = 0.0
    target_hit_probability: float = 0.0
    expected_capture_rate: float = 0.0

    # Integration signals
    position_size_modifier: float = 1.0
    risk_appetite_modifier: float = 0.0
    opportunity_rank_signal: float = 0.0

    # Metadata
    confidence: float = 0.0
    sample_count: int = 0
    is_structural: bool = True

    def to_dict(self) -> dict:
        return {
            "expected_mae_atr": round(self.expected_mae_atr, 4),
            "mae_p75_atr": round(self.mae_p75_atr, 4),
            "mae_p90_atr": round(self.mae_p90_atr, 4),
            "expected_mfe_atr": round(self.expected_mfe_atr, 4),
            "mfe_p50_atr": round(self.mfe_p50_atr, 4),
            "mfe_p75_atr": round(self.mfe_p75_atr, 4),
            "expected_rr": round(self.expected_rr, 4),
            "win_rate": round(self.win_rate, 4),
            "win_rate_at_rr2": round(self.win_rate_at_rr2, 4),
            "stop_hit_probability": round(self.stop_hit_probability, 4),
            "target_hit_probability": round(self.target_hit_probability, 4),
            "expected_capture_rate": round(self.expected_capture_rate, 4),
            "position_size_modifier": round(self.position_size_modifier, 4),
            "risk_appetite_modifier": round(self.risk_appetite_modifier, 4),
            "opportunity_rank_signal": round(self.opportunity_rank_signal, 4),
            "confidence": round(self.confidence, 4),
            "sample_count": self.sample_count,
            "is_structural": self.is_structural,
        }


# ── RewardRiskEngine ──────────────────────────────────────────────────────────


class RewardRiskEngine:
    """Learned multi-symbol reward/risk geometry engine.

    Holds one ``RewardRiskMemory`` for all instruments.  Each instrument
    accumulates its own bounded FIFO of completed-trade geometry
    observations.

    Complementary to ``ExpectedMoveEngine`` (projection):

    * **Projection**: "Given these market conditions, what was the
      *average* trajectory of similar setups?"
    * **RewardRiskEngine**: "Given these market conditions, what is the
      *distribution* of trade geometry, and how do specific stop/target
      distances interact with that distribution?"

    Integration signals (``context_for_inference``)
    ------------------------------------------------
    Returns exactly 18 ``rr_*`` keys for ``QDEInferenceInput.brain_context``:

        rr_expected_mae_atr
        rr_mae_p75_atr
        rr_mae_p90_atr
        rr_expected_mfe_atr
        rr_mfe_p50_atr
        rr_mfe_p75_atr
        rr_expected_rr
        rr_win_rate
        rr_win_rate_at_rr2
        rr_stop_hit_probability
        rr_target_hit_probability
        rr_expected_capture_rate
        rr_position_size_modifier
        rr_risk_appetite_modifier
        rr_opportunity_rank_signal
        rr_confidence
        rr_sample_count
        rr_is_structural
    """

    def __init__(
        self,
        max_records: int = _DEFAULT_MAX_RECORDS,
        k: int = _DEFAULT_K,
        recency_tau_days: float = _RECENCY_TAU_DAYS,
        min_informative: int = _MIN_INFORMATIVE,
    ) -> None:
        self._memory = RewardRiskMemory(max_records=max_records)
        self._mae_pred = AdverseExcursionPredictor()
        self._mfe_pred = FavorableExcursionPredictor()
        self._k = k
        self._tau = recency_tau_days
        self._min = min_informative

    # ── Learning ──────────────────────────────────────────────────────────

    def record_outcome(
        self,
        conditions: RewardRiskConditions,
        outcome: TradeGeometryOutcome,
        now: float | None = None,
    ) -> RewardRiskRecord:
        """Append one completed trade geometry to the learning store.

        Call once per confirmed trade exit with the realized MAE, MFE,
        reward, and outcome flags.  Use ``TradeGeometryOutcome.compute()``
        to automatically calculate ``capture_rate``.
        """
        now = now if now is not None else time.time()
        rec = self._memory.record(conditions.instrument, conditions, outcome, now)
        logger.debug(
            f"[RRE] {conditions.instrument} dir={conditions.direction:+d} "
            f"mae={outcome.mae_atr:.3f} mfe={outcome.mfe_atr:.3f} "
            f"reward={outcome.reward_atr:.3f} ATR recorded."
        )
        return rec

    # ── Scoring ───────────────────────────────────────────────────────────

    def score(
        self,
        conditions: RewardRiskConditions,
        stop_distance_atr: float = 0.0,
        target_distance_atr: float = 0.0,
        now: float | None = None,
    ) -> RewardRiskSignal:
        """Compute the full reward/risk geometry signal for one opportunity.

        Parameters
        ----------
        conditions:
            Live market conditions and direction.
        stop_distance_atr:
            Proposed stop distance in ATR units.  0 = not specified;
            ``stop_hit_probability`` will be 0.0 in that case.
        target_distance_atr:
            Proposed take-profit distance in ATR units.  0 = not specified.
        now:
            Override for the current time (for testing / replay).
        """
        now = now if now is not None else time.time()

        # ── Structural baseline ───────────────────────────────────────────
        mae_inp = AdverseExcursionInput(
            spread_atr_ratio=conditions.spread_atr_ratio,
            atr_pct=conditions.atr_pct,
            volatility=conditions.volatility,
            session_quality=conditions.session_quality,
            momentum=conditions.momentum,
            direction=conditions.direction,
            microstructure_quality=conditions.microstructure_quality,
            stop_distance_atr=stop_distance_atr,
        )
        mfe_inp = FavorableExcursionInput(
            spread_atr_ratio=conditions.spread_atr_ratio,
            atr_pct=conditions.atr_pct,
            volatility=conditions.volatility,
            session_quality=conditions.session_quality,
            momentum=conditions.momentum,
            direction=conditions.direction,
            microstructure_quality=conditions.microstructure_quality,
            target_distance_atr=target_distance_atr,
        )
        s_mae = self._mae_pred.estimate(mae_inp)
        s_mfe = self._mfe_pred.estimate(mfe_inp)

        # Structural win rate: MFE / (MFE + MAE) is the "share of the
        # excursion budget" in the favorable direction.
        struct_win_rate = s_mfe.expected_mfe_atr / max(
            s_mfe.expected_mfe_atr + s_mae.expected_mae_atr, 1e-9
        )
        struct_rr = s_mfe.expected_mfe_atr / max(s_mae.expected_mae_atr, 1e-9)

        # ── Historical lookup ─────────────────────────────────────────────
        records = self._memory.records(conditions.instrument)
        confidence = 0.0
        h_mae_mean = s_mae.expected_mae_atr
        h_mfe_mean = s_mfe.expected_mfe_atr
        h_mae_p75 = s_mae.mae_p75_atr
        h_mae_p90 = s_mae.mae_p90_atr
        h_mfe_p50 = s_mfe.mfe_p50_atr
        h_mfe_p75 = s_mfe.mfe_p75_atr
        h_win_rate = struct_win_rate
        h_win_rr2 = max(0.0, struct_win_rate - 0.15)  # structural prior for RR2
        h_capture = 0.50  # neutral prior when no history
        h_stop_hit = s_mae.stop_hit_probability
        h_tgt_hit = s_mfe.target_hit_probability
        n_used = 0

        if records:
            scored = _score_records(conditions.as_vector(), records, self._tau, now)
            top_k = scored[: self._k]

            if top_k:
                n_used = len(top_k)
                confidence = min(1.0, n_used / max(self._min, 1))
                weights = [w for w, _ in top_k]
                recs_k = [r for _, r in top_k]

                h_mae_mean = _wmean([r.outcome.mae_atr for r in recs_k], weights)
                h_mfe_mean = _wmean([r.outcome.mfe_atr for r in recs_k], weights)

                # Quantiles from empirical weighted distribution
                h_mae_p75, h_mae_p90 = _wquantiles(
                    [r.outcome.mae_atr for r in recs_k], weights, (0.75, 0.90)
                )
                h_mfe_p50, h_mfe_p75 = _wquantiles(
                    [r.outcome.mfe_atr for r in recs_k], weights, (0.50, 0.75)
                )

                # Win rate: fraction with reward > 0 (weighted)
                h_win_rate = _wfrac(weights, [r.outcome.reward_atr > 0.0 for r in recs_k])
                # Win rate at ≥2:1 RR: mfe_atr / mae_atr ≥ 2 AND profitable
                h_win_rr2 = _wfrac(
                    weights,
                    [
                        r.outcome.reward_atr > 0.0
                        and r.outcome.mfe_atr / max(r.outcome.mae_atr, 1e-9) >= 2.0
                        for r in recs_k
                    ],
                )
                # Capture rate: mean of positive-reward records' capture_rate
                cap_vals = [r.outcome.capture_rate for r in recs_k if r.outcome.reward_atr > 0.0]
                cap_ws = [w for w, r in zip(weights, recs_k) if r.outcome.reward_atr > 0.0]
                h_capture = _wmean(cap_vals, cap_ws) if cap_vals else 0.50

                # Probabilistic stop/target hit from empirical distribution
                if stop_distance_atr > 0.0:
                    h_stop_hit = _wfrac(
                        weights,
                        [r.outcome.mae_atr > stop_distance_atr for r in recs_k],
                    )
                if target_distance_atr > 0.0:
                    h_tgt_hit = _wfrac(
                        weights,
                        [r.outcome.mfe_atr > target_distance_atr for r in recs_k],
                    )

        # ── Blend structural + historical ─────────────────────────────────
        c = confidence
        s = 1.0 - c

        mae_mean = s * s_mae.expected_mae_atr + c * h_mae_mean
        mfe_mean = s * s_mfe.expected_mfe_atr + c * h_mfe_mean
        mae_p75 = s * s_mae.mae_p75_atr + c * h_mae_p75
        mae_p90 = s * s_mae.mae_p90_atr + c * h_mae_p90
        mfe_p50 = s * s_mfe.mfe_p50_atr + c * h_mfe_p50
        mfe_p75 = s * s_mfe.mfe_p75_atr + c * h_mfe_p75
        win_rate = s * struct_win_rate + c * h_win_rate
        win_rr2 = s * max(0.0, struct_win_rate - 0.15) + c * h_win_rr2
        capture = s * 0.50 + c * h_capture

        # Blended stop/target hit probabilities (structural already
        # embedded in h_* when no history, so only blend when we have
        # empirical data)
        stop_hit = h_stop_hit
        tgt_hit = h_tgt_hit

        expected_rr = mfe_mean / max(mae_mean, 1e-9)

        # ── Integration signals ────────────────────────────────────────────
        size_mod = _kelly_size_modifier(win_rate, expected_rr)
        risk_mod = _risk_appetite(mae_p75, mfe_mean, win_rate, confidence)
        opp_rank = _opportunity_rank(win_rate, expected_rr, confidence)

        return RewardRiskSignal(
            expected_mae_atr=round(max(0.0, mae_mean), 4),
            mae_p75_atr=round(max(0.0, mae_p75), 4),
            mae_p90_atr=round(max(0.0, mae_p90), 4),
            expected_mfe_atr=round(max(0.0, mfe_mean), 4),
            mfe_p50_atr=round(max(0.0, mfe_p50), 4),
            mfe_p75_atr=round(max(0.0, mfe_p75), 4),
            expected_rr=round(expected_rr, 4),
            win_rate=round(win_rate, 4),
            win_rate_at_rr2=round(win_rr2, 4),
            stop_hit_probability=round(stop_hit, 4),
            target_hit_probability=round(tgt_hit, 4),
            expected_capture_rate=round(capture, 4),
            position_size_modifier=round(size_mod, 4),
            risk_appetite_modifier=round(risk_mod, 4),
            opportunity_rank_signal=round(opp_rank, 4),
            confidence=round(confidence, 4),
            sample_count=n_used,
            is_structural=(confidence == 0.0),
        )

    def context_for_inference(
        self,
        conditions: RewardRiskConditions,
        stop_distance_atr: float = 0.0,
        target_distance_atr: float = 0.0,
        now: float | None = None,
    ) -> dict:
        """Return exactly 18 ``rr_*`` keys for ``QDEInferenceInput.brain_context``.

        Keys
        ----
        rr_expected_mae_atr        — mean MAE in ATR (structural+historical blend)
        rr_mae_p75_atr             — 75th-pct MAE (stop placement floor)
        rr_mae_p90_atr             — 90th-pct MAE (stress bound)
        rr_expected_mfe_atr        — mean MFE in ATR
        rr_mfe_p50_atr             — median MFE (conservative take-profit target)
        rr_mfe_p75_atr             — 75th-pct MFE
        rr_expected_rr             — expected_mfe / expected_mae
        rr_win_rate                — empirical win rate (0-1)
        rr_win_rate_at_rr2         — win rate achieving ≥2:1 RR
        rr_stop_hit_probability    — P(MAE > stop_distance_atr)
        rr_target_hit_probability  — P(MFE > target_distance_atr)
        rr_expected_capture_rate   — historically captured fraction of MFE
        rr_position_size_modifier  — Kelly-inspired 0.5..2.0
        rr_risk_appetite_modifier  — -0.50..+0.25
        rr_opportunity_rank_signal — 0-1 trade selection signal
        rr_confidence              — 0-1 based on similar-record count
        rr_sample_count            — records used in estimate
        rr_is_structural           — 1.0 if pure structural, 0.0 if history used
        """
        sig = self.score(conditions, stop_distance_atr, target_distance_atr, now)
        return {
            "rr_expected_mae_atr": round(sig.expected_mae_atr, 4),
            "rr_mae_p75_atr": round(sig.mae_p75_atr, 4),
            "rr_mae_p90_atr": round(sig.mae_p90_atr, 4),
            "rr_expected_mfe_atr": round(sig.expected_mfe_atr, 4),
            "rr_mfe_p50_atr": round(sig.mfe_p50_atr, 4),
            "rr_mfe_p75_atr": round(sig.mfe_p75_atr, 4),
            "rr_expected_rr": round(sig.expected_rr, 4),
            "rr_win_rate": round(sig.win_rate, 4),
            "rr_win_rate_at_rr2": round(sig.win_rate_at_rr2, 4),
            "rr_stop_hit_probability": round(sig.stop_hit_probability, 4),
            "rr_target_hit_probability": round(sig.target_hit_probability, 4),
            "rr_expected_capture_rate": round(sig.expected_capture_rate, 4),
            "rr_position_size_modifier": round(sig.position_size_modifier, 4),
            "rr_risk_appetite_modifier": round(sig.risk_appetite_modifier, 4),
            "rr_opportunity_rank_signal": round(sig.opportunity_rank_signal, 4),
            "rr_confidence": round(sig.confidence, 4),
            "rr_sample_count": sig.sample_count,
            "rr_is_structural": 1.0 if sig.is_structural else 0.0,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        """Serialise one instrument's memory for cross-session persistence."""
        return self._memory.to_snapshot(instrument)

    def load_snapshot(self, instrument: str, data: dict) -> None:
        """Restore a previously serialised instrument memory."""
        self._memory.load_snapshot(instrument, data)

    def record_count(self, instrument: str) -> int:
        return self._memory.record_count(instrument)

    def memory(self) -> RewardRiskMemory:
        return self._memory


# ── Internal helpers ──────────────────────────────────────────────────────────


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity ∈ [−1, 1].  Returns 0.0 for zero vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _score_records(
    current_vec: list[float],
    records: list[RewardRiskRecord],
    tau_days: float,
    now: float,
) -> list[tuple[float, RewardRiskRecord]]:
    """Score records by cosine_sim × recency, sorted highest first.

    Falls back to recency-only when all cosine similarities are zero
    (e.g. zero-vector conditions).
    """
    scored: list[tuple[float, RewardRiskRecord]] = []
    for rec in records:
        sim = max(0.0, _cosine_sim(current_vec, rec.conditions.as_vector()))
        recency = rec.recency_weight(now, tau_days)
        w = sim * recency
        if w > 1e-12:
            scored.append((w, rec))

    if not scored:
        # Recency-only fallback
        for rec in records:
            r = rec.recency_weight(now, tau_days)
            if r > _COSINE_FALLBACK_W:
                scored.append((r, rec))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def _wmean(values: list[float], weights: list[float]) -> float:
    """Weighted mean.  Returns 0.0 on empty input."""
    total_w = sum(weights)
    if total_w < 1e-12:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _wfrac(weights: list[float], flags: list[bool]) -> float:
    """Weighted fraction of True flags.  Returns 0.0 on empty input."""
    total_w = sum(weights)
    if total_w < 1e-12:
        return 0.0
    pos_w = sum(w for w, f in zip(weights, flags) if f)
    return max(0.0, min(1.0, pos_w / total_w))


def _wquantiles(
    values: list[float],
    weights: list[float],
    quantiles: tuple[float, ...],
) -> tuple[float, ...]:
    """Weighted empirical quantiles.

    Sorts (value, weight) pairs ascending by value, computes cumulative
    normalized weights, interpolates the requested percentiles.

    Returns the same number of values as ``quantiles``.  Falls back to
    0.0 for each quantile if the input is empty.
    """
    if not values:
        return tuple(0.0 for _ in quantiles)

    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    total_w = sum(w for _, w in pairs)
    if total_w < 1e-12:
        return tuple(0.0 for _ in quantiles)

    # Build cumulative weight CDF
    cum_w = 0.0
    cdf: list[tuple[float, float]] = []  # (cum_fraction, value)
    for v, w in pairs:
        cum_w += w
        cdf.append((cum_w / total_w, v))

    result: list[float] = []
    for q in quantiles:
        q = max(0.0, min(1.0, q))
        # Find the value where cumulative fraction first exceeds q
        found = cdf[-1][1]  # default: max value
        for frac, v in cdf:
            if frac >= q:
                found = v
                break
        result.append(found)

    return tuple(result)


def _kelly_size_modifier(win_rate: float, expected_rr: float) -> float:
    """Half-Kelly position size modifier in [0.5, 2.0].

    Kelly fraction = win_rate − (1 − win_rate) / expected_rr
    Half-Kelly    = kelly_f × 0.5
    size_modifier = clamp(1.0 + half_kelly, 0.5, 2.0)

    Interpretation:
      win_rate=0.6, rr=2.0 → kelly=0.40 → half=0.20 → size=1.20
      win_rate=0.3, rr=1.0 → kelly=−0.40 → half=−0.20 → size=0.80
      win_rate=0.2, rr=0.5 → kelly<0 → size floored at 0.50
    """
    if expected_rr < 1e-6:
        return _SIZE_MOD_MIN
    kelly_f = win_rate - (1.0 - win_rate) / expected_rr
    half_kelly = max(_KELLY_LO, min(_KELLY_HI, kelly_f * 0.5))
    return max(_SIZE_MOD_MIN, min(_SIZE_MOD_MAX, 1.0 + half_kelly))


def _risk_appetite(
    mae_p75: float,
    mfe_mean: float,
    win_rate: float,
    confidence: float,
) -> float:
    """Risk appetite modifier in [−0.50, +0.25].

    Negative (risk-reducing) when adverse excursion is large relative to
    favorable excursion.  Positive when the expected setup geometry is
    favorable.  Scaled by confidence so thin history stays near neutral.

    Formula:
        mae_share = P75_MAE / (P75_MAE + mean_MFE)   → [0, 1]
        rr_quality = clamp(mean_MFE / P75_MAE, 0, 1) → [0, 1]

        risk_drag  = mae_share × |RISK_DRAG_MAX|  × confidence
        risk_boost = rr_quality × RISK_BOOST_MAX  × confidence
        risk_mod   = clamp(risk_boost − risk_drag, DRAG_MAX, BOOST_MAX)
    """
    denom = mae_p75 + mfe_mean
    mae_share = mae_p75 / max(denom, 1e-9)
    rr_q = min(1.0, mfe_mean / max(mae_p75, 1e-9)) if mae_p75 > 0 else 0.5

    drag = mae_share * abs(_RISK_DRAG_MAX) * confidence
    boost = rr_q * _RISK_BOOST_MAX * confidence
    return max(_RISK_DRAG_MAX, min(_RISK_BOOST_MAX, boost - drag))


def _opportunity_rank(
    win_rate: float,
    expected_rr: float,
    confidence: float,
) -> float:
    """Opportunity rank signal in [0, 1].

    Combines win rate, RR quality, and confidence:

        rr_quality = clamp(expected_rr / 2.0, 0, 1)
        rank = win_rate × 0.40 + rr_quality × 0.40 + confidence × 0.20

    High when the setup is historically profitable, has strong RR, and
    there are enough similar records to trust the estimate.  Returns a
    small positive value even with no history (structural win_rate > 0).
    """
    rr_q = min(1.0, expected_rr / 2.0)
    rank = win_rate * 0.40 + rr_q * 0.40 + confidence * 0.20
    return max(0.0, min(1.0, rank))
