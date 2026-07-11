"""QDE Brain — execution: learned execution quality engine.

``ExecutionQualityEngine`` is the learning layer that answers: given the
current market conditions and the order we're about to place, what quality
of execution should we expect — and how should that expectation modify the
trade decision?

Architecture
------------
Three distinct layers:

1. **Structural layer** — ``SlippagePredictor`` and ``FillProbabilityEngine``
   produce mathematically grounded estimates purely from current conditions.
   These are reliable even with zero historical data.

2. **Historical layer** — ``ExecutionMemory`` accumulates
   ``(ExecutionObservation → ExecutionOutcome)`` pairs as trades complete.
   L2-distance nearest-neighbour lookup finds past fills under similar
   conditions and computes a weighted quality estimate.  When history is
   thin (``history_confidence ≈ 0``), the structural layer governs; as
   fills accumulate, the historical signal progressively corrects any
   systematic bias in the structural model.

3. **Integration layer** — four downstream signals derived from the
   combined estimate:

   ``timing_recommendation``      EXECUTE_NOW / WAIT / AVOID
   ``order_type_recommendation``  MARKET / LIMIT / STOP_LIMIT
   ``conviction_modifier``        −0.25..0  (reduce conviction in bad conditions)
   ``expected_reward_modifier``   −0.30..0  (subtract execution cost from reward)

One engine per symbol
---------------------
Instantiate one ``ExecutionQualityEngine`` per watched symbol.  Execution
characteristics (typical spread, microstructure, session sensitivity) are
instrument-specific — a EURUSD fill history is not transferable to
XAUUSD.

Learning loop (per completed fill)
-----------------------------------
1. At order-submission time, call ``build_observation(conditions,
   order_type, distance_to_target_atr)`` and store the returned
   ``ExecutionObservation``.
2. When the broker confirms the fill (or cancellation), build an
   ``ExecutionOutcome`` using ``ExecutionOutcome.compute_quality()``.
3. Call ``record_outcome(observation, outcome)`` to append the pair to
   ``ExecutionMemory``.

The engine learns *what execution conditions have historically produced
good or bad fills for this specific instrument*.  "Good" is defined by
the realized fill quality relative to the structural prediction — so the
model learns to correct its own biases, not to chase an external target.

What ``context_for_inference`` provides
----------------------------------------
Exactly 16 ``exec_*`` keys merged into ``QDEInferenceInput.brain_context``:

    exec_spread_atr_ratio         — current spread as fraction of ATR
    exec_spread_risk              — 0-1 normalised spread risk
    exec_spread_cost_atr          — one-way spread cost in ATR units
    exec_expected_slippage_atr    — expected total slippage in ATR
    exec_worst_case_slippage_atr  — 90th-percentile slippage in ATR
    exec_fill_probability         — probability of getting filled
    exec_fill_quality_score       — expected fill quality (0-1)
    exec_timing_score             — composite timing signal (0-1)
    exec_timing_recommendation    — EXECUTE_NOW / WAIT / AVOID
    exec_session_quality          — current session quality (0-1)
    exec_order_type_recommendation — MARKET / LIMIT / STOP_LIMIT
    exec_conviction_modifier      — conviction adjustment (−0.25..0)
    exec_expected_reward_modifier — reward adjustment (−0.30..0)
    exec_execution_grade          — EXCELLENT / GOOD / ACCEPTABLE / POOR / REJECT
    exec_historical_quality       — learned quality at this spread/vol level
    exec_history_confidence       — confidence in historical signal (0-1)

Design rules
------------
* Per-symbol state — instantiate one engine per instrument.
* No hardcoded broker constraints or broker-specific assumptions.
* No hardcoded fill/slippage thresholds — the learning loop sets them.
* No import of ``QDEDecision`` or any execution-layer action type.
* Reuses ``OrderType``, ``TimingRecommendation``, ``ExecutionGrade``
  from ``trainer.execution.execution_types``.
* Reuses ``SlippagePredictor`` and ``FillProbabilityEngine`` from this
  package — does not reimplement their structural math.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from trainer.execution.execution_types import ExecutionGrade, OrderType, TimingRecommendation
from trainer.qde.brain.execution.fill_probability_engine import FillContext, FillProbabilityEngine
from trainer.qde.brain.execution.slippage_predictor import (
    SlippageInput,
    SlippagePredictor,
)

# ── Learning hyper-parameters ──────────────────────────────────────────────────

_DEFAULT_MAX_RECORDS = 2000  # per-symbol bounded FIFO capacity
_DEFAULT_K = 20  # nearest-neighbour top-K
_MIN_INFORMATIVE = 5  # minimum records before history has weight
_RECENCY_TAU_DAYS = 21.0  # recency decay time constant (days)
_L2_SCALE = 0.40  # L2 distance at which similarity falls to 0.5

# Observation vector dimension (must match ExecutionObservation.as_vector()).
_N_DIMS = 6

# ── Integration signal bounds ──────────────────────────────────────────────────

_CONVICTION_DRAG_MAX = -0.25
_REWARD_MOD_MAX = -0.30

# Timing score thresholds → TimingRecommendation
_TIMING_EXECUTE_NOW = 0.65
_TIMING_WAIT = 0.35

# Execution grade thresholds (matching execution_types._grade_from_score convention)
_GRADE_EXCELLENT = 0.85
_GRADE_GOOD = 0.70
_GRADE_ACCEPTABLE = 0.55
_GRADE_POOR = 0.40


# ── ExecutionConditions ───────────────────────────────────────────────────────


@dataclass
class ExecutionConditions:
    """Live market conditions at the moment an order decision is made.

    Built by the caller from market-data feeds; all fields are
    pre-normalised so the engine remains instrument-agnostic.

    ``spread_atr_ratio``
        Current bid-ask spread / ATR.  Primary driver for all estimates.

    ``atr_pct``
        ATR as a fraction of price (e.g. 0.005 = 0.5%).

    ``volatility``
        Realised rolling volatility (fractional).

    ``session_quality``
        0-1 from the session-timing engine.  1 = London-NY overlap;
        0 = closed / weekend.

    ``momentum``
        Signed −1..+1.  Positive = price trending up.

    ``microstructure_quality``
        0-1 from the MIE / liquidity map.  1 = clean order flow; 0 = stressed.
    """

    instrument: str = ""
    spread_atr_ratio: float = 0.10
    atr_pct: float = 0.005
    volatility: float = 0.01
    session_quality: float = 0.80
    momentum: float = 0.0
    microstructure_quality: float = 0.50


# ── ExecutionObservation ──────────────────────────────────────────────────────


@dataclass
class ExecutionObservation:
    """Conditions snapshot captured at order-submission time.

    The 6-dimensional ``as_vector()`` is the input to the L2-distance
    nearest-neighbour lookup in ``ExecutionQualityEngine``.

    Dimensions:
        [spread_atr_ratio, volatility, session_quality,
         atr_pct, abs(momentum), order_aggressiveness]
    """

    # Market conditions
    spread_atr_ratio: float = 0.10
    volatility: float = 0.01
    session_quality: float = 0.80
    atr_pct: float = 0.005
    abs_momentum: float = 0.0  # |momentum|, always >= 0

    # Order characteristics
    order_type: OrderType = OrderType.MARKET
    order_aggressiveness: float = 1.0
    distance_to_target_atr: float = 0.0  # 0 for market orders
    direction: int = 1  # +1 long, -1 short

    # Structural predictions at entry (retained for bias tracking)
    structural_slippage_atr: float = 0.0
    structural_fill_quality: float = 0.75

    instrument: str = ""
    timestamp: float = field(default_factory=time.time)

    def as_vector(self) -> list[float]:
        """6-dimensional condition vector for L2-distance lookup."""
        return [
            self.spread_atr_ratio,
            self.volatility,
            self.session_quality,
            self.atr_pct,
            self.abs_momentum,
            self.order_aggressiveness,
        ]

    def to_dict(self) -> dict:
        return {
            "spread_atr_ratio": round(self.spread_atr_ratio, 4),
            "volatility": round(self.volatility, 4),
            "session_quality": round(self.session_quality, 4),
            "atr_pct": round(self.atr_pct, 6),
            "abs_momentum": round(self.abs_momentum, 4),
            "order_type": self.order_type.value,
            "order_aggressiveness": round(self.order_aggressiveness, 4),
            "distance_to_target_atr": round(self.distance_to_target_atr, 4),
            "direction": self.direction,
            "structural_slippage_atr": round(self.structural_slippage_atr, 6),
            "structural_fill_quality": round(self.structural_fill_quality, 4),
            "instrument": self.instrument,
            "timestamp": round(self.timestamp, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionObservation:
        ot_str = d.get("order_type", OrderType.MARKET.value)
        try:
            ot = OrderType(ot_str)
        except ValueError:
            ot = OrderType.MARKET
        return cls(
            spread_atr_ratio=d.get("spread_atr_ratio", 0.10),
            volatility=d.get("volatility", 0.01),
            session_quality=d.get("session_quality", 0.80),
            atr_pct=d.get("atr_pct", 0.005),
            abs_momentum=d.get("abs_momentum", 0.0),
            order_type=ot,
            order_aggressiveness=d.get("order_aggressiveness", 1.0),
            distance_to_target_atr=d.get("distance_to_target_atr", 0.0),
            direction=d.get("direction", 1),
            structural_slippage_atr=d.get("structural_slippage_atr", 0.0),
            structural_fill_quality=d.get("structural_fill_quality", 0.75),
            instrument=d.get("instrument", ""),
            timestamp=d.get("timestamp", 0.0),
        )


# ── ExecutionOutcome ──────────────────────────────────────────────────────────


@dataclass
class ExecutionOutcome:
    """Realized fill outcome, recorded after the broker confirms the fill.

    ``actual_slippage_atr``
        Realized slippage vs. the theoretical mid-price at entry,
        expressed in ATR units.  Positive = adverse (worse than mid);
        negative = favorable (filled better than mid).

    ``fill_quality_score``
        0-1 overall fill quality as assessed by the caller:
          1.0 = filled at better-than-expected price
          0.5 = filled at roughly expected price
          0.0 = severe slippage or no fill

    ``was_filled``
        Whether the order received any fill.  False for missed limit
        orders or rejected orders.

    ``time_to_fill_seconds``
        Elapsed seconds from order submission to fill confirmation.
        0.0 for market orders (typically sub-second).

    ``outcome_quality``
        0-1 primary learning target.  Compute with
        ``ExecutionOutcome.compute_quality()`` for consistent normalisation.
    """

    actual_slippage_atr: float = 0.0
    fill_quality_score: float = 0.75
    was_filled: bool = True
    time_to_fill_seconds: float = 0.0
    outcome_quality: float = 0.5  # 0-1, primary learning target

    def to_dict(self) -> dict:
        return {
            "actual_slippage_atr": round(self.actual_slippage_atr, 6),
            "fill_quality_score": round(self.fill_quality_score, 4),
            "was_filled": self.was_filled,
            "time_to_fill_seconds": round(self.time_to_fill_seconds, 1),
            "outcome_quality": round(self.outcome_quality, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionOutcome:
        return cls(
            actual_slippage_atr=d.get("actual_slippage_atr", 0.0),
            fill_quality_score=d.get("fill_quality_score", 0.75),
            was_filled=d.get("was_filled", True),
            time_to_fill_seconds=d.get("time_to_fill_seconds", 0.0),
            outcome_quality=d.get("outcome_quality", 0.5),
        )

    @staticmethod
    def compute_quality(
        actual_slippage_atr: float,
        fill_quality_score: float,
        was_filled: bool,
        structural_slippage_atr: float = 0.0,
    ) -> float:
        """Derive ``outcome_quality`` (0-1) from raw fill results.

        Design
        ------
        * No fill → 0.30 (bad but not catastrophic — missed limit is not
          the same as catastrophic slippage; the position size was zero).
        * Filled better than expected (negative slippage, or quality >
          structural estimate) → quality approaches 1.0.
        * Severe adverse slippage → quality falls toward 0.0.

        The normalisation is relative to the structural prediction so the
        engine learns to correct its own biases — not to hit an external
        fill-quality target.
        """
        if not was_filled:
            return 0.30

        # Slippage beyond what was predicted (bias = actual - structural)
        excess_slippage = actual_slippage_atr - structural_slippage_atr
        # Positive excess = worse than predicted → reduces quality
        # Negative excess = better than predicted → improves quality
        slip_factor = max(-0.4, min(0.4, -excess_slippage * 2.0))
        quality = fill_quality_score + slip_factor
        return max(0.0, min(1.0, quality))


# ── ExecutionRecord ───────────────────────────────────────────────────────────


@dataclass
class ExecutionRecord:
    """One completed (observation → outcome) fill record."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    observation: ExecutionObservation = field(default_factory=ExecutionObservation)
    outcome: ExecutionOutcome = field(default_factory=ExecutionOutcome)
    created_at: float = field(default_factory=time.time)

    def recency_weight(self, now: float, tau_days: float = _RECENCY_TAU_DAYS) -> float:
        """Exponential recency weight: 1.0 when fresh, decays over tau_days."""
        elapsed_days = max(0.0, now - self.created_at) / 86400.0
        return 1.0 / (1.0 + elapsed_days / max(tau_days, 1e-9))

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "observation": self.observation.to_dict(),
            "outcome": self.outcome.to_dict(),
            "created_at": round(self.created_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionRecord:
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())[:12]),
            observation=ExecutionObservation.from_dict(d.get("observation", {})),
            outcome=ExecutionOutcome.from_dict(d.get("outcome", {})),
            created_at=d.get("created_at", 0.0),
        )


# ── ExecutionMemory ───────────────────────────────────────────────────────────


class ExecutionMemory:
    """Per-symbol bounded rolling store of execution outcome records.

    Mirrors ``ExposureMemory`` and ``ProjectionMemory`` in structure:
    bounded FIFO with snapshot persistence.  No business logic — that
    lives in ``ExecutionQualityEngine``.
    """

    def __init__(self, max_records: int = _DEFAULT_MAX_RECORDS) -> None:
        self._max_records = max(1, max_records)
        self._store: deque[ExecutionRecord] = deque(maxlen=self._max_records)

    def record(
        self,
        observation: ExecutionObservation,
        outcome: ExecutionOutcome,
        now: float | None = None,
    ) -> ExecutionRecord:
        """Append one fill record.  Oldest is evicted when capacity is full."""
        now = now if now is not None else time.time()
        rec = ExecutionRecord(observation=observation, outcome=outcome, created_at=now)
        self._store.append(rec)
        return rec

    def records(self) -> list[ExecutionRecord]:
        """All current records, oldest first."""
        return list(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def to_snapshot(self) -> dict:
        return {
            "max_records": self._max_records,
            "records": [r.to_dict() for r in self._store],
        }

    @classmethod
    def load_snapshot(cls, snapshot: dict) -> ExecutionMemory:
        mem = cls(max_records=snapshot.get("max_records", _DEFAULT_MAX_RECORDS))
        for d in snapshot.get("records", []):
            try:
                mem._store.append(ExecutionRecord.from_dict(d))
            except Exception:
                pass
        return mem


# ── ExecutionQualitySignal ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionQualitySignal:
    """Complete execution quality estimate for one order scenario.

    Produced by ``ExecutionQualityEngine.score()``.

    Component signals
    -----------------
    ``spread_risk``              0-1 normalised spread risk.
    ``spread_cost_atr``          One-way spread cost in ATR.
    ``expected_slippage_atr``    Expected total slippage in ATR.
    ``worst_case_slippage_atr``  90th-percentile slippage in ATR.
    ``fill_probability``         Probability of getting filled (0-1).
    ``fill_quality_score``       Expected fill quality (0-1).
    ``timing_score``             Composite timing signal (0-1).

    Integration signals
    -------------------
    ``timing_recommendation``       EXECUTE_NOW / WAIT / AVOID
    ``order_type_recommendation``   MARKET / LIMIT / STOP_LIMIT
    ``conviction_modifier``         −0.25..0
    ``expected_reward_modifier``    −0.30..0
    ``execution_grade``             EXCELLENT / GOOD / ACCEPTABLE / POOR / REJECT

    Learning signals
    ----------------
    ``historical_quality``   Learned execution quality at these conditions (0-1).
    ``history_confidence``   Confidence in the historical signal (0-1).
    """

    # ── Conditions (echoed for convenience) ──────────────────────────────
    spread_atr_ratio: float = 0.10
    session_quality: float = 0.80

    # ── Structural components ─────────────────────────────────────────────
    spread_risk: float = 0.0
    spread_cost_atr: float = 0.0
    expected_slippage_atr: float = 0.0
    worst_case_slippage_atr: float = 0.0
    fill_probability: float = 1.0
    fill_quality_score: float = 0.75
    timing_score: float = 0.5

    # ── Integration signals ───────────────────────────────────────────────
    timing_recommendation: TimingRecommendation = TimingRecommendation.WAIT
    order_type_recommendation: OrderType = OrderType.MARKET
    conviction_modifier: float = 0.0
    expected_reward_modifier: float = 0.0
    execution_grade: ExecutionGrade = ExecutionGrade.ACCEPTABLE

    # ── Learning signals ──────────────────────────────────────────────────
    historical_quality: float = 0.5
    history_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "spread_atr_ratio": round(self.spread_atr_ratio, 4),
            "session_quality": round(self.session_quality, 4),
            "spread_risk": round(self.spread_risk, 4),
            "spread_cost_atr": round(self.spread_cost_atr, 6),
            "expected_slippage_atr": round(self.expected_slippage_atr, 6),
            "worst_case_slippage_atr": round(self.worst_case_slippage_atr, 6),
            "fill_probability": round(self.fill_probability, 4),
            "fill_quality_score": round(self.fill_quality_score, 4),
            "timing_score": round(self.timing_score, 4),
            "timing_recommendation": self.timing_recommendation.value,
            "order_type_recommendation": self.order_type_recommendation.value,
            "conviction_modifier": round(self.conviction_modifier, 4),
            "expected_reward_modifier": round(self.expected_reward_modifier, 4),
            "execution_grade": self.execution_grade.value,
            "historical_quality": round(self.historical_quality, 4),
            "history_confidence": round(self.history_confidence, 4),
        }


# ── ExecutionQualityEngine ────────────────────────────────────────────────────


class ExecutionQualityEngine:
    """Learned per-symbol execution quality engine.

    Wraps ``SlippagePredictor`` and ``FillProbabilityEngine`` (stateless
    structural layer) with ``ExecutionMemory`` (per-symbol learning store).

    State
    -----
    One ``ExecutionMemory`` per instrument, bounded FIFO.  Instantiate one
    engine per watched symbol and persist the memory across sessions with
    ``memory_snapshot()`` / ``load_memory()``.

    Typical call sequence (per tick)
    ---------------------------------
    ::

        conditions = ExecutionConditions(
            instrument="EURUSD",
            spread_atr_ratio=current_spread / atr,
            atr_pct=atr / price,
            volatility=rolling_vol,
            session_quality=session_engine.quality(),
            momentum=market_momentum,
            microstructure_quality=liq_engine.microstructure_score(),
        )
        ctx = engine.context_for_inference(
            conditions, OrderType.MARKET, distance_to_target_atr=0.0
        )
        # Merge ctx into brain_context for the inference pass

    Learning loop (per completed fill)
    ------------------------------------
    ::

        obs = engine.build_observation(conditions, order_type, distance)
        # ... store obs until fill confirmation ...
        outcome = ExecutionOutcome(
            actual_slippage_atr=measured_slippage / atr,
            fill_quality_score=measured_quality,
            was_filled=True,
            time_to_fill_seconds=elapsed,
            outcome_quality=ExecutionOutcome.compute_quality(
                measured_slippage / atr, measured_quality, True, obs.structural_slippage_atr
            ),
        )
        engine.record_outcome(obs, outcome)
    """

    def __init__(
        self,
        max_records: int = _DEFAULT_MAX_RECORDS,
        k: int = _DEFAULT_K,
        recency_tau: float = _RECENCY_TAU_DAYS,
    ) -> None:
        self._memory = ExecutionMemory(max_records=max_records)
        self._slip_pred = SlippagePredictor()
        self._fill_engine = FillProbabilityEngine()
        self._k = k
        self._tau = recency_tau

    # ── Learning ──────────────────────────────────────────────────────────

    def record_outcome(
        self,
        observation: ExecutionObservation,
        outcome: ExecutionOutcome,
        now: float | None = None,
    ) -> ExecutionRecord:
        """Append one completed fill record to the learning store.

        Call once per confirmed fill (or cancellation) for the instrument
        this engine is tracking.
        """
        return self._memory.record(
            observation,
            outcome,
            now=now if now is not None else time.time(),
        )

    # ── Primary signal computation ────────────────────────────────────────

    def score(
        self,
        conditions: ExecutionConditions,
        order_type: OrderType = OrderType.MARKET,
        distance_to_target_atr: float = 0.0,
        direction: int = 1,
        now: float | None = None,
    ) -> ExecutionQualitySignal:
        """Compute the full execution quality signal for one order scenario.

        Parameters
        ----------
        conditions:
            Live market conditions (spread, vol, session, etc.).
        order_type:
            The order type being evaluated.
        distance_to_target_atr:
            For limit/stop orders, the distance from current price to the
            order target, in ATR units.  0.0 for MARKET orders.
        direction:
            +1 long, −1 short.  Used for adverse-momentum calculation.
        """
        now = now if now is not None else time.time()

        # ── 1. Structural layer ───────────────────────────────────────────
        slip_inp = SlippageInput(
            spread_atr_ratio=conditions.spread_atr_ratio,
            atr_pct=conditions.atr_pct,
            volatility=conditions.volatility,
            session_quality=conditions.session_quality,
            momentum=conditions.momentum,
            order_type=order_type,
            microstructure_quality=conditions.microstructure_quality,
        )
        slip = self._slip_pred.estimate(slip_inp)

        fill_ctx = FillContext(
            order_type=order_type,
            distance_to_target_atr=distance_to_target_atr,
            spread_atr_ratio=conditions.spread_atr_ratio,
            volatility=conditions.volatility,
            session_quality=conditions.session_quality,
            momentum=conditions.momentum,
            direction=direction,
        )
        fill_sig = self._fill_engine.estimate(fill_ctx)

        # ── 2. Composite timing score ─────────────────────────────────────
        slippage_risk = min(1.0, slip.expected_slippage_atr / max(0.1, slip.spread_cost_atr + 0.5))
        timing_struct = (
            (1.0 - slip.spread_risk) * 0.40
            + fill_sig.fill_probability * 0.30
            + conditions.session_quality * 0.20
            + (1.0 - min(1.0, slippage_risk)) * 0.10
        )

        # ── 3. Historical layer ───────────────────────────────────────────
        hist_quality, hist_conf = self._historical_quality(conditions, order_type, now)

        # Historical adjustment: bad fills reduce timing; good fills boost it.
        # hist_quality < 0.5 → drag is negative → timing decreases.
        # hist_quality > 0.5 → drag is positive → timing increases.
        hist_drag = (hist_quality - 0.5) * 2.0 * hist_conf  # −conf..+conf
        timing_score = max(0.0, min(1.0, timing_struct + hist_drag * timing_struct))

        # ── 4. Integration signals ────────────────────────────────────────
        timing_rec = _timing_recommendation(timing_score)
        order_type_rec = self._fill_engine.recommend_order_type(
            conditions.spread_atr_ratio,
            slip.expected_slippage_atr,
            timing_budget_seconds=300.0,
        )
        conviction_mod = round(
            max(_CONVICTION_DRAG_MAX, min(0.0, _CONVICTION_DRAG_MAX * (1.0 - timing_score))), 4
        )

        # Execution cost as a fraction of "typical move" — proportional to
        # spread cost + expected slippage vs. the worst-case slippage
        exec_cost_score = min(
            1.0,
            (slip.spread_cost_atr + slip.expected_slippage_atr)
            / max(0.01, slip.worst_case_slippage_atr),
        )
        reward_mod = round(max(_REWARD_MOD_MAX, min(0.0, _REWARD_MOD_MAX * exec_cost_score)), 4)

        grade = _grade_from_timing(timing_score)

        return ExecutionQualitySignal(
            spread_atr_ratio=round(conditions.spread_atr_ratio, 4),
            session_quality=round(conditions.session_quality, 4),
            spread_risk=round(slip.spread_risk, 4),
            spread_cost_atr=round(slip.spread_cost_atr, 6),
            expected_slippage_atr=round(slip.expected_slippage_atr, 6),
            worst_case_slippage_atr=round(slip.worst_case_slippage_atr, 6),
            fill_probability=round(fill_sig.fill_probability, 4),
            fill_quality_score=round(fill_sig.expected_fill_quality, 4),
            timing_score=round(timing_score, 4),
            timing_recommendation=timing_rec,
            order_type_recommendation=order_type_rec,
            conviction_modifier=conviction_mod,
            expected_reward_modifier=reward_mod,
            execution_grade=grade,
            historical_quality=round(hist_quality, 4),
            history_confidence=round(hist_conf, 4),
        )

    def context_for_inference(
        self,
        conditions: ExecutionConditions,
        order_type: OrderType = OrderType.MARKET,
        distance_to_target_atr: float = 0.0,
        direction: int = 1,
        now: float | None = None,
    ) -> dict:
        """Return exactly 16 ``exec_*`` keys for ``QDEInferenceInput.brain_context``.

        Keys
        ----
        exec_spread_atr_ratio          — current spread as fraction of ATR
        exec_spread_risk               — 0-1 normalised spread risk
        exec_spread_cost_atr           — one-way spread cost in ATR units
        exec_expected_slippage_atr     — expected total slippage in ATR
        exec_worst_case_slippage_atr   — 90th-pct slippage in ATR
        exec_fill_probability          — fill probability (0-1)
        exec_fill_quality_score        — expected fill quality (0-1)
        exec_timing_score              — composite timing signal (0-1)
        exec_timing_recommendation     — EXECUTE_NOW / WAIT / AVOID
        exec_session_quality           — current session quality (0-1)
        exec_order_type_recommendation — MARKET / LIMIT / STOP_LIMIT
        exec_conviction_modifier       — conviction adjustment (−0.25..0)
        exec_expected_reward_modifier  — reward adjustment (−0.30..0)
        exec_execution_grade           — EXCELLENT…REJECT
        exec_historical_quality        — learned fill quality at this spread/vol
        exec_history_confidence        — confidence in historical signal (0-1)
        """
        sig = self.score(conditions, order_type, distance_to_target_atr, direction, now)
        return {
            "exec_spread_atr_ratio": round(sig.spread_atr_ratio, 4),
            "exec_spread_risk": round(sig.spread_risk, 4),
            "exec_spread_cost_atr": round(sig.spread_cost_atr, 6),
            "exec_expected_slippage_atr": round(sig.expected_slippage_atr, 6),
            "exec_worst_case_slippage_atr": round(sig.worst_case_slippage_atr, 6),
            "exec_fill_probability": round(sig.fill_probability, 4),
            "exec_fill_quality_score": round(sig.fill_quality_score, 4),
            "exec_timing_score": round(sig.timing_score, 4),
            "exec_timing_recommendation": sig.timing_recommendation.value,
            "exec_session_quality": round(sig.session_quality, 4),
            "exec_order_type_recommendation": sig.order_type_recommendation.value,
            "exec_conviction_modifier": round(sig.conviction_modifier, 4),
            "exec_expected_reward_modifier": round(sig.expected_reward_modifier, 4),
            "exec_execution_grade": sig.execution_grade.value,
            "exec_historical_quality": round(sig.historical_quality, 4),
            "exec_history_confidence": round(sig.history_confidence, 4),
        }

    # ── Convenience ──────────────────────────────────────────────────────────

    def build_observation(
        self,
        conditions: ExecutionConditions,
        order_type: OrderType = OrderType.MARKET,
        distance_to_target_atr: float = 0.0,
        direction: int = 1,
    ) -> ExecutionObservation:
        """Build an ``ExecutionObservation`` to store at order-submission time.

        Store the returned object; pass it back to ``record_outcome()``
        after the fill confirmation arrives.  The observation captures the
        *conditions at entry* — not the exit conditions — which is what the
        learning loop needs to improve future predictions.
        """
        slip_inp = SlippageInput(
            spread_atr_ratio=conditions.spread_atr_ratio,
            atr_pct=conditions.atr_pct,
            volatility=conditions.volatility,
            session_quality=conditions.session_quality,
            momentum=conditions.momentum,
            order_type=order_type,
            microstructure_quality=conditions.microstructure_quality,
        )
        slip = self._slip_pred.estimate(slip_inp)

        fill_ctx = FillContext(
            order_type=order_type,
            distance_to_target_atr=distance_to_target_atr,
            spread_atr_ratio=conditions.spread_atr_ratio,
            volatility=conditions.volatility,
            session_quality=conditions.session_quality,
            momentum=conditions.momentum,
            direction=direction,
        )
        fill_sig = self._fill_engine.estimate(fill_ctx)

        return ExecutionObservation(
            spread_atr_ratio=conditions.spread_atr_ratio,
            volatility=conditions.volatility,
            session_quality=conditions.session_quality,
            atr_pct=conditions.atr_pct,
            abs_momentum=abs(conditions.momentum),
            order_type=order_type,
            order_aggressiveness=slip.order_aggressiveness,
            distance_to_target_atr=distance_to_target_atr,
            direction=direction,
            structural_slippage_atr=slip.expected_slippage_atr,
            structural_fill_quality=fill_sig.expected_fill_quality,
            instrument=conditions.instrument,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def memory_snapshot(self) -> dict:
        """Serialise the learning store for cross-session persistence."""
        return self._memory.to_snapshot()

    def load_memory(self, snapshot: dict) -> None:
        """Restore a previously serialised learning store."""
        self._memory = ExecutionMemory.load_snapshot(snapshot)

    # ── Internal: historical quality estimation ────────────────────────────────

    def _historical_quality(
        self,
        conditions: ExecutionConditions,
        order_type: OrderType,
        now: float,
    ) -> tuple[float, float]:
        """Nearest-neighbour quality estimate from execution history.

        Returns (quality: 0-1, confidence: 0-1).
        0.5 quality = neutral / no information.
        0.0 confidence = insufficient history.
        """
        records = self._memory.records()
        if not records:
            return 0.5, 0.0

        slip_inp = SlippageInput(
            spread_atr_ratio=conditions.spread_atr_ratio,
            atr_pct=conditions.atr_pct,
            volatility=conditions.volatility,
            session_quality=conditions.session_quality,
            momentum=conditions.momentum,
            order_type=order_type,
            microstructure_quality=conditions.microstructure_quality,
        )
        agg = self._slip_pred.aggressiveness(order_type)
        query = [
            conditions.spread_atr_ratio,
            conditions.volatility,
            conditions.session_quality,
            conditions.atr_pct,
            abs(conditions.momentum),
            agg,
        ]

        scored: list[tuple[float, ExecutionRecord]] = []
        for rec in records:
            hist_vec = rec.observation.as_vector()
            dist = _l2_distance(query, hist_vec)
            sim = _similarity(dist, _L2_SCALE)
            recency = rec.recency_weight(now, tau_days=self._tau)
            w = sim * recency
            if w > 1e-10:
                scored.append((w, rec))

        if not scored:
            return 0.5, 0.0

        scored.sort(key=lambda t: t[0], reverse=True)
        top_k = scored[: self._k]
        total_w = sum(w for w, _ in top_k)
        if total_w < 1e-12:
            return 0.5, 0.0

        weighted_q = sum(w * r.outcome.outcome_quality for w, r in top_k) / total_w
        confidence = min(1.0, len(top_k) / max(_MIN_INFORMATIVE, 1))
        return max(0.0, min(1.0, weighted_q)), round(confidence, 4)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _l2_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _similarity(dist: float, scale: float) -> float:
    """Soft inverse-distance → [0, 1] similarity."""
    return 1.0 / (1.0 + dist / max(scale, 1e-9))


def _timing_recommendation(timing_score: float) -> TimingRecommendation:
    if timing_score >= _TIMING_EXECUTE_NOW:
        return TimingRecommendation.EXECUTE_NOW
    if timing_score >= _TIMING_WAIT:
        return TimingRecommendation.WAIT
    return TimingRecommendation.AVOID


def _grade_from_timing(timing_score: float) -> ExecutionGrade:
    if timing_score >= _GRADE_EXCELLENT:
        return ExecutionGrade.EXCELLENT
    if timing_score >= _GRADE_GOOD:
        return ExecutionGrade.GOOD
    if timing_score >= _GRADE_ACCEPTABLE:
        return ExecutionGrade.ACCEPTABLE
    if timing_score >= _GRADE_POOR:
        return ExecutionGrade.POOR
    return ExecutionGrade.REJECT
