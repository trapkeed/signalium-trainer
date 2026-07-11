"""QDE Brain — projection: expected move, reward, and MAE estimation.

``ExpectedMoveEngine`` is the top-level orchestrator for the projection
package.  It owns one ``ProjectionMemory`` and one
``ExpectedDurationEngine`` for all symbols and exposes four integration
points that influence the four named concerns:

1. **``context_for_inference``** — ``proj_*`` keys for
   ``QDEInferenceInput.brain_context`` (trade selection context).
2. **``opportunity_rank_signal``** — 0-1 float that lifts or depresses an
   opportunity's position in the queue (trade selection).
3. **``position_size_modifier``** — 0.5..2.0 multiplier relative to the
   baseline size (position sizing).
4. **``risk_appetite_modifier``** — −0.50..+0.25 float for risk appetite.

Learning loop
-------------
``record_outcome(instrument, direction, conditions, outcome)`` is the
single write entry point.  Callers invoke it once per completed trade
(or simulated replay) with the ground-truth ``ProjectionOutcome`` that
they build from ``TradeOutcomePath.from_outcome_path()``.

Estimation algorithm
--------------------
For each metric (move, MAE, MFE, reward):

1. Retrieve all historical ``ProjectionRecord`` entries for ``instrument``
   from ``ProjectionMemory``.
2. Score each record: ``weight = cosine_similarity(current_vec, hist_vec)
   × recency_factor``.
3. Weighted mean of the target metric across the top-K highest-weight
   records.

No hardcoded pip / ATR targets.  ``reward_sufficiency_score`` is a
*relative* 0-1 percentile rank of the expected reward within the
instrument's historical distribution — there is no minimum absolute
threshold.

Design rules
------------
* Does not import ``QDEDecision`` or any execution-layer type.
* Does not re-derive conviction, uncertainty, or regime classification —
  those are owned upstream and passed in via ``ProjectionConditions``.
* All magnitude outputs are in ATR multiples (instrument-agnostic).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.projection.expected_duration_engine import (
    DurationProjection,
    ExpectedDurationEngine,
    _cosine_sim,
    _percentile_rank,
    _weighted_mean,
)
from trainer.qde.brain.projection.projection_memory import (
    DEFAULT_RECENCY_TAU_DAYS,
    ProjectionConditions,
    ProjectionMemory,
    ProjectionOutcome,
    ProjectionRecord,
)

# ── Estimation hyper-parameters (no pip targets anywhere) ─────────────────────

_DEFAULT_K = 30  # top-K records for weighted estimation
_MIN_INFORMATIVE = 5  # confidence floor sample count
_DEFAULT_WINDOW = 500  # max records pulled from memory per call

# ── Integration-signal bounds ─────────────────────────────────────────────────

_POS_SIZE_MIN = 0.50  # smallest allowed size modifier
_POS_SIZE_MAX = 2.00  # largest allowed size modifier
_RISK_DRAG_MAX = -0.50  # most negative risk appetite modifier
_RISK_BOOST_MAX = 0.25  # most positive risk appetite modifier


# ── ExpectedProjection ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpectedProjection:
    """Forward estimates for one opportunity, all in ATR multiples.

    ``reward_sufficiency_score``
        Percentile rank of ``expected_reward_atr`` within the instrument's
        historical reward distribution, blended with confidence.  0 = this
        setup historically ranks in the bottom of all outcomes seen;
        1 = this setup ranks near the top.  No absolute reward floor is
        embedded — "sufficient" is always relative to experience.

    ``reward_risk_ratio``
        expected_reward_atr / expected_mae_atr.  Unbounded above zero;
        values < 1.0 mean MAE is historically larger than reward.

    ``opportunity_rank_signal``
        0-1 composite of reward sufficiency and confidence.  Higher means
        this opportunity looks historically more rewarding than average.

    ``position_size_modifier``
        Suggested size multiplier (0.5..2.0) relative to baseline.  High
        reward sufficiency + high confidence → larger size.

    ``risk_appetite_modifier``
        −0.50..+0.25 signal.  High expected MAE or vol → negative;
        high expected reward and low vol → positive.
    """

    instrument: str = ""
    direction: int = 0  # +1 long, −1 short

    # ── Magnitude estimates (ATR multiples) ────────────────────────────────
    expected_move_atr: float = 0.0  # |expected price move| / ATR
    expected_return_pct: float = 0.0  # signed expected return fraction
    expected_reward_atr: float = 0.0  # signed reward in ATR units
    expected_mae_atr: float = 0.0  # expected adverse excursion / ATR
    expected_mfe_atr: float = 0.0  # expected favorable excursion / ATR
    reward_risk_ratio: float = 0.0  # expected_reward_atr / expected_mae_atr

    # ── Duration / volatility ──────────────────────────────────────────────
    expected_duration_seconds: float = 0.0
    expected_volatility: float = 0.0
    volatility_percentile: float = 0.5

    # ── Learned quality signals ────────────────────────────────────────────
    reward_sufficiency_score: float = 0.5
    reward_percentile: float = 0.5  # within this instrument's history
    confidence: float = 0.0
    sample_count: int = 0

    # ── Pre-computed integration signals ──────────────────────────────────
    opportunity_rank_signal: float = 0.0
    position_size_modifier: float = 1.0
    risk_appetite_modifier: float = 0.0

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "expected_move_atr": round(self.expected_move_atr, 4),
            "expected_return_pct": round(self.expected_return_pct, 6),
            "expected_reward_atr": round(self.expected_reward_atr, 4),
            "expected_mae_atr": round(self.expected_mae_atr, 4),
            "expected_mfe_atr": round(self.expected_mfe_atr, 4),
            "reward_risk_ratio": round(self.reward_risk_ratio, 4),
            "expected_duration_seconds": round(self.expected_duration_seconds, 1),
            "expected_volatility": round(self.expected_volatility, 8),
            "volatility_percentile": round(self.volatility_percentile, 4),
            "reward_sufficiency_score": round(self.reward_sufficiency_score, 4),
            "reward_percentile": round(self.reward_percentile, 4),
            "confidence": round(self.confidence, 4),
            "sample_count": self.sample_count,
            "opportunity_rank_signal": round(self.opportunity_rank_signal, 4),
            "position_size_modifier": round(self.position_size_modifier, 4),
            "risk_appetite_modifier": round(self.risk_appetite_modifier, 4),
        }


# Default returned for unknown instruments
_DEFAULT_PROJECTION = ExpectedProjection(position_size_modifier=1.0)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _score_records(
    current_vec: list[float],
    records: list[ProjectionRecord],
    recency_tau_days: float,
    now: float,
) -> list[tuple[float, ProjectionRecord]]:
    """Return (weight, record) pairs sorted highest-weight first.

    Weight = cosine_similarity(current, historical) × recency_factor.
    Records with weight ≤ 1e-10 are excluded.  If all similarities are
    zero (e.g. identical zero vectors), falls back to recency-only
    weighting so we never return empty when records exist.
    """
    scored: list[tuple[float, ProjectionRecord]] = []
    for rec in records:
        sim = max(0.0, _cosine_sim(current_vec, rec.conditions.as_vector()))
        recency = rec.recency_weight(now, recency_tau_days)
        w = sim * recency
        if w > 1e-10:
            scored.append((w, rec))

    if not scored:
        # Recency-only fallback
        for rec in records:
            recency = rec.recency_weight(now, recency_tau_days)
            if recency > 1e-10:
                scored.append((recency, rec))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def _compute_signals(
    expected_reward_atr: float,
    expected_mae_atr: float,
    expected_volatility: float,
    volatility_percentile: float,
    reward_percentile: float,
    confidence: float,
) -> tuple[float, float, float, float]:
    """Return (reward_sufficiency, opportunity_rank, position_size, risk_appetite).

    All computed as pure functions of the estimates — no hardcoded targets.
    """
    # Reward sufficiency: blend of percentile rank and confidence.
    # High confidence + high percentile → close to 1.0.
    # Low confidence → regress toward 0.5 (neutral).
    sufficiency = reward_percentile * confidence + 0.5 * (1.0 - confidence)

    # Opportunity rank: sufficiency × confidence, with a floor so even
    # thin history gives some signal (not zero).
    opp_rank = min(1.0, sufficiency * (0.3 + 0.7 * confidence))

    # Position size modifier: [0.5, 2.0]
    # sufficiency=1, confidence=1 → 2.0; sufficiency=0, confidence=1 → 0.5
    size_mod = _POS_SIZE_MIN + (_POS_SIZE_MAX - _POS_SIZE_MIN) * sufficiency * confidence
    size_mod = max(_POS_SIZE_MIN, min(_POS_SIZE_MAX, size_mod))

    # Risk appetite: vol percentile drives negative; reward sufficiency drives positive
    # Both scaled by confidence so thin history stays near zero.
    risk_drag = volatility_percentile * abs(_RISK_DRAG_MAX) * confidence
    risk_boost = sufficiency * _RISK_BOOST_MAX * confidence
    risk_mod = risk_boost - risk_drag
    risk_mod = max(_RISK_DRAG_MAX, min(_RISK_BOOST_MAX, risk_mod))

    return sufficiency, opp_rank, size_mod, risk_mod


# ── ExpectedMoveEngine ────────────────────────────────────────────────────────


class ExpectedMoveEngine:
    """Top-level projection orchestrator for all symbols.

    Holds one ``ProjectionMemory`` and one ``ExpectedDurationEngine``.
    All five estimates (move, MAE, MFE, reward, duration/vol) are
    computed in a single ``project()`` call that both engines contribute
    to; callers then read the pre-computed integration signals from the
    returned ``ExpectedProjection``.
    """

    def __init__(
        self,
        max_records: int = 1000,
        k: int = _DEFAULT_K,
        recency_tau_days: float = DEFAULT_RECENCY_TAU_DAYS,
        min_samples: int = _MIN_INFORMATIVE,
    ) -> None:
        self._memory = ProjectionMemory(max_records=max_records)
        self._duration = ExpectedDurationEngine()
        self._k = k
        self._tau = recency_tau_days
        self._min = min_samples

        # Per-call cache: (instrument, direction, vec_tuple) → ExpectedProjection.
        # Keyed by conditions vector so different conditions for the same symbol
        # are never conflated.  Invalidated when record_outcome is called.
        self._cache: dict[tuple, ExpectedProjection] = {}
        self._dirty: set[str] = set()  # symbols whose cache entries are stale

    # ── Learning ──────────────────────────────────────────────────────────

    def record_outcome(
        self,
        instrument: str,
        direction: int,
        conditions: ProjectionConditions,
        outcome: ProjectionOutcome,
        now: float | None = None,
    ) -> ProjectionRecord:
        """Append one completed (conditions → outcome) observation.

        Called once per completed trade.  ``conditions`` should be the
        snapshot taken at the time the opportunity was identified, not
        the exit conditions.
        """
        now = now if now is not None else time.time()
        rec = self._memory.record(instrument, direction, conditions, outcome, now)
        self._dirty.add(instrument)
        logger.debug(
            f"[EME] {instrument} dir={direction:+d} "
            f"reward={outcome.reward_atr:.3f} ATR, "
            f"mae={outcome.mae_atr:.3f} ATR recorded."
        )
        return rec

    # ── Estimation ────────────────────────────────────────────────────────

    def project(
        self,
        instrument: str,
        conditions: ProjectionConditions,
        direction: int,
        now: float | None = None,
    ) -> ExpectedProjection:
        """Compute all five estimates for one opportunity.

        Returns a cached result when called again with identical
        ``(instrument, direction, conditions_vector)`` within the same
        tick and before any new ``record_outcome`` calls.  The cache is
        fully invalidated for a symbol whenever ``record_outcome`` is
        called for it.
        """
        now = now if now is not None else time.time()

        cache_key = (instrument, direction, tuple(conditions.as_vector()))

        if instrument not in self._dirty:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        result = self._compute(instrument, conditions, direction, now)
        # Evict all stale entries for this symbol before storing the new one
        if instrument in self._dirty:
            stale = [k for k in self._cache if k[0] == instrument]
            for k in stale:
                del self._cache[k]
            self._dirty.discard(instrument)
        self._cache[cache_key] = result
        return result

    def _compute(
        self,
        instrument: str,
        conditions: ProjectionConditions,
        direction: int,
        now: float,
    ) -> ExpectedProjection:
        records = self._memory.records(instrument)
        if not records:
            return ExpectedProjection(
                instrument=instrument,
                direction=direction,
                position_size_modifier=1.0,
            )

        current_vec = conditions.as_vector()
        scored = _score_records(current_vec, records, self._tau, now)
        top_k = scored[: self._k]

        if not top_k:
            return ExpectedProjection(
                instrument=instrument,
                direction=direction,
                position_size_modifier=1.0,
            )

        weights = [t[0] for t in top_k]
        recs = [t[1] for t in top_k]

        # ── Weighted metric estimates ──────────────────────────────────────
        exp_move = _weighted_mean([r.outcome.realized_move_pct for r in recs], weights)
        exp_ret = _weighted_mean([r.outcome.realized_return_pct for r in recs], weights)

        # For reward/MAE/MFE: weight more heavily toward same-direction records
        dir_weights = [
            w * (1.5 if r.direction == direction else 0.5) for w, r in zip(weights, recs)
        ]
        exp_reward = _weighted_mean([r.outcome.reward_atr for r in recs], dir_weights)
        exp_mae = _weighted_mean([r.outcome.mae_atr for r in recs], dir_weights)
        exp_mfe = _weighted_mean([r.outcome.mfe_atr for r in recs], dir_weights)

        rrr = exp_reward / max(exp_mae, 1e-6)

        # ── Percentile ranks (unweighted, vs all records) ──────────────────
        all_rewards = [r.outcome.reward_atr for r in records]
        reward_pct = _percentile_rank(exp_reward, all_rewards)

        # ── Duration + volatility from duration engine ─────────────────────
        dur_proj: DurationProjection = self._duration.project(
            instrument=instrument,
            conditions=conditions,
            memory=self._memory,
            direction=direction,
            k=self._k,
            recency_tau_days=self._tau,
            min_samples=self._min,
            now=now,
        )

        n_used = len(top_k)
        confidence = min(1.0, n_used / max(self._min, 1))

        # ── Integration signals ────────────────────────────────────────────
        sufficiency, opp_rank, size_mod, risk_mod = _compute_signals(
            expected_reward_atr=exp_reward,
            expected_mae_atr=exp_mae,
            expected_volatility=dur_proj.expected_volatility,
            volatility_percentile=dur_proj.volatility_percentile,
            reward_percentile=reward_pct,
            confidence=confidence,
        )

        return ExpectedProjection(
            instrument=instrument,
            direction=direction,
            expected_move_atr=max(0.0, exp_move),
            expected_return_pct=exp_ret,
            expected_reward_atr=exp_reward,
            expected_mae_atr=max(0.0, exp_mae),
            expected_mfe_atr=max(0.0, exp_mfe),
            reward_risk_ratio=rrr,
            expected_duration_seconds=dur_proj.expected_duration_seconds,
            expected_volatility=dur_proj.expected_volatility,
            volatility_percentile=dur_proj.volatility_percentile,
            reward_sufficiency_score=sufficiency,
            reward_percentile=reward_pct,
            confidence=confidence,
            sample_count=n_used,
            opportunity_rank_signal=opp_rank,
            position_size_modifier=size_mod,
            risk_appetite_modifier=risk_mod,
        )

    # ── Integration point 1: inference context ────────────────────────────

    def context_for_inference(
        self,
        instrument: str,
        conditions: ProjectionConditions,
        direction: int,
        now: float | None = None,
    ) -> dict:
        """Return ``proj_*`` keys for ``QDEInferenceInput.brain_context``.

        Keys
        ----
        proj_expected_move_atr       — |move| in ATR multiples
        proj_expected_return_pct     — signed expected return fraction
        proj_expected_reward_atr     — signed reward in ATR multiples
        proj_expected_mae_atr        — adverse excursion in ATR multiples
        proj_expected_mfe_atr        — favorable excursion in ATR multiples
        proj_reward_risk_ratio       — expected_reward / expected_mae
        proj_reward_sufficiency      — 0-1 learned relative score
        proj_reward_percentile       — 0-1 percentile rank in this instrument's history
        proj_expected_duration_s     — expected holding time in seconds
        proj_expected_volatility     — expected vol during trade
        proj_volatility_percentile   — where expected_vol sits vs history
        proj_confidence              — 0-1 based on similar-record count
        proj_sample_count            — number of records used
        proj_opportunity_rank_signal — 0-1 trade-selection signal
        proj_position_size_modifier  — suggested size multiplier
        proj_risk_appetite_modifier  — −0.50..+0.25 risk signal
        """
        now = now if now is not None else time.time()
        ep = self.project(instrument, conditions, direction, now)
        return {
            "proj_expected_move_atr": round(ep.expected_move_atr, 4),
            "proj_expected_return_pct": round(ep.expected_return_pct, 6),
            "proj_expected_reward_atr": round(ep.expected_reward_atr, 4),
            "proj_expected_mae_atr": round(ep.expected_mae_atr, 4),
            "proj_expected_mfe_atr": round(ep.expected_mfe_atr, 4),
            "proj_reward_risk_ratio": round(ep.reward_risk_ratio, 4),
            "proj_reward_sufficiency": round(ep.reward_sufficiency_score, 4),
            "proj_reward_percentile": round(ep.reward_percentile, 4),
            "proj_expected_duration_s": round(ep.expected_duration_seconds, 1),
            "proj_expected_volatility": round(ep.expected_volatility, 8),
            "proj_volatility_percentile": round(ep.volatility_percentile, 4),
            "proj_confidence": round(ep.confidence, 4),
            "proj_sample_count": ep.sample_count,
            "proj_opportunity_rank_signal": round(ep.opportunity_rank_signal, 4),
            "proj_position_size_modifier": round(ep.position_size_modifier, 4),
            "proj_risk_appetite_modifier": round(ep.risk_appetite_modifier, 4),
        }

    # ── Integration point 2: opportunity rank signal ──────────────────────

    def opportunity_rank_signal(
        self,
        instrument: str,
        conditions: ProjectionConditions,
        direction: int,
        now: float | None = None,
    ) -> float:
        """Return 0-1 score for opportunity ranking.

        High when this setup historically delivered above-median reward
        in similar conditions, with sufficient historical evidence.
        """
        ep = self.project(instrument, conditions, direction, now)
        return round(ep.opportunity_rank_signal, 4)

    # ── Integration point 3: position size modifier ───────────────────────

    def position_size_modifier(
        self,
        instrument: str,
        conditions: ProjectionConditions,
        direction: int,
        now: float | None = None,
    ) -> float:
        """Return a 0.5..2.0 multiplier for position sizing.

        Without history the modifier is 1.0 (neutral).
        High reward sufficiency + high confidence → toward 2.0.
        Low reward sufficiency → toward 0.5.
        """
        ep = self.project(instrument, conditions, direction, now)
        return round(ep.position_size_modifier, 4)

    # ── Integration point 4: risk appetite modifier ───────────────────────

    def risk_appetite_modifier(
        self,
        instrument: str,
        conditions: ProjectionConditions,
        direction: int,
        now: float | None = None,
    ) -> float:
        """Return a −0.50..+0.25 risk appetite signal.

        Negative when the expected volatility is historically high or MAE
        is large.  Positive when the setup is historically clean.
        Without history returns 0.0 (neutral).
        """
        ep = self.project(instrument, conditions, direction, now)
        return round(ep.risk_appetite_modifier, 4)

    # ── Accessors ─────────────────────────────────────────────────────────

    def record_count(self, instrument: str) -> int:
        return self._memory.record_count(instrument)

    def memory(self) -> ProjectionMemory:
        return self._memory

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        return self._memory.to_snapshot(instrument)

    def load_snapshot(self, instrument: str, data: dict) -> None:
        self._memory.load_snapshot(instrument, data)
        self._dirty.add(instrument)
