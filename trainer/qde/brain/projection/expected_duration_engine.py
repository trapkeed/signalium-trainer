"""QDE Brain — projection: expected duration and volatility estimation.

``ExpectedDurationEngine`` is a **stateless** analytics class (no mutable
state of its own) that estimates *how long* a trade in the current conditions
is likely to last and *how volatile* the environment will be while it is open.

These two estimates are kept in a separate module from move/reward because
they are consumed by different downstream systems:

* **Duration** → ``brain_context`` for position-management scheduling,
  opportunity-queue prioritisation (don't queue a 5-min setup if the session
  closes in 2 hours), and the ``SurpriseEngine`` timing-surprise component.

* **Volatility** → risk-appetite modifier, position-size modifier (wider
  expected vol → tighter sizing), and ``brain_context`` as a forward-looking
  risk signal distinct from the *current* ``volatility_state``.

Algorithm
---------
Given the current ``ProjectionConditions``, find the K most-similar
historical ``ProjectionRecord`` entries from ``ProjectionMemory`` using
cosine similarity on the 9-dimensional feature vector.  Weight each by
similarity × recency (``exp(−age / τ)``).  Return the weighted mean and
standard deviation of ``duration_seconds`` and ``realized_vol`` across
those K records.

No hardcoded duration targets.  The "expected" duration is whatever
the market historically produced in similar conditions.

``volatility_percentile`` answers "is the expected vol higher or lower than
in most past setups?" as a 0-1 relative rank — no absolute vol threshold
is embedded.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from trainer.qde.brain.projection.projection_memory import (
    DEFAULT_RECENCY_TAU_DAYS,
    ProjectionConditions,
    ProjectionMemory,
    ProjectionRecord,
)

# Minimum records before the estimate is considered informative
_MIN_INFORMATIVE = 5

# Number of nearest-neighbor records to use
_DEFAULT_K = 30


# ── Similarity helpers ────────────────────────────────────────────────────────


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_w = sum(weights)
    if total_w < 1e-12:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _weighted_std(values: list[float], weights: list[float], mean: float) -> float:
    total_w = sum(weights)
    if total_w < 1e-12 or len(values) < 2:
        return 0.0
    variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_w
    return math.sqrt(max(0.0, variance))


def _percentile_rank(value: float, population: list[float]) -> float:
    """Fraction of population values <= value (0-1)."""
    if not population:
        return 0.5
    return sum(1 for x in population if x <= value) / len(population)


# ── DurationProjection ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DurationProjection:
    """Expected duration and volatility for one set of conditions.

    All fields are computed from historical ``ProjectionRecord`` entries
    with similar market conditions; no values are hardcoded.

    ``duration_std_seconds``  — 1-σ dispersion around the expected duration
        (high dispersion → uncertain timing, useful for stop-time decisions).
    ``volatility_percentile`` — where the expected vol sits vs all historical
        outcomes for this instrument: 0 = lowest ever, 1 = highest ever.
    ``confidence``            — min(1, sample_count / min_informative).
    """

    expected_duration_seconds: float = 0.0
    duration_std_seconds: float = 0.0
    expected_volatility: float = 0.0
    volatility_std: float = 0.0
    volatility_percentile: float = 0.5
    confidence: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> dict:
        return {
            "expected_duration_seconds": round(self.expected_duration_seconds, 1),
            "duration_std_seconds": round(self.duration_std_seconds, 1),
            "expected_volatility": round(self.expected_volatility, 8),
            "volatility_std": round(self.volatility_std, 8),
            "volatility_percentile": round(self.volatility_percentile, 4),
            "confidence": round(self.confidence, 4),
            "sample_count": self.sample_count,
        }


# Default projection returned when no history is available
_EMPTY_DURATION = DurationProjection()


# ── ExpectedDurationEngine ────────────────────────────────────────────────────


class ExpectedDurationEngine:
    """Stateless: reads ``ProjectionMemory``, returns ``DurationProjection``.

    ``project()`` accepts the current conditions and a live
    ``ProjectionMemory`` reference; it holds no state of its own and may
    be called concurrently for different symbols without coordination.
    """

    def project(
        self,
        instrument: str,
        conditions: ProjectionConditions,
        memory: ProjectionMemory,
        direction: int | None = None,
        k: int = _DEFAULT_K,
        recency_tau_days: float = DEFAULT_RECENCY_TAU_DAYS,
        min_samples: int = _MIN_INFORMATIVE,
        now: float | None = None,
    ) -> DurationProjection:
        """Estimate expected duration and volatility for ``instrument``.

        Parameters
        ----------
        instrument:
            Symbol to estimate for.
        conditions:
            Current market conditions fingerprint.
        memory:
            Live ``ProjectionMemory`` containing historical records.
        direction:
            If +1 or −1, restrict to same-direction records only.
            ``None`` uses all records regardless of direction.
        k:
            Maximum number of similar records to use.
        recency_tau_days:
            Recency decay time constant (larger = slower decay).
        min_samples:
            Below this, confidence scales down and estimates are
            less informative.
        """
        now = now if now is not None else time.time()

        records = memory.records(
            instrument,
            direction=direction,
        )
        if not records:
            return _EMPTY_DURATION

        current_vec = conditions.as_vector()

        # ── Score all records by similarity × recency ─────────────────────
        scored: list[tuple[float, ProjectionRecord]] = []
        for rec in records:
            sim = max(0.0, _cosine_sim(current_vec, rec.conditions.as_vector()))
            recency = rec.recency_weight(now, recency_tau_days)
            weight = sim * recency
            if weight > 1e-10:
                scored.append((weight, rec))

        if not scored:
            # Fall back to recency-only weighting (no similarity signal)
            for rec in records:
                recency = rec.recency_weight(now, recency_tau_days)
                if recency > 1e-10:
                    scored.append((recency, rec))

        if not scored:
            return _EMPTY_DURATION

        # Take top-K by weight
        scored.sort(key=lambda t: t[0], reverse=True)
        top_k = scored[:k]

        weights = [t[0] for t in top_k]
        dur_vals = [t[1].outcome.duration_seconds for t in top_k]
        vol_vals = [t[1].outcome.realized_vol for t in top_k]

        exp_dur = _weighted_mean(dur_vals, weights)
        exp_vol = _weighted_mean(vol_vals, weights)
        dur_std = _weighted_std(dur_vals, weights, exp_dur)
        vol_std = _weighted_std(vol_vals, weights, exp_vol)

        # Percentile rank of expected_vol vs all historical vols (unweighted)
        all_vols = [r.outcome.realized_vol for r in records if r.outcome.realized_vol > 0]
        vol_pct = _percentile_rank(exp_vol, all_vols) if all_vols else 0.5

        n_used = len(top_k)
        confidence = min(1.0, n_used / max(min_samples, 1))

        return DurationProjection(
            expected_duration_seconds=exp_dur,
            duration_std_seconds=dur_std,
            expected_volatility=exp_vol,
            volatility_std=vol_std,
            volatility_percentile=vol_pct,
            confidence=confidence,
            sample_count=n_used,
        )
