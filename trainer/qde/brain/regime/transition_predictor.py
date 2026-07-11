"""QDE Brain — regime transition: feature extraction and prediction.

``TransitionPredictor`` is a **stateless** analytics class.  It reads a
``TransitionMemory`` and returns a ``TransitionProbabilityMap`` — a
probability distribution over the six ``TransitionTarget`` values.

Algorithm (no hardcoded thresholds)
------------------------------------
1. Extract a ``TransitionFeatureVector`` from the most recent N snapshots
   in memory (the "current window").
2. For each historical ``TransitionRecord``, compute cosine similarity
   between the current feature vector and the record's cached feature
   vector.
3. Weight each similarity by a recency factor (records that occurred more
   recently count more).
4. Accumulate weighted similarities per ``TransitionTarget`` to produce
   raw scores, then normalize to a probability simplex.

There are no threshold lines of the form "if slope > X → predict Y".
The threshold for predicting any transition is *emergent*: a target only
receives probability mass if the current feature pattern resembles (by
cosine similarity) patterns that historically preceded that target.
When no historical transitions exist the system returns a uniform prior.

``TransitionFeatureVector`` — 10-dimensional
--------------------------------------------
All components are raw-unit values; the vector is unit-normalized before
similarity computation, so absolute magnitudes do not influence the
relative direction of the comparison.

  atr_slope          — linear slope of ATR values across the window
  spread_slope       — linear slope of spread values
  momentum_slope     — linear slope of momentum values (already −1..+1)
  volatility_slope   — linear slope of volatility values
  vol_rank_slope     — linear slope of ordinal vol-state rank
  trend_valid_ratio  — fraction of ticks with trend_valid=True (0..1)
  microstructure_slope — slope of microstructure_quality
  vol_quality_slope  — slope of volatility_quality
  atr_cv             — ATR coefficient of variation (spread/mean; higher =
                       more ATR instability, a compression/expansion signal)
  window_fill        — fraction of desired window that was actually used
                       (1.0 = full window available; < 1.0 = thin history)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from trainer.qde.brain.regime.transition_memory import (
    TransitionMemory,
    TransitionRecord,
    TransitionTarget,
    RegimeSnapshot,
    _PRE_WINDOW,
)

# Default window: how many recent snapshots form the "current pattern"
_DEFAULT_WINDOW = _PRE_WINDOW          # matches pre-transition capture window

# Recency decay: a transition that occurred N hours ago has weight exp(-N/τ)
# Default τ = 336 h (≈ 2 weeks), so a 1-week-old transition retains ~50%
_DEFAULT_DECAY_TAU_HOURS = 336.0

# Minimum historical transitions before the prediction is considered
# informative (below this we blend toward the uniform prior)
_MIN_INFORMATIVE = 5


# ── Linear regression helper ──────────────────────────────────────────────────


def _linear_slope(values: list[float]) -> float:
    """Slope from OLS regression of values against indices 0..n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / max(den, 1e-12)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _coeff_of_variation(values: list[float]) -> float:
    """Coefficient of variation: std / |mean|.  Returns 0.0 if mean ≈ 0."""
    m = _mean(values)
    if abs(m) < 1e-12:
        return 0.0
    return _std(values) / abs(m)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [−1, +1].  Returns 0 if either vector is zero."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _softmax(scores: dict[TransitionTarget, float]) -> dict[TransitionTarget, float]:
    """Softmax over a dict of raw scores.  Handles empty / all-zero dicts."""
    if not scores:
        uniform = 1.0 / len(TransitionTarget)
        return {t: uniform for t in TransitionTarget}
    # Shift by max for numerical stability
    max_v = max(scores.values(), default=0.0)
    exp_scores = {t: math.exp(v - max_v) for t, v in scores.items()}
    total = sum(exp_scores.values())
    if total < 1e-12:
        uniform = 1.0 / len(TransitionTarget)
        return {t: uniform for t in TransitionTarget}
    return {t: v / total for t, v in exp_scores.items()}


# ── TransitionFeatureVector ───────────────────────────────────────────────────


@dataclass(frozen=True)
class TransitionFeatureVector:
    """10-dimensional feature vector derived from a snapshot window.

    All components are real-valued but unscaled; the vector is unit-
    normalized by ``as_vector()`` before cosine comparison, so
    cross-feature scaling differences are removed automatically.
    """

    atr_slope:            float = 0.0
    spread_slope:         float = 0.0
    momentum_slope:       float = 0.0
    volatility_slope:     float = 0.0
    vol_rank_slope:       float = 0.0
    trend_valid_ratio:    float = 0.5
    microstructure_slope: float = 0.0
    vol_quality_slope:    float = 0.0
    atr_cv:               float = 0.0
    window_fill:          float = 1.0

    def as_list(self) -> list[float]:
        return [
            self.atr_slope,
            self.spread_slope,
            self.momentum_slope,
            self.volatility_slope,
            self.vol_rank_slope,
            self.trend_valid_ratio,
            self.microstructure_slope,
            self.vol_quality_slope,
            self.atr_cv,
            self.window_fill,
        ]

    def to_dict(self) -> dict:
        return {
            "atr_slope":            round(self.atr_slope, 6),
            "spread_slope":         round(self.spread_slope, 6),
            "momentum_slope":       round(self.momentum_slope, 6),
            "volatility_slope":     round(self.volatility_slope, 6),
            "vol_rank_slope":       round(self.vol_rank_slope, 6),
            "trend_valid_ratio":    round(self.trend_valid_ratio, 4),
            "microstructure_slope": round(self.microstructure_slope, 6),
            "vol_quality_slope":    round(self.vol_quality_slope, 6),
            "atr_cv":               round(self.atr_cv, 6),
            "window_fill":          round(self.window_fill, 4),
        }


# ── TransitionProbabilityMap ──────────────────────────────────────────────────


@dataclass
class TransitionProbabilityMap:
    """Output of one prediction call: a full probability distribution over
    the six ``TransitionTarget`` values plus summary metadata.

    ``probabilities`` always sums to 1.0 (or very close due to float
    rounding).  When no historical data is available all values are 1/6.

    ``confidence`` (0-1) reflects how many historical transitions informed
    this estimate.  Low confidence does not change the probabilities but
    is a signal to downstream consumers to weight the distribution less
    strongly (treat it closer to the uniform prior themselves).

    ``entropy`` is the normalised Shannon entropy of the distribution
    (0 = perfectly concentrated on one target; 1 = fully uniform).
    High entropy means the system sees no clear directional signal.
    """

    probabilities:  dict[str, float]    # target.value → probability
    confidence:     float               # 0-1
    top_target:     str                 # TransitionTarget.value
    top_probability: float              # 0-1
    entropy:        float               # 0-1 normalised
    sample_count:   int                 # how many historical records used
    feature_vector: TransitionFeatureVector | None = None

    # ── Convenience accessors ─────────────────────────────────────────────

    def probability(self, target: TransitionTarget) -> float:
        return self.probabilities.get(target.value, 0.0)

    def to_dict(self) -> dict:
        return {
            "probabilities":   {k: round(v, 4) for k, v in self.probabilities.items()},
            "confidence":      round(self.confidence, 4),
            "top_target":      self.top_target,
            "top_probability": round(self.top_probability, 4),
            "entropy":         round(self.entropy, 4),
            "sample_count":    self.sample_count,
            "feature_vector":  self.feature_vector.to_dict() if self.feature_vector else None,
        }


def _uniform_map(
    confidence: float = 0.0,
    sample_count: int = 0,
    fv: TransitionFeatureVector | None = None,
) -> TransitionProbabilityMap:
    n = len(TransitionTarget)
    uniform = 1.0 / n
    return TransitionProbabilityMap(
        probabilities={t.value: uniform for t in TransitionTarget},
        confidence=confidence,
        top_target=TransitionTarget.TREND_FORMATION.value,
        top_probability=uniform,
        entropy=1.0,
        sample_count=sample_count,
        feature_vector=fv,
    )


def _entropy(probs: dict[str, float]) -> float:
    """Normalised Shannon entropy of a probability distribution, 0..1."""
    n = len(probs)
    if n <= 1:
        return 0.0
    h = -sum(p * math.log(p + 1e-12) for p in probs.values())
    h_max = math.log(n)
    return h / max(h_max, 1e-12)


# ── TransitionPredictor ───────────────────────────────────────────────────────


class TransitionPredictor:
    """Stateless prediction engine.

    ``extract_feature_vector`` and ``predict`` are both pure functions of
    their arguments — they hold no mutable state and can be called from
    any number of symbols concurrently.
    """

    def extract_feature_vector(
        self,
        snapshots: list[RegimeSnapshot],
        desired_window: int = _DEFAULT_WINDOW,
    ) -> TransitionFeatureVector:
        """Derive the 10-dimensional feature vector from a snapshot window.

        Uses up to ``desired_window`` of the most recent snapshots.
        If fewer are available the ``window_fill`` component reflects the
        coverage fraction.
        """
        if not snapshots:
            return TransitionFeatureVector()

        window = snapshots[-desired_window:]
        n      = len(window)

        atr_vals     = [s.atr                   for s in window]
        spread_vals  = [s.spread                 for s in window]
        mom_vals     = [s.momentum               for s in window]
        vol_vals     = [s.volatility             for s in window]
        rank_vals    = [float(s.vol_state_rank)  for s in window]
        tv_vals      = [1.0 if s.trend_valid else 0.0 for s in window]
        micro_vals   = [s.microstructure_quality for s in window]
        vq_vals      = [s.volatility_quality     for s in window]

        return TransitionFeatureVector(
            atr_slope=_linear_slope(atr_vals),
            spread_slope=_linear_slope(spread_vals),
            momentum_slope=_linear_slope(mom_vals),
            volatility_slope=_linear_slope(vol_vals),
            vol_rank_slope=_linear_slope(rank_vals),
            trend_valid_ratio=_mean(tv_vals),
            microstructure_slope=_linear_slope(micro_vals),
            vol_quality_slope=_linear_slope(vq_vals),
            atr_cv=_coeff_of_variation(atr_vals),
            window_fill=n / max(desired_window, 1),
        )

    def predict(
        self,
        instrument:          str,
        memory:              TransitionMemory,
        window:              int   = _DEFAULT_WINDOW,
        recency_tau_hours:   float = _DEFAULT_DECAY_TAU_HOURS,
        min_samples:         int   = _MIN_INFORMATIVE,
        now:                 float | None = None,
    ) -> TransitionProbabilityMap:
        """Return a probability distribution over ``TransitionTarget``.

        The distribution is built entirely from cosine similarities between
        the current feature vector and the feature vectors of historical
        transition records.  No threshold or direction rule is embedded;
        the signal is learned from accumulated history.

        Parameters
        ----------
        window:
            Number of recent snapshots forming the "current" feature vector.
        recency_tau_hours:
            Exponential decay time constant for historical records.
            A record that occurred ``τ`` hours ago retains weight ``e^−1``
            relative to a fresh record.
        min_samples:
            Below this many historical records, confidence scales down
            linearly and the returned probabilities blend toward the
            uniform prior.
        """
        now = now if now is not None else time.time()

        snaps       = memory.snapshots(instrument)
        transitions = memory.transitions(instrument)

        # Extract current feature vector
        fv = self.extract_feature_vector(snaps, desired_window=window)
        current_vec = fv.as_list()

        if not transitions:
            return _uniform_map(confidence=0.0, sample_count=0, fv=fv)

        # ── Accumulate weighted-similarity scores per target ──────────────
        raw_scores: dict[TransitionTarget, float] = {t: 0.0 for t in TransitionTarget}
        total_weight = 0.0

        for rec in transitions:
            if not rec.feature_vector:
                continue
            age_h     = rec.age_hours(now)
            recency   = math.exp(-age_h / max(recency_tau_hours, 1e-3))
            sim       = max(0.0, _cosine_sim(current_vec, rec.feature_vector))
            raw_scores[rec.target] += sim * recency
            total_weight            += recency

        # ── Confidence: proportion of informative history available ───────
        n_used     = len(transitions)
        confidence = min(1.0, n_used / max(min_samples, 1))

        # ── Check for degenerate case (all zeros) ─────────────────────────
        total_raw = sum(raw_scores.values())
        if total_raw < 1e-12:
            # No similarity signal at all — return uniform
            return _uniform_map(confidence=confidence, sample_count=n_used, fv=fv)

        # ── Softmax over raw scores → probability simplex ─────────────────
        softmax_probs = _softmax(raw_scores)

        # Blend toward uniform prior when confidence is low
        uniform = 1.0 / len(TransitionTarget)
        final_probs: dict[str, float] = {}
        for t, p in softmax_probs.items():
            blended = confidence * p + (1.0 - confidence) * uniform
            final_probs[t.value] = blended

        top_target, top_prob = max(final_probs.items(), key=lambda kv: kv[1])
        ent = _entropy(final_probs)

        return TransitionProbabilityMap(
            probabilities=final_probs,
            confidence=confidence,
            top_target=top_target,
            top_probability=top_prob,
            entropy=ent,
            sample_count=n_used,
            feature_vector=fv,
        )
