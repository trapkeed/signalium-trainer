"""QDE Brain — exposure: learned trade saturation engine.

``TradeSaturationEngine`` is the learning layer that answers: given the
portfolio's *current* concentration and the *marginal* concentration a
candidate trade would add, is this the kind of situation that has
historically produced good or bad portfolio outcomes?

No hardcoded limits
-------------------
The engine never says "you have more than N USD positions, denied."  It
computes a ``saturation_score`` (0-1) from two components:

1. **Structural signal** — the marginal composite HHI increase the
   candidate would introduce.  A trade that meaningfully increases
   currency, correlation, and regional concentration gets a higher
   structural saturation score than one that is genuinely orthogonal.

2. **Historical signal** — from ``ExposureMemory``.  The engine finds
   records where the portfolio *previously* had a similar concentration
   state and a similar marginal exposure was being considered, and asks:
   how good were the outcomes?  Portfolios at high concentration that
   historically produced good outcomes reduce the penalty; portfolios at
   high concentration that produced bad outcomes increase it.

   When there is no history (early in operation), ``history_confidence``
   is 0.0 and the historical signal defaults to neutral (0.5 quality),
   so the structural signal governs.  As experience accumulates the model
   progressively trusts its own learned thresholds over the structural
   defaults.

Learning loop
-------------
``record_outcome(observation, outcome)`` is the single write entry
point.  Callers build an ``ExposureObservation`` at trade-entry time
(using ``build_observation()``) and store it, then after the trade
closes they compute the ``ExposureOutcome`` (using
``ExposureOutcome.compute_quality()``) and call ``record_outcome()``.

Integration points
------------------
Four signals produced for downstream consumers:

  ``size_modifier``         0.5..1.0  — multiply baseline lot size
  ``conviction_modifier``   −0.3..0   — subtract from conviction
  ``rank_penalty``          0..0.25   — subtract from opportunity rank
  ``risk_appetite_modifier`` −0.3..0  — add to risk appetite

The severity of all four is a monotone function of ``saturation_score``
— the learning signal determines severity, not a fixed threshold.

``context_for_inference()`` emits 14 ``exp_*`` keys for
``QDEInferenceInput.brain_context``, following the same convention as
the projection (``proj_*``), regime (``regime_*``), liquidity (``liq_*``),
and opportunity (``opp_*``) packages.

Design rules
------------
* Does not import ``QDEDecision`` or any execution-layer type.
* Does not re-derive currency/sector/correlation/regional exposure —
  reads from ``ExposureClusterEngine.compute()``.
* Does not duplicate ``PortfolioScorer`` concentration math — the HHI is
  applied to different data.
* Stateless except for ``ExposureMemory``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trainer.portfolio.portfolio_instruments import InstrumentRegistry
from trainer.qde.brain.exposure.exposure_cluster_engine import (
    ExposureCluster,
    ExposureClusterEngine,
    PositionExposure,
    _N_POS_CAP,
)
from trainer.qde.brain.exposure.exposure_memory import (
    ExposureMemory,
    ExposureObservation,
    ExposureOutcome,
    ExposureRecord,
)

if TYPE_CHECKING:
    from trainer.portfolio.correlation_engine import CorrelationMatrix

# ── Learning hyper-parameters ──────────────────────────────────────────────────

# Top-K nearest-neighbour records used to estimate historical quality.
_DEFAULT_K          = 20
# Minimum records before the historical signal gets non-zero weight.
_MIN_INFORMATIVE    = 5
# Recency half-life for outcome records (days).
_RECENCY_TAU_DAYS   = 30.0
# L2 distance at which similarity falls to 0.5 (soft-scale parameter).
_DISTANCE_SCALE     = 0.5

# Integration-signal bounds — derived from saturation_score, not from
# hardcoded thresholds.
_SIZE_MOD_MIN          = 0.50
_SIZE_MOD_MAX          = 1.00   # concentration can only shrink size, never grow it
_CONVICTION_DRAG_MAX   = -0.30
_RANK_PENALTY_MAX      =  0.25
_RISK_MOD_MAX          = -0.30

# Composite concentration dimension weights (must match exposure_cluster_engine)
_W_CURRENCY     = 0.35
_W_SECTOR       = 0.20
_W_CORRELATION  = 0.30
_W_REGIONAL     = 0.15


# ── SaturationSignal ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SaturationSignal:
    """Learned concentration-risk signal for one candidate trade.

    ``saturation_score`` (0-1)
        Combined structural + historical measure of how risky adding this
        trade is from a portfolio-concentration perspective.  0 = no
        concentration concern; 1 = maximum learned concern.

    ``historical_outcome_quality`` (0-1)
        Weighted mean outcome quality of the K nearest historical records.
        0.5 = neutral / insufficient history.

    ``history_confidence`` (0-1)
        min(1, n_used / _MIN_INFORMATIVE).  0 when no history; 1 when
        K+ informative records exist.  Controls how much the historical
        signal governs vs the structural signal.

    Integration signals
    -------------------
    All four are monotone functions of ``saturation_score``; severity
    grows with saturation but never blocks a trade outright (that
    boundary belongs to ``qde_inference``).
    """

    instrument:                 str   = ""
    direction:                  int   = 1

    # ── Cluster snapshots ────────────────────────────────────────────────
    cluster_before:             ExposureCluster = field(
        default_factory=ExposureCluster)
    cluster_after:              ExposureCluster = field(
        default_factory=ExposureCluster)

    # ── Marginal deltas ──────────────────────────────────────────────────
    marginal_currency_hhi:     float = 0.0
    marginal_sector_hhi:       float = 0.0
    marginal_correlation_risk: float = 0.0
    marginal_regional_hhi:     float = 0.0
    marginal_composite:        float = 0.0

    # ── Learning signals ─────────────────────────────────────────────────
    historical_outcome_quality: float = 0.5   # 0-1, 0.5 = neutral/thin
    history_confidence:         float = 0.0   # 0-1
    structural_saturation:      float = 0.0   # raw marginal composite HHI

    # ── Combined score ───────────────────────────────────────────────────
    saturation_score:          float = 0.0   # 0-1, primary output

    # ── Integration signals ──────────────────────────────────────────────
    size_modifier:             float = 1.0   # 0.5..1.0
    conviction_modifier:       float = 0.0   # -0.3..0
    rank_penalty:              float = 0.0   # 0..0.25
    risk_appetite_modifier:    float = 0.0   # -0.3..0

    def to_dict(self) -> dict:
        return {
            "instrument":                 self.instrument,
            "direction":                  self.direction,
            "cluster_before":             self.cluster_before.to_dict(),
            "cluster_after":              self.cluster_after.to_dict(),
            "marginal_currency_hhi":      round(self.marginal_currency_hhi, 4),
            "marginal_sector_hhi":        round(self.marginal_sector_hhi, 4),
            "marginal_correlation_risk":  round(self.marginal_correlation_risk, 4),
            "marginal_regional_hhi":      round(self.marginal_regional_hhi, 4),
            "marginal_composite":         round(self.marginal_composite, 4),
            "historical_outcome_quality": round(self.historical_outcome_quality, 4),
            "history_confidence":         round(self.history_confidence, 4),
            "structural_saturation":      round(self.structural_saturation, 4),
            "saturation_score":           round(self.saturation_score, 4),
            "size_modifier":              round(self.size_modifier, 4),
            "conviction_modifier":        round(self.conviction_modifier, 4),
            "rank_penalty":               round(self.rank_penalty, 4),
            "risk_appetite_modifier":     round(self.risk_appetite_modifier, 4),
        }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _l2_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _similarity(dist: float, scale: float = _DISTANCE_SCALE) -> float:
    """Converts L2 distance to a 0-1 similarity weight.

    Uses a soft inverse-distance formula so nearby records get high
    weight while very distant records still contribute a little.
    """
    return 1.0 / (1.0 + dist / max(scale, 1e-9))


def _integration_signals(saturation_score: float) -> tuple[float, float, float, float]:
    """Derive the four integration signals from saturation_score.

    All are monotone functions of saturation_score — no independent
    thresholds.  Returns (size_modifier, conviction_modifier,
    rank_penalty, risk_appetite_modifier).
    """
    s = max(0.0, min(1.0, saturation_score))
    size_mod  = _SIZE_MOD_MAX + (_SIZE_MOD_MIN - _SIZE_MOD_MAX) * s
    conv_mod  = _CONVICTION_DRAG_MAX * s
    rank_pen  = _RANK_PENALTY_MAX * s
    risk_mod  = _RISK_MOD_MAX * s
    return (
        round(max(_SIZE_MOD_MIN, min(_SIZE_MOD_MAX, size_mod)), 4),
        round(max(_CONVICTION_DRAG_MAX, min(0.0, conv_mod)), 4),
        round(max(0.0, min(_RANK_PENALTY_MAX, rank_pen)), 4),
        round(max(_RISK_MOD_MAX, min(0.0, risk_mod)), 4),
    )


# ── TradeSaturationEngine ─────────────────────────────────────────────────────


class TradeSaturationEngine:
    """Learned portfolio-concentration risk engine.

    State
    -----
    Holds one ``ExposureMemory`` (the rolling learning store) and one
    ``ExposureClusterEngine`` (stateless concentration computer).

    The engine is the *only* component in this package that holds state;
    ``ExposureClusterEngine`` is stateless and may be called independently.

    Learning loop (per closed trade)
    ---------------------------------
    1. At trade-entry time, call ``build_observation(candidate,
       current_positions)`` and store the returned
       ``ExposureObservation``.
    2. When the trade closes, compute an ``ExposureOutcome`` using
       ``ExposureOutcome.compute_quality(portfolio_return,
       max_drawdown)``.
    3. Call ``record_outcome(observation, outcome)`` to append the pair
       to ``ExposureMemory``.

    The engine progressively learns which concentration levels are
    acceptable by observing what happened when similar exposures were
    taken in the past.
    """

    def __init__(
        self,
        registry:       InstrumentRegistry | None = None,
        max_records:    int   = 2000,
        k:              int   = _DEFAULT_K,
        recency_tau:    float = _RECENCY_TAU_DAYS,
    ) -> None:
        self._cluster_engine = ExposureClusterEngine(registry=registry)
        self._memory         = ExposureMemory(max_records=max_records)
        self._k              = k
        self._tau            = recency_tau

    # ── Learning ──────────────────────────────────────────────────────────

    def record_outcome(
        self,
        observation: ExposureObservation,
        outcome:     ExposureOutcome,
        now:         float | None = None,
    ) -> ExposureRecord:
        """Append one (observation → outcome) pair to the learning store.

        ``observation`` must have been built by ``build_observation()`` at
        trade-entry time (not exit time — concentration at entry is what
        matters for learning).
        """
        return self._memory.record(observation, outcome,
                                   now=now if now is not None else time.time())

    # ── Signal computation ────────────────────────────────────────────────

    def score(
        self,
        candidate:         PositionExposure,
        current_positions: list[PositionExposure],
        corr_matrix:       "CorrelationMatrix | None" = None,
        now:               float | None = None,
    ) -> SaturationSignal:
        """Compute the full saturation signal for a candidate trade.

        Parameters
        ----------
        candidate:
            The trade being considered (instrument, direction, weight).
        current_positions:
            All currently open positions in the portfolio.
        corr_matrix:
            Optional live correlation matrix; improves the correlation-
            concentration dimension when present.
        """
        now = now if now is not None else time.time()

        before, after = self._cluster_engine.marginal(
            candidate, current_positions, corr_matrix=corr_matrix
        )

        # ── Marginal deltas ────────────────────────────────────────────
        d_currency = after.currency_hhi       - before.currency_hhi
        d_sector   = after.sector_hhi         - before.sector_hhi
        d_corr     = after.correlation_concentration - before.correlation_concentration
        d_regional = after.regional_hhi       - before.regional_hhi
        d_composite = after.composite_concentration - before.composite_concentration

        # ── Structural saturation ─────────────────────────────────────
        # Primary signal: how concentrated is the portfolio RIGHT NOW
        # (before adding this trade)?  Secondary signal: does this trade
        # make it materially worse?
        #
        # Weighting before-state (0.65) vs marginal change (0.35) means
        # a first trade into an empty portfolio (before=0, large delta)
        # scores low, while an additional trade on top of an already-
        # concentrated portfolio (before>0, small delta) scores high.
        # This reflects the economic reality: concentration risk is a
        # property of the *existing* portfolio, not of the instrument alone.
        structural = (
            0.65 * before.composite_concentration
            + 0.35 * max(0.0, d_composite)
        )
        structural = min(1.0, structural)

        # ── Historical signal ──────────────────────────────────────────
        hist_quality, hist_conf = self._historical_quality(
            before=before,
            n_open=len(current_positions),
            now=now,
        )

        # ── Combined saturation score ──────────────────────────────────
        # When history is thin (hist_conf ≈ 0): structural governs.
        # As history accumulates: the historical penalty/boost modulates.
        # hist_quality < 0.5  → concentration hurt historically → boost saturation
        # hist_quality > 0.5  → concentration helped historically → reduce saturation
        hist_adjustment = (0.5 - hist_quality) * 2.0 * hist_conf  # range: −conf..+conf
        saturation = min(1.0, max(0.0, structural + hist_adjustment * structural))

        size_mod, conv_mod, rank_pen, risk_mod = _integration_signals(saturation)

        return SaturationSignal(
            instrument=candidate.instrument,
            direction=candidate.direction,
            cluster_before=before,
            cluster_after=after,
            marginal_currency_hhi=round(d_currency, 6),
            marginal_sector_hhi=round(d_sector, 6),
            marginal_correlation_risk=round(d_corr, 6),
            marginal_regional_hhi=round(d_regional, 6),
            marginal_composite=round(d_composite, 6),
            historical_outcome_quality=round(hist_quality, 4),
            history_confidence=round(hist_conf, 4),
            structural_saturation=round(structural, 4),
            saturation_score=round(saturation, 4),
            size_modifier=size_mod,
            conviction_modifier=conv_mod,
            rank_penalty=rank_pen,
            risk_appetite_modifier=risk_mod,
        )

    def context_for_inference(
        self,
        candidate:         PositionExposure,
        current_positions: list[PositionExposure],
        corr_matrix:       "CorrelationMatrix | None" = None,
        now:               float | None = None,
    ) -> dict:
        """Return 14 ``exp_*`` keys for ``QDEInferenceInput.brain_context``.

        Keys
        ----
        exp_currency_hhi           — portfolio currency HHI before trade
        exp_sector_hhi             — portfolio sector HHI before trade
        exp_correlation_conc       — portfolio correlation concentration
        exp_regional_hhi           — portfolio regional HHI before trade
        exp_composite_conc         — portfolio composite concentration
        exp_dominant_currency      — most exposed currency (string)
        exp_marginal_currency_hhi  — how much this trade adds to currency HHI
        exp_marginal_composite     — how much this trade adds to composite
        exp_saturation_score       — learned saturation signal
        exp_size_modifier          — suggested size modifier (0.5..1.0)
        exp_conviction_modifier    — conviction adjustment (−0.3..0)
        exp_rank_penalty           — opportunity rank penalty (0..0.25)
        exp_historical_quality     — learned outcome quality at this conc
        exp_history_confidence     — confidence in historical signal
        """
        sig = self.score(candidate, current_positions,
                         corr_matrix=corr_matrix, now=now)
        b   = sig.cluster_before
        return {
            "exp_currency_hhi":          round(b.currency_hhi, 4),
            "exp_sector_hhi":            round(b.sector_hhi, 4),
            "exp_correlation_conc":      round(b.correlation_concentration, 4),
            "exp_regional_hhi":          round(b.regional_hhi, 4),
            "exp_composite_conc":        round(b.composite_concentration, 4),
            "exp_dominant_currency":     b.dominant_currency,
            "exp_marginal_currency_hhi": round(sig.marginal_currency_hhi, 4),
            "exp_marginal_composite":    round(sig.marginal_composite, 4),
            "exp_saturation_score":      round(sig.saturation_score, 4),
            "exp_size_modifier":         round(sig.size_modifier, 4),
            "exp_conviction_modifier":   round(sig.conviction_modifier, 4),
            "exp_rank_penalty":          round(sig.rank_penalty, 4),
            "exp_historical_quality":    round(sig.historical_outcome_quality, 4),
            "exp_history_confidence":    round(sig.history_confidence, 4),
        }

    # ── Convenience ──────────────────────────────────────────────────────

    def current_cluster(
        self,
        positions:   list[PositionExposure],
        corr_matrix: "CorrelationMatrix | None" = None,
    ) -> ExposureCluster:
        """Current portfolio exposure cluster without a candidate trade."""
        return self._cluster_engine.compute(positions, corr_matrix=corr_matrix)

    def build_observation(
        self,
        candidate:         PositionExposure,
        current_positions: list[PositionExposure],
        corr_matrix:       "CorrelationMatrix | None" = None,
    ) -> ExposureObservation:
        """Build an ``ExposureObservation`` to store at trade-entry time.

        Callers store this and pass it back to ``record_outcome()`` when
        the trade closes — it must capture the state AT ENTRY, not exit.
        """
        before, _ = self._cluster_engine.marginal(
            candidate, current_positions, corr_matrix=corr_matrix
        )
        sig = self.score(candidate, current_positions, corr_matrix=corr_matrix)
        n_pos = len(current_positions)
        return ExposureObservation(
            currency_hhi=before.currency_hhi,
            sector_hhi=before.sector_hhi,
            correlation_concentration=before.correlation_concentration,
            regional_hhi=before.regional_hhi,
            n_positions_norm=min(1.0, n_pos / _N_POS_CAP),
            candidate_instrument=candidate.instrument,
            candidate_direction=candidate.direction,
            marginal_currency_hhi=sig.marginal_currency_hhi,
            marginal_sector_hhi=sig.marginal_sector_hhi,
            marginal_correlation_risk=sig.marginal_correlation_risk,
            marginal_regional_hhi=sig.marginal_regional_hhi,
            marginal_composite=sig.marginal_composite,
        )

    # ── Persistence helpers ────────────────────────────────────────────────

    def memory_snapshot(self) -> dict:
        return self._memory.to_snapshot()

    def load_memory(self, snapshot: dict) -> None:
        self._memory = ExposureMemory.load_snapshot(snapshot)

    # ── Internal: historical quality estimation ────────────────────────────

    def _historical_quality(
        self,
        before: ExposureCluster,
        n_open: int,
        now:    float,
    ) -> tuple[float, float]:
        """Estimate quality at the current concentration from memory.

        Returns (quality: 0-1, confidence: 0-1).
        0.5 quality = neutral (no information).
        0.0 confidence = insufficient history.
        """
        records = self._memory.records()
        if not records:
            return 0.5, 0.0

        # Query vector: the before-state (without the candidate)
        query = [
            before.currency_hhi,
            before.sector_hhi,
            before.correlation_concentration,
            before.regional_hhi,
            min(1.0, n_open / _N_POS_CAP),
        ]

        # Score each record by L2-similarity × recency
        scored: list[tuple[float, ExposureRecord]] = []
        for rec in records:
            hist_vec = rec.observation.as_vector()
            dist     = _l2_distance(query, hist_vec)
            sim      = _similarity(dist)
            recency  = rec.recency_weight(now, tau_days=self._tau)
            w        = sim * recency
            if w > 1e-10:
                scored.append((w, rec))

        if not scored:
            return 0.5, 0.0

        # Top-K by weight
        scored.sort(key=lambda t: t[0], reverse=True)
        top_k = scored[: self._k]

        total_w  = sum(w for w, _ in top_k)
        if total_w < 1e-12:
            return 0.5, 0.0

        weighted_quality = sum(w * r.outcome.outcome_quality for w, r in top_k) / total_w
        confidence       = min(1.0, len(top_k) / max(_MIN_INFORMATIVE, 1))

        return max(0.0, min(1.0, weighted_quality)), round(confidence, 4)
