"""QDE Brain — institutional: structural distribution detector.

``DistributionDetector`` is the symmetric counterpart to
``AccumulationDetector``.  It estimates the probability that a large
participant is gradually exiting a long position (or building a short)
by observing price-range behaviour at highs.

What is distribution (structurally)?
--------------------------------------
Distribution is the gradual exit of a large long position (or entry of
a large short).  Its structural signatures are symmetric to
accumulation but at price highs:

  * **Range compression** — price compresses near a high zone as large
    sell orders are absorbed into buy flow from retail participants,
    preventing a sharp downside move that would signal the distribution.

  * **Resistance rejection** — the price repeatedly tests a high zone
    and closes below it.  Each "failed breakout" represents
    distribution: available buy demand is met by persistent selling.

  * **Volume sell dominance** — within the range, down-close bars tend
    to carry more volume than up-close bars.

  * **Momentum divergence** — price is near the top of its recent
    range but momentum is flattening or turning downward.  Divergence
    suggests distribution is neutralising the directional buy pressure.

Design rules
------------
* Symmetric to ``AccumulationDetector`` by construction — the formulas
  are exact mirrors at highs rather than lows.
* No ICT-specific rules; no hardcoded patterns.
* Stateless — safe to call on every tick.

``BarSnapshot`` is imported from ``accumulation_detector`` and shared
across the entire institutional package.
"""

from __future__ import annotations

from dataclasses import dataclass

from trainer.qde.brain.institutional.accumulation_detector import (
    BarSnapshot,
    _mean,
    _MAX_TOUCHES,
    _LONG_WINDOW,
    _SHORT_WINDOW,
    _TOUCH_TOLERANCE,
)

# ── DistributionInput ─────────────────────────────────────────────────────────


@dataclass
class DistributionInput:
    """Structural inputs for distribution probability estimation.

    Symmetric to ``AccumulationInput``.  ``bars`` override pre-computed
    signals when supplied.

    Context scalars
    ---------------
    ``momentum``
        Signed −1..+1.  Near-zero or falling momentum at a high-zone
        supports distribution.

    ``session_quality``
        0-1.  Distribution during liquid sessions is structurally more
        significant than thin off-hours activity.

    ``microstructure_quality``
        0-1.  Poor microstructure mimics rejection noise.

    ``level_proximity``
        0-1 proximity to the nearest known structural high level.

    Pre-computed signal overrides
    ------------------------------
    ``range_compression``
        0-1: recent price range vs. baseline.

    ``rejection_strength``
        0-1: how persistently the price bounced off the high zone.

    ``volume_sell_dominance``
        0-1: 0 = only buy volume, 0.5 = balanced, 1 = only selling.

    ``momentum_divergence``
        0-1: momentum diverging negatively while price is near highs.
    """

    # Optional raw bars
    bars: list[BarSnapshot] | None = None

    # Context scalars
    momentum:               float = 0.0
    session_quality:        float = 0.80
    microstructure_quality: float = 0.80
    level_proximity:        float = 0.0

    # Pre-computed signal overrides
    range_compression:     float = 0.0
    rejection_strength:    float = 0.0
    volume_sell_dominance: float = 0.5
    momentum_divergence:   float = 0.0


# ── DistributionEstimate ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class DistributionEstimate:
    """Structural distribution probability and component scores.

    ``distribution_prob``
        Composite probability (0-1) that institutional distribution is
        in progress.

    ``range_compression``
        0-1: recent range compressed vs. baseline.

    ``rejection_strength``
        0-1: fraction of high-zone tests that closed below the zone.

    ``volume_sell_dominance``
        0-1: volume-weighted sell/buy balance.

    ``momentum_divergence``
        0-1: momentum diverging negatively at price highs.

    ``structural_quality``
        0-1 reliability of the structural observation.
    """

    distribution_prob:     float = 0.0
    range_compression:     float = 0.0
    rejection_strength:    float = 0.0
    volume_sell_dominance: float = 0.5
    momentum_divergence:   float = 0.0
    structural_quality:    float = 0.5

    def to_dict(self) -> dict:
        return {
            "distribution_prob":     round(self.distribution_prob, 4),
            "range_compression":     round(self.range_compression, 4),
            "rejection_strength":    round(self.rejection_strength, 4),
            "volume_sell_dominance": round(self.volume_sell_dominance, 4),
            "momentum_divergence":   round(self.momentum_divergence, 4),
            "structural_quality":    round(self.structural_quality, 4),
        }


# ── DistributionDetector ──────────────────────────────────────────────────────


class DistributionDetector:
    """Stateless structural distribution probability estimator.

    Symmetric to ``AccumulationDetector`` at price highs.  Detects
    range compression, resistance rejection, sell-volume dominance, and
    downside momentum divergence.
    """

    def __init__(
        self,
        short_window:    int   = _SHORT_WINDOW,
        long_window:     int   = _LONG_WINDOW,
        touch_tolerance: float = _TOUCH_TOLERANCE,
        max_touches:     int   = _MAX_TOUCHES,
    ) -> None:
        self._sw   = short_window
        self._lw   = long_window
        self._ttol = touch_tolerance
        self._mt   = max_touches

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, inp: DistributionInput) -> DistributionEstimate:
        """Compute the structural distribution probability."""
        if inp.bars and len(inp.bars) >= max(4, self._sw):
            rc, rj, vs, md = self._compute_bar_signals(inp.bars)
        else:
            rc = inp.range_compression
            rj = inp.rejection_strength
            vs = inp.volume_sell_dominance
            md = inp.momentum_divergence

        prob = self._distribution_prob(rc, rj, vs, md, inp.level_proximity)
        sq   = 0.5 * inp.session_quality + 0.5 * inp.microstructure_quality

        return DistributionEstimate(
            distribution_prob=round(max(0.0, min(1.0, prob)), 4),
            range_compression=round(max(0.0, min(1.0, rc)), 4),
            rejection_strength=round(max(0.0, min(1.0, rj)), 4),
            volume_sell_dominance=round(max(0.0, min(1.0, vs)), 4),
            momentum_divergence=round(max(0.0, min(1.0, md)), 4),
            structural_quality=round(max(0.0, min(1.0, sq)), 4),
        )

    def from_bars(
        self,
        bars:                   list[BarSnapshot],
        momentum:               float = 0.0,
        session_quality:        float = 0.80,
        microstructure_quality: float = 0.80,
        level_proximity:        float = 0.0,
    ) -> DistributionEstimate:
        """Convenience: compute signals from raw bars and detect."""
        inp = DistributionInput(
            bars=bars,
            momentum=momentum,
            session_quality=session_quality,
            microstructure_quality=microstructure_quality,
            level_proximity=level_proximity,
        )
        return self.detect(inp)

    # ── Core formula ──────────────────────────────────────────────────────

    def _distribution_prob(
        self,
        range_compression:     float,
        rejection_strength:    float,
        volume_sell_dominance: float,
        momentum_divergence:   float,
        level_proximity:       float,
    ) -> float:
        """Structural distribution probability from four components.

        Mirror of the accumulation formula at price highs.

        ``sell_vol_component``
            The fraction of sell volume above the neutral 0.5 midpoint
            mapped to [0, 1].  At pure sell dominance (vs=1.0) → full
            0.25 contribution.  At neutral → 0.  At buy dominance → 0.
        """
        sell_vol_component = max(0.0, 2.0 * (volume_sell_dominance - 0.5))  # 0-1
        return (
            0.30 * range_compression
            + 0.30 * rejection_strength
            + 0.25 * sell_vol_component
            + 0.10 * momentum_divergence
            + 0.05 * level_proximity
        )

    # ── Bar-derived signal computation ────────────────────────────────────

    def _compute_bar_signals(
        self,
        bars: list[BarSnapshot],
    ) -> tuple[float, float, float, float]:
        """Compute (range_compression, rejection, vol_sell_dom, momentum_divergence).

        Exact mirror of ``AccumulationDetector._compute_bar_signals``
        with the low-zone logic replaced by a high-zone logic.
        """
        n = len(bars)
        sw = min(self._sw, n // 2)

        recent_bars   = bars[max(0, n - sw):]
        baseline_bars = bars[max(0, n - self._lw): max(0, n - sw)]
        window_bars   = bars[max(0, n - self._lw):]

        # ── Range compression (same logic as accumulation) ────────────────
        recent_range   = _mean([b.full_range for b in recent_bars])
        baseline_range = _mean([b.full_range for b in baseline_bars]) if baseline_bars else recent_range
        if baseline_range > 1e-12:
            range_compression = max(0.0, 1.0 - recent_range / baseline_range)
        else:
            range_compression = 0.0

        # ── Rejection: touches of the HIGH zone that close BELOW it ───────
        # A bar "touches" the zone if its high is within tolerance of the
        # window's highest high.  "Closes below" means close < high_zone.
        if window_bars:
            high_zone  = max(b.high for b in window_bars)
            touch_floor = high_zone * (1.0 - self._ttol)  # zone lower edge
            touches     = sum(
                1 for b in window_bars
                if b.high >= touch_floor and b.close < high_zone
            )
            rejection = min(1.0, touches / max(self._mt, 1))
        else:
            rejection = 0.0

        # ── Volume sell dominance (inverse of buy dominance) ─────────────
        up_bars   = [b for b in recent_bars if b.is_up]
        down_bars = [b for b in recent_bars if not b.is_up]
        avg_up_v  = _mean([b.volume for b in up_bars])   if up_bars   else 0.0
        avg_dn_v  = _mean([b.volume for b in down_bars]) if down_bars else 0.0
        total_v   = avg_up_v + avg_dn_v
        vs        = (avg_dn_v / total_v) if total_v > 1e-12 else 0.5

        # ── Momentum divergence: price near highs but range compressing ───
        window_high = max((b.high for b in window_bars), default=0.0)
        window_low  = min((b.low  for b in window_bars), default=0.0)
        price_range = window_high - window_low
        if price_range > 1e-12 and window_bars:
            close_position    = (window_bars[-1].close - window_low) / price_range
            price_at_high     = close_position    # high when close is near high
            momentum_divergence = price_at_high * range_compression
        else:
            momentum_divergence = 0.0

        return range_compression, rejection, vs, momentum_divergence
