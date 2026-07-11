"""QDE Brain — institutional: structural accumulation detector.

``AccumulationDetector`` is a **stateless, structural** model that
estimates the probability of institutional accumulation from bar-pattern
observations alone — range compression, support absorption, volume
dominance, and price-momentum divergence — without requiring any
historical trade outcomes.

What is accumulation (structurally)?
--------------------------------------
Accumulation is the gradual building of a large long position by a
well-capitalised participant who must spread entries over time to avoid
adverse price impact.  Its structural signatures:

  * **Range compression** — price coils as large buy orders absorb
    available sell flow, suppressing downside moves without driving
    price sharply upward (which would reveal the accumulator's hand).

  * **Support absorption** — the price repeatedly tests a low-zone
    and closes above it.  Each "failed breakdown" represents absorption:
    the available sell pressure was met by persistent buying.

  * **Volume buy dominance** — within the range, up-close bars tend to
    carry more volume than down-close bars.  This reflects more
    aggressive buying on rallies relative to selling on pullbacks, even
    if the net price range is unchanged.

  * **Momentum divergence** — price is near the bottom of its recent
    range but momentum is flattening or turning upward.  Divergence
    suggests accumulation is neutralising the directional sell pressure
    that brought price to this zone.

Design rules
------------
* No ICT-specific rules.  No hardcoded "accumulation zones" or named
  structural patterns by proprietary names.  The detector observes
  generic bar-pattern features that any participant could measure.
* The QDE model learns what combinations actually predict upside follow-
  through from the ``InstitutionalIntentEngine`` learning loop.
* Stateless — safe to call on every tick.
* ``BarSnapshot`` is defined here and shared across all detectors in
  this package.  Callers construct ``BarSnapshot`` from whatever OHLCV
  format their feed produces.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Tuneable defaults ─────────────────────────────────────────────────────────

_SHORT_WINDOW = 8  # bars for recent range/volume measurement
_LONG_WINDOW = 24  # bars for baseline range measurement
_TOUCH_TOLERANCE = 0.002  # 0.2%: price within this of low-zone counts as a touch
_MAX_TOUCHES = 5  # touch count that saturates absorption signal
_PROB_FLOOR = 0.0  # minimum structural probability
_MOMENTUM_UP_BIAS = 0.0  # no directional bias in structural model


# ── BarSnapshot ───────────────────────────────────────────────────────────────


@dataclass
class BarSnapshot:
    """Minimal OHLCV bar for institutional pattern detection.

    All four price fields are required; ``volume`` is optional (defaults
    to 1.0, i.e. equal weight per bar).  ``timestamp`` is for ordering
    and recency computations only.

    Using a local type avoids importing from the liquidity package
    (which has its own ``OHLCBar``), keeping this package's dependency
    graph clean.
    """

    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 1.0  # unit volume when actual volume is unavailable
    timestamp: float = 0.0

    @property
    def is_up(self) -> bool:
        """True when the bar closed above its open (up-close bar)."""
        return self.close > self.open

    @property
    def body_size(self) -> float:
        """Absolute bar body size (|close − open|)."""
        return abs(self.close - self.open)

    @property
    def full_range(self) -> float:
        """High-to-low bar range."""
        return max(0.0, self.high - self.low)


# ── AccumulationInput ─────────────────────────────────────────────────────────


@dataclass
class AccumulationInput:
    """Structural inputs for accumulation probability estimation.

    Callers may provide either raw bars (for automatic signal extraction
    via ``AccumulationDetector.from_bars()``) or pre-computed scalar
    signals, or both (bar-derived signals take precedence when bars are
    supplied).

    Raw bars
    --------
    ``bars``
        Recent price history, oldest first.  Minimum 10 bars required
        for meaningful detection; fewer bars → lower signal quality.

    Context scalars
    ---------------
    ``momentum``
        Signed −1..+1 from the MIE momentum engine.  Near-zero or
        rising momentum at a low-zone supports accumulation.

    ``session_quality``
        0-1 session-quality score.  Accumulation during liquid sessions
        is structurally more significant than thin off-hours activity.

    ``microstructure_quality``
        0-1 order-flow quality.  Poor microstructure can mimic
        absorption without institutional involvement (noise absorption).

    ``level_proximity``
        0-1 distance to the nearest known structural level.  Supplied
        by the engine from the liquidity map; 1 = price exactly at a
        level, 0 = price far from any known level.

    Pre-computed signal overrides
    ------------------------------
    When bars are not available (e.g. in testing or replay), these
    scalars can be set directly.  They are also set automatically when
    ``from_bars()`` is called.

    ``range_compression``
        0-1: recent price range / baseline range.  1 = recent range has
        fully compressed to a fraction of its baseline width.

    ``absorption_strength``
        0-1: how persistently has the price bounced off the low zone?

    ``volume_buy_dominance``
        0-1: 0 = only selling volume, 0.5 = balanced, 1 = only buying.

    ``momentum_divergence``
        0-1: how strongly is momentum diverging positively from price
        position?  0 = no divergence, 1 = maximum divergence.
    """

    # Optional raw bars — used when present
    bars: list[BarSnapshot] | None = None

    # Context scalars
    momentum: float = 0.0
    session_quality: float = 0.80
    microstructure_quality: float = 0.80
    level_proximity: float = 0.0

    # Pre-computed signal overrides (auto-filled by from_bars)
    range_compression: float = 0.0
    absorption_strength: float = 0.0
    volume_buy_dominance: float = 0.5
    momentum_divergence: float = 0.0


# ── AccumulationEstimate ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class AccumulationEstimate:
    """Structural accumulation probability and component scores.

    ``accumulation_prob``
        Composite probability (0-1) that institutional accumulation is
        in progress based on structural bar patterns.  This is a prior;
        the ``InstitutionalIntentEngine`` adjusts it via historical
        learning.

    ``range_compression``
        0-1: how much the recent price range has compressed relative to
        the baseline.  Isolated compression is not accumulation by
        itself — it requires concurrent absorption and volume evidence.

    ``absorption_strength``
        0-1: fraction of low-zone tests that closed above the zone,
        normalized to the expected number of touches.

    ``volume_buy_dominance``
        0-1: volume-weighted buy/sell balance.

    ``momentum_divergence``
        0-1: the extent to which momentum is diverging positively while
        price is at / near a low zone.

    ``structural_quality``
        0-1 estimate of how reliable the structural observation is,
        based on session quality and microstructure quality.  Used by
        the engine to weight the structural prior.
    """

    accumulation_prob: float = 0.0
    range_compression: float = 0.0
    absorption_strength: float = 0.0
    volume_buy_dominance: float = 0.5
    momentum_divergence: float = 0.0
    structural_quality: float = 0.5

    def to_dict(self) -> dict:
        return {
            "accumulation_prob": round(self.accumulation_prob, 4),
            "range_compression": round(self.range_compression, 4),
            "absorption_strength": round(self.absorption_strength, 4),
            "volume_buy_dominance": round(self.volume_buy_dominance, 4),
            "momentum_divergence": round(self.momentum_divergence, 4),
            "structural_quality": round(self.structural_quality, 4),
        }


# ── AccumulationDetector ──────────────────────────────────────────────────────


class AccumulationDetector:
    """Stateless structural accumulation probability estimator.

    Call ``detect(inp)`` directly if you have pre-computed signals.
    Call ``from_bars(bars, **context)`` to compute signals from raw bars
    and then detect in one step.

    Configuration
    -------------
    Defaults are calibrated for typical M15 FX intraday bars.  Adjust
    ``short_window`` and ``long_window`` for other timeframes.
    """

    def __init__(
        self,
        short_window: int = _SHORT_WINDOW,
        long_window: int = _LONG_WINDOW,
        touch_tolerance: float = _TOUCH_TOLERANCE,
        max_touches: int = _MAX_TOUCHES,
    ) -> None:
        self._sw = short_window
        self._lw = long_window
        self._ttol = touch_tolerance
        self._mt = max_touches

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, inp: AccumulationInput) -> AccumulationEstimate:
        """Compute the structural accumulation probability.

        If ``inp.bars`` is provided, bar-derived signals are computed
        automatically and override any pre-set scalar values on ``inp``.
        """
        if inp.bars and len(inp.bars) >= max(4, self._sw):
            rc, ab, vb, md = self._compute_bar_signals(inp.bars)
        else:
            rc = inp.range_compression
            ab = inp.absorption_strength
            vb = inp.volume_buy_dominance
            md = inp.momentum_divergence

        prob = self._accumulation_prob(rc, ab, vb, md, inp.level_proximity)
        sq = self._structural_quality(inp.session_quality, inp.microstructure_quality)

        return AccumulationEstimate(
            accumulation_prob=round(max(0.0, min(1.0, prob)), 4),
            range_compression=round(max(0.0, min(1.0, rc)), 4),
            absorption_strength=round(max(0.0, min(1.0, ab)), 4),
            volume_buy_dominance=round(max(0.0, min(1.0, vb)), 4),
            momentum_divergence=round(max(0.0, min(1.0, md)), 4),
            structural_quality=round(max(0.0, min(1.0, sq)), 4),
        )

    def from_bars(
        self,
        bars: list[BarSnapshot],
        momentum: float = 0.0,
        session_quality: float = 0.80,
        microstructure_quality: float = 0.80,
        level_proximity: float = 0.0,
    ) -> AccumulationEstimate:
        """Convenience: compute signals from raw bars and detect."""
        inp = AccumulationInput(
            bars=bars,
            momentum=momentum,
            session_quality=session_quality,
            microstructure_quality=microstructure_quality,
            level_proximity=level_proximity,
        )
        return self.detect(inp)

    # ── Core formula ──────────────────────────────────────────────────────

    def _accumulation_prob(
        self,
        range_compression: float,
        absorption_strength: float,
        volume_buy_dominance: float,
        momentum_divergence: float,
        level_proximity: float,
    ) -> float:
        """Structural accumulation probability from four components.

        Component weights (must sum to 1.0):

        ``range_compression`` (0.30)
            Price coiling is necessary but not sufficient for
            accumulation.  Receives the highest single weight because
            it is the most consistently observable structural feature.

        ``absorption_strength`` (0.30)
            Repeated bouncing from a low-zone is the clearest sign of
            persistent buy orders being placed there.  Equal weight to
            range compression.

        ``buy_vol_component`` (0.25)
            The fraction of buy volume relative to sell volume above
            the neutral 0.5 midpoint, mapped to a [0, 1] contribution.
            At pure buy dominance (vb=1.0) → full 0.25 contribution.
            At neutral (vb=0.5) → 0.  At sell dominance (vb<0.5) → 0
            (distribution, not accumulation).

        ``momentum_divergence`` (0.10) + ``level_proximity`` (0.05)
            Supporting signals that strengthen the interpretation when
            present but should not drive the overall estimate alone.
        """
        buy_vol_component = max(0.0, 2.0 * (volume_buy_dominance - 0.5))  # 0-1
        return (
            0.30 * range_compression
            + 0.30 * absorption_strength
            + 0.25 * buy_vol_component
            + 0.10 * momentum_divergence
            + 0.05 * level_proximity
        )

    def _structural_quality(self, session_quality: float, microstructure_quality: float) -> float:
        """Reliability weight for the structural observation.

        Higher quality = we trust the structural signals more.
        Poor session / microstructure = the signals are noisier.
        """
        return 0.5 * session_quality + 0.5 * microstructure_quality

    # ── Bar-derived signal computation ────────────────────────────────────

    def _compute_bar_signals(
        self,
        bars: list[BarSnapshot],
    ) -> tuple[float, float, float, float]:
        """Compute (range_compression, absorption, vol_buy_dom, momentum_divergence)
        from raw bars.

        Returns
        -------
        range_compression
            How compressed recent range is vs. baseline.
        absorption_strength
            How persistently price bounced from the low zone.
        volume_buy_dominance
            0-1 volume-weighted buy/sell balance.
        momentum_divergence
            Price near lows but range compressing upward.
        """
        n = len(bars)
        sw = min(self._sw, n // 2)
        lw = min(self._lw, n)

        recent_bars = bars[max(0, n - sw) :]
        baseline_bars = bars[max(0, n - lw) : max(0, n - sw)]

        # ── Range compression ─────────────────────────────────────────────
        recent_range = _mean([b.full_range for b in recent_bars])
        baseline_range = (
            _mean([b.full_range for b in baseline_bars]) if baseline_bars else recent_range
        )
        if baseline_range > 1e-12:
            range_compression = max(0.0, 1.0 - recent_range / baseline_range)
        else:
            range_compression = 0.0

        # ── Absorption: touches of the low zone that close above it ────────
        # A bar "touches" the zone if its low is within tolerance of the
        # window's lowest low.  "Closes above" means close > low_zone
        # (the actual zone price), not above the wider tolerance band.
        window_bars = bars[max(0, n - self._lw) :]
        if window_bars:
            low_zone = min(b.low for b in window_bars)
            touch_ceil = low_zone * (1.0 + self._ttol)  # zone upper edge
            touches = sum(1 for b in window_bars if b.low <= touch_ceil and b.close > low_zone)
            absorption = min(1.0, touches / max(self._mt, 1))
        else:
            absorption = 0.0

        # ── Volume buy dominance ──────────────────────────────────────────
        up_bars = [b for b in recent_bars if b.is_up]
        down_bars = [b for b in recent_bars if not b.is_up]
        avg_up_v = _mean([b.volume for b in up_bars]) if up_bars else 0.0
        avg_dn_v = _mean([b.volume for b in down_bars]) if down_bars else 0.0
        total_v = avg_up_v + avg_dn_v
        vb = (avg_up_v / total_v) if total_v > 1e-12 else 0.5

        # ── Momentum divergence: price near lows but range compressing ────
        # Divergence = price_position_in_range is LOW but range_compression
        # is HIGH.  High divergence suggests buy pressure absorbing downside.
        window_high = max((b.high for b in window_bars), default=0.0)
        window_low = min((b.low for b in window_bars), default=0.0)
        price_range = window_high - window_low
        if price_range > 1e-12 and window_bars:
            close_position = (window_bars[-1].close - window_low) / price_range
            # Divergence: price is LOW (close_position near 0) but range_compression is HIGH
            price_at_low_score = 1.0 - close_position  # high when close is near low
            momentum_divergence = price_at_low_score * range_compression
        else:
            momentum_divergence = 0.0

        return range_compression, absorption, vb, momentum_divergence


# ── Internal helper ───────────────────────────────────────────────────────────


def _mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
