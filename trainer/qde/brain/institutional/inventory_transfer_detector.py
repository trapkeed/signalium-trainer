"""QDE Brain — institutional: structural inventory transfer detector.

``InventoryTransferDetector`` estimates four related probabilities from
bar-pattern observations:

  * **Stop hunt** — an engineered excursion that sweeps stops above/below
    a structural level before reversing sharply.

  * **Liquidity acquisition** — a directional move toward a known zone
    of resting orders (stop clusters, equal highs/lows) where a large
    participant needs to fill a position against that pooled liquidity.

  * **Inventory transfer** — a sustained directional move that transfers
    an accumulated or distributed position to counterparty retail flow,
    typically on expanding volume into a liquidity zone.

  * **Position rebalancing** — low-conviction directional activity in
    quiet sessions that reflects portfolio adjustments rather than
    directional intent.  A residual category: present when no stronger
    signal dominates.

These four are conceptually distinct but structurally related:

  Stop hunt → engineered sweep → reversal (short duration, sharp)
  Liquidity acquisition → sustained move toward a known level cluster
  Inventory transfer → breakout-like move *through* a level on volume
  Rebalancing → drifting, low-session activity near familiar levels

What is NOT asserted
---------------------
* The detector never claims "this WAS a stop hunt."  It produces
  ``stop_hunt_prob: float ∈ [0, 1]`` — a structural prior that
  experienced behaviour patterns in the bar history.
* There are no ICT-specific rules.  The detector observes: "the price
  wick exceeded a known level and the bar closed back below it within N
  bars" — a neutral structural observation.  The model learns what that
  predicts from historical trade outcomes.

``SweepEvent``
--------------
The detector accepts ``SweepEvent`` records that summarise recent
structural sweeps already identified by the liquidity layer.  Callers
populate these from ``LiquidityEventDetector`` output (LEVEL_SWEPT
events) without the institutional package needing to import the
liquidity package.  ``SweepEvent`` is a minimal projection of the
information we need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trainer.qde.brain.institutional.accumulation_detector import BarSnapshot, _mean

# ── Tuneable defaults ─────────────────────────────────────────────────────────

_REVERSAL_WINDOW = 5  # bars after sweep to look for reversal close
_HUNT_VELOCITY_MULT = 1.5  # wick must be this × avg_range to count as aggressive
_TRANSFER_MIN_BARS = 3  # minimum bars of sustained directional move
_VOL_SPIKE_MULT = 1.8  # volume on transfer bar vs. recent average
_SWEEP_DECAY_BARS = 20  # after this many bars, sweep recency → 0


# ── SweepEvent ────────────────────────────────────────────────────────────────


@dataclass
class SweepEvent:
    """Minimal descriptor of a recently observed structural sweep.

    Callers construct this from ``LiquidityEventDetector`` LEVEL_SWEPT
    output without importing the liquidity package directly.

    ``side``
        ``"high"`` if the sweep was above a HIGH-side level,
        ``"low"`` if below a LOW-side level.

    ``bars_ago``
        How many bars ago the sweep was detected.  0 = current bar.

    ``level_price``
        The price of the swept level.

    ``reversal_confirmed``
        True when a subsequent bar closed back through the level
        (within ``_REVERSAL_WINDOW`` bars of the sweep).  Set by the
        caller from LEVEL_SWEPT event context or from subsequent
        observation.

    ``wick_to_range_ratio``
        The wick length that exceeded the level divided by the average
        recent bar range.  Higher values = more aggressive sweep.
    """

    side: str = "high"
    bars_ago: int = 0
    level_price: float = 0.0
    reversal_confirmed: bool = False
    wick_to_range_ratio: float = 1.0


# ── InventoryTransferInput ────────────────────────────────────────────────────


@dataclass
class InventoryTransferInput:
    """Structural inputs for inventory-transfer pattern detection.

    ``bars``
        Recent OHLCV history, oldest first.  Minimum 6 bars required
        for meaningful signal computation.

    ``sweep_events``
        Recent structural sweeps from the liquidity layer.  May be
        empty; when empty, stop_hunt_prob falls back to bar-pattern
        analysis only.

    ``momentum``
        Signed −1..+1.  Direction of current momentum.

    ``session_quality``
        0-1 session quality.  Stop hunts are more common during liquid
        sessions (London, NY); rebalancing is more common off-hours.

    ``microstructure_quality``
        0-1.  Clean microstructure makes liquidity acquisition more
        visible.

    ``level_proximity``
        0-1 proximity to the nearest structural level.  Liquidity
        acquisition accelerates as price approaches a known level.

    ``level_proximity_direction``
        +1 if price is moving toward the nearest level, −1 if moving
        away.  Used to distinguish approach (acquisition) vs. departure
        (post-transfer).
    """

    bars: list[BarSnapshot] | None = None
    sweep_events: list[SweepEvent] = field(default_factory=list)

    momentum: float = 0.0
    session_quality: float = 0.80
    microstructure_quality: float = 0.80
    level_proximity: float = 0.0
    level_proximity_direction: int = 0  # +1 approaching, −1 departing


# ── InventoryTransferEstimate ─────────────────────────────────────────────────


@dataclass(frozen=True)
class InventoryTransferEstimate:
    """Structural probabilities for inventory-transfer patterns.

    All four probabilities are independent priors.  Their sum may
    exceed 1.0 — they are not a probability distribution over mutually
    exclusive hypotheses, but rather four independent structural
    likelihoods.

    ``stop_hunt_prob``
        P(a stop hunt occurred or is in progress).  Elevated when a
        recent structural sweep reversed quickly and aggressively.

    ``liquidity_acq_prob``
        P(a liquidity acquisition move is underway).  Elevated when
        price is moving toward a level proximity zone with sustained
        directional momentum.

    ``transfer_prob``
        P(an inventory transfer is in progress).  Elevated when there
        is a sustained breakout-like directional move on above-average
        volume through a structural level.

    ``rebalance_prob``
        P(the activity reflects position rebalancing).  A residual
        estimate; elevated in quiet sessions near familiar levels with
        low directionality.

    ``sweep_recency``
        0-1: how recently a structural sweep was observed.  1 = very
        recent (0-2 bars ago), decaying to 0 over ``_SWEEP_DECAY_BARS``.

    ``volume_signature``
        0-1: volume anomaly score.  High when current volume is
        significantly above recent average.

    ``directional_consistency``
        0-1: how consistently the bars in the recent window are
        directional (all up or all down).

    ``structural_quality``
        0-1 reliability weight.
    """

    stop_hunt_prob: float = 0.0
    liquidity_acq_prob: float = 0.0
    transfer_prob: float = 0.0
    rebalance_prob: float = 0.0

    sweep_recency: float = 0.0
    volume_signature: float = 0.0
    directional_consistency: float = 0.0
    structural_quality: float = 0.5

    def to_dict(self) -> dict:
        return {
            "stop_hunt_prob": round(self.stop_hunt_prob, 4),
            "liquidity_acq_prob": round(self.liquidity_acq_prob, 4),
            "transfer_prob": round(self.transfer_prob, 4),
            "rebalance_prob": round(self.rebalance_prob, 4),
            "sweep_recency": round(self.sweep_recency, 4),
            "volume_signature": round(self.volume_signature, 4),
            "directional_consistency": round(self.directional_consistency, 4),
            "structural_quality": round(self.structural_quality, 4),
        }


# ── InventoryTransferDetector ─────────────────────────────────────────────────


class InventoryTransferDetector:
    """Stateless structural estimator for inventory-transfer patterns.

    Estimates stop-hunt, liquidity-acquisition, transfer, and
    rebalancing probabilities from bars and optional sweep events.
    """

    def __init__(
        self,
        reversal_window: int = _REVERSAL_WINDOW,
        hunt_velocity_mult: float = _HUNT_VELOCITY_MULT,
        transfer_min_bars: int = _TRANSFER_MIN_BARS,
        vol_spike_mult: float = _VOL_SPIKE_MULT,
        sweep_decay_bars: int = _SWEEP_DECAY_BARS,
    ) -> None:
        self._rev_win = reversal_window
        self._hunt_mult = hunt_velocity_mult
        self._xfer_min = transfer_min_bars
        self._vol_mult = vol_spike_mult
        self._decay = sweep_decay_bars

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, inp: InventoryTransferInput) -> InventoryTransferEstimate:
        """Compute the structural inventory-transfer pattern probabilities."""
        bars = inp.bars or []
        n = len(bars)

        # Structural signals from bars (when available)
        if n >= 4:
            vol_sig = self._volume_signature(bars)
            dir_con = self._directional_consistency(bars)
        else:
            vol_sig = 0.0
            dir_con = 0.5

        # Sweep recency from explicit events (takes precedence over bar scan)
        if inp.sweep_events:
            sweep_rec = self._sweep_recency_from_events(inp.sweep_events)
            hunt_prob = self._stop_hunt_from_events(inp.sweep_events, dir_con)
        else:
            sweep_rec = self._sweep_recency_from_bars(bars) if n >= 4 else 0.0
            hunt_prob = self._stop_hunt_from_bars(bars, sweep_rec) if n >= 4 else 0.0

        liq_prob = self._liquidity_acq_prob(
            inp.level_proximity,
            inp.level_proximity_direction,
            dir_con,
            inp.session_quality,
        )
        xfer_prob = self._transfer_prob(dir_con, vol_sig, inp.level_proximity)
        rebal_prob = self._rebalance_prob(
            hunt_prob,
            liq_prob,
            xfer_prob,
            inp.session_quality,
            inp.level_proximity,
        )

        sq = 0.5 * inp.session_quality + 0.5 * inp.microstructure_quality

        return InventoryTransferEstimate(
            stop_hunt_prob=round(max(0.0, min(1.0, hunt_prob)), 4),
            liquidity_acq_prob=round(max(0.0, min(1.0, liq_prob)), 4),
            transfer_prob=round(max(0.0, min(1.0, xfer_prob)), 4),
            rebalance_prob=round(max(0.0, min(1.0, rebal_prob)), 4),
            sweep_recency=round(max(0.0, min(1.0, sweep_rec)), 4),
            volume_signature=round(max(0.0, min(1.0, vol_sig)), 4),
            directional_consistency=round(max(0.0, min(1.0, dir_con)), 4),
            structural_quality=round(max(0.0, min(1.0, sq)), 4),
        )

    # ── Stop hunt ─────────────────────────────────────────────────────────

    def _stop_hunt_from_events(
        self,
        events: list[SweepEvent],
        dir_con: float,
    ) -> float:
        """Stop-hunt probability from explicit sweep events.

        Combines:
        * recency: how recently was the sweep?
        * aggression: was the wick larger than average range?
        * reversal: did price close back through the level?

        Reversal confirmation is the strongest single signal: a sweep
        that immediately reverses is the archetypal stop hunt.
        """
        if not events:
            return 0.0

        best = 0.0
        for ev in events:
            recency = max(0.0, 1.0 - ev.bars_ago / max(self._decay, 1))
            aggression = min(1.0, ev.wick_to_range_ratio / max(self._hunt_mult, 1e-9))
            reversal_w = 1.0 if ev.reversal_confirmed else 0.3
            p = recency * aggression * reversal_w
            best = max(best, p)
        return best

    def _stop_hunt_from_bars(self, bars: list[BarSnapshot], sweep_recency: float) -> float:
        """Bar-only stop-hunt proxy when no explicit sweep events exist.

        Heuristic: a bar with a very long wick relative to body that
        closes sharply back into the previous range.  This is a neutral
        structural observation (the LiquidityEventDetector produces the
        authoritative sweep classification; this is a fallback).
        """
        if len(bars) < 3:
            return 0.0

        recent = bars[-8:]
        avg_range = _mean([b.full_range for b in recent]) or 1e-9
        last = bars[-1]

        # Long upper wick: (high − max(open, close)) / full_range
        upper_wick = max(0.0, last.high - max(last.open, last.close))
        lower_wick = max(0.0, min(last.open, last.close) - last.low)
        dominant_wick = max(upper_wick, lower_wick)
        wick_ratio = dominant_wick / max(last.full_range, 1e-9)

        # Strong wick + sweep_recency = elevated stop-hunt prior
        wick_signal = min(1.0, wick_ratio * 2.0)  # saturates at 50% wick
        return sweep_recency * 0.6 + wick_signal * 0.4

    # ── Sweep recency helpers ─────────────────────────────────────────────

    def _sweep_recency_from_events(self, events: list[SweepEvent]) -> float:
        """0-1 recency weight from the most recent sweep event."""
        if not events:
            return 0.0
        min_bars_ago = min(ev.bars_ago for ev in events)
        return max(0.0, 1.0 - min_bars_ago / max(self._decay, 1))

    def _sweep_recency_from_bars(self, bars: list[BarSnapshot]) -> float:
        """Approximate sweep recency from wick patterns when no sweep events exist.

        Looks for the most recent bar whose dominant wick exceeds
        ``_HUNT_VELOCITY_MULT × avg_range`` — a proxy for a sweep bar.
        """
        if len(bars) < 4:
            return 0.0
        window = bars[-min(self._decay, len(bars)) :]
        avg_range = _mean([b.full_range for b in window]) or 1e-9
        threshold = avg_range * self._hunt_mult

        for i, bar in enumerate(reversed(window)):
            upper = max(0.0, bar.high - max(bar.open, bar.close))
            lower = max(0.0, min(bar.open, bar.close) - bar.low)
            if max(upper, lower) >= threshold:
                bars_ago = i
                return max(0.0, 1.0 - bars_ago / max(self._decay, 1))
        return 0.0

    # ── Liquidity acquisition ─────────────────────────────────────────────

    def _liquidity_acq_prob(
        self,
        level_proximity: float,
        level_proximity_direction: int,
        directional_consistency: float,
        session_quality: float,
    ) -> float:
        """Liquidity acquisition probability.

        Elevated when:
        * Price is approaching a known level (proximity high, direction +1)
        * The approach is directionally consistent (sustained move)
        * The session is liquid (institutional participants active)

        Formula:
            p = proximity × approach_factor × consistency × session_factor
            approach_factor: 1.0 when approaching (+1), 0.3 when departing (−1)
            session_factor:  [0.5, 1.0] — capped not zeroed (off-hours liq.
                             acquisition happens too, just less likely)
        """
        approach_factor = 1.0 if level_proximity_direction >= 0 else 0.3
        session_factor = 0.5 + 0.5 * session_quality
        return level_proximity * approach_factor * directional_consistency * session_factor

    # ── Inventory transfer ────────────────────────────────────────────────

    def _transfer_prob(
        self,
        directional_consistency: float,
        volume_signature: float,
        level_proximity: float,
    ) -> float:
        """Inventory transfer probability.

        Transfer is a sustained directional move (high consistency) on
        elevated volume (high volume_signature) at or through a
        structural level (high proximity).

        Transfer differs from liquidity acquisition in that it *breaks
        through* levels rather than approaching them, and requires
        expanding volume (the retail counterparty is being lured in by
        the move).
        """
        return 0.40 * directional_consistency + 0.40 * volume_signature + 0.20 * level_proximity

    # ── Position rebalancing ──────────────────────────────────────────────

    def _rebalance_prob(
        self,
        hunt_prob: float,
        liq_prob: float,
        xfer_prob: float,
        session_quality: float,
        level_proximity: float,
    ) -> float:
        """Residual position-rebalancing probability.

        Rebalancing is the weakest signal: it occurs when no stronger
        institutional pattern is present.  High during off-hours sessions
        near familiar levels, low when other patterns dominate.

        Formula:
            residual = 1 − max(hunt, liq, xfer)
            p = residual × off_hours_weight × proximity_weight
        """
        dominant_signal = max(hunt_prob, liq_prob, xfer_prob)
        residual = max(0.0, 1.0 - dominant_signal)
        off_hours = 1.0 - session_quality  # higher in quiet sessions
        return residual * (0.5 * off_hours + 0.5 * level_proximity)

    # ── Volume and direction ───────────────────────────────────────────────

    def _volume_signature(self, bars: list[BarSnapshot]) -> float:
        """0-1 volume anomaly: how elevated is recent volume vs. baseline?

        Compares the last bar's volume to the rolling mean.  Elevated
        volume (> ``_VOL_SPIKE_MULT × avg``) saturates at 1.0.
        When all bars have unit volume (no volume data), returns 0.0
        (neutral) rather than a false signal.
        """
        if not bars:
            return 0.0
        # If all volumes are 1.0 (default), we have no real volume data
        volumes = [b.volume for b in bars]
        if max(volumes) - min(volumes) < 1e-9:
            return 0.0  # no volume variance → no volume signal

        baseline_vols = volumes[:-1] if len(volumes) > 1 else volumes
        avg_vol = _mean(baseline_vols) or 1e-9
        ratio = bars[-1].volume / avg_vol
        return min(1.0, max(0.0, (ratio - 1.0) / max(self._vol_mult - 1.0, 0.1)))

    def _directional_consistency(self, bars: list[BarSnapshot]) -> float:
        """0-1: how consistently directional are the recent bars?

        Fraction of bars in the recent window that share the same
        direction (all up or all down), normalized so 0.5 = half-and-half
        and 1.0 = all bars the same direction.

        A value > 0.7 suggests a sustained directional push; a value
        near 0.5 suggests sideways or mixed price action.
        """
        window = bars[-min(8, len(bars)) :]
        if not window:
            return 0.5
        n_up = sum(1 for b in window if b.is_up)
        n_down = len(window) - n_up
        return max(n_up, n_down) / max(len(window), 1)
