"""C10 — Signal enricher for the four new attention factors.

Produces the four factors missing from the existing six-signal
``OpportunitySignals`` that ``opportunity_scanner.scan()`` generates:

  ``volume_anomaly``       — Unusual volume relative to recent baseline.
                             High volume anomaly means the market is
                             "voting with size" — worth elevating attention.

  ``key_level_proximity``  — How close price is to a significant structural
                             level (support, resistance, prior high/low,
                             round number).  Proximity raises setup probability
                             and risk both, so the QDE must watch more closely.

  ``spread_change``        — Spread widening or narrowing relative to the
                             symbol's recent baseline.  A sudden spread spike
                             signals a liquidity condition change that may
                             affect fill quality and should trigger tighter
                             monitoring.

  ``failed_move``          — A recent failed breakout or momentum rejection:
                             price moved beyond a key level but reversed,
                             trapping participants.  Failed moves frequently
                             precede strong counter-moves and deserve elevated
                             attention.

Design constraints
------------------
* Pure signal-detection layer — no state mutation, no Brain access.
* Accepts an ``ExtendedSignalContext`` with the additional market data
  not present in ``SymbolObservationInput`` (spread, volume, key levels,
  recent failed-move info).  All fields are optional (default to neutral);
  callers supply what they have.
* Returns a new ``OpportunitySignals`` with the four additional fields
  populated.  The first six fields are passed through unchanged from the
  ``base`` argument so this module never re-derives the original signals.
* Mirrors the style of ``opportunity_scanner.py`` — each signal has its
  own private function, the public entry point is ``enrich_signals()``.
"""

from __future__ import annotations

from dataclasses import dataclass

from trainer.qde.brain.opportunity_scanner import OpportunitySignals

# ── Thresholds (kept as module constants so they're easy to tune/test) ────────

# Volume: ratio of current volume to baseline beyond which we flag an anomaly.
_VOLUME_HIGH_RATIO = 2.0      # >2× baseline → strong anomaly
_VOLUME_MOD_RATIO  = 1.5      # >1.5× baseline → moderate anomaly

# Key level: distance thresholds as a fraction of current price.
_LEVEL_CRITICAL_PCT = 0.001   # within 0.1% → very close (0.9)
_LEVEL_NEAR_PCT     = 0.003   # within 0.3% → near (0.6)
_LEVEL_WATCH_PCT    = 0.007   # within 0.7% → on radar (0.3)

# Spread: normalized change thresholds (spread_normalized is 0-1).
_SPREAD_SPIKE_DELTA    = 0.30  # ≥0.30 normalized delta → strong change
_SPREAD_NOTABLE_DELTA  = 0.15  # ≥0.15 normalized delta → notable change

# Failed move: wick/body ratio beyond which we flag a significant rejection.
_FAILED_MOVE_STRONG_RATIO  = 2.0
_FAILED_MOVE_NOTABLE_RATIO = 1.2


# ─────────────────────────────────────────────────────────────────────────────
# Extended signal context (the additional inputs for the four new factors)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExtendedSignalContext:
    """Optional additional market data for the four new attention signals.

    All fields have neutral defaults so callers can supply only what they
    have without breaking the enrichment.  A caller that populates none of
    these fields will get 0.0 for all four new signals — equivalent to the
    pre-C10 behavior.

    ``volume_ratio``            — Current bar volume divided by the recent
                                  baseline (N-bar average).  1.0 = normal;
                                  2.0 = twice the baseline (strong anomaly).
    ``spread_normalized``       — Current spread as a 0-1 normalized value
                                  relative to the symbol's historical range
                                  (0=tight, 1=wide).  From ``SpreadPoint``.
    ``prior_spread_normalized`` — Spread normalized value from the previous
                                  tick, for detecting changes.
    ``nearest_level_price``     — Price of the nearest structural level
                                  (support/resistance/round number).
    ``current_price``           — Current mid-price, for calculating
                                  distance to ``nearest_level_price``.
    ``recent_failed_breakout``  — Whether a ``FailedBreakout`` event was
                                  recorded in the last N bars.
    ``failed_move_wick_ratio``  — The wick/body ratio of the most recent
                                  failed move (from ``FailedBreakout``).
                                  0.0 if no recent failed move.
    ``opportunity_quality``     — A 0-1 quality score from the opportunity
                                  scanner / hypothesis layer, distinct from
                                  hypothesis_activity.  Can be left at 0.0
                                  if not available; it augments rather than
                                  replaces ``hypothesis_activity``.
    """

    volume_ratio: float = 1.0               # 1.0 = baseline-normal
    spread_normalized: float = 0.5          # 0-1 normalized spread
    prior_spread_normalized: float = 0.5    # previous tick's normalized spread
    nearest_level_price: float = 0.0        # 0 = no level data available
    current_price: float = 0.0              # 0 = no price data available
    recent_failed_breakout: bool = False
    failed_move_wick_ratio: float = 0.0     # 0 = no recent failed move
    opportunity_quality: float = 0.0        # 0-1 direct quality score


# ─────────────────────────────────────────────────────────────────────────────
# Per-signal detection functions (private, stateless)
# ─────────────────────────────────────────────────────────────────────────────


def _volume_anomaly(ctx: ExtendedSignalContext) -> float:
    """Unusual volume relative to the recent baseline."""
    ratio = max(0.0, ctx.volume_ratio)
    if ratio >= _VOLUME_HIGH_RATIO:
        return 1.0
    if ratio >= _VOLUME_MOD_RATIO:
        # Linear interpolation between mod and high thresholds.
        t = (ratio - _VOLUME_MOD_RATIO) / (_VOLUME_HIGH_RATIO - _VOLUME_MOD_RATIO)
        return round(0.5 + 0.5 * t, 4)
    if ratio >= 1.0:
        return 0.0   # at or below baseline — no anomaly
    # Below-baseline volume (thin market) is also notable.
    below = max(0.0, 1.0 - ratio)
    return min(0.4, below * 0.6)


def _key_level_proximity(ctx: ExtendedSignalContext) -> float:
    """Price proximity to the nearest structural level."""
    if ctx.nearest_level_price <= 0 or ctx.current_price <= 0:
        return 0.0
    distance_pct = abs(ctx.current_price - ctx.nearest_level_price) / ctx.current_price
    if distance_pct <= _LEVEL_CRITICAL_PCT:
        return 0.9
    if distance_pct <= _LEVEL_NEAR_PCT:
        # Interpolate 0.6–0.9 between NEAR and CRITICAL.
        t = (distance_pct - _LEVEL_CRITICAL_PCT) / (_LEVEL_NEAR_PCT - _LEVEL_CRITICAL_PCT)
        return round(0.9 - 0.3 * t, 4)
    if distance_pct <= _LEVEL_WATCH_PCT:
        # Interpolate 0.3–0.6 between WATCH and NEAR.
        t = (distance_pct - _LEVEL_NEAR_PCT) / (_LEVEL_WATCH_PCT - _LEVEL_NEAR_PCT)
        return round(0.6 - 0.3 * t, 4)
    return 0.0


def _spread_change(ctx: ExtendedSignalContext) -> float:
    """Spread widening or tightening relative to the previous tick."""
    delta = abs(ctx.spread_normalized - ctx.prior_spread_normalized)
    if delta >= _SPREAD_SPIKE_DELTA:
        return 1.0
    if delta >= _SPREAD_NOTABLE_DELTA:
        t = (delta - _SPREAD_NOTABLE_DELTA) / (_SPREAD_SPIKE_DELTA - _SPREAD_NOTABLE_DELTA)
        return round(0.5 + 0.5 * t, 4)
    # Scale linearly below the notable threshold.
    return min(0.5, delta / _SPREAD_NOTABLE_DELTA * 0.5)


def _failed_move(ctx: ExtendedSignalContext) -> float:
    """Recent failed breakout or momentum rejection."""
    if not ctx.recent_failed_breakout:
        return 0.0
    ratio = max(0.0, ctx.failed_move_wick_ratio)
    if ratio >= _FAILED_MOVE_STRONG_RATIO:
        return 1.0
    if ratio >= _FAILED_MOVE_NOTABLE_RATIO:
        t = (ratio - _FAILED_MOVE_NOTABLE_RATIO) / (
            _FAILED_MOVE_STRONG_RATIO - _FAILED_MOVE_NOTABLE_RATIO
        )
        return round(0.5 + 0.5 * t, 4)
    # Failed breakout confirmed but wick/body ratio not extreme → mild signal.
    return 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def enrich_signals(
    base: OpportunitySignals,
    ctx: ExtendedSignalContext,
) -> OpportunitySignals:
    """Return a new ``OpportunitySignals`` with the four C10 factors populated.

    The first six fields are passed through unchanged from ``base`` so this
    function never re-derives the original signals — it only adds the four
    new ones computed from ``ctx``.

    Typical usage::

        base = scan(observation, previous)
        ctx  = ExtendedSignalContext(
            volume_ratio=current_volume / baseline_volume,
            spread_normalized=spread_point.spread_normalized,
            ...
        )
        enriched = enrich_signals(base, ctx)
        score = allocator.update(instrument, enriched)
    """
    return OpportunitySignals(
        # Pass-through the six existing signals unchanged.
        hypothesis_activity=base.hypothesis_activity,
        volatility_expansion=base.volatility_expansion,
        liquidity_event=base.liquidity_event,
        news_impact=base.news_impact,
        unusual_behavior=base.unusual_behavior,
        session_transition=base.session_transition,
        # The four new C10 factors.
        volume_anomaly=_volume_anomaly(ctx),
        key_level_proximity=_key_level_proximity(ctx),
        spread_change=_spread_change(ctx),
        failed_move=_failed_move(ctx),
    )


class SignalEnricher:
    """Stateful wrapper that caches ``ExtendedSignalContext`` between calls.

    Useful when the caller builds up the context incrementally (e.g. pulling
    spread history from ``WorkingMemoryStore``, volume from MIE, levels from
    the liquidity layer) and wants a single object to hand to the attention
    pipeline rather than re-constructing ``ExtendedSignalContext`` at each
    call site.

    Call ``update_context(instrument, ctx)`` whenever new market data
    arrives, then ``enrich(instrument, base_signals)`` to produce the
    full ``OpportunitySignals``.
    """

    def __init__(self) -> None:
        self._contexts: dict[str, ExtendedSignalContext] = {}

    def update_context(self, instrument: str, ctx: ExtendedSignalContext) -> None:
        """Store the latest extended context for ``instrument``."""
        self._contexts[instrument] = ctx

    def enrich(self, instrument: str, base: OpportunitySignals) -> OpportunitySignals:
        """Enrich ``base`` with the stored context for ``instrument``.

        Returns ``base`` unchanged if no context has been stored yet (i.e.
        the four new signals will be 0.0 — graceful degradation on first tick).
        """
        ctx = self._contexts.get(instrument)
        if ctx is None:
            return base
        return enrich_signals(base, ctx)

    def remove(self, instrument: str) -> None:
        self._contexts.pop(instrument, None)


__all__ = [
    "ExtendedSignalContext",
    "enrich_signals",
    "SignalEnricher",
]
