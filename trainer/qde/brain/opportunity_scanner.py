"""QDE Brain — opportunity scanner.

Pure signal-detection layer for the attention system: given a symbol's
current and previous `SymbolObservationInput` (plus a little hypothesis
context), detects the six conditions that should raise a symbol's
attention score:

  1. developing hypotheses  — new/reinforced hypotheses, or several
     active at once
  2. volatility expansion   — volatility state/quality escalating
  3. liquidity events       — a sudden liquidity regime shift
  4. news impact            — `MarketContext.news_risk` (reused as-is,
     never re-derived — news risk classification is owned upstream)
  5. unusual behavior       — regime flip, invalidated trend, or a
     momentum/volatility mismatch inconsistent with the current regime
  6. session transitions    — session identity change or a session
     quality bucket crossing

This module produces raw signal strengths only — it never decides
weights, decay, or cross-symbol normalization (that's
`attention_allocator.py`), and never mutates any Brain state (that's
`attention_engine.py`'s job as orchestrator).
"""

from __future__ import annotations

from dataclasses import dataclass

from trainer.qde.qde_observer_models import SymbolObservationInput

# Ordinal ranks for state escalation detection — mirrors the bucket-style
# comparisons already used descriptively in `narratives.py`, kept local
# since these are trivial ordering conventions, not shared business logic.
_VOLATILITY_RANK = {"LOW": 0, "NORMAL": 1, "ELEVATED": 2, "HIGH": 3, "EXTREME": 4}
_LIQUIDITY_RANK = {"POOR": 0, "THIN": 0, "LOW": 0, "NORMAL": 1, "GOOD": 2, "EXCELLENT": 2}

_SESSION_QUALITY_BUCKETS = (0.75, 0.45)  # prime / moderate / thin boundaries


@dataclass(frozen=True)
class OpportunitySignals:
    """Raw, unweighted 0-1 strength for each attention-worthy condition
    observed on one tick for one symbol.

    The first six fields are produced by ``scan()`` from a plain
    ``SymbolObservationInput``.  The four additional C10 fields
    (``volume_anomaly``, ``key_level_proximity``, ``spread_change``,
    ``failed_move``) default to 0.0 and are populated by
    ``trainer.qde.brain.attention.signal_enricher.enrich_signals()``
    when the caller supplies an ``ExtendedSignalContext`` with spread,
    volume, and key-level data.  Backward-compatible: all existing code
    that constructs ``OpportunitySignals`` with keyword arguments and
    omits the four new fields continues to work unchanged.
    """

    # ── Original six signals ─────────────────────────────────────────────────
    hypothesis_activity: float = 0.0
    volatility_expansion: float = 0.0
    liquidity_event: float = 0.0
    news_impact: float = 0.0
    unusual_behavior: float = 0.0
    session_transition: float = 0.0

    # ── C10: four new attention factors ──────────────────────────────────────
    # Populated by signal_enricher.enrich_signals(); remain 0.0 when the
    # caller does not supply an ExtendedSignalContext.
    volume_anomaly: float = 0.0       # unusual volume vs recent baseline
    key_level_proximity: float = 0.0  # price closeness to a structural level
    spread_change: float = 0.0        # spread widening/narrowing signal
    failed_move: float = 0.0          # recent failed breakout or rejection

    def to_dict(self) -> dict:
        return {
            "hypothesis_activity": round(self.hypothesis_activity, 4),
            "volatility_expansion": round(self.volatility_expansion, 4),
            "liquidity_event": round(self.liquidity_event, 4),
            "news_impact": round(self.news_impact, 4),
            "unusual_behavior": round(self.unusual_behavior, 4),
            "session_transition": round(self.session_transition, 4),
            # C10 factors
            "volume_anomaly": round(self.volume_anomaly, 4),
            "key_level_proximity": round(self.key_level_proximity, 4),
            "spread_change": round(self.spread_change, 4),
            "failed_move": round(self.failed_move, 4),
        }

    def max_signal(self) -> float:
        return max(
            self.hypothesis_activity,
            self.volatility_expansion,
            self.liquidity_event,
            self.news_impact,
            self.unusual_behavior,
            self.session_transition,
            self.volume_anomaly,
            self.key_level_proximity,
            self.spread_change,
            self.failed_move,
        )


def _session_bucket(session_quality: float) -> int:
    if session_quality >= _SESSION_QUALITY_BUCKETS[0]:
        return 2
    if session_quality >= _SESSION_QUALITY_BUCKETS[1]:
        return 1
    return 0


def _volatility_expansion(
    current: SymbolObservationInput, previous: SymbolObservationInput | None
) -> float:
    if previous is None:
        return 0.0
    cur_rank = _VOLATILITY_RANK.get(current.volatility_state.upper(), 1)
    prev_rank = _VOLATILITY_RANK.get(previous.volatility_state.upper(), 1)
    escalation = max(0, cur_rank - prev_rank) / 4.0
    quality_drop = max(0.0, previous.volatility_quality - current.volatility_quality)
    return min(1.0, escalation + 0.3 * quality_drop)


def _liquidity_event(
    current: SymbolObservationInput, previous: SymbolObservationInput | None
) -> float:
    if previous is None:
        return 0.0
    cur_rank = _LIQUIDITY_RANK.get(current.liquidity_state.upper(), 1)
    prev_rank = _LIQUIDITY_RANK.get(previous.liquidity_state.upper(), 1)
    state_shift = abs(cur_rank - prev_rank) / 2.0
    quality_shift = abs(current.liquidity_quality - previous.liquidity_quality)
    return min(1.0, state_shift + quality_shift)


def _unusual_behavior(
    current: SymbolObservationInput, previous: SymbolObservationInput | None
) -> float:
    score = 0.0
    if not current.market.trend_valid:
        score += 0.5
    if previous is not None:
        cur_regime = current.market_regime.upper()
        prev_regime = previous.market_regime.upper()
        flipped = {cur_regime, prev_regime} == {"TREND_UP", "TREND_DOWN"}
        if flipped:
            score += 0.6
        momentum_jump = abs(current.market.momentum - previous.market.momentum)
        if momentum_jump > 0.5:
            score += 0.4
    # A trending regime with markedly poor microstructure quality is an
    # inconsistency worth a closer look (momentum without clean structure).
    if (
        current.market_regime.upper() in ("TREND_UP", "TREND_DOWN")
        and current.microstructure_quality < 0.3
    ):
        score += 0.3
    return min(1.0, score)


def _session_transition(
    current: SymbolObservationInput, previous: SymbolObservationInput | None
) -> float:
    if previous is None:
        return 0.0
    if current.market.session != previous.market.session:
        return 1.0
    if _session_bucket(current.session_quality) != _session_bucket(previous.session_quality):
        return 0.6
    return 0.0


def _hypothesis_activity(
    active_hypothesis_count: int,
    new_hypothesis_formed: bool,
    hypothesis_reinforced: bool,
) -> float:
    if new_hypothesis_formed:
        return 1.0
    if hypothesis_reinforced:
        return 0.6
    return min(1.0, active_hypothesis_count * 0.2)


def scan(
    current: SymbolObservationInput,
    previous: SymbolObservationInput | None = None,
    active_hypothesis_count: int = 0,
    new_hypothesis_formed: bool = False,
    hypothesis_reinforced: bool = False,
) -> OpportunitySignals:
    """Detect this tick's attention-worthy conditions for one symbol.

    `previous` is the prior tick's `SymbolObservationInput` for the same
    symbol (caller-supplied — this module holds no state of its own), so
    all "did X change" detections are one-tick deltas; `None` on the
    first observed tick simply yields no delta-based signals yet.
    """
    return OpportunitySignals(
        hypothesis_activity=_hypothesis_activity(
            active_hypothesis_count, new_hypothesis_formed, hypothesis_reinforced
        ),
        volatility_expansion=_volatility_expansion(current, previous),
        liquidity_event=_liquidity_event(current, previous),
        news_impact=max(0.0, min(1.0, current.market.news_risk)),
        unusual_behavior=_unusual_behavior(current, previous),
        session_transition=_session_transition(current, previous),
    )
