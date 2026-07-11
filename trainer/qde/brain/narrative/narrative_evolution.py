"""QDE Brain — narrative evolution.

Stateless functions that update an *existing* `SymbolNarrative` when a
new market observation arrives for the same kind.  The distinction from
`NarrativeBuilder` is intentional:

* `NarrativeBuilder.detect` → fires when a *new* narrative kind is
  emerging for the first time (or re-emerging after expiry).
* `NarrativeEvolution.evolve` → fires when that kind already has a
  living narrative and the goal is incremental revision — reinforcing
  or weakening confidence, updating regime/session context, and
  recomputing maturity — without discarding accumulated evidence history.

This mirrors how `NarrativeEngine.update` in `narratives.py` revises the
single-instance `MarketNarrative` incrementally rather than rebuilding it
from scratch, and for the same reason: narrative continuity matters.  A
QDE that forgets its story every tick cannot reason from sustained context.

All methods are pure — they mutate the passed `SymbolNarrative` in place
(following the `Hypothesis.reinforce / weaken` convention) and return it
for chaining.  No storage interaction happens here.
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain_models import Hypothesis, MarketNarrative
from trainer.qde.brain.brain_types import BeliefDirection, TrendMaturity
from trainer.qde.brain.narrative.narrative_store import (
    NarrativeKind,
    NarrativeMaturity,
    SymbolNarrative,
)
from trainer.qde.qde_observer_models import SymbolObservationInput

_VOL_RANK: dict[str, int] = {"LOW": 0, "NORMAL": 1, "ELEVATED": 2, "HIGH": 3, "EXTREME": 4}
_LIQ_RANK: dict[str, int] = {"POOR": 0, "THIN": 0, "LOW": 0, "NORMAL": 1, "GOOD": 2, "EXCELLENT": 2}


class NarrativeEvolution:
    """Incrementally revise one `SymbolNarrative` from a new market tick.

    Stateless — safe to reuse across symbols and instantiate once per engine.
    """

    def evolve(
        self,
        narrative: SymbolNarrative,
        inp: SymbolObservationInput,
        existing_narrative: MarketNarrative | None,
        active_hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        now: float | None = None,
    ) -> SymbolNarrative:
        """Update `narrative` in place from this tick's observation.

        Returns the same object for caller convenience.  Does not create a
        new narrative; that is `NarrativeBuilder`'s responsibility.
        """
        if narrative.maturity == NarrativeMaturity.EXPIRED:
            return narrative

        now = now if now is not None else time.time()
        regime = inp.market_regime.upper()
        vol_rank = _VOL_RANK.get(inp.volatility_state.upper(), 1)
        liq_rank = _LIQ_RANK.get(inp.liquidity_state.upper(), 1)
        trend_maturity = (
            existing_narrative.trend_maturity if existing_narrative else TrendMaturity.NONE
        )

        # Dispatch to the per-kind evolution method.
        dispatcher = {
            NarrativeKind.TREND_CONTINUATION: self._evolve_trend_continuation,
            NarrativeKind.REVERSAL: self._evolve_reversal,
            NarrativeKind.ACCUMULATION: self._evolve_accumulation,
            NarrativeKind.DISTRIBUTION: self._evolve_distribution,
            NarrativeKind.BREAKOUT: self._evolve_breakout,
            NarrativeKind.EXHAUSTION: self._evolve_exhaustion,
            NarrativeKind.LIQUIDITY_GRAB: self._evolve_liquidity_grab,
        }
        fn = dispatcher.get(narrative.kind)
        if fn is not None:
            fn(
                narrative,
                inp,
                regime,
                vol_rank,
                liq_rank,
                trend_maturity,
                active_hypotheses,
                observer_state,
                readiness_score,
                dwell_seconds,
                now,
            )

        # Always refresh context strings so they reflect the latest regime/session.
        narrative.regime_context = regime
        narrative.session_context = _session_context(inp.session_quality)
        return narrative

    # ── Per-kind evolution logic ──────────────────────────────────────────

    def _evolve_trend_continuation(
        self,
        n: SymbolNarrative,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        liq_rank: int,
        trend_maturity: TrendMaturity,
        hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        now: float,
    ) -> None:
        in_trend = regime in ("TREND_UP", "TREND_DOWN")
        trend_valid = inp.market.trend_valid

        if in_trend and trend_valid and readiness_score >= 0.4:
            boost = 0.08 + readiness_score * 0.06
            n.reinforce(f"readiness={readiness_score:.2f}, regime={regime}", boost=boost, now=now)
        elif not in_trend:
            n.weaken(f"regime flipped to {regime}", penalty=0.20, now=now)
        elif not trend_valid:
            n.weaken("trend_valid=False", penalty=0.12, now=now)
        elif trend_maturity in (TrendMaturity.LATE, TrendMaturity.EXHAUSTED):
            n.weaken(f"trend_maturity={trend_maturity.value}", penalty=0.10, now=now)
        else:
            # Neutral tick: small passive decay (confidence erodes without confirmation)
            n.confidence = max(0.0, n.confidence - 0.02)
            n.last_updated_at = now

    def _evolve_reversal(
        self,
        n: SymbolNarrative,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        liq_rank: int,
        trend_maturity: TrendMaturity,
        hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        now: float,
    ) -> None:
        in_trend = regime in ("TREND_UP", "TREND_DOWN")
        if not in_trend:
            # Regime already flipped — reversal materialised or cancelled
            n.weaken(f"regime={regime} (no longer in trend)", penalty=0.25, now=now)
            return

        reversal_signals = 0
        evidence_parts: list[str] = []
        if not inp.market.trend_valid:
            reversal_signals += 2
            evidence_parts.append("trend_valid=False")
        if trend_maturity in (TrendMaturity.LATE, TrendMaturity.EXHAUSTED):
            reversal_signals += 1
            evidence_parts.append(f"maturity={trend_maturity.value}")
        if inp.microstructure_quality < 0.35:
            reversal_signals += 1
            evidence_parts.append(f"microstructure={inp.microstructure_quality:.2f}")

        bull = sum(1 for h in hypotheses if h.direction == BeliefDirection.BULLISH)
        bear = sum(1 for h in hypotheses if h.direction == BeliefDirection.BEARISH)
        if bull > 0 and bear > 0:
            reversal_signals += 1
            evidence_parts.append(f"mixed_hypotheses={bull}b/{bear}B")

        if reversal_signals >= 2:
            n.reinforce("; ".join(evidence_parts), boost=0.06 * reversal_signals, now=now)
        else:
            n.weaken("reversal signals fading", penalty=0.10, now=now)

    def _evolve_accumulation(
        self,
        n: SymbolNarrative,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        liq_rank: int,
        trend_maturity: TrendMaturity,
        hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        now: float,
    ) -> None:
        in_range = regime in ("RANGE", "CHOP", "CHOPPY")
        if not in_range:
            n.weaken(f"regime={regime} (left range)", penalty=0.25, now=now)
            return

        score = 0
        if vol_rank <= 1:
            score += 1
        if liq_rank >= 1 and inp.liquidity_quality >= 0.45:
            score += 1
        if inp.microstructure_quality >= 0.45:
            score += 1
        if any(h.direction == BeliefDirection.BULLISH for h in hypotheses):
            score += 1

        if score >= 2:
            n.reinforce(f"quiet_range vol={inp.volatility_state.lower()}", boost=0.06, now=now)
        else:
            n.confidence = max(0.0, n.confidence - 0.03)
            n.last_updated_at = now

    def _evolve_distribution(
        self,
        n: SymbolNarrative,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        liq_rank: int,
        trend_maturity: TrendMaturity,
        hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        now: float,
    ) -> None:
        in_range = regime in ("RANGE", "CHOP", "CHOPPY")
        if not in_range:
            n.weaken(f"regime={regime} (left range)", penalty=0.25, now=now)
            return

        score = 0
        if vol_rank >= 2:
            score += 1
        if liq_rank <= 1 or inp.liquidity_quality < 0.45:
            score += 1
        if inp.microstructure_quality < 0.40:
            score += 1
        if any(h.direction == BeliefDirection.BEARISH for h in hypotheses):
            score += 1

        if score >= 2:
            n.reinforce(f"supply_pressure vol={inp.volatility_state.lower()}", boost=0.06, now=now)
        else:
            n.confidence = max(0.0, n.confidence - 0.03)
            n.last_updated_at = now

    def _evolve_breakout(
        self,
        n: SymbolNarrative,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        liq_rank: int,
        trend_maturity: TrendMaturity,
        hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        now: float,
    ) -> None:
        armed = observer_state in ("ARMED", "ENTRY_WINDOW", "CONFIRMING", "READY")
        if armed and readiness_score >= 0.60:
            boost = 0.08 + readiness_score * 0.08
            n.reinforce(
                f"observer={observer_state} readiness={readiness_score:.2f}", boost=boost, now=now
            )
        elif observer_state == "COOLDOWN":
            # Breakout window closed
            n.weaken("observer entered cooldown", penalty=0.30, now=now)
        elif readiness_score < 0.40:
            n.weaken(f"readiness dropped to {readiness_score:.2f}", penalty=0.12, now=now)
        else:
            n.confidence = max(0.0, n.confidence - 0.05)
            n.last_updated_at = now

    def _evolve_exhaustion(
        self,
        n: SymbolNarrative,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        liq_rank: int,
        trend_maturity: TrendMaturity,
        hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        now: float,
    ) -> None:
        in_trend = regime in ("TREND_UP", "TREND_DOWN")

        if in_trend and trend_maturity in (TrendMaturity.LATE, TrendMaturity.EXHAUSTED):
            n.reinforce(f"trend_maturity={trend_maturity.value} continuing", boost=0.07, now=now)
        elif in_trend and not inp.market.trend_valid:
            n.reinforce("trend_valid=False in trend regime", boost=0.08, now=now)
        elif in_trend and inp.market.trend_valid and readiness_score >= 0.60:
            # Trend re-validated at high readiness — exhaustion is fading
            n.weaken(f"trend re-validated, readiness={readiness_score:.2f}", penalty=0.15, now=now)
        else:
            n.confidence = max(0.0, n.confidence - 0.03)
            n.last_updated_at = now

    def _evolve_liquidity_grab(
        self,
        n: SymbolNarrative,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        liq_rank: int,
        trend_maturity: TrendMaturity,
        hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        now: float,
    ) -> None:
        # Liquidity grab is inherently transient; it either resolves quickly
        # (liquidity normalises) or becomes a structural problem.
        if vol_rank >= 3 and liq_rank == 0:
            n.reinforce(
                f"vol={inp.volatility_state.lower()} liq={inp.liquidity_state.lower()} persisting",
                boost=0.06,
                now=now,
            )
        elif liq_rank >= 1 and vol_rank <= 2:
            # Normalised — liquidity grab resolved
            n.weaken("liquidity normalised", penalty=0.35, now=now)
        else:
            # Passive decay (liq grab is a short-lived event by nature)
            n.confidence = max(0.0, n.confidence - 0.08)
            n.last_updated_at = now


def _session_context(session_quality: float) -> str:
    if session_quality >= 0.75:
        return "prime session"
    if session_quality >= 0.45:
        return "moderate session"
    return "thin/off-hours"


__all__ = ["NarrativeEvolution"]
