"""QDE Brain — narrative builder.

Pure, stateless detection: given a `SymbolObservationInput`, the existing
`MarketNarrative` summary, and the currently active `Hypothesis` list,
returns a list of `NarrativeSeed` objects — one for each narrative kind
whose detection conditions are met on this tick.

Why seeds rather than `SymbolNarrative` directly
-------------------------------------------------
The builder never overwrites an existing narrative.  The caller
(`MarketNarrativeEngine`) decides whether a seed's kind already has a
living narrative (in which case `NarrativeEvolution.evolve()` updates it)
or not (in which case the engine creates a new `SymbolNarrative` from the
seed).  This separation keeps detection and storage concerns cleanly apart.

Detection principles
--------------------
* Every detector reads *only* from `SymbolObservationInput` fields and the
  existing `MarketNarrative` context (regime, tone, trend maturity, session).
  Regime / volatility / liquidity classification is never re-derived here —
  those fields come pre-classified from upstream engines.
* Detectors are independent: multiple narrative kinds can be seeded on the
  same tick (e.g. EXHAUSTION + REVERSAL for a late trend at entry window).
* Confidence values are initial estimates, not final — `NarrativeEvolution`
  reinforces or decays them as subsequent ticks confirm or contradict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from trainer.qde.brain.brain_models import Hypothesis, MarketNarrative
from trainer.qde.brain.brain_types import BeliefDirection, NarrativeTone, TrendMaturity
from trainer.qde.brain.narrative.narrative_store import (
    _DEFAULT_MAX_AGE,
    NarrativeKind,
    NarrativeMaturity,
    SymbolNarrative,
)
from trainer.qde.qde_observer_models import SymbolObservationInput

# ── Volatility / liquidity rank helpers (local, mirrors opportunity_scanner) ──

_VOL_RANK: dict[str, int] = {"LOW": 0, "NORMAL": 1, "ELEVATED": 2, "HIGH": 3, "EXTREME": 4}
_LIQ_RANK: dict[str, int] = {"POOR": 0, "THIN": 0, "LOW": 0, "NORMAL": 1, "GOOD": 2, "EXCELLENT": 2}

# Minimum readiness before directional narratives can be seeded.
_DIRECTIONAL_READINESS_FLOOR = 0.3


# ── Seed dataclass ────────────────────────────────────────────────────────


@dataclass
class NarrativeSeed:
    """All information needed to open a fresh `SymbolNarrative`."""

    kind: NarrativeKind
    title: str
    initial_confidence: float
    evidence: str  # first supporting-evidence string
    regime_context: str  # market regime when the seed fired
    session_context: str  # session quality description
    max_age_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.max_age_seconds = _DEFAULT_MAX_AGE[self.kind]

    def to_narrative(self, instrument: str, now: float | None = None) -> SymbolNarrative:
        """Materialise this seed into a new `SymbolNarrative`."""
        now = now if now is not None else time.time()
        n = SymbolNarrative(
            instrument=instrument,
            kind=self.kind,
            title=self.title,
            confidence=self.initial_confidence,
            maturity=NarrativeMaturity.FORMING,
            regime_context=self.regime_context,
            session_context=self.session_context,
            formed_at=now,
            last_updated_at=now,
            max_age_seconds=self.max_age_seconds,
        )
        n.supporting_evidence.append(self.evidence)
        return n

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "title": self.title,
            "initial_confidence": round(self.initial_confidence, 4),
            "evidence": self.evidence,
            "regime_context": self.regime_context,
            "session_context": self.session_context,
            "max_age_seconds": self.max_age_seconds,
        }


# ── Builder ───────────────────────────────────────────────────────────────


class NarrativeBuilder:
    """Detect which narrative kinds are emerging on this market tick.

    All methods are pure functions of their arguments; no instance state
    is held so the builder is safe to share across symbols.
    """

    def detect(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        existing_narrative: MarketNarrative | None,
        active_hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        now: float | None = None,
    ) -> list[NarrativeSeed]:
        """Return narrative seeds for all kinds detected on this tick.

        The caller is responsible for deciding whether to open a new
        narrative from a seed or reinforce an existing one.
        """
        now = now if now is not None else time.time()
        regime = inp.market_regime.upper()
        vol_rank = _VOL_RANK.get(inp.volatility_state.upper(), 1)
        liq_rank = _LIQ_RANK.get(inp.liquidity_state.upper(), 1)
        tone = existing_narrative.tone if existing_narrative else NarrativeTone.NEUTRAL
        trend_maturity = (
            existing_narrative.trend_maturity if existing_narrative else TrendMaturity.NONE
        )
        session_ctx = self._session_context(inp.session_quality)
        regime_ctx = regime

        seeds: list[NarrativeSeed] = []

        # Each detector is independent; order does not determine precedence.
        s = self._detect_trend_continuation(
            instrument,
            inp,
            regime,
            vol_rank,
            tone,
            trend_maturity,
            observer_state,
            readiness_score,
            regime_ctx,
            session_ctx,
        )
        if s:
            seeds.append(s)

        s = self._detect_reversal(
            instrument,
            inp,
            regime,
            tone,
            trend_maturity,
            active_hypotheses,
            observer_state,
            readiness_score,
            dwell_seconds,
            regime_ctx,
            session_ctx,
        )
        if s:
            seeds.append(s)

        s = self._detect_accumulation(
            instrument, inp, regime, vol_rank, liq_rank, active_hypotheses, regime_ctx, session_ctx
        )
        if s:
            seeds.append(s)

        s = self._detect_distribution(
            instrument, inp, regime, vol_rank, liq_rank, active_hypotheses, regime_ctx, session_ctx
        )
        if s:
            seeds.append(s)

        s = self._detect_breakout(
            instrument,
            inp,
            regime,
            vol_rank,
            observer_state,
            readiness_score,
            regime_ctx,
            session_ctx,
        )
        if s:
            seeds.append(s)

        s = self._detect_exhaustion(
            instrument,
            inp,
            regime,
            trend_maturity,
            observer_state,
            readiness_score,
            dwell_seconds,
            regime_ctx,
            session_ctx,
        )
        if s:
            seeds.append(s)

        s = self._detect_liquidity_grab(
            instrument, inp, vol_rank, liq_rank, regime_ctx, session_ctx
        )
        if s:
            seeds.append(s)

        return seeds

    # ── Individual detectors ──────────────────────────────────────────────

    def _detect_trend_continuation(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        tone: NarrativeTone,
        trend_maturity: TrendMaturity,
        observer_state: str,
        readiness_score: float,
        regime_ctx: str,
        session_ctx: str,
    ) -> NarrativeSeed | None:
        if regime not in ("TREND_UP", "TREND_DOWN"):
            return None
        if readiness_score < _DIRECTIONAL_READINESS_FLOOR:
            return None
        if not inp.market.trend_valid:
            return None
        if trend_maturity in (TrendMaturity.LATE, TrendMaturity.EXHAUSTED):
            return None  # late trend is exhaustion, not continuation

        direction = "upward" if regime == "TREND_UP" else "downward"
        confidence = 0.35 + readiness_score * 0.25
        if inp.microstructure_quality >= 0.6:
            confidence += 0.05
        if vol_rank <= 2:  # not extreme vol
            confidence += 0.05

        return NarrativeSeed(
            kind=NarrativeKind.TREND_CONTINUATION,
            title=f"{instrument} {direction} trend continuation",
            initial_confidence=min(0.75, confidence),
            evidence=(
                f"regime={regime}, readiness={readiness_score:.2f}, "
                f"trend_valid=True, microstructure={inp.microstructure_quality:.2f}"
            ),
            regime_context=regime_ctx,
            session_context=session_ctx,
        )

    def _detect_reversal(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        regime: str,
        tone: NarrativeTone,
        trend_maturity: TrendMaturity,
        active_hypotheses: list[Hypothesis],
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        regime_ctx: str,
        session_ctx: str,
    ) -> NarrativeSeed | None:
        in_trend = regime in ("TREND_UP", "TREND_DOWN")
        if not in_trend:
            return None

        reversal_signals = 0
        evidence_parts: list[str] = []

        if not inp.market.trend_valid:
            reversal_signals += 2
            evidence_parts.append("trend_valid=False")

        if trend_maturity in (TrendMaturity.LATE, TrendMaturity.EXHAUSTED):
            reversal_signals += 2
            evidence_parts.append(f"trend_maturity={trend_maturity.value}")

        if inp.microstructure_quality < 0.35:
            reversal_signals += 1
            evidence_parts.append(f"microstructure={inp.microstructure_quality:.2f}")

        # A conflicted hypothesis set (both bullish and bearish active) suggests
        # the directional picture is breaking down.
        bull_count = sum(1 for h in active_hypotheses if h.direction == BeliefDirection.BULLISH)
        bear_count = sum(1 for h in active_hypotheses if h.direction == BeliefDirection.BEARISH)
        if bull_count > 0 and bear_count > 0:
            reversal_signals += 1
            evidence_parts.append(f"conflicting_hypotheses={bull_count}b/{bear_count}B")

        if observer_state == "COOLDOWN" and dwell_seconds > 1800:
            reversal_signals += 1
            evidence_parts.append("post-entry cooldown")

        if reversal_signals < 2:
            return None

        confidence = min(0.65, 0.25 + reversal_signals * 0.08)
        direction_lbl = "bearish" if regime == "TREND_UP" else "bullish"
        return NarrativeSeed(
            kind=NarrativeKind.REVERSAL,
            title=f"{instrument} potential {direction_lbl} reversal",
            initial_confidence=confidence,
            evidence="; ".join(evidence_parts) or "reversal_signals detected",
            regime_context=regime_ctx,
            session_context=session_ctx,
        )

    def _detect_accumulation(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        liq_rank: int,
        active_hypotheses: list[Hypothesis],
        regime_ctx: str,
        session_ctx: str,
    ) -> NarrativeSeed | None:
        if regime not in ("RANGE", "CHOP", "CHOPPY"):
            return None

        score = 0
        evidence_parts: list[str] = []

        # Quiet, low-volatility consolidation
        if vol_rank <= 1:
            score += 2
            evidence_parts.append(f"vol={inp.volatility_state.lower()}")

        # Healthy or improving liquidity (bids building)
        if liq_rank >= 1 and inp.liquidity_quality >= 0.5:
            score += 1
            evidence_parts.append(f"liq={inp.liquidity_state.lower()}")

        # Momentum compressed near neutral or slightly positive
        momentum = getattr(inp.market, "momentum", 0.0)
        if -0.2 <= momentum <= 0.4:
            score += 1
            evidence_parts.append(f"momentum={momentum:.2f}")

        # Microstructure decent (not noise-dominated)
        if inp.microstructure_quality >= 0.45:
            score += 1
            evidence_parts.append(f"microstructure={inp.microstructure_quality:.2f}")

        # Any bullish hypothesis active adds conviction
        if any(h.direction == BeliefDirection.BULLISH for h in active_hypotheses):
            score += 1
            evidence_parts.append("bullish_hypothesis_active")

        if score < 3:
            return None

        confidence = min(0.60, 0.25 + score * 0.07)
        return NarrativeSeed(
            kind=NarrativeKind.ACCUMULATION,
            title=f"{instrument} accumulation in range",
            initial_confidence=confidence,
            evidence="; ".join(evidence_parts),
            regime_context=regime_ctx,
            session_context=session_ctx,
        )

    def _detect_distribution(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        liq_rank: int,
        active_hypotheses: list[Hypothesis],
        regime_ctx: str,
        session_ctx: str,
    ) -> NarrativeSeed | None:
        if regime not in ("RANGE", "CHOP", "CHOPPY"):
            return None

        score = 0
        evidence_parts: list[str] = []

        # Slightly elevated vol relative to range (selling pressure)
        if vol_rank >= 2:
            score += 2
            evidence_parts.append(f"vol={inp.volatility_state.lower()}")

        # Liquidity thinning (offers absorbed, thinning bid)
        if liq_rank <= 1 or inp.liquidity_quality < 0.45:
            score += 1
            evidence_parts.append(f"liq_thinning={inp.liquidity_state.lower()}")

        # Negative or neutral momentum
        momentum = getattr(inp.market, "momentum", 0.0)
        if -0.4 <= momentum <= 0.1:
            score += 1
            evidence_parts.append(f"momentum={momentum:.2f}")

        # Any bearish hypothesis active
        if any(h.direction == BeliefDirection.BEARISH for h in active_hypotheses):
            score += 1
            evidence_parts.append("bearish_hypothesis_active")

        # Declining microstructure quality
        if inp.microstructure_quality < 0.40:
            score += 1
            evidence_parts.append(f"microstructure={inp.microstructure_quality:.2f}")

        if score < 3:
            return None

        confidence = min(0.60, 0.25 + score * 0.07)
        return NarrativeSeed(
            kind=NarrativeKind.DISTRIBUTION,
            title=f"{instrument} distribution in range",
            initial_confidence=confidence,
            evidence="; ".join(evidence_parts),
            regime_context=regime_ctx,
            session_context=session_ctx,
        )

    def _detect_breakout(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        regime: str,
        vol_rank: int,
        observer_state: str,
        readiness_score: float,
        regime_ctx: str,
        session_ctx: str,
    ) -> NarrativeSeed | None:
        # A breakout narrative requires the observer to be in an armed /
        # entry-window state with high readiness — it is the "about to happen
        # or actively happening" transition out of range or continuation.
        if observer_state not in ("ARMED", "ENTRY_WINDOW", "CONFIRMING", "READY"):
            return None
        if readiness_score < 0.60:
            return None

        score = 0
        evidence_parts: list[str] = [f"observer={observer_state}"]

        if readiness_score >= 0.80:
            score += 2
        elif readiness_score >= 0.65:
            score += 1
        evidence_parts.append(f"readiness={readiness_score:.2f}")

        if vol_rank >= 2:
            score += 1
            evidence_parts.append(f"vol={inp.volatility_state.lower()}")

        if inp.microstructure_quality >= 0.55:
            score += 1
            evidence_parts.append(f"microstructure={inp.microstructure_quality:.2f}")

        if inp.market.trend_valid:
            score += 1
            evidence_parts.append("trend_valid=True")

        if score < 2:
            return None

        confidence = min(0.80, 0.40 + readiness_score * 0.30)
        return NarrativeSeed(
            kind=NarrativeKind.BREAKOUT,
            title=f"{instrument} breakout developing",
            initial_confidence=confidence,
            evidence="; ".join(evidence_parts),
            regime_context=regime_ctx,
            session_context=session_ctx,
        )

    def _detect_exhaustion(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        regime: str,
        trend_maturity: TrendMaturity,
        observer_state: str,
        readiness_score: float,
        dwell_seconds: float,
        regime_ctx: str,
        session_ctx: str,
    ) -> NarrativeSeed | None:
        in_trend = regime in ("TREND_UP", "TREND_DOWN")

        score = 0
        evidence_parts: list[str] = []

        if in_trend and trend_maturity in (TrendMaturity.LATE, TrendMaturity.EXHAUSTED):
            score += 3
            evidence_parts.append(f"trend_maturity={trend_maturity.value}")

        if not inp.market.trend_valid and in_trend:
            score += 2
            evidence_parts.append("trend_valid=False in trend")

        if observer_state == "COOLDOWN" and readiness_score < 0.20:
            score += 2
            evidence_parts.append(f"cooldown readiness={readiness_score:.2f}")

        # Trend very long-lived without retracement — dwell > 6 h
        if in_trend and dwell_seconds > 6 * 3600:
            score += 1
            evidence_parts.append(f"dwell={dwell_seconds/3600:.1f}h")

        if inp.microstructure_quality < 0.30 and in_trend:
            score += 1
            evidence_parts.append(f"microstructure={inp.microstructure_quality:.2f}")

        if score < 3:
            return None

        confidence = min(0.70, 0.30 + score * 0.08)
        return NarrativeSeed(
            kind=NarrativeKind.EXHAUSTION,
            title=f"{instrument} move exhaustion",
            initial_confidence=confidence,
            evidence="; ".join(evidence_parts),
            regime_context=regime_ctx,
            session_context=session_ctx,
        )

    def _detect_liquidity_grab(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        vol_rank: int,
        liq_rank: int,
        regime_ctx: str,
        session_ctx: str,
    ) -> NarrativeSeed | None:
        # A liquidity grab is a sudden spike + thin liquidity combination —
        # indicative of a stop-hunt or wick that often precedes reversal.
        if vol_rank < 3:  # requires HIGH or EXTREME volatility
            return None
        if liq_rank > 0:  # liquidity must be thin/poor
            return None

        evidence_parts = [
            f"vol={inp.volatility_state.lower()}",
            f"liq={inp.liquidity_state.lower()}",
            f"microstructure={inp.microstructure_quality:.2f}",
        ]
        confidence = 0.40 + min(0.20, (vol_rank - 3) * 0.10)
        return NarrativeSeed(
            kind=NarrativeKind.LIQUIDITY_GRAB,
            title=f"{instrument} liquidity grab",
            initial_confidence=min(0.65, confidence),
            evidence="; ".join(evidence_parts),
            regime_context=regime_ctx,
            session_context=session_ctx,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _session_context(session_quality: float) -> str:
        if session_quality >= 0.75:
            return "prime session"
        if session_quality >= 0.45:
            return "moderate session"
        return "thin/off-hours"


__all__ = ["NarrativeSeed", "NarrativeBuilder"]
