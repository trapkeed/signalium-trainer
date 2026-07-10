"""QDE Brain — uncertainty engine.

Computes the QDE's degree of **market ambiguity** — distinct from both
confidence (probability a prediction is correct) and conviction
(strength of belief in a setup). A symbol can have a confident,
high-conviction hypothesis while the *market itself* is behaving
ambiguously (conflicting signals, unstable volatility/liquidity,
elevated news risk, invalid trend structure) — uncertainty is what lets
the QDE recognize that condition explicitly and prefer NO_TRADE, rather
than trusting a hypothesis formed in confusing conditions.

Every input here is read from data already produced elsewhere
(`MarketContext.news_risk`, volatility/liquidity/regime state already
classified upstream, active `Hypothesis` objects already tracked by the
Brain) — this module never re-derives volatility/liquidity/regime
classification itself, it only interprets what's already known as a
single ambiguity signal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.brain_types import BeliefDirection

HIGH_VOLATILITY_STATES = {"HIGH", "EXTREME"}
POOR_LIQUIDITY_STATES = {"POOR", "THIN", "LOW"}


class UncertaintyLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"  # QDE does not understand the current market


_LEVEL_THRESHOLDS = (
    (0.75, UncertaintyLevel.EXTREME),
    (0.55, UncertaintyLevel.HIGH),
    (0.30, UncertaintyLevel.MODERATE),
)


def classify_level(composite: float) -> UncertaintyLevel:
    for threshold, level in _LEVEL_THRESHOLDS:
        if composite >= threshold:
            return level
    return UncertaintyLevel.LOW


@dataclass(frozen=True)
class UncertaintySignals:
    """Raw 0-1 contributors to the composite ambiguity score."""

    news_ambiguity: float = 0.0
    volatility_instability: float = 0.0
    liquidity_instability: float = 0.0
    regime_conflict: float = 0.0
    structural_invalidity: float = 0.0
    evidence_conflict: float = 0.0

    def to_dict(self) -> dict:
        return {
            "news_ambiguity": round(self.news_ambiguity, 4),
            "volatility_instability": round(self.volatility_instability, 4),
            "liquidity_instability": round(self.liquidity_instability, 4),
            "regime_conflict": round(self.regime_conflict, 4),
            "structural_invalidity": round(self.structural_invalidity, 4),
            "evidence_conflict": round(self.evidence_conflict, 4),
        }

    def max_signal(self) -> float:
        return max(
            self.news_ambiguity,
            self.volatility_instability,
            self.liquidity_instability,
            self.regime_conflict,
            self.structural_invalidity,
            self.evidence_conflict,
        )


@dataclass
class UncertaintyScore:
    """Ambiguity reading for one symbol at one point in time."""

    instrument: str
    composite: float
    level: UncertaintyLevel
    signals: UncertaintySignals = field(default_factory=UncertaintySignals)
    explanation: str = ""
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "composite": round(self.composite, 4),
            "level": self.level.value,
            "signals": self.signals.to_dict(),
            "explanation": self.explanation,
            "computed_at": round(self.computed_at, 3),
        }


def _regime_conflict(active_hypotheses: list[Hypothesis]) -> float:
    """Detect the market sending genuinely conflicting directional
    signals — multiple active hypotheses with comparable confidence
    pointing opposite ways.
    """
    bullish = [h.confidence for h in active_hypotheses if h.direction == BeliefDirection.BULLISH]
    bearish = [h.confidence for h in active_hypotheses if h.direction == BeliefDirection.BEARISH]
    if not bullish or not bearish:
        return 0.0
    bull_strength = max(bullish)
    bear_strength = max(bearish)
    total = bull_strength + bear_strength
    if total <= 1e-9:
        return 0.0
    balance = 1.0 - abs(bull_strength - bear_strength) / total
    return min(1.0, balance * min(1.0, total))


def _evidence_conflict(active_hypotheses: list[Hypothesis]) -> float:
    """Detect individual hypotheses whose own evidence is internally
    indecisive (supporting and contradicting evidence roughly balanced),
    which signals the market isn't confirming or denying the thesis.
    """
    scores = []
    for h in active_hypotheses:
        supporting = len(h.supporting_evidence)
        contradicting = len(h.contradicting_evidence)
        total = supporting + contradicting
        if total == 0:
            continue
        scores.append(1.0 - abs(supporting - contradicting) / total)
    return max(scores) if scores else 0.0


def compute_uncertainty(
    instrument: str,
    news_risk: float = 0.0,
    volatility_state: str = "NORMAL",
    volatility_quality: float = 0.5,
    liquidity_state: str = "NORMAL",
    liquidity_quality: float = 0.5,
    trend_valid: bool = True,
    microstructure_quality: float = 0.5,
    active_hypotheses: list[Hypothesis] | None = None,
    now: float | None = None,
) -> UncertaintyScore:
    """Compute the QDE's current degree of market ambiguity for a symbol.

    Pure function over data already produced by upstream engines
    (observer/microstructure classification, the Brain's tracked
    hypotheses) — never re-derives that classification itself.
    """
    now = now if now is not None else time.time()
    active_hypotheses = active_hypotheses or []

    news_ambiguity = max(0.0, min(1.0, news_risk))

    volatility_instability = (
        0.7 if volatility_state.upper() in HIGH_VOLATILITY_STATES else 0.0
    ) + 0.3 * max(0.0, 1.0 - volatility_quality)
    volatility_instability = min(1.0, volatility_instability)

    liquidity_instability = (
        0.7 if liquidity_state.upper() in POOR_LIQUIDITY_STATES else 0.0
    ) + 0.3 * max(0.0, 1.0 - liquidity_quality)
    liquidity_instability = min(1.0, liquidity_instability)

    structural_invalidity = (0.7 if not trend_valid else 0.0) + 0.3 * max(
        0.0, 1.0 - microstructure_quality
    )
    structural_invalidity = min(1.0, structural_invalidity)

    regime_conflict = _regime_conflict(active_hypotheses)
    evidence_conflict = _evidence_conflict(active_hypotheses)

    composite = max(
        0.0,
        min(
            1.0,
            0.20 * news_ambiguity
            + 0.20 * volatility_instability
            + 0.15 * liquidity_instability
            + 0.20 * regime_conflict
            + 0.15 * structural_invalidity
            + 0.10 * evidence_conflict,
        ),
    )

    level = classify_level(composite)
    signals = UncertaintySignals(
        news_ambiguity=news_ambiguity,
        volatility_instability=volatility_instability,
        liquidity_instability=liquidity_instability,
        regime_conflict=regime_conflict,
        structural_invalidity=structural_invalidity,
        evidence_conflict=evidence_conflict,
    )

    if level == UncertaintyLevel.EXTREME:
        explanation = (
            f"{instrument}: market ambiguity is extreme ({composite:.0%}) — "
            "QDE does not currently understand this market; NO_TRADE preferred."
        )
    elif level == UncertaintyLevel.HIGH:
        explanation = (
            f"{instrument}: market ambiguity is high ({composite:.0%}); "
            "trade selection should lean toward NO_TRADE."
        )
    else:
        explanation = f"{instrument}: market ambiguity is {level.value.lower()} ({composite:.0%})."

    return UncertaintyScore(
        instrument=instrument,
        composite=composite,
        level=level,
        signals=signals,
        explanation=explanation,
        computed_at=now,
    )


def prefers_no_trade(score: UncertaintyScore, threshold: float = 0.55) -> bool:
    """Whether current market ambiguity is high enough that NO_TRADE
    should be preferred over acting on any hypothesis, however confident
    or well-reinforced.
    """
    return score.composite >= threshold or score.level in (
        UncertaintyLevel.HIGH,
        UncertaintyLevel.EXTREME,
    )


def does_not_understand_market(score: UncertaintyScore, threshold: float = 0.75) -> bool:
    """The strongest form of uncertainty recognition: the QDE explicitly
    flags that it does not understand current conditions well enough to
    trust any hypothesis at all.
    """
    return score.composite >= threshold or score.level == UncertaintyLevel.EXTREME


__all__ = [
    "UncertaintyLevel",
    "UncertaintySignals",
    "UncertaintyScore",
    "classify_level",
    "compute_uncertainty",
    "prefers_no_trade",
    "does_not_understand_market",
]
