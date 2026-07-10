"""QDE Brain — self-awareness engine.

Converts a `metacognition_engine.MetacognitiveAssessment` into bounded,
slowly-tracking behavioral adjustments: aggression, risk appetite, trade
frequency, and evidence requirements. This is the module that actually
*changes QDE's own behavior* in response to how it has been performing —
`MetacognitionEngine` only measures, this module decides and acts.

Distinct from `personality_profiles.risk_appetite_multiplier` /
`setup_requirement_overrides` / `confidence_adjustment`: those reflect how
one *symbol* tends to move (an external fact), independent of how QDE has
been doing lately. `SelfAwarenessState` reflects QDE's own recent track
record and is an orthogonal multiplicative factor a caller may combine
with the personality-driven ones — neither re-derives the other.

Design constraints (from the product requirement, enforced structurally
rather than just by convention):

  - Poor recent performance must automatically increase caution — `caution`
    (from `MetacognitiveAssessment.caution_level`, plus regime/symbol
    weakness bumps) directly and unconditionally pulls every multiplier
    toward its cautious bound. There is no performance level at which
    caution fails to act.
  - Strong recent performance must never cause *uncontrolled* risk
    expansion. Expansion is driven by a *separate*, deliberately harder to
    trigger `performance_signal` (needs an above-baseline recent win rate,
    reinforced by a positive trend) that is itself damped by `(1 - caution)`
    so any live caution suppresses it, and every multiplier has a hard
    `MAX_*` ceiling far tighter than its `MIN_*` floor is loose. Adjustment
    targets are clamped to bounds before EMA-blending with the prior state
    (itself always within bounds), and a convex combination of two values
    that are each <= a bound is itself <= that bound — so no sequence of
    good ticks can ever compound past the ceiling, for all time, not just
    per-tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainer.qde.brain.metacognition_engine import MetacognitiveAssessment

# Bounds are intentionally asymmetric: caution can pull behavior down much
# further than confidence is ever allowed to push it up.
MIN_AGGRESSION = 0.35
MAX_AGGRESSION = 1.15
MIN_RISK_APPETITE = 0.35
MAX_RISK_APPETITE = 1.10
MIN_TRADE_FREQUENCY = 0.30
MAX_TRADE_FREQUENCY = 1.10
MIN_EVIDENCE_REQUIREMENT = 0.85  # evidence bar can never be relaxed far below baseline
MAX_EVIDENCE_REQUIREMENT = 1.75

ADJUSTMENT_ALPHA = 0.2  # EMA smoothing so behavior tracks slowly, not whipsaws tick to tick

WEAK_REGIME_CAUTION_BUMP = 0.15
SYMBOL_WEAKNESS_CAUTION_WEIGHT = 0.20


@dataclass
class SelfAwarenessState:
    """Persisted, slowly-tracking behavioral-adjustment state for one symbol."""

    instrument: str = ""
    aggression: float = 1.0
    risk_appetite: float = 1.0
    trade_frequency: float = 1.0
    evidence_requirement: float = 1.0
    caution_level: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "aggression": round(self.aggression, 4),
            "risk_appetite": round(self.risk_appetite, 4),
            "trade_frequency": round(self.trade_frequency, 4),
            "evidence_requirement": round(self.evidence_requirement, 4),
            "caution_level": round(self.caution_level, 4),
            "created_at": round(self.created_at, 3),
            "last_updated_at": round(self.last_updated_at, 3),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "SelfAwarenessState":
        return cls(
            instrument=raw.get("instrument", ""),
            aggression=raw.get("aggression", 1.0),
            risk_appetite=raw.get("risk_appetite", 1.0),
            trade_frequency=raw.get("trade_frequency", 1.0),
            evidence_requirement=raw.get("evidence_requirement", 1.0),
            caution_level=raw.get("caution_level", 0.5),
            created_at=raw.get("created_at", time.time()),
            last_updated_at=raw.get("last_updated_at", time.time()),
        )


class SelfAwarenessEngine:
    """Turns a `MetacognitiveAssessment` into the next `SelfAwarenessState`."""

    def adjust(
        self,
        prior: SelfAwarenessState | None,
        assessment: "MetacognitiveAssessment",
        current_regime: str = "",
        now: float | None = None,
    ) -> SelfAwarenessState:
        now = now if now is not None else time.time()
        prior = prior or SelfAwarenessState(instrument=assessment.instrument, last_updated_at=now)

        caution = assessment.caution_level
        if current_regime and current_regime in assessment.weak_regimes:
            caution = min(1.0, caution + WEAK_REGIME_CAUTION_BUMP)
        if assessment.symbol_weakness > 0:
            caution = min(
                1.0, caution + assessment.symbol_weakness * SYMBOL_WEAKNESS_CAUTION_WEIGHT
            )

        # Reduction: caution directly and unconditionally pulls every
        # multiplier toward its cautious bound. No performance level makes
        # caution stop acting.
        reduction_pull = min(1.0, max(0.0, caution))

        # Expansion: requires an explicit positive-performance signal (an
        # above-baseline recent win rate, optionally reinforced by a
        # positive trend) — being merely "not cautious" is not enough on
        # its own. Any live caution damps expansion toward zero.
        performance_signal = 0.0
        if assessment.recent_win_rate is not None:
            performance_signal = max(0.0, (assessment.recent_win_rate - 0.5) * 2.0)
            if assessment.performance_trend > 0:
                performance_signal = min(1.0, performance_signal + assessment.performance_trend)
        expansion_pull = min(1.0, performance_signal) * (1.0 - reduction_pull)

        target_aggression = (
            1.0 - reduction_pull * (1.0 - MIN_AGGRESSION) + expansion_pull * (MAX_AGGRESSION - 1.0)
        )
        target_risk_appetite = (
            1.0
            - reduction_pull * (1.0 - MIN_RISK_APPETITE)
            + expansion_pull * (MAX_RISK_APPETITE - 1.0)
        )
        target_trade_frequency = (
            1.0
            - reduction_pull * (1.0 - MIN_TRADE_FREQUENCY)
            + expansion_pull * (MAX_TRADE_FREQUENCY - 1.0)
        )
        target_evidence_requirement = (
            1.0
            + reduction_pull * (MAX_EVIDENCE_REQUIREMENT - 1.0)
            - expansion_pull * (1.0 - MIN_EVIDENCE_REQUIREMENT)
        )

        def _ema(prior_value: float, target_value: float, lo: float, hi: float) -> float:
            clamped_target = min(hi, max(lo, target_value))
            blended = (1.0 - ADJUSTMENT_ALPHA) * prior_value + ADJUSTMENT_ALPHA * clamped_target
            return round(min(hi, max(lo, blended)), 4)

        return SelfAwarenessState(
            instrument=assessment.instrument,
            aggression=_ema(prior.aggression, target_aggression, MIN_AGGRESSION, MAX_AGGRESSION),
            risk_appetite=_ema(
                prior.risk_appetite, target_risk_appetite, MIN_RISK_APPETITE, MAX_RISK_APPETITE
            ),
            trade_frequency=_ema(
                prior.trade_frequency,
                target_trade_frequency,
                MIN_TRADE_FREQUENCY,
                MAX_TRADE_FREQUENCY,
            ),
            evidence_requirement=_ema(
                prior.evidence_requirement,
                target_evidence_requirement,
                MIN_EVIDENCE_REQUIREMENT,
                MAX_EVIDENCE_REQUIREMENT,
            ),
            caution_level=round(caution, 4),
            created_at=prior.created_at,
            last_updated_at=now,
        )
