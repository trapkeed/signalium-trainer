"""QDE Brain — cognitive decision engine (Phase C6).

Pure functions that translate accumulated cognitive state into one of the
five ``CognitiveAction`` values (BUY / SELL / WAIT / MONITOR / IGNORE) and
compute the six signal heads.

Design principles
-----------------
* Pure functions only — no I/O, no state, no mutation of inputs.
* Delegates uncertainty to ``compute_uncertainty()`` (already exists).
* Delegates conviction to ``compute_conviction()`` (already exists).
* Applies ``CalibrationProfile`` biases rather than reimplementing them.
* Reads ``DebateRecord`` as-is (from C5) — never re-evaluates the debate.
* Never imports anything from ``qde_inference`` — this is cognitive
  vocabulary, not execution vocabulary.

Decision priority order
-----------------------
  1. IGNORE — if market is incomprehensible (EXTREME uncertainty) or
     there is no cognitive basis (no hypotheses, no narrative).
  2. WAIT — if uncertainty is HIGH, or a risk limit is reached, or
     overconfidence was flagged on a non-conclusive debate.
  3. MONITOR — if a debate exists but is not conclusive, or hypotheses
     are forming but conviction is below threshold.
  4. BUY / SELL — if the debate is conclusive, conviction meets the
     threshold, and uncertainty is LOW or MODERATE.

Signal-head formulas
--------------------
  risk_signal       = (1 − uncertainty.composite) × portfolio_risk_factor
  entry_urgency     = conviction × (1 − uncertainty.composite) if actionable else 0
  hold_horizon      = persistence_strength × 14400 seconds (4h base)
  size_signal       = conviction × risk_signal if actionable else 0
  calibrated_conf   = clamp(raw + calibration_bias, 0.05, 0.95)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.conviction_engine import (
    ConvictionProfile,
    ConvictionScore,
    compute_conviction,
)
from trainer.qde.brain.decision.decision_types import (
    CognitiveAction,
    CognitiveDecision,
    DecisionInput,
)
from trainer.qde.brain.uncertainty_engine import (
    UncertaintyLevel,
    UncertaintyScore,
    does_not_understand_market,
    prefers_no_trade,
)

if TYPE_CHECKING:
    from trainer.qde.brain.brain_models import CalibrationProfile, Hypothesis

# -------------------------------------------------------------------
# Thresholds
# -------------------------------------------------------------------

CONVICTION_ACTION_THRESHOLD  = 0.45   # conviction ≥ this → eligible for BUY/SELL
CONVICTION_MONITOR_THRESHOLD = 0.20   # conviction ≥ this → MONITOR rather than IGNORE
MIN_HYPOTHESES_FOR_MONITOR   = 1      # need at least this many active hypotheses
MAX_OPEN_POSITIONS_SOFT      = 4      # soft limit: reduce size above this count
DAILY_LOSS_LIMIT_PCT         = -0.03  # reduce risk below this daily P&L
HOLD_HORIZON_BASE_SECONDS    = 14_400.0  # 4 hours baseline
SIZE_MAX                     = 1.0
SIZE_POSITION_PENALTY        = 0.3    # deduct from size if already in the same direction


# -------------------------------------------------------------------
# Primary entry point
# -------------------------------------------------------------------


def compute_decision(
    inp:      DecisionInput,
    calib:    "CalibrationProfile | None" = None,
    now:      float | None                = None,
) -> CognitiveDecision:
    """Translate a fully-populated ``DecisionInput`` into a
    ``CognitiveDecision``.

    All branching logic is in the private helpers below — this function
    is the stable public entry point that callers can unit-test.
    """
    now = now if now is not None else time.time()

    action, direction, hyp = _classify_action(inp)

    raw_confidence  = hyp.confidence if hyp else 0.0
    calib_bias      = calib.confidence_bias if calib else 0.0
    calibrated_conf = max(0.05, min(0.95, raw_confidence + calib_bias))

    conv_score      = inp.conviction.conviction if inp.conviction else 0.0
    conv_profile    = inp.conviction.profile.value if inp.conviction else ConvictionProfile.MIXED.value

    risk_signal, entry_urgency, hold_horizon, size_signal = _compute_signals(
        action=action,
        uncertainty=inp.uncertainty,
        conviction=inp.conviction,
        portfolio_open_positions=inp.portfolio_open_positions,
        portfolio_daily_pnl_pct=inp.portfolio_daily_pnl_pct,
        has_existing_position=inp.has_existing_position,
        existing_position_direction=inp.existing_position_direction,
        direction=direction,
    )

    oc_flagged  = bool(inp.debate_record and inp.debate_record.has_overconfidence)
    ie_flagged  = bool(inp.debate_record and inp.debate_record.has_ignored_evidence)
    debate_id   = inp.debate_record.debate_id if inp.debate_record else None

    reasoning = _brief_reasoning(action, direction, inp, hyp)

    return CognitiveDecision(
        instrument=inp.instrument,
        action=action,
        direction=direction,
        confidence=raw_confidence,
        calibrated_confidence=calibrated_conf,
        risk_signal=risk_signal,
        entry_urgency=entry_urgency,
        hold_horizon=hold_horizon,
        size_signal=size_signal,
        uncertainty_score=inp.uncertainty.composite,
        uncertainty_level=inp.uncertainty.level.value,
        conviction_score=conv_score,
        conviction_profile=conv_profile,
        reasoning=reasoning,
        primary_hypothesis_id=hyp.hypothesis_id if hyp else None,
        debate_id=debate_id,
        overconfidence_flagged=oc_flagged,
        ignored_evidence_flagged=ie_flagged,
        created_at=now,
    )


# -------------------------------------------------------------------
# Action classification
# -------------------------------------------------------------------


def _classify_action(
    inp: DecisionInput,
) -> tuple[CognitiveAction, BeliefDirection, "Hypothesis | None"]:
    """Return (action, direction, dominant_hypothesis).

    Priority: IGNORE > WAIT > MONITOR > BUY/SELL.
    """
    hyp = _select_hypothesis(inp.active_hypotheses)

    # ── 1. IGNORE ────────────────────────────────────────────────────
    # Market incomprehensible or completely empty cognitive state
    if does_not_understand_market(inp.uncertainty):
        return CognitiveAction.IGNORE, BeliefDirection.NEUTRAL, hyp

    if not inp.active_hypotheses and inp.narrative is None:
        return CognitiveAction.IGNORE, BeliefDirection.NEUTRAL, hyp

    # ── 2. WAIT ──────────────────────────────────────────────────────
    # Uncertainty too high to act
    if prefers_no_trade(inp.uncertainty):
        return CognitiveAction.WAIT, BeliefDirection.NEUTRAL, hyp

    # Risk limit: heavy daily loss → step back
    if inp.portfolio_daily_pnl_pct < DAILY_LOSS_LIMIT_PCT:
        return CognitiveAction.WAIT, BeliefDirection.NEUTRAL, hyp

    # Overconfidence + inconclusive debate → WAIT (don't trust the signal)
    if (
        inp.debate_record
        and inp.debate_record.has_overconfidence
        and not inp.debate_record.conclusive
    ):
        return CognitiveAction.WAIT, BeliefDirection.NEUTRAL, hyp

    # ── 3. MONITOR ───────────────────────────────────────────────────
    direction = _dominant_direction(inp.active_hypotheses, inp.debate_record, inp.narrative)

    # No hypothesis yet or conviction below the monitoring floor
    conv_score = inp.conviction.conviction if inp.conviction else 0.0
    if not inp.active_hypotheses or conv_score < CONVICTION_MONITOR_THRESHOLD:
        return CognitiveAction.MONITOR, direction, hyp

    # Debate present but not yet conclusive
    if inp.debate_record is not None and not inp.debate_record.conclusive:
        return CognitiveAction.MONITOR, direction, hyp

    # No debate at all yet → still forming
    if inp.debate_record is None:
        return CognitiveAction.MONITOR, direction, hyp

    # ── 4. BUY / SELL ────────────────────────────────────────────────
    # Debate is conclusive + conviction above action threshold
    if conv_score >= CONVICTION_ACTION_THRESHOLD and inp.debate_record.conclusive:
        if direction == BeliefDirection.BULLISH:
            return CognitiveAction.BUY, direction, hyp
        if direction == BeliefDirection.BEARISH:
            return CognitiveAction.SELL, direction, hyp

    # Conviction above action threshold but debate not conclusive → MONITOR
    return CognitiveAction.MONITOR, direction, hyp


# -------------------------------------------------------------------
# Signal head computation
# -------------------------------------------------------------------


def _compute_signals(
    action:                    CognitiveAction,
    uncertainty:               UncertaintyScore,
    conviction:                ConvictionScore | None,
    portfolio_open_positions:  int,
    portfolio_daily_pnl_pct:   float,
    has_existing_position:     bool,
    existing_position_direction: BeliefDirection | None,
    direction:                 BeliefDirection,
) -> tuple[float, float, float, float]:
    """Return (risk_signal, entry_urgency, hold_horizon, size_signal)."""

    # risk_signal: how much risk the current uncertainty headroom permits
    risk_signal = max(0.0, min(1.0, 1.0 - uncertainty.composite))

    # Portfolio-level penalty: too many open positions
    if portfolio_open_positions > MAX_OPEN_POSITIONS_SOFT:
        excess = portfolio_open_positions - MAX_OPEN_POSITIONS_SOFT
        risk_signal = max(0.0, risk_signal - 0.10 * excess)

    # Daily loss floor penalty
    if portfolio_daily_pnl_pct < 0:
        risk_signal = max(0.0, risk_signal + portfolio_daily_pnl_pct * 5.0)

    conv = conviction.conviction if conviction else 0.0
    persist = (
        conviction.components.persistence_strength if conviction else 0.0
    )

    if action.is_passive():
        return risk_signal, 0.0, 0.0, 0.0

    # entry_urgency: how urgently to act (conviction × certainty)
    entry_urgency = max(0.0, min(1.0, conv * risk_signal))

    # hold_horizon: expected hold in seconds based on setup persistence
    hold_horizon = max(0.0, persist * HOLD_HORIZON_BASE_SECONDS)

    # size_signal: conviction-weighted risk headroom
    size_signal = max(0.0, min(SIZE_MAX, conv * risk_signal))

    # Existing-position penalty: if already in this direction, reduce size
    if has_existing_position and existing_position_direction == direction:
        size_signal = max(0.0, size_signal - SIZE_POSITION_PENALTY)

    return risk_signal, entry_urgency, hold_horizon, size_signal


# -------------------------------------------------------------------
# Hypothesis selection
# -------------------------------------------------------------------


def _select_hypothesis(
    active_hypotheses: "list[Hypothesis]",
) -> "Hypothesis | None":
    """Pick the dominant (highest-confidence, non-terminal) hypothesis."""
    if not active_hypotheses:
        return None
    return max(active_hypotheses, key=lambda h: (h.confidence, h.reinforcement_count))


def _dominant_direction(
    active_hypotheses: "list[Hypothesis]",
    debate_record:     object | None,
    narrative:         object | None,
) -> BeliefDirection:
    """Best available direction signal, priority: debate > hypotheses > narrative."""
    # Debate dominates if conclusive
    if debate_record is not None and getattr(debate_record, "conclusive", False):
        verdict = debate_record.verdict if isinstance(debate_record.verdict, dict) else {}
        dom = verdict.get("dominant_side")
        if dom == BeliefDirection.BULLISH.value:
            return BeliefDirection.BULLISH
        if dom == BeliefDirection.BEARISH.value:
            return BeliefDirection.BEARISH

    # Hypothesis aggregate
    if active_hypotheses:
        hyp = _select_hypothesis(active_hypotheses)
        if hyp:
            return hyp.direction

    # Narrative tone proxy
    if narrative is not None:
        tone = getattr(narrative, "tone", None)
        if tone is not None:
            tone_val = tone.value if hasattr(tone, "value") else str(tone)
            if "BULL" in tone_val.upper():
                return BeliefDirection.BULLISH
            if "BEAR" in tone_val.upper():
                return BeliefDirection.BEARISH

    return BeliefDirection.NEUTRAL


# -------------------------------------------------------------------
# Brief reasoning builder
# -------------------------------------------------------------------


def _brief_reasoning(
    action:    CognitiveAction,
    direction: BeliefDirection,
    inp:       DecisionInput,
    hyp:       "Hypothesis | None",
) -> str:
    """One-sentence summary used in CognitiveDecision.reasoning.

    The full chain-of-thought is built by ``ReasoningEngine`` in
    decision_reasoning.py.
    """
    unc_pct = f"{inp.uncertainty.composite:.0%}"

    if action == CognitiveAction.IGNORE:
        return (
            f"{inp.instrument}: IGNORE — uncertainty is extreme ({unc_pct}); "
            "market conditions incomprehensible."
        )
    if action == CognitiveAction.WAIT:
        if inp.portfolio_daily_pnl_pct < DAILY_LOSS_LIMIT_PCT:
            return (
                f"{inp.instrument}: WAIT — daily P&L limit reached "
                f"({inp.portfolio_daily_pnl_pct:.1%})."
            )
        if inp.debate_record and inp.debate_record.has_overconfidence:
            return (
                f"{inp.instrument}: WAIT — overconfidence detected on "
                f"inconclusive debate; uncertainty {unc_pct}."
            )
        return (
            f"{inp.instrument}: WAIT — uncertainty {unc_pct} (HIGH); "
            "conditions not suitable for engagement."
        )
    if action == CognitiveAction.MONITOR:
        conv = inp.conviction.conviction if inp.conviction else 0.0
        thesis = f'"{hyp.thesis[:50]}..."' if hyp else "developing hypothesis"
        return (
            f"{inp.instrument}: MONITOR — {thesis}; "
            f"conviction={conv:.0%} (below action threshold); "
            f"uncertainty={unc_pct}."
        )
    # BUY or SELL
    conv = inp.conviction.conviction if inp.conviction else 0.0
    thesis = f'"{hyp.thesis[:50]}..."' if hyp else "strong hypothesis"
    return (
        f"{inp.instrument}: {action.value} — {direction.value.lower()} {thesis}; "
        f"conviction={conv:.0%}; uncertainty={unc_pct}."
    )


__all__ = [
    "compute_decision",
    "CONVICTION_ACTION_THRESHOLD",
    "CONVICTION_MONITOR_THRESHOLD",
]
