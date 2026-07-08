"""Phase 21.5 — Deterministic Finite State Machine.

Every transition has: entry conditions, exit conditions, validation rules,
timeout rules, confidence, expected benefit, risk rating, and reason.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_models import (
    MarketContext,
    TransitionResult,
)
from trainer.lifecycle.lifecycle_states import (
    STATE_EXPECTED_DURATION_HOURS,
    LifecycleState,
    can_transition,
)

# ── Transition Rule ───────────────────────────────────────────────────────────


@dataclass
class TransitionRule:
    """Declarative definition of one state transition."""

    from_state: LifecycleState
    to_state: LifecycleState
    entry_conditions: list[str]  # human-readable required conditions
    exit_conditions: list[str]  # conditions that trigger exit from from_state
    validation_rules: list[str]  # rules that must hold for the transition
    timeout_hours: float  # max time to wait for conditions
    base_confidence: float  # 0-1 default confidence when conditions met
    expected_benefit: float  # 0-1 upside if transition proceeds
    risk_rating: str  # LOW | MEDIUM | HIGH | CRITICAL
    description: str


# ── Rule Registry ─────────────────────────────────────────────────────────────

_RULES: list[TransitionRule] = [
    TransitionRule(
        LifecycleState.IDEA,
        LifecycleState.WATCHLIST,
        entry_conditions=["Instrument identified", "Initial screening passed"],
        exit_conditions=["Setup pattern detected", "Momentum alignment observed"],
        validation_rules=["confidence >= 0.40", "news_risk < 0.70"],
        timeout_hours=72.0,
        base_confidence=0.50,
        expected_benefit=0.30,
        risk_rating="LOW",
        description="Promote idea to watchlist on initial setup detection",
    ),
    TransitionRule(
        LifecycleState.WATCHLIST,
        LifecycleState.PREPARATION,
        entry_conditions=["Setup strengthening", "Trend direction clear"],
        exit_conditions=["Entry zone approaching", "Conditions improving"],
        validation_rules=["confidence >= 0.55", "trend_valid = True", "spread_ok"],
        timeout_hours=48.0,
        base_confidence=0.60,
        expected_benefit=0.45,
        risk_rating="LOW",
        description="Move to preparation when setup is developing",
    ),
    TransitionRule(
        LifecycleState.PREPARATION,
        LifecycleState.ENTRY_READY,
        entry_conditions=["Price near entry zone", "All conditions confirmed"],
        exit_conditions=["Spread acceptable", "Liquidity adequate", "No news risk"],
        validation_rules=["confidence >= 0.65", "news_risk < 0.40", "spread_ok", "trend_valid"],
        timeout_hours=24.0,
        base_confidence=0.70,
        expected_benefit=0.60,
        risk_rating="MEDIUM",
        description="Entry ready when all pre-trade conditions satisfied",
    ),
    TransitionRule(
        LifecycleState.ENTRY_READY,
        LifecycleState.PENDING_ENTRY,
        entry_conditions=["Execution timing favourable", "Risk budget available"],
        exit_conditions=["Order placed", "Awaiting fill"],
        validation_rules=["confidence >= 0.65", "session_ok", "spread_ok"],
        timeout_hours=8.0,
        base_confidence=0.72,
        expected_benefit=0.65,
        risk_rating="MEDIUM",
        description="Order submitted — awaiting fill confirmation",
    ),
    TransitionRule(
        LifecycleState.PENDING_ENTRY,
        LifecycleState.ACTIVE,
        entry_conditions=["Fill confirmed (notional)"],
        exit_conditions=["Position open", "Risk management active"],
        validation_rules=["confidence >= 0.60"],
        timeout_hours=4.0,
        base_confidence=0.75,
        expected_benefit=0.70,
        risk_rating="MEDIUM",
        description="Trade now active — position management begins",
    ),
    TransitionRule(
        LifecycleState.ACTIVE,
        LifecycleState.POSITION_BUILDING,
        entry_conditions=["High confidence", "Momentum aligned", "Risk budget allows"],
        exit_conditions=["Scale-in triggered at pullback", "Position size increased"],
        validation_rules=["confidence >= 0.75", "momentum_aligned", "news_risk < 0.30"],
        timeout_hours=12.0,
        base_confidence=0.70,
        expected_benefit=0.55,
        risk_rating="MEDIUM",
        description="Adding to position on confirmed pullback",
    ),
    TransitionRule(
        LifecycleState.ACTIVE,
        LifecycleState.PROTECTED,
        entry_conditions=["Price reached TP1", "Breakeven triggered"],
        exit_conditions=["Stop moved to breakeven", "Minimum profit locked"],
        validation_rules=["price_beyond_tp1"],
        timeout_hours=48.0,
        base_confidence=0.80,
        expected_benefit=0.75,
        risk_rating="LOW",
        description="Stop moved to breakeven — worst case is scratch",
    ),
    TransitionRule(
        LifecycleState.ACTIVE,
        LifecycleState.PARTIAL_EXIT,
        entry_conditions=["TP1 reached", "Partial profit secured"],
        exit_conditions=["50% position closed", "Remaining at reduced risk"],
        validation_rules=["price_beyond_tp1"],
        timeout_hours=24.0,
        base_confidence=0.78,
        expected_benefit=0.65,
        risk_rating="LOW",
        description="Partial close at TP1 — profit locked on half position",
    ),
    TransitionRule(
        LifecycleState.PROTECTED,
        LifecycleState.PARTIAL_EXIT,
        entry_conditions=["TP2 approaching", "Additional profit opportunity"],
        exit_conditions=["30% more closed at TP2"],
        validation_rules=["price_beyond_tp1"],
        timeout_hours=48.0,
        base_confidence=0.75,
        expected_benefit=0.60,
        risk_rating="LOW",
        description="Secured position — now targeting TP2",
    ),
    TransitionRule(
        LifecycleState.PROTECTED,
        LifecycleState.RUNNER,
        entry_conditions=["TP2 reached", "Runner position remains"],
        exit_conditions=["20% position running to full target"],
        validation_rules=["confidence >= 0.70"],
        timeout_hours=72.0,
        base_confidence=0.72,
        expected_benefit=0.80,
        risk_rating="LOW",
        description="Runner activated — allowing position to reach full target",
    ),
    TransitionRule(
        LifecycleState.PARTIAL_EXIT,
        LifecycleState.RUNNER,
        entry_conditions=["TP2 hit", "Final runner remaining"],
        exit_conditions=["Remaining 20% running with trail"],
        validation_rules=["confidence >= 0.65"],
        timeout_hours=72.0,
        base_confidence=0.70,
        expected_benefit=0.75,
        risk_rating="LOW",
        description="Partial exit complete — runner active",
    ),
    TransitionRule(
        LifecycleState.RUNNER,
        LifecycleState.TRAILING,
        entry_conditions=["Price extending — trail stop activated"],
        exit_conditions=["Dynamic trail following price"],
        validation_rules=["momentum_aligned"],
        timeout_hours=48.0,
        base_confidence=0.75,
        expected_benefit=0.70,
        risk_rating="LOW",
        description="Trail stop now managing the runner",
    ),
    TransitionRule(
        LifecycleState.TRAILING,
        LifecycleState.EXIT_READY,
        entry_conditions=["Trail stop hit OR runner target reached"],
        exit_conditions=["Exit signal confirmed"],
        validation_rules=[],
        timeout_hours=24.0,
        base_confidence=0.85,
        expected_benefit=0.90,
        risk_rating="LOW",
        description="Exit conditions met — preparing to close",
    ),
    TransitionRule(
        LifecycleState.EXIT_READY,
        LifecycleState.CLOSING,
        entry_conditions=["Exit order being executed (notional)"],
        exit_conditions=["All positions closing"],
        validation_rules=[],
        timeout_hours=1.0,
        base_confidence=0.95,
        expected_benefit=0.95,
        risk_rating="LOW",
        description="Trade closing — final execution",
    ),
    TransitionRule(
        LifecycleState.CLOSING,
        LifecycleState.CLOSED,
        entry_conditions=["All positions closed"],
        exit_conditions=["Trade completed"],
        validation_rules=[],
        timeout_hours=0.5,
        base_confidence=1.0,
        expected_benefit=1.0,
        risk_rating="LOW",
        description="Trade fully closed",
    ),
    # Stopped / emergency paths
    TransitionRule(
        LifecycleState.ACTIVE,
        LifecycleState.STOPPED,
        entry_conditions=["Stop loss hit"],
        exit_conditions=["Position closed at stop"],
        validation_rules=[],
        timeout_hours=0.0,
        base_confidence=0.95,
        expected_benefit=0.0,
        risk_rating="HIGH",
        description="Stop loss triggered — position closed",
    ),
    TransitionRule(
        LifecycleState.ACTIVE,
        LifecycleState.EMERGENCY_EXIT,
        entry_conditions=["Critical risk event", "Emergency stop triggered"],
        exit_conditions=["Position closed immediately"],
        validation_rules=["news_risk > 0.80 OR volatility_extreme"],
        timeout_hours=0.0,
        base_confidence=1.0,
        expected_benefit=0.0,
        risk_rating="CRITICAL",
        description="Emergency exit — critical market condition",
    ),
]

# Build fast lookup: (from, to) → rule
_RULE_LOOKUP: dict[tuple[LifecycleState, LifecycleState], TransitionRule] = {
    (r.from_state, r.to_state): r for r in _RULES
}


# ── State Machine ─────────────────────────────────────────────────────────────


class StateMachine:
    """Validates lifecycle state transitions and reports context.

    Pure deterministic FSM — does not place trades. Structural graph
    validity (`hard_block`) is the only hard invariant enforced here.
    Contextual/confidence rule checks (news risk, spread, trend, session,
    momentum, confidence thresholds) are advisory context for the decision
    layer (QDE) — a failed rule does NOT itself veto a transition; it is
    surfaced via `conditions_failed` for QDE to weigh.
    """

    def validate_transition(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        context: MarketContext | None = None,
        confidence: float = 0.65,
    ) -> TransitionResult:
        """Check whether a transition is structurally and contextually valid."""

        # 1. Structural check — hard invariant (data integrity, not a
        # strategy decision). This is the only case that may block a
        # transition outright.
        if not can_transition(from_state, to_state):
            return TransitionResult(
                from_state=from_state,
                to_state=to_state,
                allowed=False,
                hard_block=True,
                conditions_failed=[
                    f"{from_state.value} → {to_state.value} not in VALID_TRANSITIONS"
                ],
                reason="Transition not permitted by state machine",
            )

        # 2. Rule lookup
        rule = _RULE_LOOKUP.get((from_state, to_state))
        if rule is None:
            # Allowed by graph but no detailed rule — permit with defaults
            return TransitionResult(
                from_state=from_state,
                to_state=to_state,
                allowed=True,
                confidence=confidence,
                expected_benefit=0.50,
                risk_rating="MEDIUM",
                reason="Transition allowed (no detailed rule defined)",
                timeout_hours=STATE_EXPECTED_DURATION_HOURS.get(from_state, 24.0),
            )

        # 3. Contextual validation
        met: list[str] = []
        failed: list[str] = []

        if context is not None:
            # news_risk check
            for v in rule.validation_rules:
                if "news_risk < 0.40" in v:
                    if context.news_risk < 0.40:
                        met.append("news_risk acceptable")
                    else:
                        failed.append(f"news_risk {context.news_risk:.2f} ≥ 0.40")
                elif "news_risk < 0.30" in v:
                    if context.news_risk < 0.30:
                        met.append("news_risk low")
                    else:
                        failed.append(f"news_risk {context.news_risk:.2f} ≥ 0.30")
                elif "spread_ok" in v:
                    if context.spread < context.atr * 0.30:
                        met.append("spread acceptable")
                    else:
                        failed.append("spread too wide vs ATR")
                elif "trend_valid" in v:
                    if context.trend_valid:
                        met.append("trend valid")
                    else:
                        failed.append("trend invalidated")
                elif "session_ok" in v:
                    if context.session not in ("CLOSED",):
                        met.append("session active")
                    else:
                        failed.append("session closed")
                elif "momentum_aligned" in v:
                    if abs(context.momentum) > 0.20:
                        met.append("momentum aligned")
                    else:
                        failed.append("momentum weak")
        else:
            met.append("no market context provided — assuming conditions met")

        # Confidence check
        if "confidence >= 0.65" in rule.validation_rules and confidence < 0.65:
            failed.append(f"confidence {confidence:.2f} < 0.65")
        elif "confidence >= 0.60" in rule.validation_rules and confidence < 0.60:
            failed.append(f"confidence {confidence:.2f} < 0.60")

        allowed = len(failed) == 0

        # Effective confidence
        eff_conf = rule.base_confidence * (0.8 + 0.2 * confidence)

        logger.debug(
            f"StateMachine: {from_state.value} → {to_state.value} "
            f"allowed={allowed} failed={failed}"
        )

        return TransitionResult(
            from_state=from_state,
            to_state=to_state,
            allowed=allowed,
            conditions_met=met or list(rule.entry_conditions),
            conditions_failed=failed,
            confidence=round(min(eff_conf, 1.0), 4),
            expected_benefit=rule.expected_benefit,
            risk_rating=rule.risk_rating,
            reason=rule.description,
            timeout_hours=rule.timeout_hours,
        )

    def all_rules(self) -> list[TransitionRule]:
        return list(_RULES)

    def rules_for_state(self, state: LifecycleState) -> list[TransitionRule]:
        return [r for r in _RULES if r.from_state == state]

    def get_rule(
        self, from_state: LifecycleState, to_state: LifecycleState
    ) -> TransitionRule | None:
        return _RULE_LOOKUP.get((from_state, to_state))
