"""Phase 28.3 — QDE Intelligence.

THE SINGLE DECISION-MAKING INTELLIGENCE OF THE QUANT DECISION ENGINE.

This is the only module in the trainer that may produce QDEDecision objects.
All 11 capability modules are internal faculties consumed here.
All 13 QDE responsibilities are resolved here and only here.

ARCHITECTURAL INVARIANTS
-------------------------
1. QDEDecision is constructed ONLY inside this module.
2. No capability module may import or construct QDEDecision.
3. Only QDEIntelligence.reason() returns QDEDecision objects.
4. All capability outputs are advisory — they inform, never instruct.
5. The QDE is a single intelligence — capabilities are its internal faculties,
   not separate engines and not separate decision-makers.

RESPONSIBILITIES (all resolved inside reason())
------------------------------------------------
Entry              — should the QDE open a new position?
Exit               — should the QDE close an open position?
Position Size      — what lot size?
Scaling            — should the QDE add to a winning position?
Partial Close      — should the QDE take partial profits?
Break Even         — should the QDE move stop to entry?
Exposure           — is current exposure within bounds?
Capital Preservation — are any circuit breakers active?
Trade Duration     — is any trade overdue for closure?
Reversal Detection — is a trend reversing against open positions?
Continuation Detection — is the trend continuing for open positions?
Regime Understanding — what market regime are we in?
Market Understanding — what is the complete market context?
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.qde.capabilities.cap_models import (
    BreakEvenSignal,
    CircuitBreakerStatus,
    DecisionType,
    Direction,
    EntrySignal,
    ExitSignal,
    MarketContext,
    MarketSnapshot,
    PartialCloseSignal,
    QDEDecision,
    RegimeType,
    ScalingSignal,
)
from trainer.qde.capabilities.capital_preservation import CapitalPreservationCapability
from trainer.qde.capabilities.expected_value_reasoning import ExpectedValueReasoningCapability
from trainer.qde.capabilities.exposure_reasoning import ExposureReasoningCapability
from trainer.qde.capabilities.learning import LearningCapability
from trainer.qde.capabilities.market_memory import MarketMemoryCapability
from trainer.qde.capabilities.market_understanding import MarketUnderstandingCapability
from trainer.qde.capabilities.opportunity_discovery import OpportunityDiscoveryCapability
from trainer.qde.capabilities.position_management import PositionManagementCapability
from trainer.qde.capabilities.reflection import ReflectionCapability
from trainer.qde.capabilities.risk_reasoning import RiskReasoningCapability
from trainer.qde.capabilities.trade_management import TradeManagementCapability


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class QDEConfig:
    """Runtime configuration for the QDE Intelligence."""

    # Capital Preservation circuit breakers
    daily_loss_limit_pct: float = 0.03
    drawdown_limit_pct: float = 0.10
    consecutive_loss_limit: int = 5

    # Risk parameters
    risk_per_trade_pct: float = 0.01
    max_risk_per_trade_pct: float = 0.02

    # Opportunity thresholds
    min_confluence_to_enter: float = 0.55
    min_ev_to_enter: float = 0.0          # EV must be ≥ this (in pips)
    min_rr_to_enter: float = 1.5

    # Exposure limits
    max_gross_exposure_pct: float = 0.06

    # Pip value assumption
    pip_value_per_lot: float = 10.0

    # Account equity fallback if not in snapshot
    default_equity: float = 10_000.0

    # Minimum overall confidence to enter (across all capabilities)
    min_overall_confidence: float = 0.50

    # Capability trust weights (can be updated by Learning)
    capability_weights: dict[str, float] = field(default_factory=lambda: {
        "market_understanding":    1.00,
        "opportunity_discovery":   0.90,
        "expected_value_reasoning": 0.85,
        "risk_reasoning":          0.95,
        "exposure_reasoning":      0.90,
        "trade_management":        0.95,
        "position_management":     0.90,
        "capital_preservation":    1.00,   # Always full weight — safety critical
        "market_memory":           0.75,
        "reflection":              0.80,
        "learning":                0.80,
    })


# ── QDE Intelligence ──────────────────────────────────────────────────────────


class QDEIntelligence:
    """The Quant Decision Engine — a single trading intelligence.

    Internally orchestrates 11 cognitive capabilities. Produces QDEDecision
    objects as the sole output. Nothing else in the trainer produces decisions.

    Usage::

        qde = QDEIntelligence(config=QDEConfig())
        snapshot = MarketSnapshot(instrument="EURUSD", ...)
        decision = qde.reason(snapshot)

        # decision.decision_type tells you what the QDE decided
        # decision.entry is populated if decision_type == ENTER
        # decision.exits is populated if decision_type == EXIT
        # etc.

    The QDE is stateful: Market Memory and Learning persist across calls.
    """

    def __init__(self, config: QDEConfig | None = None) -> None:
        self.config = config or QDEConfig()

        # ── Initialise all 11 capabilities ────────────────────────────────────
        self._market_understanding = MarketUnderstandingCapability()
        self._opportunity_discovery = OpportunityDiscoveryCapability()
        self._ev_reasoning = ExpectedValueReasoningCapability()
        self._risk_reasoning = RiskReasoningCapability()
        self._exposure_reasoning = ExposureReasoningCapability()
        self._trade_management = TradeManagementCapability()
        self._position_management = PositionManagementCapability()
        self._capital_preservation = CapitalPreservationCapability(
            daily_loss_limit_pct=self.config.daily_loss_limit_pct,
            drawdown_limit_pct=self.config.drawdown_limit_pct,
            consecutive_loss_limit=self.config.consecutive_loss_limit,
        )
        self._market_memory = MarketMemoryCapability()
        self._reflection = ReflectionCapability()
        self._learning = LearningCapability()

        # ── Decision audit trail ──────────────────────────────────────────────
        self._decision_count: int = 0
        self._last_decision: QDEDecision | None = None

    # ── Core reasoning cycle ──────────────────────────────────────────────────

    def reason(self, snapshot: MarketSnapshot) -> QDEDecision:
        """Execute the full QDE reasoning cycle.

        This is the ONLY entry point that produces a QDEDecision.

        The reasoning cycle proceeds in six stages:
          Stage 1 — Preservation check (safety gate — may short-circuit)
          Stage 2 — Market understanding and regime analysis
          Stage 3 — Trade management evaluation (open positions)
          Stage 4 — Opportunity discovery and EV scoring (entry path)
          Stage 5 — Exposure and position sizing
          Stage 6 — Decision synthesis (all 13 responsibilities resolved)

        Args:
            snapshot: Complete market state at this reasoning instant.

        Returns:
            QDEDecision — the single, unified decision of the QDE.
        """
        self._decision_count += 1
        equity = snapshot.account.equity if snapshot.account else self.config.default_equity
        t0 = time.monotonic()

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 1 — CAPITAL PRESERVATION (SAFETY GATE)
        # ═════════════════════════════════════════════════════════════════════
        preservation = self._capital_preservation.evaluate(snapshot)

        if preservation.forced_exit_required:
            # Circuit breaker triggered — produce PRESERVE decision immediately
            forced_exits = [
                ExitSignal(
                    trade_id=t.trade_id,
                    exit_price=snapshot.mid,
                    reason=preservation.forced_exit_reason,
                    urgency="FORCED",
                )
                for t in snapshot.open_trades
            ]
            decision = QDEDecision.preserve(
                snapshot=snapshot,
                reason=preservation.forced_exit_reason,
                circuit_breaker=preservation.circuit_breaker,
                forced_exits=forced_exits,
            )
            decision.capability_contributions["capital_preservation"] = 1.0
            self._last_decision = decision
            logger.warning(
                f"QDE PRESERVE: {preservation.forced_exit_reason} "
                f"[{snapshot.instrument}] decision_id={decision.decision_id}"
            )
            return decision

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 2 — MARKET UNDERSTANDING
        # ═════════════════════════════════════════════════════════════════════
        market_ctx = self._market_understanding.analyse(snapshot)
        memory_ctx = self._market_memory.recall(snapshot, None, market_ctx)

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 3 — TRADE MANAGEMENT (open position evaluation)
        # ═════════════════════════════════════════════════════════════════════
        trade_signals = self._trade_management.evaluate(snapshot, market_ctx)

        # Resolve trade management responsibilities
        exits = self._resolve_exits(snapshot, trade_signals, market_ctx)
        break_evens = self._resolve_break_evens(snapshot, trade_signals)
        partial_closes = self._resolve_partial_closes(snapshot, trade_signals)

        # If there are urgent exits, emit a decision now (management priority)
        urgent_exits = [e for e in exits if e.urgency == "URGENT"]
        if urgent_exits and not exits.__class__.__name__:  # non-blocking — fall through
            pass  # exits are accumulated and emitted in Stage 6

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 4 — OPPORTUNITY DISCOVERY & EV SCORING (entry path)
        # ═════════════════════════════════════════════════════════════════════
        entry_signal: EntrySignal | None = None
        scaling_signals: list[ScalingSignal] = []
        opportunity = None
        ev_score = None
        risk_assess = None

        if preservation.new_entries_allowed:
            opportunity = self._opportunity_discovery.discover(snapshot, market_ctx)
            memory_ctx = self._market_memory.recall(snapshot, opportunity.best_candidate, market_ctx)

            if opportunity.market_is_tradeable and opportunity.best_candidate is not None:
                candidate = opportunity.best_candidate
                risk_assess = self._risk_reasoning.assess(
                    snapshot, candidate, market_ctx,
                    risk_per_trade_pct=self.config.risk_per_trade_pct,
                )
                ev_score = self._ev_reasoning.score(
                    snapshot, candidate, risk_assess, memory_ctx, market_ctx
                )

                if ev_score.ev_threshold_met and ev_score.risk_reward_ratio >= self.config.min_rr_to_enter:
                    exposure = self._exposure_reasoning.analyse(snapshot, equity)
                    if exposure.adding_exposure_is_safe:
                        pos_size = self._position_management.compute(
                            snapshot, risk_assess, ev_score, market_ctx, equity,
                            risk_pct=self.config.risk_per_trade_pct,
                        )
                        entry_signal = EntrySignal(
                            direction=candidate.direction,
                            entry_price=snapshot.mid,
                            stop_loss=risk_assess.stop_loss_price,
                            take_profit=risk_assess.take_profit_price,
                            lot_size=pos_size.lot_size,
                            setup_type=candidate.setup_type,
                            confluence_score=candidate.confluence_score,
                            ev_score=ev_score.ev,
                            risk_pct=pos_size.risk_pct,
                        )

        # Scaling: only if preservation allows and trend management signals it
        if preservation.scaling_allowed and trade_signals.trades_eligible_for_scaling:
            scaling_signals = self._resolve_scaling(snapshot, trade_signals, market_ctx, equity)

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 5 — EXPOSURE CHECK (post-entry sizing validation)
        # ═════════════════════════════════════════════════════════════════════
        if entry_signal is not None:
            exposure = self._exposure_reasoning.analyse(snapshot, equity)
            if not exposure.adding_exposure_is_safe:
                entry_signal = None  # Veto entry due to exposure
                logger.info(
                    f"QDE vetoed entry: exposure headroom exhausted "
                    f"[{snapshot.instrument}]"
                )

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 6 — DECISION SYNTHESIS (all 13 responsibilities resolved)
        # ═════════════════════════════════════════════════════════════════════
        decision_type = self._classify_decision_type(
            entry_signal, exits, scaling_signals, partial_closes, break_evens
        )

        cap_contributions = self._compute_contributions(
            market_ctx, opportunity, ev_score, risk_assess, preservation
        )
        overall_confidence = self._compute_overall_confidence(cap_contributions)

        reasoning = self._build_reasoning_summary(
            market_ctx, opportunity, ev_score, preservation,
            entry_signal, exits, scaling_signals, partial_closes, break_evens,
            decision_type,
        )

        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

        decision = QDEDecision(
            decision_id=f"dec-{snapshot.snapshot_id[-8:] if snapshot.snapshot_id else 'xxxx'}-{self._decision_count:04d}",
            snapshot_id=snapshot.snapshot_id,
            instrument=snapshot.instrument,
            decision_type=decision_type,
            decided_at=self._now_iso(),

            # 13 responsibilities
            entry=entry_signal,
            exits=exits,
            scaling=scaling_signals,
            partial_closes=partial_closes,
            break_evens=break_evens,
            regime=market_ctx.regime,
            regime_transition=market_ctx.regime_transition,
            trend_direction=market_ctx.trend_direction,
            trend_strength=market_ctx.trend_strength,
            reversal_detected=trade_signals.reversal_detected,
            continuation_detected=trade_signals.continuation_detected,
            circuit_breaker=preservation.circuit_breaker,
            preservation_active=False,

            capability_contributions=cap_contributions,
            reasoning_summary=reasoning,
            overall_confidence=overall_confidence,
        )

        self._last_decision = decision
        logger.info(
            f"QDE {decision_type.value}: {snapshot.instrument} "
            f"regime={market_ctx.regime.value} "
            f"confidence={overall_confidence:.2f} "
            f"elapsed={elapsed_ms}ms "
            f"decision_id={decision.decision_id}"
        )
        return decision

    # ── Responsibility resolvers ──────────────────────────────────────────────
    # Each resolver maps one capability output to one QDE responsibility.
    # No resolver produces a QDEDecision — only builds signal components.

    def _resolve_exits(
        self,
        snapshot: MarketSnapshot,
        trade_signals: Any,
        market_ctx: MarketContext,
    ) -> list[ExitSignal]:
        exits: list[ExitSignal] = []
        for trade in snapshot.open_trades:
            if trade.trade_id in trade_signals.trades_needing_exit:
                urgency = "URGENT" if trade.r_multiple < -1.0 else "NORMAL"
                exits.append(ExitSignal(
                    trade_id=trade.trade_id,
                    exit_price=snapshot.mid,
                    reason=trade_signals.per_trade_notes.get(trade.trade_id, "Management signal"),
                    urgency=urgency,
                ))
            elif trade.trade_id in trade_signals.trades_exceeding_duration:
                exits.append(ExitSignal(
                    trade_id=trade.trade_id,
                    exit_price=snapshot.mid,
                    reason="Trade duration limit exceeded",
                    urgency="NORMAL",
                ))
        return exits

    def _resolve_break_evens(
        self,
        snapshot: MarketSnapshot,
        trade_signals: Any,
    ) -> list[BreakEvenSignal]:
        result: list[BreakEvenSignal] = []
        for trade in snapshot.open_trades:
            if trade.trade_id in trade_signals.trades_needing_break_even:
                result.append(BreakEvenSignal(
                    trade_id=trade.trade_id,
                    new_stop_loss=trade.entry_price,
                    reason=f"Break-even at {trade.r_multiple:.1f}R profit",
                ))
        return result

    def _resolve_partial_closes(
        self,
        snapshot: MarketSnapshot,
        trade_signals: Any,
    ) -> list[PartialCloseSignal]:
        result: list[PartialCloseSignal] = []
        for trade in snapshot.open_trades:
            if trade.trade_id in trade_signals.trades_needing_partial_close:
                result.append(PartialCloseSignal(
                    trade_id=trade.trade_id,
                    close_fraction=0.50,  # Close 50% at 2R
                    reason=f"Partial profit lock at {trade.r_multiple:.1f}R",
                ))
        return result

    def _resolve_scaling(
        self,
        snapshot: MarketSnapshot,
        trade_signals: Any,
        market_ctx: MarketContext,
        equity: float,
    ) -> list[ScalingSignal]:
        result: list[ScalingSignal] = []
        for trade in snapshot.open_trades:
            if trade.trade_id not in trade_signals.trades_eligible_for_scaling:
                continue
            # Compute a scaling position size (half of original risk)
            scale_lot = round(trade.lot_size * 0.50, 2)
            if scale_lot < 0.01:
                continue
            # Trailing stop: move to 0.5R below current price
            atr = snapshot.atr_14 or abs(trade.entry_price - trade.stop_loss)
            if trade.direction == Direction.LONG:
                new_stop = snapshot.mid - atr * 1.0
            else:
                new_stop = snapshot.mid + atr * 1.0
            result.append(ScalingSignal(
                trade_id=trade.trade_id,
                additional_lot_size=scale_lot,
                new_stop_loss=round(new_stop, 6),
                reason=f"Scale-in at {trade.r_multiple:.1f}R; trend={market_ctx.trend_strength.value}",
            ))
        return result

    # ── Synthesis helpers ─────────────────────────────────────────────────────

    def _classify_decision_type(
        self,
        entry: EntrySignal | None,
        exits: list[ExitSignal],
        scaling: list[ScalingSignal],
        partials: list[PartialCloseSignal],
        break_evens: list[BreakEvenSignal],
    ) -> DecisionType:
        if exits:
            return DecisionType.EXIT
        if entry:
            return DecisionType.ENTER
        if scaling:
            return DecisionType.SCALE
        if partials:
            return DecisionType.PARTIAL_CLOSE
        if break_evens:
            return DecisionType.BREAK_EVEN
        return DecisionType.HOLD

    def _compute_contributions(
        self,
        market_ctx: Any,
        opportunity: Any,
        ev_score: Any,
        risk_assess: Any,
        preservation: Any,
    ) -> dict[str, float]:
        weights = self.config.capability_weights
        contributions: dict[str, float] = {
            "market_understanding": market_ctx.confidence * weights.get("market_understanding", 1.0),
            "capital_preservation": preservation.confidence * weights.get("capital_preservation", 1.0),
        }
        if opportunity:
            contributions["opportunity_discovery"] = opportunity.confidence * weights.get("opportunity_discovery", 0.9)
        if ev_score:
            contributions["expected_value_reasoning"] = ev_score.confidence * weights.get("expected_value_reasoning", 0.85)
        if risk_assess:
            contributions["risk_reasoning"] = risk_assess.confidence * weights.get("risk_reasoning", 0.95)
        return {k: round(v, 4) for k, v in contributions.items()}

    def _compute_overall_confidence(self, contributions: dict[str, float]) -> float:
        if not contributions:
            return 0.0
        total_weight = sum(self.config.capability_weights.get(k, 1.0) for k in contributions)
        weighted_sum = sum(v for v in contributions.values())
        return min(1.0, weighted_sum / total_weight) if total_weight > 0 else 0.0

    def _build_reasoning_summary(
        self,
        market_ctx: Any,
        opportunity: Any,
        ev_score: Any,
        preservation: Any,
        entry: Any,
        exits: list,
        scaling: list,
        partials: list,
        break_evens: list,
        decision_type: DecisionType,
    ) -> str:
        parts = [
            f"Regime={market_ctx.regime.value}/{market_ctx.regime_transition.value}",
            f"Trend={market_ctx.trend_direction.value}/{market_ctx.trend_strength.value}",
            f"Preservation={preservation.circuit_breaker.value}",
        ]
        if opportunity:
            parts.append(f"Opportunity={opportunity.opportunity_quality:.2f}")
        if ev_score:
            parts.append(f"EV={ev_score.ev:.4f} RR={ev_score.risk_reward_ratio:.2f} WinP={ev_score.win_probability:.0%}")
        if entry:
            parts.append(f"Entry={entry.direction.value} lot={entry.lot_size:.2f}")
        if exits:
            parts.append(f"Exits={len(exits)}")
        if scaling:
            parts.append(f"Scale={len(scaling)}")
        if partials:
            parts.append(f"Partial={len(partials)}")
        if break_evens:
            parts.append(f"BreakEven={len(break_evens)}")
        parts.append(f"Decision={decision_type.value}")
        return " | ".join(parts)

    # ── Post-trade learning interface ─────────────────────────────────────────

    def reflect_and_learn(
        self,
        outcome: Any,
        predicted_win_probability: float = 0.5,
        predicted_ev: float = 0.0,
    ) -> None:
        """Process a completed trade outcome through Reflection → Learning.

        Called post-trade. Updates internal priors.
        Does NOT produce a QDEDecision — this is a maintenance operation.
        """
        from trainer.qde.capabilities.market_memory import HistoricalOutcome
        import time as _time
        import uuid as _uuid

        insight = self._reflection.reflect(outcome, predicted_win_probability, predicted_ev)
        update = self._learning.learn(insight)

        # Apply updated capability weights from learning
        for cap, trust in update.capability_trust_scores.items():
            if cap in self.config.capability_weights:
                self.config.capability_weights[cap] = trust

        # Store outcome in memory
        result = "WIN" if outcome.pnl_r > 0 else ("LOSS" if outcome.pnl_r < 0 else "NEUTRAL")
        hist = HistoricalOutcome(
            outcome_id=f"hist-{_uuid.uuid4().hex[:8]}",
            setup_type=outcome.setup_type,
            regime=outcome.regime_at_entry,
            direction=outcome.direction.value,
            result=result,
            pnl_r=outcome.pnl_r,
            rsi_at_entry=outcome.capability_scores_at_entry.get("rsi_at_entry", 50.0),
            adx_at_entry=outcome.capability_scores_at_entry.get("adx_at_entry", 0.0),
            atr_ratio=outcome.capability_scores_at_entry.get("atr_ratio", 1.0),
            timestamp=outcome.closed_at,
        )
        self._market_memory.store(hist)

        logger.info(
            f"QDE reflect_and_learn: trade={outcome.trade_id} "
            f"result={result} pnl_r={outcome.pnl_r:.2f} "
            f"delta={update.convergence_delta:.6f}"
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return the current state of the QDE for monitoring."""
        return {
            "decision_count": self._decision_count,
            "capability_weights": {k: round(v, 4) for k, v in self.config.capability_weights.items()},
            "capability_trust_scores": self._learning.get_trust_scores(),
            "regime_priors": self._learning.get_regime_priors(),
            "memory_size": len(self._market_memory._outcomes),
            "last_decision_id": self._last_decision.decision_id if self._last_decision else None,
        }

    @staticmethod
    def _now_iso() -> str:
        from datetime import UTC, datetime
        return datetime.now(UTC).isoformat()
