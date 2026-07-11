"""QDE Brain — decision API (Phase C6).

Three spec-required methods:

    ``make_decision(instrument, brain_state, debate_record, portfolio_state, now)``
        Translates accumulated cognition into a ``CognitiveDecision``.

          1. Build ``DecisionInput`` from ``BrainState`` + supplements.
          2. Compute uncertainty via ``compute_uncertainty()`` (delegates).
          3. Compute conviction via ``compute_conviction()`` (delegates).
          4. Apply ``CalibrationProfile`` bias.
          5. Call ``compute_decision()`` (decision engine).
          6. Store to memory (episodic + working memory + local deque).

    ``explain_decision(decision, brain_state, debate_record, now)``
        Produce a structured chain-of-thought ``DecisionReasoning`` via
        ``ReasoningEngine.explain()`` (delegates).

    ``confidence_report(decision, brain_state, now)``
        Produce a ``ConfidenceReport`` that breaks down confidence vs.
        conviction, calibration bias, and historical accuracy.

What this module does NOT do
----------------------------
* Never imports or references ``QDEDecision`` (execution vocabulary).
* Never reimplements ``compute_uncertainty()`` or ``compute_conviction()``.
* Never places trades or calls ``ExecutionPlanner``.
* The decision is advisory — how the inference layer uses it is up to
  the caller.

Dependency direction
--------------------
``DecisionAPI``
    → ``compute_uncertainty()`` (uncertainty_engine)
    → ``compute_conviction()`` (conviction_engine)
    → ``compute_decision()`` (decision_engine — internal)
    → ``ReasoningEngine`` (decision_reasoning — internal)
    → ``DecisionMemoryBridge`` (decision_memory — internal)
    → ``CalibrationProfile`` (brain_models — read-only)
    → ``DebateRecord`` (C5 debate package — read-only)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.conviction_engine import (
    ConvictionProfile,
    ConvictionScore,
    compute_conviction,
)
from trainer.qde.brain.decision.decision_engine import compute_decision
from trainer.qde.brain.decision.decision_memory import DecisionMemoryBridge
from trainer.qde.brain.decision.decision_reasoning import ReasoningEngine
from trainer.qde.brain.decision.decision_types import (
    CognitiveDecision,
    ConfidenceReport,
    DecisionInput,
    DecisionReasoning,
)
from trainer.qde.brain.uncertainty_engine import compute_uncertainty

if TYPE_CHECKING:
    from trainer.qde.brain.brain_models import BrainState
    from trainer.qde.brain.debate.debate_store import DebateRecord


class DecisionAPI:
    """Phase C6 public API: ``make_decision``, ``explain_decision``,
    ``confidence_report``.

    One instance is shared across all instruments.

    Parameters
    ----------
    memory_bridge:
        ``DecisionMemoryBridge`` for persistence.  None → a fresh bridge
        with no C2 episodic backend is created (local deque still works).
    """

    def __init__(
        self,
        memory_bridge: DecisionMemoryBridge | None = None,
    ) -> None:
        self._bridge = memory_bridge or DecisionMemoryBridge()
        self._reasoning = ReasoningEngine()

    # ── Primary API ───────────────────────────────────────────────────

    def make_decision(
        self,
        instrument: str,
        brain_state: "BrainState",
        debate_record: "DebateRecord | None" = None,
        portfolio_state: dict | None = None,
        now: float | None = None,
    ) -> CognitiveDecision:
        """Translate accumulated cognition into a ``CognitiveDecision``.

        Parameters
        ----------
        instrument:
            Symbol to decide on.
        brain_state:
            The full ``BrainState`` for the symbol (narrative, hypotheses,
            calibration, personality, etc.).
        debate_record:
            Optional ``DebateRecord`` from ``DebateAPI.create_debate()``
            (C5).  If None, MONITOR is the highest reachable action.
        portfolio_state:
            Optional dict with portfolio-level context::

                {
                    "open_positions_count":  int,     # default 0
                    "daily_pnl_pct":        float,    # default 0.0
                    "has_existing_position": bool,    # default False
                    "existing_direction":    str,     # "BULLISH"/"BEARISH"/None
                    "memory_win_rate":       float,   # default None
                    "memory_recent_count":   int,     # default 0
                }
        now:
            Override current time.

        Returns
        -------
        CognitiveDecision
            Stored to memory immediately; also returned for the caller
            to inspect or pass to ``explain_decision()`` /
            ``confidence_report()``.
        """
        now = now if now is not None else time.time()
        ps = portfolio_state or {}

        active_hyps = brain_state.active_hypotheses()

        # ── Step 1: uncertainty (delegates entirely) ──────────────────
        narrative = brain_state.narrative
        uncertainty = compute_uncertainty(
            instrument=instrument,
            news_risk=0.0,
            volatility_state=narrative.volatility_state if narrative else "NORMAL",
            volatility_quality=0.5,
            liquidity_state=narrative.liquidity_state if narrative else "NORMAL",
            liquidity_quality=0.5,
            trend_valid=True,
            microstructure_quality=0.5,
            active_hypotheses=active_hyps,
            now=now,
        )

        # ── Step 2: conviction (delegates entirely) ───────────────────
        conviction: ConvictionScore | None = None
        if active_hyps:
            dominant_hyp = max(active_hyps, key=lambda h: (h.confidence, h.reinforcement_count))
            conviction = compute_conviction(dominant_hyp, now=now)

        # ── Step 3: parse portfolio_state ─────────────────────────────
        existing_dir_str = ps.get("existing_direction")
        existing_dir: BeliefDirection | None = None
        if existing_dir_str:
            try:
                existing_dir = BeliefDirection(existing_dir_str.upper())
            except ValueError:
                pass

        inp = DecisionInput(
            instrument=instrument,
            narrative=narrative,
            active_hypotheses=active_hyps,
            debate_record=debate_record,
            uncertainty=uncertainty,
            conviction=conviction,
            portfolio_open_positions=int(ps.get("open_positions_count", 0)),
            portfolio_daily_pnl_pct=float(ps.get("daily_pnl_pct", 0.0)),
            has_existing_position=bool(ps.get("has_existing_position", False)),
            existing_position_direction=existing_dir,
            memory_win_rate=ps.get("memory_win_rate"),
            memory_recent_count=int(ps.get("memory_recent_count", 0)),
        )

        # ── Step 4: compute decision (pure function) ───────────────────
        decision = compute_decision(inp, calib=brain_state.calibration, now=now)

        # ── Step 5: store to memory ───────────────────────────────────
        self._bridge.store(decision, brain_state=brain_state, now=now)

        logger.debug(
            f"[DecisionAPI] {instrument}: {decision.action.value} "
            f"dir={decision.direction.value} "
            f"conv={decision.conviction_score:.2f} "
            f"unc={decision.uncertainty_score:.2f} "
            f"oc={decision.overconfidence_flagged}"
        )
        return decision

    def explain_decision(
        self,
        decision: CognitiveDecision,
        brain_state: "BrainState",
        debate_record: "DebateRecord | None" = None,
        now: float | None = None,
    ) -> DecisionReasoning:
        """Produce a structured chain-of-thought for ``decision``.

        Delegates entirely to ``ReasoningEngine.explain()`` — no logic
        is reimplemented here.

        Parameters
        ----------
        decision:
            The ``CognitiveDecision`` to explain.
        brain_state:
            The ``BrainState`` at the time the decision was made.
        debate_record:
            The C5 ``DebateRecord`` (if one was run this cycle).
        now:
            Override current time.

        Returns
        -------
        DecisionReasoning
            Structured chain-of-thought with per-dimension explanations.
        """
        return self._reasoning.explain(
            decision=decision,
            brain_state=brain_state,
            debate_record=debate_record,
            now=now,
        )

    def confidence_report(
        self,
        decision: CognitiveDecision,
        brain_state: "BrainState",
        now: float | None = None,
    ) -> ConfidenceReport:
        """Produce a full confidence breakdown for ``decision``.

        Reads calibration bias and conviction components from
        ``brain_state.calibration`` and the stored conviction score.
        Never re-computes conviction — uses what ``make_decision()``
        already computed (carried in ``decision.conviction_score`` /
        ``decision.conviction_profile``).

        Parameters
        ----------
        decision:
            The ``CognitiveDecision`` to report on.
        brain_state:
            The ``BrainState`` for calibration context.
        now:
            Override current time.

        Returns
        -------
        ConfidenceReport
            Full confidence accounting including calibration bias,
            historical accuracy delta, and alignment warning.
        """
        now = now if now is not None else time.time()
        calib = brain_state.calibration

        raw = decision.confidence
        bias = calib.confidence_bias
        calibrated = max(0.05, min(0.95, raw + bias))

        # Historical accuracy delta (requires recent episodic recall)
        recent_win_rate = self._estimate_win_rate(decision.instrument)
        accuracy_delta: float | None = None
        if recent_win_rate is not None:
            accuracy_delta = raw - recent_win_rate

        # Conviction component breakdown — read from conviction_engine output
        # stored in decision fields; avoid re-computing.
        conv_breakdown: dict = {
            "conviction_score": round(decision.conviction_score, 4),
            "conviction_profile": decision.conviction_profile,
            "uncertainty_score": round(decision.uncertainty_score, 4),
            "risk_signal": round(decision.risk_signal, 4),
        }

        alignment_warning = (
            decision.conviction_profile == ConvictionProfile.HIGH_CONFIDENCE_LOW_CONVICTION.value
        )

        return ConfidenceReport(
            decision_id=decision.decision_id,
            instrument=decision.instrument,
            raw_confidence=raw,
            calibration_bias=bias,
            calibrated_confidence=calibrated,
            conviction_score=decision.conviction_score,
            conviction_profile=decision.conviction_profile,
            calibration_sample_count=calib.sample_count,
            alignment_warning=alignment_warning,
            historical_win_rate=recent_win_rate,
            accuracy_delta=accuracy_delta,
            breakdown=conv_breakdown,
            generated_at=now,
        )

    # ── Convenience helpers ───────────────────────────────────────────

    def recent_decisions(
        self,
        instrument: str,
        n: int = 20,
        action: str | None = None,
    ) -> list[CognitiveDecision]:
        """Recent decisions from the local deque, newest first."""
        return self._bridge.recent_decisions(instrument, n=n, action=action)

    def action_distribution(self, instrument: str) -> dict[str, int]:
        """Count of decisions by action type for ``instrument``."""
        return self._bridge.action_distribution(instrument)

    def decision_count(self, instrument: str) -> int:
        return self._bridge.decision_count(instrument)

    # ── Persistence ───────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        return {"memory_bridge": self._bridge.to_snapshot()}

    def load_snapshot(self, data: dict) -> None:
        if "memory_bridge" in data:
            self._bridge.load_snapshot(data["memory_bridge"])

    # ── Sub-component access ──────────────────────────────────────────

    @property
    def memory_bridge(self) -> DecisionMemoryBridge:
        return self._bridge

    # ── Internal ──────────────────────────────────────────────────────

    def _estimate_win_rate(self, instrument: str) -> float | None:
        """Estimate recent win rate from local decision history.

        We use ACTIONABLE decisions that ended up being committed (BUY/SELL)
        as a rough proxy for historical confidence accuracy.  Returns None
        if fewer than 4 actionable decisions are recorded.
        """
        actionable = [d for d in self._bridge.recent_decisions(instrument, n=50) if d.is_actionable]
        if len(actionable) < 4:
            return None
        # "Won" = decision had lower-than-average uncertainty (proxy heuristic;
        # true outcome tracking lives in DecisionMemory / KnowledgeBase)
        avg_uncertainty = sum(d.uncertainty_score for d in actionable) / len(actionable)
        wins = sum(1 for d in actionable if d.uncertainty_score < avg_uncertainty)
        return wins / len(actionable)


__all__ = ["DecisionAPI"]
