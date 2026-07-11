"""QDE Brain — decision subpackage (Phase C6).

Conscious decision layer: decisions emerge from cognition.

Five first-class actions
-------------------------
  BUY     — debate is conclusive BULLISH, conviction is above threshold,
             uncertainty is acceptable (LOW or MODERATE).
  SELL    — debate is conclusive BEARISH, conviction is above threshold,
             uncertainty is acceptable.
  WAIT    — conditions understood but unfavourable (HIGH uncertainty,
             risk limit reached, overconfidence flagged on non-conclusive
             debate).
  MONITOR — a developing setup exists; debate is ongoing or conviction
             is building toward threshold.  No commitment yet.
  IGNORE  — market is incomprehensible (EXTREME uncertainty) or no
             cognitive basis exists for engaging this instrument.

WAIT and MONITOR are first-class — they carry full signal heads and
are stored in memory just like BUY and SELL.

What was already there (untouched)
-------------------------------------
  uncertainty_engine.py   ``compute_uncertainty()`` / ``UncertaintyScore``
  conviction_engine.py    ``compute_conviction()`` / ``ConvictionScore``
  brain_models.py         ``CalibrationProfile.nudge()`` / ``BrainState``
  debate/ (C5)            ``DebateRecord`` / ``DebateAPI``
  episodic/ (C2)          ``EpisodicMemory``

What this package adds (Phase C6)
-------------------------------------
  decision_types.py      ``CognitiveAction``, ``DecisionInput``,
                         ``CognitiveDecision``, ``DecisionReasoning``,
                         ``ConfidenceReport``

  decision_engine.py     Pure decision logic: classify action, compute
                         signal heads (risk_signal, entry_urgency,
                         hold_horizon, size_signal), select dominant
                         hypothesis.

  decision_reasoning.py  ``ReasoningEngine`` — structured chain-of-thought
                         from BrainState + CognitiveDecision + DebateRecord.

  decision_memory.py     ``DecisionMemoryBridge`` — soft C2 episodic
                         storage + local deque + working-memory log.

  decision_api.py        ``DecisionAPI`` — three spec methods:
                           make_decision(instrument, brain_state, ...)
                           explain_decision(decision, brain_state, ...)
                           confidence_report(decision, brain_state, ...)

Integration example
-------------------
::

    # At initialization:
    api = DecisionAPI(
        memory_bridge=DecisionMemoryBridge(episodic_memory=emem)
    )

    # On each tick — after brain update + optional debate:
    decision = api.make_decision(
        "EURUSD", brain_state, debate_record=debate_record,
        portfolio_state={"open_positions_count": 2, "daily_pnl_pct": -0.01},
    )

    if decision.action == CognitiveAction.BUY:
        # Hand off to inference layer with the brain's cognitive context
        pass
    elif decision.action == CognitiveAction.MONITOR:
        # Continue tracking — no execution
        pass

    # Explain any decision for dashboards / reflection:
    reasoning = api.explain_decision(decision, brain_state, debate_record)
    print(reasoning.full_text())

    # Confidence breakdown:
    report = api.confidence_report(decision, brain_state)
    if report.alignment_warning:
        print("Warning: high confidence in poorly-formed setup")
"""

from trainer.qde.brain.decision.decision_api import DecisionAPI
from trainer.qde.brain.decision.decision_engine import (
    CONVICTION_ACTION_THRESHOLD,
    CONVICTION_MONITOR_THRESHOLD,
    compute_decision,
)
from trainer.qde.brain.decision.decision_memory import DecisionMemoryBridge
from trainer.qde.brain.decision.decision_reasoning import ReasoningEngine
from trainer.qde.brain.decision.decision_types import (
    CognitiveAction,
    CognitiveDecision,
    ConfidenceReport,
    DecisionInput,
    DecisionReasoning,
)

__all__ = [
    # ── Primary API (Phase C6) ─────────────────────────────────────────
    "DecisionAPI",
    # ── Action vocabulary ─────────────────────────────────────────────
    "CognitiveAction",
    # ── Data types ────────────────────────────────────────────────────
    "DecisionInput",
    "CognitiveDecision",
    "DecisionReasoning",
    "ConfidenceReport",
    # ── Engine + memory ───────────────────────────────────────────────
    "compute_decision",
    "ReasoningEngine",
    "DecisionMemoryBridge",
    # ── Constants ─────────────────────────────────────────────────────
    "CONVICTION_ACTION_THRESHOLD",
    "CONVICTION_MONITOR_THRESHOLD",
]
