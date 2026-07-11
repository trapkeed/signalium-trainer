"""QDE Brain — decision reasoning engine (Phase C6).

``ReasoningEngine`` produces the full structured chain-of-thought for a
``CognitiveDecision`` — the detailed ``DecisionReasoning`` that explains
*why* the brain chose its action, step by step.

This is distinct from the one-line ``reasoning`` field on
``CognitiveDecision`` (which is a quick summary).  ``DecisionReasoning``
is the deep explanation, meant for:

  * Dashboard display ("why is QDE recommending WAIT?")
  * Reflection engine input ("what reasoning chain preceded this outcome?")
  * Overconfidence/ignored-evidence audit trails

Design constraints
------------------
* Pure function (no state).  ``ReasoningEngine`` is stateless.
* Reads from existing ``CognitiveDecision`` + ``BrainState`` + optional
  ``DebateRecord``.  Never re-derives conviction or uncertainty.
* No ``QDEDecision`` imports.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from trainer.qde.brain.conviction_engine import ConvictionProfile
from trainer.qde.brain.decision.decision_types import (
    CognitiveAction,
    CognitiveDecision,
    DecisionReasoning,
)
from trainer.qde.brain.uncertainty_engine import UncertaintyLevel

if TYPE_CHECKING:
    from trainer.qde.brain.brain_models import BrainState
    from trainer.qde.brain.debate.debate_store import DebateRecord


class ReasoningEngine:
    """Stateless: all methods are pure over their inputs."""

    def explain(
        self,
        decision:     CognitiveDecision,
        brain_state:  "BrainState",
        debate_record: "DebateRecord | None" = None,
        now:          float | None           = None,
    ) -> DecisionReasoning:
        """Build a full ``DecisionReasoning`` for a completed decision.

        Parameters
        ----------
        decision:
            The ``CognitiveDecision`` being explained.
        brain_state:
            The ``BrainState`` at the time the decision was made.
        debate_record:
            The C5 ``DebateRecord`` (if one was run this cycle).
        now:
            Override current time.
        """
        now = now if now is not None else time.time()

        chain          = self._build_chain(decision, brain_state, debate_record)
        hyp_basis      = self._hypothesis_basis(decision, brain_state)
        debate_summary = self._debate_summary(debate_record)
        unc_basis      = self._uncertainty_basis(decision, brain_state)
        mem_influence  = self._memory_influence(decision, brain_state)
        portfolio_con  = self._portfolio_constraint(decision, brain_state)
        flags          = self._collect_flags(decision)

        return DecisionReasoning(
            decision_id=decision.decision_id,
            instrument=decision.instrument,
            action=decision.action.value,
            chain=chain,
            hypothesis_basis=hyp_basis,
            debate_summary=debate_summary,
            uncertainty_basis=unc_basis,
            memory_influence=mem_influence,
            portfolio_constraint=portfolio_con,
            flags=flags,
            generated_at=now,
        )

    # ── Private builders ──────────────────────────────────────────────

    def _build_chain(
        self,
        decision:      CognitiveDecision,
        brain_state:   "BrainState",
        debate_record: "DebateRecord | None",
    ) -> list[str]:
        """Build an ordered list of reasoning steps."""
        steps: list[str] = []
        action = decision.action

        # Step 1: market ambiguity
        unc_pct = f"{decision.uncertainty_score:.0%}"
        unc_lvl = decision.uncertainty_level
        steps.append(
            f"Market uncertainty is {unc_lvl.lower()} ({unc_pct})."
        )

        # Step 2: hypothesis context
        active = brain_state.active_hypotheses()
        if active:
            top = max(active, key=lambda h: h.confidence)
            steps.append(
                f"{len(active)} active hypothesis/es; dominant: "
                f'"{top.thesis[:60]}" '
                f"(confidence={top.confidence:.0%}, "
                f"reinforcements={top.reinforcement_count})."
            )
        else:
            steps.append("No active hypotheses — cognitive basis is absent.")

        # Step 3: narrative context
        narrative = brain_state.narrative
        if narrative:
            steps.append(
                f"Market narrative: {narrative.tone.value.lower()} tone; "
                f"regime={narrative.market_regime}; "
                f"session={narrative.session_context or 'unknown'}."
            )

        # Step 4: conviction reading
        conv_pct     = f"{decision.conviction_score:.0%}"
        conv_profile = decision.conviction_profile.replace("_", " ").lower()
        steps.append(
            f"Conviction is {conv_pct} ({conv_profile})."
        )

        # Step 5: calibrated confidence
        if abs(decision.calibrated_confidence - decision.confidence) > 0.01:
            steps.append(
                f"Confidence calibrated from {decision.confidence:.0%} → "
                f"{decision.calibrated_confidence:.0%} "
                f"(historical bias correction)."
            )
        else:
            steps.append(
                f"Confidence is {decision.confidence:.0%} "
                f"(no material calibration bias)."
            )

        # Step 6: debate
        if debate_record is not None:
            v = debate_record.verdict
            steps.append(
                f"Debate: bull strength={v.get('bull_strength',0):.0%} "
                f"vs bear={v.get('bear_strength',0):.0%}; "
                f"margin={v.get('margin',0):.0%}; "
                f"conclusive={debate_record.conclusive}; "
                f"quality={debate_record.quality_grade}."
            )
        else:
            steps.append("No debate record available this cycle.")

        # Step 7: action rationale
        if action == CognitiveAction.IGNORE:
            steps.append(
                "Decision: IGNORE — uncertainty is extreme or no cognitive "
                "basis exists for this instrument."
            )
        elif action == CognitiveAction.WAIT:
            steps.append(
                "Decision: WAIT — conditions understood but unfavourable "
                "(high uncertainty, risk limit, or overconfidence flagged)."
            )
        elif action == CognitiveAction.MONITOR:
            steps.append(
                f"Decision: MONITOR — setup is developing but debate is not "
                f"yet conclusive or conviction ({conv_pct}) is below action threshold."
            )
        else:
            steps.append(
                f"Decision: {action.value} — debate conclusive with "
                f"{decision.direction.value.lower()} dominance; "
                f"risk={decision.risk_signal:.0%}, "
                f"urgency={decision.entry_urgency:.0%}, "
                f"size={decision.size_signal:.0%}."
            )

        return steps

    def _hypothesis_basis(
        self,
        decision:    CognitiveDecision,
        brain_state: "BrainState",
    ) -> str:
        hyp_id = decision.primary_hypothesis_id
        if hyp_id is None:
            return "No hypothesis drove this decision."
        hyp = brain_state.hypotheses.get(hyp_id)
        if hyp is None:
            return f"Primary hypothesis {hyp_id} (no longer in state)."
        return (
            f'Hypothesis: "{hyp.thesis}" — '
            f"{hyp.direction.value.lower()}, "
            f"confidence={hyp.confidence:.0%}, "
            f"status={hyp.status.value}, "
            f"reinforcements={hyp.reinforcement_count}, "
            f"supporting={len(hyp.supporting_evidence)}, "
            f"contradicting={len(hyp.contradicting_evidence)}."
        )

    def _debate_summary(self, debate_record: "DebateRecord | None") -> str:
        if debate_record is None:
            return "No debate ran this cycle."
        q = debate_record.quality
        v = debate_record.verdict
        parts = [
            f"Quality={q.get('grade','?')} (overall={q.get('overall_score',0):.0%}); "
            f"margin={v.get('margin',0):.0%}; "
            f"conclusive={debate_record.conclusive}."
        ]
        if debate_record.has_overconfidence:
            oe = debate_record.overconfidence_event or {}
            parts.append(
                f"Overconfidence event: trigger={oe.get('trigger','?')} "
                f"(conviction={oe.get('conviction_at_event',0):.0%})."
            )
        if debate_record.has_ignored_evidence:
            ie = debate_record.ignored_evidence or {}
            signals = ie.get("ignored_signals", [])[:3]
            parts.append(
                f"Ignored evidence ({len(signals)} signals): "
                + ", ".join(f'"{s}"' for s in signals)
                + ("…" if len(ie.get("ignored_signals", [])) > 3 else ".")
            )
        return " ".join(parts)

    def _uncertainty_basis(
        self,
        decision:    CognitiveDecision,
        brain_state: "BrainState",
    ) -> str:
        # The full explanation is already computed by compute_uncertainty()
        # and sits in the UncertaintyScore that was used to make the decision.
        # We reconstruct a brief summary from the stored level + score.
        unc_pct = f"{decision.uncertainty_score:.0%}"
        unc_lvl = decision.uncertainty_level
        return (
            f"Market uncertainty is {unc_lvl.lower()} ({unc_pct}). "
            f"See UncertaintyScore for full signal breakdown."
        )

    def _memory_influence(
        self,
        decision:    CognitiveDecision,
        brain_state: "BrainState",
    ) -> str:
        # LongTermMemory entries carry accumulated outcome quality
        ltm = brain_state.long_term_memory
        if not ltm:
            return "No long-term memory entries — decision made without historical context."
        recent = ltm[-5:]  # last 5 entries
        avg_quality = sum(e.get("outcome_quality", 0.0) if isinstance(e, dict)
                          else getattr(e, "outcome_quality", 0.0)
                          for e in recent) / len(recent)
        return (
            f"{len(ltm)} long-term memory entries; "
            f"recent average outcome quality={avg_quality:+.2f} "
            f"(-1=bad, +1=good)."
        )

    def _portfolio_constraint(
        self,
        decision:    CognitiveDecision,
        brain_state: "BrainState",
    ) -> str:
        parts = []
        if decision.risk_signal < 0.5:
            parts.append(
                f"Risk signal is reduced ({decision.risk_signal:.0%}) "
                "due to portfolio constraints or uncertainty."
            )
        if decision.size_signal < decision.conviction_score * 0.5:
            parts.append(
                "Size signal is below conviction level — portfolio concentration "
                "or existing-position penalty applied."
            )
        if not parts:
            return "No material portfolio constraints active."
        return " ".join(parts)

    def _collect_flags(self, decision: CognitiveDecision) -> list[str]:
        flags = []
        if decision.overconfidence_flagged:
            flags.append("OVERCONFIDENCE_DETECTED")
        if decision.ignored_evidence_flagged:
            flags.append("IGNORED_EVIDENCE_DETECTED")
        if decision.conviction_profile == ConvictionProfile.HIGH_CONFIDENCE_LOW_CONVICTION.value:
            flags.append("HIGH_CONFIDENCE_LOW_CONVICTION")
        if decision.uncertainty_level == UncertaintyLevel.HIGH.value:
            flags.append("HIGH_UNCERTAINTY")
        if decision.uncertainty_level == UncertaintyLevel.EXTREME.value:
            flags.append("EXTREME_UNCERTAINTY")
        return flags


__all__ = ["ReasoningEngine"]
