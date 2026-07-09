"""Phase 27.7 — Inference traffic splitting (dual-dispatch shadow testing).

Every live `QDEInferenceInput` can be fanned out to both the active champion
decision model and a designated challenger simultaneously (`ShadowDispatcher`).
Neither dispatch places a trade: `ExecutionPlanner` only ever executes the
champion's decision. The challenger's decision is observed and compared via a
widened `ShadowDecisionRecord` (decision agreement, confidence/PnL/latency/risk
differences, and a single computed `promotion_score`).

That promotion score feeds `evaluate_and_maybe_promote`, which wires live
shadow-traffic results into automated promote/reject/extend decisions
*alongside* — not instead of — the existing backtest-based
`QDEPromotionEngine`. A PROMOTE recommendation from live traffic still has to
clear every backtest dimension check before the challenger is ever activated.

This module intentionally lives under `trainer.qde` (not `trainer.research_lab`)
so the dependency only goes one way: `trainer.research_lab` never needs to
import anything from `trainer.qde`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.qde_inference import QDEDecision, QDEInferenceInput
from trainer.qde.qde_promotion import QDEPromotionEngine
from trainer.qde.qde_registry import QDERegistry
from trainer.research_lab.rl_models import (
    ShadowDecisionRecord,
    ShadowTestRecommendation,
    ShadowTestRun,
    ShadowTestStatus,
)


def expected_pnl_proxy(decision: QDEDecision) -> float:
    """A deterministic, comparable proxy for a decision's expected outcome.

    The challenger's decision is never executed, so there is no realised PnL
    to diff against the champion at dispatch time. This proxy — position size
    times allocated risk times conviction — lets both sides be compared on
    equal footing using only fields `QDEDecisionModel.decide()` already
    produces. Once a decision's real trade outcome is known (always true for
    the champion, since only it is ever executed), callers should record that
    realised PnL directly instead of relying on this proxy — see
    `ShadowDecisionRecord.new(pnl_diff=...)`.
    """
    return decision.position_size_lots * decision.risk_allocation_pct * decision.confidence


@dataclass
class DualDispatchResult:
    """The outcome of sending one live inference input to both models."""

    champion_decision: QDEDecision
    challenger_decision: QDEDecision
    champion_latency_ms: float
    challenger_latency_ms: float
    record: ShadowDecisionRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion_decision": self.champion_decision.to_dict(),
            "challenger_decision": self.challenger_decision.to_dict(),
            "champion_latency_ms": round(self.champion_latency_ms, 3),
            "challenger_latency_ms": round(self.challenger_latency_ms, 3),
            "record": self.record.to_dict(),
        }


class ShadowDispatcher:
    """Fans a single live `QDEInferenceInput` out to champion + challenger.

    Usage::

        dispatcher = ShadowDispatcher(qde_registry)
        result = dispatcher.dispatch(test_id, challenger_version_id, inputs)
        research_lab_api.record_dual_dispatch_decision(result.record)
    """

    def __init__(self, registry: QDERegistry) -> None:
        self._registry = registry

    def dispatch(
        self,
        test_id: str,
        challenger_version_id: str,
        inputs: QDEInferenceInput,
        regime: str = "",
    ) -> DualDispatchResult:
        champion_model = self._registry.get_active_model()
        challenger_model = self._registry.get_model_for_version(challenger_version_id)

        t0 = time.perf_counter()
        champion_decision = champion_model.decide(inputs)
        champion_latency_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        challenger_decision = challenger_model.decide(inputs)
        challenger_latency_ms = (time.perf_counter() - t0) * 1000.0
        # Structural (not conventional) guarantee: mark the challenger side
        # as observation-only. `ExecutionPlanner.plan()` refuses to execute
        # any `QDEDecision` with `is_shadow=True`, so even if a caller
        # mistakenly wires a challenger decision into the execution path
        # downstream, it is rejected rather than silently traded.
        #
        # `dataclasses.replace` (rather than mutating `challenger_decision`
        # in place) so this never risks flipping the flag on a decision
        # instance a custom `QDEDecisionModel` implementation might reuse
        # or share elsewhere — the champion side of this same dispatch, in
        # particular, must never be affected by tagging the challenger.
        challenger_decision = replace(challenger_decision, is_shadow=True)

        pnl_diff = expected_pnl_proxy(challenger_decision) - expected_pnl_proxy(champion_decision)
        latency_diff_ms = challenger_latency_ms - champion_latency_ms
        risk_diff = challenger_decision.risk_allocation_pct - champion_decision.risk_allocation_pct

        record = ShadowDecisionRecord.new(
            test_id=test_id,
            instrument=inputs.forecast.instrument,
            champion_decision_type=champion_decision.entry_action.value,
            challenger_decision_type=challenger_decision.entry_action.value,
            champion_confidence=champion_decision.confidence,
            challenger_confidence=challenger_decision.confidence,
            regime=regime,
            market_context=inputs.market.to_dict(),
            pnl_diff=pnl_diff,
            latency_diff_ms=latency_diff_ms,
            risk_diff=risk_diff,
        )

        logger.debug(
            f"ShadowDispatcher.dispatch: {inputs.forecast.instrument} "
            f"agree={record.decisions_agree} score={record.promotion_score}"
        )

        return DualDispatchResult(
            champion_decision=champion_decision,
            challenger_decision=challenger_decision,
            champion_latency_ms=champion_latency_ms,
            challenger_latency_ms=challenger_latency_ms,
            record=record,
        )


def evaluate_and_maybe_promote(
    shadow_test: ShadowTestRun,
    promotion_engine: QDEPromotionEngine,
    notes: str = "",
) -> dict[str, Any]:
    """Feed a concluded shadow test's live-traffic recommendation into the
    backtest-based `QDEPromotionEngine`, without replacing its own gates.

    - The shadow test must be CONCLUDED (see `ShadowTestRun.conclude()`,
      which computes `avg_promotion_score` and `recommendation` from live
      traffic) — this function does not conclude it.
    - PROMOTE  → calls `promotion_engine.auto_evaluate_and_promote`. The
      challenger is only actually activated if it *also* clears every
      backtest dimension check; the live signal triggers evaluation, it does
      not bypass it.
    - REJECT / EXTEND → no promotion action; returned as-is so callers (e.g.
      `QDELifecycleAPI`) can log or notify.
    """
    if shadow_test.status != ShadowTestStatus.CONCLUDED:
        return {
            "action": "none",
            "reason": f"shadow test is {shadow_test.status.value}, not CONCLUDED",
            "shadow_recommendation": shadow_test.recommendation.value,
            "avg_promotion_score": shadow_test.avg_promotion_score,
        }

    combined_notes = (
        f"Shadow test {shadow_test.test_id}: avg_promotion_score="
        f"{shadow_test.avg_promotion_score}, agreement_rate={shadow_test.agreement_rate}. {notes}"
    ).strip()

    if shadow_test.recommendation == ShadowTestRecommendation.PROMOTE:
        decision = promotion_engine.auto_evaluate_and_promote(
            shadow_test.challenger_version, notes=combined_notes
        )
        return {
            "action": "promoted" if decision.is_promoted else "not_promoted",
            "reason": (
                "backtest gates also passed"
                if decision.is_promoted
                else "live traffic recommended PROMOTE but backtest gates did not clear"
            ),
            "shadow_recommendation": shadow_test.recommendation.value,
            "avg_promotion_score": shadow_test.avg_promotion_score,
            "promotion_decision": decision.to_dict(),
        }

    if shadow_test.recommendation == ShadowTestRecommendation.REJECT:
        logger.info(
            f"evaluate_and_maybe_promote: {shadow_test.challenger_version} rejected by "
            f"live shadow traffic (score={shadow_test.avg_promotion_score})"
        )
        return {
            "action": "rejected",
            "reason": "live shadow traffic recommended REJECT",
            "shadow_recommendation": shadow_test.recommendation.value,
            "avg_promotion_score": shadow_test.avg_promotion_score,
        }

    return {
        "action": "extended",
        "reason": "live shadow traffic inconclusive or below promotion thresholds",
        "shadow_recommendation": shadow_test.recommendation.value,
        "avg_promotion_score": shadow_test.avg_promotion_score,
    }
