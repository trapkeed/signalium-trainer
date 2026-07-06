"""Phase 21.6 — Decision Quality Engine.

Calculates a Decision Quality Score across 10 factors.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_types import (
    QualityFactor,
    QUALITY_WEIGHTS,
    grade_from_score,
    POSITIVE_OUTCOMES,
    NEGATIVE_OUTCOMES,
    OutcomeType,
)
from trainer.memory.decision_models import DecisionRecord, DecisionScore


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class QualityEngine:
    """Scores decisions across 10 quality factors and computes an overall grade."""

    def score(self, record: DecisionRecord) -> DecisionScore:
        sub: Dict[str, float] = {}
        notes: List[str] = []

        snap = record.snapshot
        out = record.outcome
        rea = record.reasoning

        # 1. Forecast Accuracy
        if out is not None:
            if out.outcome_type in POSITIVE_OUTCOMES:
                fa = 0.85 + 0.15 * (1.0 if out.hit_tp else 0.5)
            elif out.outcome_type in NEGATIVE_OUTCOMES:
                fa = 0.15 + 0.10 * (1.0 - snap.forecast_confidence)
            else:
                fa = 0.50
        else:
            fa = snap.tp_probability  # no outcome yet — use forecast probability
        sub[QualityFactor.FORECAST_ACCURACY.value] = _clamp(fa)

        # 2. Execution Quality — proxy from spread vs ATR
        if snap.atr > 0:
            spread_ratio = snap.spread / snap.atr
            eq = _clamp(1.0 - spread_ratio * 2.0)
        else:
            eq = 0.70
        eq *= snap.liquidity_estimate
        sub[QualityFactor.EXECUTION_QUALITY.value] = _clamp(eq)

        # 3. Risk Management — based on SL probability and risk metrics
        rm = _clamp(1.0 - snap.sl_probability * 1.5)
        if out and out.hit_sl:
            rm = min(rm, 0.40)
            notes.append("Stop loss hit — risk management score penalised")
        sub[QualityFactor.RISK_MANAGEMENT.value] = _clamp(rm)

        # 4. Lifecycle Management — based on holding time vs expected
        if out and out.holding_time_h > 0 and snap.expected_holding_h > 0:
            ratio = out.holding_time_h / snap.expected_holding_h
            # Penalise if held 3× too long or exited 3× too early
            lm = _clamp(1.0 - abs(1.0 - ratio) * 0.3)
        else:
            lm = 0.65
        sub[QualityFactor.LIFECYCLE_MANAGEMENT.value] = _clamp(lm)

        # 5. Portfolio Impact — based on exposure and correlation risk
        pi = _clamp(1.0 - snap.portfolio_exposure - snap.correlation_risk * 0.5)
        sub[QualityFactor.PORTFOLIO_IMPACT.value] = _clamp(pi)

        # 6. Calibration — P(TP) vs actual TP hit
        if out is not None:
            actual_tp = 1.0 if out.hit_tp else 0.0
            cal_err = abs(snap.tp_probability - actual_tp)
            cal = _clamp(1.0 - cal_err)
        else:
            cal = snap.tp_probability  # default to forecast confidence
        sub[QualityFactor.CALIBRATION.value] = _clamp(cal)

        # 7. Confidence Reliability — was confidence warranted?
        conf = snap.forecast_confidence
        if out is not None:
            if out.outcome_type in POSITIVE_OUTCOMES and conf > 0.65:
                cr = min(1.0, conf + 0.10)
            elif out.outcome_type in NEGATIVE_OUTCOMES and conf > 0.75:
                cr = max(0.0, 1.0 - conf)  # overconfident in a loser
                notes.append("High confidence on a losing decision")
            else:
                cr = conf
        else:
            cr = conf
        sub[QualityFactor.CONFIDENCE_RELIABILITY.value] = _clamp(cr)

        # 8. Expected Value — TP prob × expected move vs SL prob × expected risk
        if snap.expected_move > 0:
            ev_ratio = (snap.tp_probability * snap.expected_move) / max(
                snap.sl_probability * snap.expected_move * 0.5, 1e-9
            )
            ev = _clamp(ev_ratio / 3.0)  # normalise: 3× RR = 1.0
        else:
            ev = snap.tp_probability
        sub[QualityFactor.EXPECTED_VALUE.value] = _clamp(ev)

        # 9. Consistency — reasoning completeness
        reason_items = (
            len(rea.primary_reasons)
            + len(rea.supporting_factors)
            + len(rea.contradicting_factors)
            + (1 if rea.final_rationale else 0)
        )
        con = _clamp(min(reason_items / 8.0, 1.0))
        sub[QualityFactor.CONSISTENCY.value] = _clamp(con)

        # 10. Decision Robustness — alternatives considered + alternatives count
        alt_count = len(rea.alternatives_considered)
        rob = _clamp(min(alt_count / 3.0, 1.0) * 0.70 + 0.30)
        sub[QualityFactor.DECISION_ROBUSTNESS.value] = _clamp(rob)

        # Weighted overall
        overall = sum(QUALITY_WEIGHTS[QualityFactor(k)] * v for k, v in sub.items())
        overall = _clamp(overall)
        grade = grade_from_score(overall)

        logger.debug(
            f"QualityEngine [{record.decision_id[:8]}]: "
            f"overall={overall:.3f} grade={grade.value}"
        )

        return DecisionScore(
            overall=round(overall, 4),
            grade=grade,
            sub_scores={k: round(v, 4) for k, v in sub.items()},
            notes=notes,
        )

    def score_and_attach(self, record: DecisionRecord) -> DecisionScore:
        """Score and attach result to record in-place."""
        ds = self.score(record)
        record.score = ds
        return ds

    def batch_score(self, records: List[DecisionRecord]) -> List[DecisionScore]:
        scores = []
        for r in records:
            s = self.score_and_attach(r)
            scores.append(s)
        return scores

    def quality_weights(self) -> Dict[str, float]:
        return {k.value: v for k, v in QUALITY_WEIGHTS.items()}
