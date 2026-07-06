"""Phase 21.6 — Self-Evaluation Engine.

Generates periodic institutional AI self-assessments:
- Strengths
- Weaknesses
- Common Mistakes
- Recurring Failure Modes
- Best/Worst Performing Conditions
- Improvement Opportunities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_models import DecisionRecord
from trainer.memory.decision_types import (
    NEGATIVE_OUTCOMES,
    POSITIVE_OUTCOMES,
    DecisionGrade,
    OutcomeType,
)


@dataclass
class SelfAssessment:
    """Periodic AI self-assessment of decision quality."""

    period: str
    total_decisions: int = 0
    decisions_with_outcome: int = 0
    overall_win_rate: float = 0.0
    overall_quality_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    best_conditions: List[str] = field(default_factory=list)
    worst_conditions: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    grade: str = "ACCEPTABLE"
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "period": self.period,
            "total_decisions": self.total_decisions,
            "decisions_with_outcome": self.decisions_with_outcome,
            "overall_win_rate": round(self.overall_win_rate, 4),
            "overall_quality_score": round(self.overall_quality_score, 4),
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "common_mistakes": self.common_mistakes,
            "failure_modes": self.failure_modes,
            "best_conditions": self.best_conditions,
            "worst_conditions": self.worst_conditions,
            "opportunities": self.opportunities,
            "grade": self.grade,
            "summary": self.summary,
        }


class SelfEvaluationEngine:
    """Generates periodic institutional self-assessments from decision history."""

    def evaluate(
        self,
        records: List[DecisionRecord],
        period: str = "CURRENT",
    ) -> SelfAssessment:

        n = len(records)
        assessment = SelfAssessment(period=period, total_decisions=n)

        completed = [r for r in records if r.has_outcome]
        assessment.decisions_with_outcome = len(completed)

        if not completed:
            assessment.summary = "No completed decisions — self-evaluation not available."
            return assessment

        # Overall metrics
        wins = sum(1 for r in completed if r.outcome.outcome_type in POSITIVE_OUTCOMES)
        assessment.overall_win_rate = round(wins / len(completed), 4) if completed else 0.0

        scored = [r for r in records if r.has_score]
        if scored:
            assessment.overall_quality_score = round(
                sum(r.score.overall for r in scored) / len(scored), 4
            )

        # ── Strengths ─────────────────────────────────────────────────────────
        strengths = []
        if assessment.overall_win_rate > 0.60:
            strengths.append(f"Win rate of {assessment.overall_win_rate:.0%} exceeds 60% threshold")
        tp_rate = sum(1 for r in completed if r.outcome.hit_tp) / len(completed)
        if tp_rate > 0.50:
            strengths.append(f"TP hit rate of {tp_rate:.0%} — good target selection")

        # High-confidence decisions performing well
        hi_conf = [r for r in completed if r.snapshot.forecast_confidence >= 0.75]
        if hi_conf:
            hi_wins = sum(1 for r in hi_conf if r.outcome.outcome_type in POSITIVE_OUTCOMES)
            if hi_wins / len(hi_conf) > 0.65:
                strengths.append(
                    f"High-confidence decisions win at {hi_wins/len(hi_conf):.0%} — "
                    f"confidence is well-calibrated"
                )

        # Best session
        sess_wins = self._group_metric(completed, lambda r: r.snapshot.session)
        best_sess = max(sess_wins, key=lambda s: sess_wins[s]["win_rate"], default=None)
        if best_sess and sess_wins[best_sess]["count"] >= 3:
            wr = sess_wins[best_sess]["win_rate"]
            if wr > 0.60:
                strengths.append(f"{best_sess} session: {wr:.0%} win rate — strongest session")

        assessment.strengths = strengths if strengths else ["No clear strengths identified yet"]

        # ── Weaknesses ────────────────────────────────────────────────────────
        weaknesses = []
        if assessment.overall_win_rate < 0.45:
            weaknesses.append(
                f"Overall win rate {assessment.overall_win_rate:.0%} is below 45% minimum threshold"
            )
        sl_rate = sum(1 for r in completed if r.outcome.hit_sl) / len(completed)
        if sl_rate > 0.40:
            weaknesses.append(f"Stop loss hit rate {sl_rate:.0%} is high — review stop placement")

        # Worst session
        worst_sess = min(sess_wins, key=lambda s: sess_wins[s]["win_rate"], default=None)
        if worst_sess and sess_wins[worst_sess]["count"] >= 3:
            wr = sess_wins[worst_sess]["win_rate"]
            if wr < 0.40:
                weaknesses.append(
                    f"{worst_sess} session: only {wr:.0%} win rate — consistently weak"
                )

        # Overconfidence check
        overconf = [
            r
            for r in completed
            if r.snapshot.forecast_confidence > 0.80 and r.outcome.outcome_type in NEGATIVE_OUTCOMES
        ]
        if len(overconf) / len(completed) > 0.20:
            weaknesses.append(
                f"{len(overconf)/len(completed):.0%} of high-confidence decisions are losers — overconfidence risk"
            )

        assessment.weaknesses = (
            weaknesses if weaknesses else ["No significant weaknesses identified"]
        )

        # ── Common Mistakes ───────────────────────────────────────────────────
        mistakes = []
        # Check: low-confidence entries
        lo_conf_losses = [
            r
            for r in completed
            if r.snapshot.forecast_confidence < 0.55 and r.outcome.outcome_type in NEGATIVE_OUTCOMES
        ]
        if len(lo_conf_losses) >= 2:
            mistakes.append(
                f"Taking {len(lo_conf_losses)} low-confidence entries that resulted in losses"
            )

        # Check: incorrect direction
        wrong_dir = [r for r in completed if r.outcome.actual_direction_correct is False]
        if wrong_dir:
            mistakes.append(f"{len(wrong_dir)} decisions with incorrect directional forecast")

        assessment.common_mistakes = mistakes if mistakes else ["No systematic mistakes detected"]

        # ── Failure Modes ─────────────────────────────────────────────────────
        failure_modes = []
        missed = sum(
            1 for r in completed if r.outcome.outcome_type == OutcomeType.MISSED_OPPORTUNITY
        )
        if missed >= 2:
            failure_modes.append(
                f"{missed} missed opportunities — review entry criteria stringency"
            )
        unexpected = sum(
            1 for r in completed if r.outcome.outcome_type == OutcomeType.UNEXPECTED_MARKET_EVENT
        )
        if unexpected >= 2:
            failure_modes.append(
                f"{unexpected} decisions disrupted by unexpected market events — news risk management needed"
            )

        assessment.failure_modes = (
            failure_modes if failure_modes else ["No recurring failure modes identified"]
        )

        # ── Best Conditions ───────────────────────────────────────────────────
        reg_wins = self._group_metric(completed, lambda r: r.snapshot.market_regime)
        best_reg = max(reg_wins, key=lambda r: reg_wins[r]["win_rate"], default=None)
        if best_reg:
            assessment.best_conditions.append(
                f"{best_reg} regime: {reg_wins[best_reg]['win_rate']:.0%} win rate"
            )
        if best_sess:
            assessment.best_conditions.append(
                f"{best_sess} session: {sess_wins[best_sess]['win_rate']:.0%} win rate"
            )
        if not assessment.best_conditions:
            assessment.best_conditions = ["Insufficient data to identify best conditions"]

        # ── Worst Conditions ──────────────────────────────────────────────────
        worst_reg = min(reg_wins, key=lambda r: reg_wins[r]["win_rate"], default=None)
        if worst_reg and reg_wins[worst_reg]["count"] >= 3:
            assessment.worst_conditions.append(
                f"{worst_reg} regime: only {reg_wins[worst_reg]['win_rate']:.0%} win rate"
            )
        if worst_sess and sess_wins[worst_sess]["count"] >= 3:
            assessment.worst_conditions.append(
                f"{worst_sess} session: only {sess_wins[worst_sess]['win_rate']:.0%} win rate"
            )
        if not assessment.worst_conditions:
            assessment.worst_conditions = ["Insufficient data to identify worst conditions"]

        # ── Improvement Opportunities ─────────────────────────────────────────
        opps = []
        if sl_rate > 0.35:
            opps.append("Reduce stop loss hit rate by widening stops by 0.5× ATR")
        if assessment.overall_win_rate < 0.55:
            opps.append("Increase minimum confidence threshold from 0.55 to 0.65 before entry")
        if len(completed) < 20:
            opps.append("Increase sample size — more decisions needed for reliable patterns")
        opps.append("Review counterfactual analysis for systematic improvements to TP sizing")
        assessment.opportunities = opps

        # ── Grade ─────────────────────────────────────────────────────────────
        if assessment.overall_quality_score >= 0.80:
            assessment.grade = DecisionGrade.EXCELLENT.value
        elif assessment.overall_quality_score >= 0.65:
            assessment.grade = DecisionGrade.GOOD.value
        elif assessment.overall_quality_score >= 0.50:
            assessment.grade = DecisionGrade.ACCEPTABLE.value
        elif assessment.overall_quality_score >= 0.35:
            assessment.grade = DecisionGrade.POOR.value
        else:
            assessment.grade = DecisionGrade.REJECT.value

        assessment.summary = (
            f"Self-evaluation [{period}]: {n} decisions, "
            f"{len(completed)} completed, "
            f"win_rate={assessment.overall_win_rate:.0%}, "
            f"quality={assessment.overall_quality_score:.2f}, "
            f"grade={assessment.grade}."
        )

        logger.debug(
            f"SelfEvaluationEngine [{period}]: "
            f"grade={assessment.grade} win_rate={assessment.overall_win_rate:.0%}"
        )
        return assessment

    @staticmethod
    def _group_metric(records, key_fn) -> Dict[str, Dict]:
        groups: Dict[str, List] = {}
        for r in records:
            k = key_fn(r)
            groups.setdefault(k, [])
            groups[k].append(r)
        result = {}
        for k, group in groups.items():
            wins = sum(1 for r in group if r.outcome.outcome_type in POSITIVE_OUTCOMES)
            result[k] = {
                "count": len(group),
                "win_rate": round(wins / len(group), 4) if group else 0.0,
            }
        return result
