"""Phase 21.5 — Lifecycle Optimizer.

Analyses historical lifecycle paths to generate optimisation
recommendations:
- Does earlier breakeven improve expectancy?
- Does delayed trailing improve profit?
- Does scaling improve Sharpe-like ratio?
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_models import StateRecord
from trainer.lifecycle.lifecycle_states import LifecycleState


@dataclass
class OptimisationFinding:
    """One optimisation finding from lifecycle path analysis."""

    finding_id: str
    category: str  # BREAKEVEN | TRAILING | SCALING | TIMING | RISK
    recommendation: str
    evidence: str
    expected_impact: str  # POSITIVE | NEGATIVE | NEUTRAL
    confidence: float  # 0-1
    priority: int  # 1 = highest

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "category": self.category,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "expected_impact": self.expected_impact,
            "confidence": round(self.confidence, 4),
            "priority": self.priority,
        }


@dataclass
class OptimisationReport:
    """Complete optimisation report."""

    total_analysed: int
    findings: list[OptimisationFinding] = field(default_factory=list)
    summary: str = ""
    key_insight: str = ""
    top_priority: str | None = None

    def to_dict(self) -> dict:
        return {
            "total_analysed": self.total_analysed,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "key_insight": self.key_insight,
            "top_priority": self.top_priority,
        }


class LifecycleOptimizer:
    """Generates optimisation recommendations from lifecycle patterns."""

    def optimise(
        self,
        histories: list[list[StateRecord]],
        final_states: list[LifecycleState] | None = None,
    ) -> OptimisationReport:
        """Analyse lifecycle paths and generate optimisation recommendations."""

        if not histories:
            return OptimisationReport(total_analysed=0, summary="No data")

        n = len(histories)
        findings: list[OptimisationFinding] = []
        fid = 1

        # ── 1. Breakeven timing analysis ──────────────────────────────────────
        protected_count = sum(
            1 for h in histories if any(r.state == LifecycleState.PROTECTED for r in h)
        )
        stopped_from_active = sum(
            1
            for h in histories
            if any(r.state == LifecycleState.STOPPED for r in h)
            and not any(r.state == LifecycleState.PROTECTED for r in h)
        )

        if n > 0:
            prot_rate = protected_count / n
            stop_no_prot = stopped_from_active / n

            if prot_rate < 0.40 and stop_no_prot > 0.20:
                findings.append(
                    OptimisationFinding(
                        finding_id=f"OPT-{fid:03d}",
                        category="BREAKEVEN",
                        recommendation="Move stop to breakeven earlier — before TP1",
                        evidence=(
                            f"Only {prot_rate:.0%} of trades reach PROTECTED state, "
                            f"but {stop_no_prot:.0%} are stopped before breakeven."
                        ),
                        expected_impact="POSITIVE",
                        confidence=0.70,
                        priority=1,
                    )
                )
                fid += 1
            elif prot_rate > 0.80:
                findings.append(
                    OptimisationFinding(
                        finding_id=f"OPT-{fid:03d}",
                        category="BREAKEVEN",
                        recommendation="Breakeven timing is optimal — maintain current threshold",
                        evidence=f"{prot_rate:.0%} of trades reach PROTECTED state.",
                        expected_impact="NEUTRAL",
                        confidence=0.75,
                        priority=3,
                    )
                )
                fid += 1

        # ── 2. Trailing stop analysis ─────────────────────────────────────────
        runner_count = sum(1 for h in histories if any(r.state == LifecycleState.RUNNER for r in h))
        trailing_count = sum(
            1 for h in histories if any(r.state == LifecycleState.TRAILING for r in h)
        )
        closed_count = sum(1 for h in histories if any(r.state == LifecycleState.CLOSED for r in h))

        if runner_count > 0 and trailing_count > 0:
            trail_activation_rate = trailing_count / runner_count
            if trail_activation_rate < 0.50:
                findings.append(
                    OptimisationFinding(
                        finding_id=f"OPT-{fid:03d}",
                        category="TRAILING",
                        recommendation="Activate trailing stop more aggressively on runner positions",
                        evidence=(
                            f"Only {trail_activation_rate:.0%} of runner positions activate "
                            f"trailing — profit may be left at risk."
                        ),
                        expected_impact="POSITIVE",
                        confidence=0.65,
                        priority=2,
                    )
                )
                fid += 1

        # Delayed trailing analysis
        if n > 0:
            runner_to_closed = sum(
                1
                for h in histories
                if any(r.state == LifecycleState.RUNNER for r in h)
                and any(r.state == LifecycleState.CLOSED for r in h)
                and not any(r.state == LifecycleState.TRAILING for r in h)
            )
            if runner_to_closed > 0:
                findings.append(
                    OptimisationFinding(
                        finding_id=f"OPT-{fid:03d}",
                        category="TRAILING",
                        recommendation=(
                            "Some runners closed without trailing stop — "
                            "delayed trailing may improve captured profit"
                        ),
                        evidence=f"{runner_to_closed} trade(s) closed from RUNNER without TRAILING.",
                        expected_impact="POSITIVE",
                        confidence=0.60,
                        priority=2,
                    )
                )
                fid += 1

        # ── 3. Scaling analysis ───────────────────────────────────────────────
        building_count = sum(
            1 for h in histories if any(r.state == LifecycleState.POSITION_BUILDING for r in h)
        )

        if n > 0:
            scaling_rate = building_count / n
            if scaling_rate > 0:
                # Trades that scaled and reached runner
                scaled_to_runner = sum(
                    1
                    for h in histories
                    if any(r.state == LifecycleState.POSITION_BUILDING for r in h)
                    and any(r.state == LifecycleState.RUNNER for r in h)
                )
                if building_count > 0:
                    runner_rate = scaled_to_runner / building_count
                    if runner_rate > 0.60:
                        findings.append(
                            OptimisationFinding(
                                finding_id=f"OPT-{fid:03d}",
                                category="SCALING",
                                recommendation=(
                                    "Scaling strategy effective — consider increasing scale-in "
                                    "threshold from 75% to 80% confidence"
                                ),
                                evidence=(
                                    f"{runner_rate:.0%} of scaled trades reached RUNNER state. "
                                    "Scaling is adding value."
                                ),
                                expected_impact="POSITIVE",
                                confidence=0.68,
                                priority=2,
                            )
                        )
                        fid += 1
                    else:
                        findings.append(
                            OptimisationFinding(
                                finding_id=f"OPT-{fid:03d}",
                                category="SCALING",
                                recommendation="Reduce scaling frequency — low conversion to runner",
                                evidence=(
                                    f"Only {runner_rate:.0%} of scaled trades reach RUNNER. "
                                    "Review confidence threshold."
                                ),
                                expected_impact="POSITIVE",
                                confidence=0.62,
                                priority=2,
                            )
                        )
                        fid += 1

        # ── 4. Expiry / timeout analysis ──────────────────────────────────────
        expired_count = sum(
            1
            for i, h in enumerate(histories)
            if (
                final_states and i < len(final_states) and final_states[i] == LifecycleState.EXPIRED
            )
            or any(r.state == LifecycleState.EXPIRED for r in h)
        )
        if n > 0 and expired_count / n > 0.20:
            findings.append(
                OptimisationFinding(
                    finding_id=f"OPT-{fid:03d}",
                    category="TIMING",
                    recommendation="High expiry rate — reduce entry timeout or entry zone width",
                    evidence=f"{expired_count / n:.0%} of trades expire before entry.",
                    expected_impact="POSITIVE",
                    confidence=0.72,
                    priority=1,
                )
            )
            fid += 1

        # ── 5. Emergency exit analysis ────────────────────────────────────────
        emergency_count = sum(
            1 for h in histories if any(r.state == LifecycleState.EMERGENCY_EXIT for r in h)
        )
        if n > 0 and emergency_count / n > 0.05:
            findings.append(
                OptimisationFinding(
                    finding_id=f"OPT-{fid:03d}",
                    category="RISK",
                    recommendation="Elevated emergency exits — improve news-risk screening pre-entry",
                    evidence=f"{emergency_count / n:.0%} of trades triggered emergency exits.",
                    expected_impact="POSITIVE",
                    confidence=0.78,
                    priority=1,
                )
            )
            fid += 1

        # Sort by priority
        findings.sort(key=lambda f: f.priority)

        top = findings[0].recommendation if findings else None
        summary = (
            f"Analysed {n} trade lifecycles. "
            f"Generated {len(findings)} optimisation findings. "
            f"Trades reaching PROTECTED: {protected_count}/{n}. "
            f"Trailing activations: {trailing_count}/{n}."
        )
        insight = findings[0].evidence if findings else "Insufficient data for insights."

        logger.debug(f"LifecycleOptimizer: {len(findings)} findings from {n} trades")

        return OptimisationReport(
            total_analysed=n,
            findings=findings,
            summary=summary,
            key_insight=insight,
            top_priority=top,
        )
