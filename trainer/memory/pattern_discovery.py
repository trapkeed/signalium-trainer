"""Phase 21.6 — Decision Pattern Discovery.

Automatically discovers performance patterns across decisions:
- Best/worst decisions by market regime
- Most successful entry timing
- Most profitable stop strategy
- Weakest instruments, sessions, regimes
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_models import DecisionRecord
from trainer.memory.decision_types import NEGATIVE_OUTCOMES, POSITIVE_OUTCOMES, DecisionType


@dataclass
class DiscoveredPattern:
    pattern_id: str
    category: str  # REGIME | SESSION | INSTRUMENT | TIMING | STOP | ENTRY
    description: str
    evidence: str
    win_rate: float
    avg_pnl: float
    sample_size: int
    strength: str  # STRONG | MODERATE | WEAK
    direction: str  # POSITIVE | NEGATIVE

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "category": self.category,
            "description": self.description,
            "evidence": self.evidence,
            "win_rate": round(self.win_rate, 4),
            "avg_pnl": round(self.avg_pnl, 5),
            "sample_size": self.sample_size,
            "strength": self.strength,
            "direction": self.direction,
        }


@dataclass
class PatternReport:
    total_decisions: int
    patterns: List[DiscoveredPattern] = field(default_factory=list)
    top_positive: List[str] = field(default_factory=list)  # pattern_ids
    top_negative: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "total_decisions": self.total_decisions,
            "patterns": [p.to_dict() for p in self.patterns],
            "top_positive": self.top_positive,
            "top_negative": self.top_negative,
            "summary": self.summary,
        }


class PatternDiscovery:
    """Discovers performance patterns across a portfolio of decisions."""

    _MIN_SAMPLE = 3  # minimum decisions to call a pattern

    def discover(self, records: List[DecisionRecord]) -> PatternReport:
        """Discover all patterns across the provided decision records."""
        completed = [r for r in records if r.has_outcome]
        report = PatternReport(total_decisions=len(records))

        if len(completed) < self._MIN_SAMPLE:
            report.summary = f"Insufficient data: {len(completed)} completed decisions"
            return report

        pid = 1
        patterns: List[DiscoveredPattern] = []

        # ── 1. By market regime ───────────────────────────────────────────────
        for reg, group in self._group_by(completed, lambda r: r.snapshot.market_regime):
            if len(group) >= self._MIN_SAMPLE:
                p = self._make_pattern(
                    f"PAT-{pid:04d}",
                    "REGIME",
                    f"Performance in {reg} regime",
                    group,
                )
                patterns.append(p)
                pid += 1

        # ── 2. By session ─────────────────────────────────────────────────────
        for sess, group in self._group_by(completed, lambda r: r.snapshot.session):
            if len(group) >= self._MIN_SAMPLE:
                p = self._make_pattern(
                    f"PAT-{pid:04d}",
                    "SESSION",
                    f"Performance during {sess} session",
                    group,
                )
                patterns.append(p)
                pid += 1

        # ── 3. By instrument ──────────────────────────────────────────────────
        for instr, group in self._group_by(completed, lambda r: r.instrument):
            if len(group) >= self._MIN_SAMPLE:
                p = self._make_pattern(
                    f"PAT-{pid:04d}",
                    "INSTRUMENT",
                    f"Performance on {instr}",
                    group,
                )
                patterns.append(p)
                pid += 1

        # ── 4. By volatility regime ───────────────────────────────────────────
        for vreg, group in self._group_by(completed, lambda r: r.snapshot.volatility_regime):
            if len(group) >= self._MIN_SAMPLE:
                p = self._make_pattern(
                    f"PAT-{pid:04d}",
                    "REGIME",
                    f"Performance in {vreg} volatility regime",
                    group,
                )
                patterns.append(p)
                pid += 1

        # ── 5. By decision type ───────────────────────────────────────────────
        for dtype, group in self._group_by(completed, lambda r: r.decision_type.value):
            if len(group) >= self._MIN_SAMPLE:
                p = self._make_pattern(
                    f"PAT-{pid:04d}",
                    "TIMING",
                    f"Performance of {dtype} decisions",
                    group,
                )
                patterns.append(p)
                pid += 1

        # ── 6. High-confidence vs low-confidence ─────────────────────────────
        hi_conf = [r for r in completed if r.snapshot.forecast_confidence >= 0.75]
        lo_conf = [r for r in completed if r.snapshot.forecast_confidence < 0.55]
        if len(hi_conf) >= self._MIN_SAMPLE:
            p = self._make_pattern(
                f"PAT-{pid:04d}",
                "ENTRY",
                "High-confidence decisions (≥75%)",
                hi_conf,
            )
            patterns.append(p)
            pid += 1
        if len(lo_conf) >= self._MIN_SAMPLE:
            p = self._make_pattern(
                f"PAT-{pid:04d}",
                "ENTRY",
                "Low-confidence decisions (<55%)",
                lo_conf,
            )
            patterns.append(p)
            pid += 1

        # ── 7. TP-hit vs SL-hit strategy ─────────────────────────────────────
        tp_group = [r for r in completed if r.outcome.hit_tp]
        sl_group = [r for r in completed if r.outcome.hit_sl]
        if len(tp_group) >= self._MIN_SAMPLE:
            p = self._make_pattern(
                f"PAT-{pid:04d}",
                "STOP",
                "Decisions that hit Take Profit",
                tp_group,
            )
            patterns.append(p)
            pid += 1
        if len(sl_group) >= self._MIN_SAMPLE:
            p = self._make_pattern(
                f"PAT-{pid:04d}",
                "STOP",
                "Decisions that hit Stop Loss",
                sl_group,
            )
            patterns.append(p)
            pid += 1

        # Sort: positive strong first, then by win_rate desc
        patterns.sort(key=lambda p: (-1 if p.direction == "POSITIVE" else 1, -p.win_rate))

        report.patterns = patterns
        report.top_positive = [p.pattern_id for p in patterns if p.direction == "POSITIVE"][:3]
        report.top_negative = [p.pattern_id for p in patterns if p.direction == "NEGATIVE"][:3]

        pos_count = sum(1 for p in patterns if p.direction == "POSITIVE")
        neg_count = sum(1 for p in patterns if p.direction == "NEGATIVE")
        report.summary = (
            f"Discovered {len(patterns)} patterns from {len(completed)} completed decisions. "
            f"Positive: {pos_count}, Negative: {neg_count}."
        )

        logger.debug(f"PatternDiscovery: {len(patterns)} patterns discovered")
        return report

    @staticmethod
    def _group_by(records, key_fn) -> List[Tuple[str, List[DecisionRecord]]]:
        groups: Dict[str, List[DecisionRecord]] = collections.defaultdict(list)
        for r in records:
            groups[key_fn(r)].append(r)
        return list(groups.items())

    def _make_pattern(
        self,
        pid: str,
        category: str,
        description: str,
        group: List[DecisionRecord],
    ) -> DiscoveredPattern:
        n = len(group)
        wins = sum(1 for r in group if r.outcome.outcome_type in POSITIVE_OUTCOMES)
        pnls = [r.outcome.actual_pnl for r in group]
        win_rate = round(wins / n, 4) if n else 0.0
        avg_pnl = round(sum(pnls) / n, 5) if n else 0.0

        strength = (
            "STRONG"
            if n >= 10 and abs(win_rate - 0.5) > 0.20
            else "MODERATE" if n >= 5 and abs(win_rate - 0.5) > 0.10 else "WEAK"
        )
        direction = "POSITIVE" if win_rate > 0.50 and avg_pnl > 0 else "NEGATIVE"
        evidence = f"{n} decisions, win_rate={win_rate:.0%}, avg_pnl={avg_pnl:.5f}"

        return DiscoveredPattern(
            pattern_id=pid,
            category=category,
            description=description,
            evidence=evidence,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            sample_size=n,
            strength=strength,
            direction=direction,
        )
