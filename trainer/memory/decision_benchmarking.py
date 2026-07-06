"""Phase 21.6 — Decision Benchmarking.

Benchmarks decisions against historical champions, previous generations,
and alternative strategies.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_models import DecisionRecord
from trainer.memory.decision_types import POSITIVE_OUTCOMES


@dataclass
class BenchmarkBaseline:
    """A stored benchmark baseline for comparison."""

    name: str
    win_rate: float
    avg_quality: float
    avg_pnl: float
    sample_size: int
    period: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "win_rate": round(self.win_rate, 4),
            "avg_quality": round(self.avg_quality, 4),
            "avg_pnl": round(self.avg_pnl, 5),
            "sample_size": self.sample_size,
            "period": self.period,
        }


@dataclass
class BenchmarkResult:
    """Comparison of current decisions vs a baseline."""

    current_win_rate: float
    current_avg_quality: float
    current_avg_pnl: float
    current_sample_size: int
    baseline: BenchmarkBaseline
    win_rate_delta: float
    quality_delta: float
    pnl_delta: float
    verdict: str  # BETTER | WORSE | SIMILAR
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current_win_rate": round(self.current_win_rate, 4),
            "current_avg_quality": round(self.current_avg_quality, 4),
            "current_avg_pnl": round(self.current_avg_pnl, 5),
            "current_sample_size": self.current_sample_size,
            "baseline": self.baseline.to_dict(),
            "win_rate_delta": round(self.win_rate_delta, 4),
            "quality_delta": round(self.quality_delta, 4),
            "pnl_delta": round(self.pnl_delta, 5),
            "verdict": self.verdict,
            "notes": self.notes,
        }


@dataclass
class BenchmarkReport:
    """Full benchmarking report across all baselines."""

    total_decisions: int
    comparisons: list[BenchmarkResult] = field(default_factory=list)
    overall_verdict: str = "INCONCLUSIVE"
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "total_decisions": self.total_decisions,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "overall_verdict": self.overall_verdict,
            "summary": self.summary,
        }


class DecisionBenchmarking:
    """Benchmarks current decision performance against stored baselines."""

    def __init__(self):
        self._baselines: dict[str, BenchmarkBaseline] = {}
        self._populate_defaults()

    def _populate_defaults(self) -> None:
        """Pre-populate with reasonable institutional benchmarks."""
        self._baselines["RANDOM"] = BenchmarkBaseline(
            name="Random Strategy",
            win_rate=0.50,
            avg_quality=0.50,
            avg_pnl=0.0,
            sample_size=1000,
            period="theoretical",
        )
        self._baselines["GOOD_TRADER"] = BenchmarkBaseline(
            name="Good Institutional Trader",
            win_rate=0.60,
            avg_quality=0.70,
            avg_pnl=0.0005,
            sample_size=500,
            period="industry_benchmark",
        )
        self._baselines["EXCELLENT_TRADER"] = BenchmarkBaseline(
            name="Excellent Institutional Trader",
            win_rate=0.70,
            avg_quality=0.85,
            avg_pnl=0.0010,
            sample_size=200,
            period="industry_benchmark",
        )

    def add_baseline(self, baseline: BenchmarkBaseline) -> None:
        self._baselines[baseline.name] = baseline
        logger.debug(f"DecisionBenchmarking: added baseline '{baseline.name}'")

    def create_baseline_from_records(
        self,
        name: str,
        records: list[DecisionRecord],
        period: str = "",
    ) -> BenchmarkBaseline:
        """Create a baseline from a set of historical records."""
        completed = [r for r in records if r.has_outcome]
        scored = [r for r in records if r.has_score]
        n = len(completed)

        win_rate = 0.0
        avg_quality = 0.0
        avg_pnl = 0.0

        if n > 0:
            wins = sum(1 for r in completed if r.outcome.outcome_type in POSITIVE_OUTCOMES)
            win_rate = round(wins / n, 4)
            avg_pnl = round(sum(r.outcome.actual_pnl for r in completed) / n, 5)

        if scored:
            avg_quality = round(sum(r.score.overall for r in scored) / len(scored), 4)

        baseline = BenchmarkBaseline(
            name=name,
            win_rate=win_rate,
            avg_quality=avg_quality,
            avg_pnl=avg_pnl,
            sample_size=n,
            period=period,
        )
        self.add_baseline(baseline)
        return baseline

    def benchmark(
        self,
        records: list[DecisionRecord],
        baselines: list[str] | None = None,
    ) -> BenchmarkReport:
        """Benchmark current records against specified (or all) baselines."""

        completed = [r for r in records if r.has_outcome]
        scored = [r for r in records if r.has_score]
        n_comp = len(completed)
        n = len(records)

        report = BenchmarkReport(total_decisions=n)

        if n_comp == 0:
            report.summary = "No completed decisions to benchmark."
            return report

        # Current metrics
        wins = sum(1 for r in completed if r.outcome.outcome_type in POSITIVE_OUTCOMES)
        curr_wr = round(wins / n_comp, 4)
        curr_pnl = round(sum(r.outcome.actual_pnl for r in completed) / n_comp, 5)
        curr_quality = (
            round(sum(r.score.overall for r in scored) / len(scored), 4) if scored else 0.0
        )

        # Which baselines to compare against
        targets = baselines or list(self._baselines.keys())
        comparisons: list[BenchmarkResult] = []

        for bl_name in targets:
            bl = self._baselines.get(bl_name)
            if not bl:
                continue

            wr_delta = round(curr_wr - bl.win_rate, 4)
            q_delta = round(curr_quality - bl.avg_quality, 4)
            pnl_delta = round(curr_pnl - bl.avg_pnl, 5)

            # Verdict: better if two of three metrics improve meaningfully
            improvements = sum(
                [
                    wr_delta >= 0.02,
                    q_delta >= 0.02,
                    pnl_delta >= 0.00005,
                ]
            )
            if improvements >= 2:
                verdict = "BETTER"
            elif improvements == 0:
                verdict = "WORSE"
            else:
                verdict = "SIMILAR"

            notes = []
            if wr_delta > 0.05:
                notes.append(f"Win rate exceeds '{bl.name}' by {wr_delta:.0%}")
            elif wr_delta < -0.05:
                notes.append(f"Win rate trails '{bl.name}' by {abs(wr_delta):.0%}")
            if q_delta > 0.05:
                notes.append(f"Decision quality exceeds '{bl.name}' by {q_delta:.2f}")

            comparisons.append(
                BenchmarkResult(
                    current_win_rate=curr_wr,
                    current_avg_quality=curr_quality,
                    current_avg_pnl=curr_pnl,
                    current_sample_size=n_comp,
                    baseline=bl,
                    win_rate_delta=wr_delta,
                    quality_delta=q_delta,
                    pnl_delta=pnl_delta,
                    verdict=verdict,
                    notes=notes,
                )
            )

        # Overall verdict: majority vote
        verdicts = [c.verdict for c in comparisons]
        counts = collections.Counter(verdicts)
        overall = counts.most_common(1)[0][0] if counts else "INCONCLUSIVE"

        report.comparisons = comparisons
        report.overall_verdict = overall
        report.summary = (
            f"Benchmarked {n_comp} completed decisions vs {len(comparisons)} baselines. "
            f"Overall: {overall}. "
            f"Win rate: {curr_wr:.0%}, Quality: {curr_quality:.2f}."
        )

        logger.debug(
            f"DecisionBenchmarking: {len(comparisons)} comparisons, " f"overall_verdict={overall}"
        )
        return report

    def list_baselines(self) -> list[dict]:
        return [b.to_dict() for b in self._baselines.values()]
