"""Leaderboard — automatic model ranking with configurable scoring criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trainer.research.benchmark.runner import BenchmarkResult


@dataclass
class ScoringWeights:
    """Configurable weights for the composite Overall Score.

    Each field is a non-negative weight. Weights are normalised internally
    so they need not sum to 1.0. Set a weight to 0 to exclude that metric.

    Attributes:
        accuracy: Weight for validation accuracy.
        f1_macro: Weight for macro-averaged F1 score.
        calibration_error: Weight for Expected Calibration Error (lower = better).
        inference_speed: Weight for inference throughput (inverse of latency).
        f1_weighted: Weight for weighted F1 score.
    """

    accuracy: float = 0.25
    f1_macro: float = 0.30
    f1_weighted: float = 0.15
    calibration_error: float = 0.15
    inference_speed: float = 0.15


@dataclass
class LeaderboardEntry:
    """A single row in the leaderboard.

    Attributes:
        rank: 1-indexed position (1 = best).
        model_name: Display name of the model.
        architecture: Architecture string from ModelSpec (if available).
        accuracy: Validation accuracy.
        f1_macro: Macro-averaged F1.
        f1_weighted: Weighted F1.
        calibration_error: Expected Calibration Error (lower = better).
        inference_latency_ms: Median inference latency per sample.
        training_time_seconds: Total wall-clock training time.
        model_size_mb: Parameter size in MB.
        overall_score: Weighted composite score (0–1, higher = better).
        gpu_memory_mb: Peak GPU memory during training.
        status: Run status (``"completed"`` or ``"failed"``).
    """

    rank: int
    model_name: str
    architecture: str = ""
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    calibration_error: float = 0.0
    inference_latency_ms: float = 0.0
    training_time_seconds: float = 0.0
    model_size_mb: float = 0.0
    overall_score: float = 0.0
    gpu_memory_mb: float = 0.0
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "rank": self.rank,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "f1_weighted": round(self.f1_weighted, 4),
            "calibration_error": round(self.calibration_error, 4),
            "inference_latency_ms": round(self.inference_latency_ms, 4),
            "training_time_seconds": round(self.training_time_seconds, 3),
            "model_size_mb": round(self.model_size_mb, 4),
            "gpu_memory_mb": round(self.gpu_memory_mb, 2),
            "overall_score": round(self.overall_score, 4),
            "status": self.status,
        }


class Leaderboard:
    """Ranks models from a BenchmarkResult using configurable scoring weights.

    Usage::

        lb = Leaderboard(weights=ScoringWeights(f1_macro=0.5, accuracy=0.3))
        lb.ingest(benchmark_result)
        entries = lb.ranked_entries()
        print(lb.format_table())

    Args:
        weights: ScoringWeights for the composite overall score.
                 Defaults to equal weighting of accuracy, F1, and speed.
    """

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self._weights = weights or ScoringWeights()
        self._entries: list[LeaderboardEntry] = []
        self._benchmark_id: str = ""

    def ingest(self, result: BenchmarkResult) -> None:
        """Populate the leaderboard from a BenchmarkResult.

        Only completed runs are ranked. Failed runs appear at the bottom
        with an overall_score of 0.

        Args:
            result: Output from BenchmarkRunner.run().
        """
        self._benchmark_id = result.benchmark_id
        raw: list[LeaderboardEntry] = []

        completed = result.completed_runs()
        all_latencies = [r.metrics.get("inference_latency_ms", 1.0) for r in completed]
        max_latency = max(all_latencies) if all_latencies else 1.0

        for run in result.runs:
            if run.status != "completed":
                entry = LeaderboardEntry(rank=0, model_name=run.model_name, status="failed")
                raw.append(entry)
                continue

            m = run.metrics
            ece = m.get("calibration_error", 0.0)
            latency = m.get("inference_latency_ms", max_latency)
            speed_score = 1.0 - (latency / (max_latency + 1e-9))

            score = self._composite_score(
                accuracy=m.get("accuracy", 0.0),
                f1_macro=m.get("f1_macro", 0.0),
                f1_weighted=m.get("f1_weighted", 0.0),
                calibration_error=ece,
                speed_score=speed_score,
            )

            arch = self._lookup_architecture(run.model_name)
            raw.append(
                LeaderboardEntry(
                    rank=0,
                    model_name=run.model_name,
                    architecture=arch,
                    accuracy=m.get("accuracy", 0.0),
                    f1_macro=m.get("f1_macro", 0.0),
                    f1_weighted=m.get("f1_weighted", 0.0),
                    calibration_error=ece,
                    inference_latency_ms=latency,
                    training_time_seconds=run.training_time_seconds,
                    model_size_mb=run.model_size_mb,
                    gpu_memory_mb=run.gpu_memory_mb,
                    overall_score=score,
                    status="completed",
                )
            )

        # Sort: completed by score desc, failed at the end
        completed_entries = sorted(
            [e for e in raw if e.status == "completed"],
            key=lambda e: e.overall_score,
            reverse=True,
        )
        failed_entries = [e for e in raw if e.status != "completed"]

        for i, entry in enumerate(completed_entries, start=1):
            entry.rank = i
        for i, entry in enumerate(failed_entries, start=len(completed_entries) + 1):
            entry.rank = i

        self._entries = completed_entries + failed_entries

    def ranked_entries(self) -> list[LeaderboardEntry]:
        """Return all entries sorted by rank (best first).

        Returns:
            List of LeaderboardEntry ordered by rank.
        """
        return list(self._entries)

    def top_n(self, n: int = 5) -> list[LeaderboardEntry]:
        """Return the top N completed entries.

        Args:
            n: Number of top entries to return.

        Returns:
            At most N LeaderboardEntry objects.
        """
        completed = [e for e in self._entries if e.status == "completed"]
        return completed[:n]

    def champion(self) -> LeaderboardEntry | None:
        """Return the top-ranked completed entry (rank 1).

        Returns:
            LeaderboardEntry with rank 1, or None if no completed runs.
        """
        top = self.top_n(1)
        return top[0] if top else None

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialise all entries to a list of dicts."""
        return [e.to_dict() for e in self._entries]

    def format_table(self, max_col_width: int = 20) -> str:
        """Return a formatted ASCII table of the leaderboard.

        Args:
            max_col_width: Maximum width for text columns.

        Returns:
            Multi-line string table.
        """
        headers = [
            "Rank",
            "Model",
            "Arch",
            "Acc",
            "F1",
            "Cal Err",
            "Lat(ms)",
            "Train(s)",
            "MB",
            "Score",
        ]
        rows = []
        for e in self._entries:
            rows.append(
                [
                    str(e.rank),
                    e.model_name[:max_col_width],
                    (e.architecture or "-")[:max_col_width],
                    f"{e.accuracy:.4f}",
                    f"{e.f1_macro:.4f}",
                    f"{e.calibration_error:.4f}",
                    f"{e.inference_latency_ms:.2f}",
                    f"{e.training_time_seconds:.1f}",
                    f"{e.model_size_mb:.2f}",
                    f"{e.overall_score:.4f}",
                ]
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

        lines = [sep, fmt.format(*headers), sep]
        for row in rows:
            lines.append(fmt.format(*row))
        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _composite_score(
        self,
        accuracy: float,
        f1_macro: float,
        f1_weighted: float,
        calibration_error: float,
        speed_score: float,
    ) -> float:
        w = self._weights
        total_weight = (
            w.accuracy + w.f1_macro + w.f1_weighted + w.calibration_error + w.inference_speed
        )
        if total_weight == 0:
            return 0.0
        calibration_score = max(0.0, 1.0 - calibration_error)
        score = (
            w.accuracy * accuracy
            + w.f1_macro * f1_macro
            + w.f1_weighted * f1_weighted
            + w.calibration_error * calibration_score
            + w.inference_speed * speed_score
        ) / total_weight
        return round(min(1.0, max(0.0, score)), 6)

    @staticmethod
    def _lookup_architecture(model_name: str) -> str:
        """Try to resolve architecture from the ModelRegistry spec."""
        try:
            from trainer.registry.registry import ModelRegistry

            spec = ModelRegistry.get_spec(model_name)
            return spec.architecture if spec else ""
        except Exception:
            return ""
