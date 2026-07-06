"""ReportGenerator — produces experiment, benchmark, and leaderboard reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from trainer.research.benchmark.runner import BenchmarkResult
from trainer.research.experiment.record import ExperimentRecord
from trainer.research.leaderboard.leaderboard import Leaderboard


@dataclass
class ReportConfig:
    """Configuration for report generation.

    Attributes:
        title: Report title string.
        include_history: Include epoch-by-epoch training curves in the report.
        include_hyperparameters: Include hyperparameter tables.
        formats: Output formats to generate (``"html"``, ``"markdown"``, ``"json"``, ``"csv"``).
        output_dir: Directory to write report files.
    """

    title: str = "Signalium Research Report"
    include_history: bool = True
    include_hyperparameters: bool = True
    formats: list[str] = field(default_factory=lambda: ["markdown", "json"])
    output_dir: Path = Path("reports")


class ReportGenerator:
    """Generates experiment and benchmark reports in multiple formats.

    Reports include:
    - Experiment summary (ID, model, git hash, metrics, duration)
    - Benchmark comparison table
    - Leaderboard
    - Failure analysis
    - Training comparison (epoch curves)

    Usage::

        gen = ReportGenerator(ReportConfig(
            title="LSTM Baseline",
            formats=["markdown", "html", "json", "csv"],
            output_dir=Path("reports/run-001"),
        ))
        paths = gen.generate_benchmark_report(benchmark_result, leaderboard)

    Args:
        config: ReportConfig.
    """

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        self._cfg = config or ReportConfig()

    # ------------------------------------------------------------------
    # Public report methods
    # ------------------------------------------------------------------

    def generate_experiment_report(
        self,
        record: ExperimentRecord,
    ) -> list[Path]:
        """Generate a report for a single experiment.

        Args:
            record: Completed ExperimentRecord.

        Returns:
            List of paths to written report files.
        """
        data = self._experiment_report_data(record)
        return self._write_all(data, stem=f"experiment-{record.experiment_id[:8]}")

    def generate_benchmark_report(
        self,
        result: BenchmarkResult,
        leaderboard: Optional[Leaderboard] = None,
    ) -> list[Path]:
        """Generate a full benchmark comparison report.

        Args:
            result: BenchmarkResult from BenchmarkRunner.run().
            leaderboard: Optional pre-built Leaderboard (built from result if omitted).

        Returns:
            List of paths to written report files.
        """
        if leaderboard is None:
            lb = Leaderboard()
            lb.ingest(result)
        else:
            lb = leaderboard

        data = self._benchmark_report_data(result, lb)
        return self._write_all(data, stem=f"benchmark-{result.benchmark_id[:8]}")

    def generate_leaderboard_report(self, leaderboard: Leaderboard) -> list[Path]:
        """Generate a standalone leaderboard report.

        Args:
            leaderboard: Pre-built and populated Leaderboard.

        Returns:
            List of paths to written report files.
        """
        data = {
            "title": f"{self._cfg.title} — Leaderboard",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "leaderboard": leaderboard.to_dict(),
        }
        return self._write_all(data, stem="leaderboard")

    # ------------------------------------------------------------------
    # Data builders
    # ------------------------------------------------------------------

    def _experiment_report_data(self, rec: ExperimentRecord) -> dict[str, Any]:
        return {
            "title": f"{self._cfg.title} — Experiment {rec.experiment_id[:8]}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": rec.to_dict(),
        }

    def _benchmark_report_data(
        self,
        result: BenchmarkResult,
        lb: Leaderboard,
    ) -> dict[str, Any]:
        failure_analysis = [
            {"model_name": r.model_name, "error": r.error}
            for r in result.runs
            if r.status == "failed"
        ]
        training_comparison = {}
        if self._cfg.include_history:
            for run in result.completed_runs():
                training_comparison[run.model_name] = run.history

        return {
            "title": f"{self._cfg.title} — Benchmark {result.benchmark_id[:8]}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "conditions": result.conditions.to_dict(),
            "runs_summary": result.to_dict()["runs"],
            "leaderboard": lb.to_dict(),
            "failure_analysis": failure_analysis,
            "training_comparison": training_comparison,
            "total_duration_seconds": result.total_duration_seconds,
        }

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _write_all(self, data: dict[str, Any], stem: str) -> list[Path]:
        """Write the data dict in all configured formats."""
        from trainer.research.report.exporters import (
            export_csv,
            export_html,
            export_json,
            export_markdown,
        )

        out_dir = Path(self._cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        dispatch = {
            "json": (export_json, ".json"),
            "markdown": (export_markdown, ".md"),
            "html": (export_html, ".html"),
            "csv": (export_csv, ".csv"),
        }
        for fmt in self._cfg.formats:
            if fmt not in dispatch:
                continue
            fn, ext = dispatch[fmt]
            path = out_dir / (stem + ext)
            fn(data, path)
            paths.append(path)

        return paths
