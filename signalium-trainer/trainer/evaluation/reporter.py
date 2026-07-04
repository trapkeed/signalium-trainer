"""Evaluation report generator."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


class EvaluationReporter:
    """Writes evaluation results to a structured report directory.

    Outputs:
    - ``report.json`` — machine-readable full results.
    - ``summary.txt`` — human-readable summary.

    Args:
        output_dir: Directory to write reports into.
    """

    def __init__(self, output_dir: Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, results: dict[str, Any]) -> Path:
        """Write all report files.

        Args:
            results: Dict returned by StandardEvaluator.evaluate().

        Returns:
            Path to the main report.json file.
        """
        report_path = self._dir / "report.json"
        summary_path = self._dir / "summary.txt"

        serialisable = self._make_serialisable(results)
        with report_path.open("w") as fh:
            json.dump(serialisable, fh, indent=2)

        with summary_path.open("w") as fh:
            fh.write(self._format_summary(results))

        logger.info("Evaluation report written to {}", self._dir)
        return report_path

    def _make_serialisable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serialisable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return str(obj)
        return obj

    def _format_summary(self, results: dict[str, Any]) -> str:
        lines = [
            "=" * 60,
            "SIGNALIUM TRAINER — EVALUATION REPORT",
            f"Generated at: {datetime.utcnow().isoformat()}Z",
            "=" * 60,
        ]
        for key, value in results.items():
            if key in ("confusion_matrix", "classification_report", "prediction_distribution"):
                lines.append(f"\n{key.upper()}:")
                lines.append(str(value))
            elif isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            elif isinstance(value, (int, str)):
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
