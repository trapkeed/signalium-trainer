"""Phase 15 — Report Exporter.

Exports validation results in multiple formats:
  - JSON   — machine-readable full results
  - CSV    — tabular leaderboard and metrics
  - HTML   — interactive dashboard (pure HTML, no dependencies)
  - PDF    — text-based summary (via HTML → text conversion)

No external PDF library required — PDF output is a formatted text report.

Usage::

    exporter = ReportExporter(output_dir=Path("reports/run-001"))
    paths = exporter.export_all(validation_run_record, leaderboard)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Export config
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """Configuration for report export.

    Attributes:
        output_dir: Directory to write exported files.
        formats: List of formats to generate ("json", "csv", "html", "pdf").
        title: Report title.
        include_charts: Include chart data in JSON output.
        include_raw_metrics: Include all raw metric values.
    """

    output_dir: Path = Path("reports")
    formats: list[str] = field(default_factory=lambda: ["json", "csv", "html"])
    title: str = "Signalium Institutional Validation Report"
    include_charts: bool = True
    include_raw_metrics: bool = True


# ---------------------------------------------------------------------------
# Report Exporter
# ---------------------------------------------------------------------------


class ReportExporter:
    """Exports validation run reports in multiple formats.

    Args:
        output_dir: Output directory (created if it doesn't exist).
        config: Optional ``ExportConfig``.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        config: ExportConfig | None = None,
    ) -> None:
        self._cfg = config or ExportConfig()
        if output_dir:
            self._cfg.output_dir = Path(output_dir)
        self._cfg.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        run_record: Any,
        leaderboard: Any | None = None,
    ) -> list[Path]:
        """Export a validation run in all configured formats.

        Args:
            run_record: ``ValidationRunRecord``.
            leaderboard: Optional ``InstitutionalLeaderboard``.

        Returns:
            List of paths to all written files.
        """
        written: list[Path] = []
        fmt_to_fn = {
            "json": self._export_json,
            "csv": self._export_csv,
            "html": self._export_html,
            "pdf": self._export_pdf,
        }
        for fmt in self._cfg.formats:
            fn = fmt_to_fn.get(fmt)
            if fn:
                try:
                    path = fn(run_record, leaderboard)
                    written.append(path)
                    logger.info(f"ReportExporter: wrote {fmt.upper()} → {path}")
                except Exception as exc:
                    logger.warning(f"ReportExporter: {fmt.upper()} export failed: {exc}")
        return written

    def export_leaderboard_csv(self, leaderboard: Any) -> Path:
        """Export the leaderboard as a standalone CSV."""
        path = self._cfg.output_dir / "leaderboard.csv"
        entries = leaderboard.all_entries()
        if not entries:
            path.write_text("no entries\n")
            return path

        fieldnames = list(entries[0].to_dict().keys())
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                writer.writerow(e.to_dict())
        return path

    # ── Format implementations ────────────────────────────────────────────

    def _export_json(self, run_record: Any, leaderboard: Any | None) -> Path:
        path = self._cfg.output_dir / f"validation_run_{run_record.run_id[:8]}.json"
        data: dict[str, Any] = {
            "run_id": run_record.run_id,
            "model_id": run_record.model_id,
            "model_version": run_record.model_version,
            "run_timestamp": run_record.run_timestamp,
            "status": run_record.status,
            "git_commit": run_record.git_commit,
            "experiment_id": run_record.experiment_id,
            "dataset_version": run_record.dataset_version,
            "feature_version": run_record.feature_version,
            "label_version": run_record.label_version,
            "statistical_metrics": run_record.statistical_metrics,
            "trading_metrics": run_record.trading_metrics,
            "calibration": run_record.calibration,
            "drift": run_record.drift,
            "monte_carlo": run_record.monte_carlo,
            "robustness": run_record.robustness,
            "promotion_decision": run_record.promotion_decision,
            "model_metadata": run_record.model_metadata,
        }
        if self._cfg.include_charts:
            data["visualization_data"] = run_record.visualization_data
        if leaderboard:
            data["leaderboard_snapshot"] = leaderboard.to_dict()
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def _export_csv(self, run_record: Any, leaderboard: Any | None) -> Path:
        path = self._cfg.output_dir / f"metrics_{run_record.run_id[:8]}.csv"
        rows: list[dict[str, Any]] = []

        # Flatten statistical metrics
        stat = run_record.statistical_metrics
        trade = run_record.trading_metrics
        combined = {
            "run_id": run_record.run_id,
            "model_id": run_record.model_id,
            "model_version": run_record.model_version,
            "run_timestamp": run_record.run_timestamp,
            **{f"stat_{k}": v for k, v in stat.items()},
            **{f"trade_{k}": v for k, v in trade.items()},
        }
        rows.append(combined)

        if rows:
            fieldnames = list(rows[0].keys())
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        return path

    def _export_html(self, run_record: Any, leaderboard: Any | None) -> Path:
        path = self._cfg.output_dir / f"report_{run_record.run_id[:8]}.html"
        stat = run_record.statistical_metrics
        trade = run_record.trading_metrics
        calib = run_record.calibration

        def _metric_table(title: str, metrics: dict) -> str:
            rows = "".join(
                f"<tr><td style='padding:4px 12px'>{k}</td>"
                f"<td style='padding:4px 12px;text-align:right'>{round(v, 4) if isinstance(v, float) else v}</td></tr>"
                for k, v in metrics.items()
                if not isinstance(v, (dict, list))
            )
            return (
                f"<h3 style='margin-top:24px'>{title}</h3>"
                f"<table border='1' cellspacing='0' style='border-collapse:collapse;font-size:13px'>"
                f"<tr><th style='padding:4px 12px;background:#f5f5f5'>Metric</th>"
                f"<th style='padding:4px 12px;background:#f5f5f5'>Value</th></tr>"
                f"{rows}</table>"
            )

        promo = run_record.promotion_decision
        promo_status = promo.get("is_eligible", "N/A")
        promo_color = "#2d8a4e" if promo_status is True else "#c0392b"

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{self._cfg.title}</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#222;}}
h1{{color:#1a1a2e;}} h2{{color:#16213e;border-bottom:2px solid #0f3460;padding-bottom:8px;}}
.badge{{display:inline-block;padding:4px 12px;border-radius:4px;color:#fff;font-weight:bold;}}
</style>
</head>
<body>
<h1>{self._cfg.title}</h1>
<p><b>Model:</b> {run_record.model_id} v{run_record.model_version} &nbsp;|&nbsp;
<b>Run:</b> {run_record.run_id[:8]} &nbsp;|&nbsp;
<b>Timestamp:</b> {run_record.run_timestamp} &nbsp;|&nbsp;
<b>Status:</b> <span class="badge" style="background:#27ae60">{run_record.status.upper()}</span></p>
<p><b>Experiment:</b> {run_record.experiment_id or 'N/A'} &nbsp;|&nbsp;
<b>Git:</b> {run_record.git_commit[:8] if run_record.git_commit else 'N/A'} &nbsp;|&nbsp;
<b>Dataset:</b> {run_record.dataset_version or 'N/A'}</p>

<h2>Promotion Decision</h2>
<p>Eligible: <span class="badge" style="background:{promo_color}">{str(promo_status).upper()}</span>
&nbsp; Promoted: <span class="badge" style="background:{'#27ae60' if promo.get('is_promoted') else '#95a5a6'}">{str(promo.get('is_promoted','N/A')).upper()}</span></p>
{_metric_table('Statistical Metrics', stat)}
{_metric_table('Trading Metrics', trade)}
{_metric_table('Calibration', {k: v for k, v in calib.items() if not isinstance(v, (dict, list))})}
"""
        if leaderboard:
            top = leaderboard.top(10)
            html += "<h2>Leaderboard (Top 10)</h2><table border='1' cellspacing='0' style='border-collapse:collapse;font-size:13px'>"
            html += (
                "<tr>"
                + "".join(
                    f"<th style='padding:4px 8px;background:#f5f5f5'>{h}</th>"
                    for h in [
                        "Name",
                        "Version",
                        "Accuracy",
                        "F1",
                        "Sharpe",
                        "PF",
                        "Score",
                        "Status",
                    ]
                )
                + "</tr>"
            )
            for e in top:
                champion = "★ " if e.is_champion else ""
                html += (
                    f"<tr><td>{champion}{e.model_name or e.model_id}</td>"
                    f"<td>{e.model_version}</td>"
                    f"<td>{e.validation_accuracy:.4f}</td>"
                    f"<td>{e.validation_f1:.4f}</td>"
                    f"<td>{e.sharpe_ratio:.3f}</td>"
                    f"<td>{e.profit_factor:.3f}</td>"
                    f"<td>{e.overall_score:.4f}</td>"
                    f"<td>{e.promotion_status}</td></tr>"
                )
            html += "</table>"

        html += "</body></html>"
        path.write_text(html)
        return path

    def _export_pdf(self, run_record: Any, leaderboard: Any | None) -> Path:
        """Export as a text-based 'PDF' report (plain text formatted for clarity)."""
        path = self._cfg.output_dir / f"report_{run_record.run_id[:8]}.pdf.txt"
        stat = run_record.statistical_metrics
        trade = run_record.trading_metrics

        lines = [
            "=" * 72,
            self._cfg.title.center(72),
            "=" * 72,
            "",
            f"Model ID     : {run_record.model_id}",
            f"Version      : {run_record.model_version}",
            f"Run ID       : {run_record.run_id}",
            f"Timestamp    : {run_record.run_timestamp}",
            f"Status       : {run_record.status.upper()}",
            f"Experiment   : {run_record.experiment_id or 'N/A'}",
            f"Git Commit   : {run_record.git_commit[:8] if run_record.git_commit else 'N/A'}",
            f"Dataset V    : {run_record.dataset_version or 'N/A'}",
            "",
            "─" * 72,
            "STATISTICAL METRICS",
            "─" * 72,
        ]
        for k, v in stat.items():
            if not isinstance(v, (dict, list)):
                lines.append(f"  {k:<30s}: {round(v, 4) if isinstance(v, float) else v}")

        lines += ["", "─" * 72, "TRADING METRICS", "─" * 72]
        for k, v in trade.items():
            if not isinstance(v, (dict, list)):
                lines.append(f"  {k:<30s}: {round(v, 4) if isinstance(v, float) else v}")

        promo = run_record.promotion_decision
        lines += [
            "",
            "─" * 72,
            "PROMOTION DECISION",
            "─" * 72,
            f"  Eligible  : {promo.get('is_eligible', 'N/A')}",
            f"  Promoted  : {promo.get('is_promoted', 'N/A')}",
        ]
        for rule in promo.get("failed_rules", []):
            lines.append(f"  [FAIL] {rule}")
        for rule in promo.get("passed_rules", []):
            lines.append(f"  [PASS] {rule}")

        lines += ["", "=" * 72, "End of Report", "=" * 72]
        path.write_text("\n".join(lines))
        return path
