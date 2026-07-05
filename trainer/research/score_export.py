"""Phase 19.2 — IRSE Score Exporter.

Exports Research Score data in multiple formats:

  JSON  — machine-readable, complete breakdown
  CSV   — flat row per experiment, one column per category
  HTML  — styled report with tables and score bars

Usage::

    exporter = IRSEScoreExporter(leaderboard, trend_tracker, benchmarker, database)
    json_str = exporter.export_json(experiment_id="exp_001")
    csv_str  = exporter.export_csv(top_n=50)
    html_str = exporter.export_html(top_n=20)
    full_report = exporter.export_research_report(experiment_id="exp_001")
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# IRSEScoreExporter
# ---------------------------------------------------------------------------


class IRSEScoreExporter:
    """Multi-format exporter for IRSE scoring data."""

    def __init__(
        self,
        leaderboard: Any,
        trend_tracker: Any,
        benchmarker: Any,
        database: Any,
    ) -> None:
        self._lb      = leaderboard
        self._trends  = trend_tracker
        self._bm      = benchmarker
        self._db      = database

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def export_json(self, experiment_id: Optional[str] = None) -> str:
        """Export full research score data as JSON."""
        if experiment_id:
            row = self._db.get_latest_for_experiment(experiment_id)
            payload: Any = row or {}
        else:
            payload = {
                "leaderboard": self._lb.rank_by("research_score")[:100],
                "trend_report": self._trends.trend_report(),
                "benchmark_report": self._bm.full_benchmark_report(),
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }
        return json.dumps(payload, indent=2, default=str)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_csv(self, top_n: int = 100) -> str:
        """Export top-N leaderboard as CSV."""
        entries = self._lb.rank_by("research_score")[:top_n]
        if not entries:
            return ""
        buf = io.StringIO()
        # Flatten — include category scores as separate columns
        fieldnames = [
            "rank",
            "experiment_id",
            "model_id",
            "name",
            "composite",
            "grade_label",
            "promotion_eligible",
            "deployment_eligible",
            "confidence_score",
            "validation_quality_score",
            "trading_performance_score",
            "robustness_score",
            "calibration_score",
            "generalization_score",
            "efficiency_score",
            "explainability_score",
            "operational_readiness_score",
            "instrument",
            "timeframe",
            "backbone",
            "architecture",
            "dataset_version",
            "feature_version",
            "model_family",
            "scored_at",
        ]
        writer = csv.DictWriter(
            buf, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for e in entries:
            writer.writerow(e)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # HTML export
    # ------------------------------------------------------------------

    def export_html(self, top_n: int = 20) -> str:
        """Export top-N leaderboard as a styled HTML report."""
        entries = self._lb.rank_by("research_score")[:top_n]
        dist = self._trends.score_distribution()
        trend = self._trends.improvement_trend()
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        def score_bar(score: float) -> str:
            pct = max(0.0, min(100.0, score))
            if pct >= 90:
                color = "#22c55e"
            elif pct >= 80:
                color = "#3b82f6"
            elif pct >= 70:
                color = "#eab308"
            else:
                color = "#ef4444"
            return (
                f'<div style="background:#e5e7eb;border-radius:4px;width:100%;height:8px;">'
                f'<div style="background:{color};width:{pct:.1f}%;height:8px;'
                f'border-radius:4px;"></div></div>'
            )

        rows_html = ""
        for e in entries:
            rank = e.get("rank", "—")
            name = e.get("name") or e.get("experiment_id", "")
            comp = e.get("composite", 0.0)
            grade = e.get("grade_label", "")
            promo = "✓" if e.get("promotion_eligible") else "—"
            rows_html += f"""
            <tr>
                <td style="text-align:center;font-weight:bold;">{rank}</td>
                <td>{name}</td>
                <td style="text-align:center;">
                    {comp:.1f}
                    {score_bar(comp)}
                </td>
                <td style="text-align:center;">{grade}</td>
                <td style="text-align:center;">{promo}</td>
                <td style="text-align:center;">{e.get('validation_quality_score',0):.1f}</td>
                <td style="text-align:center;">{e.get('trading_performance_score',0):.1f}</td>
                <td style="text-align:center;">{e.get('robustness_score',0):.1f}</td>
                <td style="text-align:center;">{e.get('calibration_score',0):.1f}</td>
            </tr>"""

        stat_rows = ""
        for k, v in {
            "Total Scored": dist.get("n", 0),
            "Mean Score":   f"{dist.get('mean', 0):.1f}",
            "Max Score":    f"{dist.get('max', 0):.1f}",
            "Min Score":    f"{dist.get('min', 0):.1f}",
            "P75":          f"{dist.get('p75', 0):.1f}",
            "Trend Slope":  f"{trend.get('slope', 0):.4f}",
            "Trend":        trend.get("trend", "—"),
        }.items():
            stat_rows += f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IRSE Leaderboard Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 24px; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; margin-bottom: 4px; }}
  .subtitle {{ color: #64748b; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
  th {{ background: #1e293b; color: white; padding: 10px 12px; text-align: left;
        font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; font-size: 13px; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f1f5f9; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;
                 margin-bottom: 24px; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 16px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat-card table {{ box-shadow: none; margin: 0; }}
  .footer {{ color: #94a3b8; font-size: 12px; text-align: center;
             margin-top: 24px; }}
</style>
</head>
<body>
<h1>Institutional Research Scoring Engine</h1>
<div class="subtitle">Leaderboard Report — {generated}</div>

<div class="stats-grid">
  <div class="stat-card">
    <h3 style="margin:0 0 12px;font-size:14px;color:#475569;">Summary Statistics</h3>
    <table>
      <tbody>{stat_rows}</tbody>
    </table>
  </div>
</div>

<h2 style="font-size:16px;margin-bottom:12px;">Top {top_n} — Research Score Leaderboard</h2>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Experiment</th>
      <th>Research Score</th>
      <th>Grade</th>
      <th>Promote</th>
      <th>Validation</th>
      <th>Trading</th>
      <th>Robustness</th>
      <th>Calibration</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>

<div class="footer">Generated by Signalium IRSE v0.1 · {generated}</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Full research report (Markdown)
    # ------------------------------------------------------------------

    def export_research_report(
        self,
        experiment_id: Optional[str] = None,
        as_markdown: bool = True,
    ) -> str:
        """Generate a full Markdown or JSON research report."""
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        dist = self._trends.score_distribution()
        trend = self._trends.improvement_trend()
        bm_report = self._bm.full_benchmark_report()
        top_10 = self._lb.rank_by("research_score")[:10]

        if not as_markdown:
            return json.dumps({
                "generated_at": generated,
                "score_distribution": dist,
                "improvement_trend": trend,
                "benchmark_summary": bm_report,
                "top_10": top_10,
            }, indent=2, default=str)

        lines = [
            "# Institutional Research Scoring Engine — Research Report",
            f"\n_Generated: {generated}_\n",
            "## Executive Summary",
            "",
            f"- **Total experiments scored:** {dist.get('n', 0)}",
            f"- **Mean Research Score:** {dist.get('mean', 0):.1f}/100",
            f"- **Best Score:** {dist.get('max', 0):.1f}/100",
            f"- **Score Trend:** {trend.get('trend', 'N/A')} "
            f"(slope: {trend.get('slope', 0):.4f} pts/experiment)",
            "",
            "## Score Distribution",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Count  | {dist.get('n', 0)} |",
            f"| Mean   | {dist.get('mean', 0):.1f} |",
            f"| Std    | {dist.get('std', 0):.2f} |",
            f"| Min    | {dist.get('min', 0):.1f} |",
            f"| P25    | {dist.get('p25', 0):.1f} |",
            f"| P50    | {dist.get('p50', 0):.1f} |",
            f"| P75    | {dist.get('p75', 0):.1f} |",
            f"| P90    | {dist.get('p90', 0):.1f} |",
            f"| Max    | {dist.get('max', 0):.1f} |",
            "",
            "## Improvement Trend",
            "",
            f"- **Trend:** {trend.get('trend', 'N/A')}",
            f"- **Slope:** {trend.get('slope', 0):.4f} pts/experiment",
            f"- **Sample:** {trend.get('n', 0)} experiments",
            "",
            "## Top 10 Experiments",
            "",
            "| Rank | Experiment | Score | Grade | Promotion |",
            "|------|-----------|-------|-------|-----------|",
        ]
        for e in top_10:
            rank = e.get("rank", "—")
            name = e.get("name") or e.get("experiment_id", "")[:20]
            score = e.get("composite", 0.0)
            grade = e.get("grade_label", "")
            promo = "Yes" if e.get("promotion_eligible") else "No"
            lines.append(f"| {rank} | {name} | {score:.1f} | {grade} | {promo} |")

        lines += [
            "",
            "## Benchmark Summary",
            "",
            f"- **Total registered:** {bm_report.get('total_registered', 0)}",
            f"- **Champion ID:** {bm_report.get('champion_id', 'None')}",
            "",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Score card (one experiment)
    # ------------------------------------------------------------------

    def experiment_score_card(self, experiment_id: str) -> str:
        """Generate a Markdown score card for one experiment."""
        row = self._db.get_latest_for_experiment(experiment_id)
        if not row:
            return f"# No score found for experiment: {experiment_id}\n"

        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# Research Score Card — {row.get('name', experiment_id)}",
            f"\n_Generated: {generated}_\n",
            f"**Composite Score:** {row.get('composite', 0):.1f}/100  "
            f"[{row.get('grade_label', '')}]",
            f"**Promotion Eligible:** {'Yes' if row.get('promotion_eligible') else 'No'}",
            f"**Deployment Eligible:** {'Yes' if row.get('deployment_eligible') else 'No'}",
            f"**Confidence:** {row.get('confidence_score', 0):.0%}  "
            f"[{row.get('confidence_level', '')}]",
            "",
            "## Category Breakdown",
            "",
            "| Category | Score | Weight | Weighted Contribution |",
            "|----------|-------|--------|-----------------------|",
        ]
        for cs in row.get("category_scores", []):
            lines.append(
                f"| {cs['category']} | {cs['score']:.1f} | "
                f"{cs['weight']*100:.0f}% | {cs['weighted_contribution']:.1f} |"
            )
        lines += [
            "",
            "## Metadata",
            "",
            f"- Instrument: {row.get('instrument', '—')}",
            f"- Timeframe: {row.get('timeframe', '—')}",
            f"- Backbone: {row.get('backbone', '—')}",
            f"- Architecture: {row.get('architecture', '—')}",
            f"- Dataset version: {row.get('dataset_version', '—')}",
            f"- Feature version: {row.get('feature_version', '—')}",
            f"- Model family: {row.get('model_family', '—')}",
            f"- Scored at: {row.get('scored_at', '—')}",
        ]
        return "\n".join(lines)
