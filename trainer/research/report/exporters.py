"""Report exporters — HTML, Markdown, JSON, and CSV output formats."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


def export_json(data: dict[str, Any], path: Path) -> None:
    """Write data as pretty-printed JSON.

    Args:
        data: Report data dict.
        path: Output file path.
    """
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def export_markdown(data: dict[str, Any], path: Path) -> None:
    """Write data as a structured Markdown report.

    Args:
        data: Report data dict.
        path: Output .md file path.
    """
    lines = [
        f"# {data.get('title', 'Report')}",
        "",
        f"**Generated:** {data.get('generated_at', '')}",
        "",
    ]

    if "experiment" in data:
        exp = data["experiment"]
        lines += [
            "## Experiment Summary",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| ID | `{exp.get('experiment_id', '')[:8]}` |",
            f"| Model | {exp.get('model_name', '')} |",
            f"| Version | {exp.get('model_version', '')} |",
            f"| Git Commit | `{exp.get('git_commit_hash', '')}` |",
            f"| Status | {exp.get('status', '')} |",
            f"| Duration | {exp.get('training_duration_seconds', 0):.1f}s |",
            f"| Dataset | {exp.get('dataset_version', '')} |",
            f"| Feature | {exp.get('feature_version', '')} |",
            f"| Seed | {exp.get('random_seed', '')} |",
            "",
            "### Metrics",
            "",
        ]
        for k, v in exp.get("metrics", {}).items():
            lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
        lines.append("")

    if "leaderboard" in data and data["leaderboard"]:
        lines += [
            "## Leaderboard",
            "",
            "| Rank | Model | Arch | Accuracy | F1 | Score |",
            "|---|---|---|---|---|---|",
        ]
        for row in data["leaderboard"]:
            lines.append(
                f"| {row.get('rank','')} "
                f"| {row.get('model_name','')} "
                f"| {row.get('architecture','') or '-'} "
                f"| {row.get('accuracy', 0):.4f} "
                f"| {row.get('f1_macro', 0):.4f} "
                f"| {row.get('overall_score', 0):.4f} |"
            )
        lines.append("")

    if "failure_analysis" in data and data["failure_analysis"]:
        lines += ["## Failure Analysis", ""]
        for f in data["failure_analysis"]:
            lines.append(f"- **{f['model_name']}**: {f.get('error', 'unknown error')}")
        lines.append("")

    if "conditions" in data:
        cond = data["conditions"]
        lines += [
            "## Benchmark Conditions",
            "",
            f"| Parameter | Value |",
            f"|---|---|",
        ]
        for k, v in cond.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def export_html(data: dict[str, Any], path: Path) -> None:
    """Write data as a self-contained HTML report.

    Args:
        data: Report data dict.
        path: Output .html file path.
    """
    title = data.get("title", "Report")
    generated = data.get("generated_at", "")

    sections: list[str] = []

    if "experiment" in data:
        exp = data["experiment"]
        rows = "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in {
                "ID": exp.get("experiment_id", "")[:8],
                "Model": exp.get("model_name", ""),
                "Version": exp.get("model_version", ""),
                "Git": exp.get("git_commit_hash", ""),
                "Status": exp.get("status", ""),
                "Duration (s)": f"{exp.get('training_duration_seconds', 0):.1f}",
                "Seed": exp.get("random_seed", ""),
            }.items()
        )
        metric_rows = "".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" if isinstance(v, float)
            else f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in exp.get("metrics", {}).items()
        )
        sections.append(
            f"<h2>Experiment Summary</h2>"
            f"<table border='1'>{rows}</table>"
            f"<h3>Metrics</h3>"
            f"<table border='1'><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>"
        )

    if "leaderboard" in data and data["leaderboard"]:
        lb_rows = "".join(
            "<tr>"
            + "".join(f"<td>{row.get(c, '')}</td>" for c in
                      ["rank", "model_name", "architecture", "accuracy",
                       "f1_macro", "f1_weighted", "inference_latency_ms",
                       "training_time_seconds", "model_size_mb", "overall_score"])
            + "</tr>"
            for row in data["leaderboard"]
        )
        headers = "".join(
            f"<th>{h}</th>" for h in
            ["Rank", "Model", "Arch", "Accuracy", "F1", "F1w",
             "Lat(ms)", "Train(s)", "MB", "Score"]
        )
        sections.append(
            f"<h2>Leaderboard</h2>"
            f"<table border='1'><tr>{headers}</tr>{lb_rows}</table>"
        )

    if "failure_analysis" in data and data["failure_analysis"]:
        fail_rows = "".join(
            f"<tr><td>{f['model_name']}</td><td>{f.get('error', '')}</td></tr>"
            for f in data["failure_analysis"]
        )
        sections.append(
            f"<h2>Failure Analysis</h2>"
            f"<table border='1'><tr><th>Model</th><th>Error</th></tr>{fail_rows}</table>"
        )

    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
    table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
    th, td {{ padding: 6px 12px; border: 1px solid #ccc; text-align: left; }}
    th {{ background: #f0f0f0; }}
    h1 {{ color: #1a1a2e; }}
    h2 {{ color: #16213e; }}
    .meta {{ color: #555; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Generated: {generated}</p>
  {body}
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def export_csv(data: dict[str, Any], path: Path) -> None:
    """Write the leaderboard or run summary as CSV.

    Exports the ``leaderboard`` key if present, otherwise ``runs_summary``.

    Args:
        data: Report data dict.
        path: Output .csv file path.
    """
    rows: list[dict[str, Any]] = []

    if "leaderboard" in data and data["leaderboard"]:
        rows = data["leaderboard"]
    elif "runs_summary" in data:
        rows = data["runs_summary"]
    elif "experiment" in data:
        rows = [data["experiment"].get("metrics", {})]

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    buf = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")
