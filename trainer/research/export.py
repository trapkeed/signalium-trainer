"""Phase 19.1 — Research Export.

Exports research artifacts to structured dicts (JSON-serialisable)
and Markdown text.

Exporters:

  ExperimentCardExporter   — one-page experiment card per lineage node
  ResearchReportExporter   — full research report
  ExperimentTreeExporter   — hierarchical experiment tree
  KnowledgeGraphExporter   — full knowledge graph export
  LineageReportExporter    — complete lineage report for one experiment

Usage::

    exporter = ResearchExporter(tree, timeline, cost_tracker, kg, differ)

    card    = exporter.experiment_card("exp_001")
    report  = exporter.research_report()
    tree    = exporter.experiment_tree()
    kg_exp  = exporter.knowledge_graph_export()
    lineage = exporter.lineage_report("exp_001")

    # Markdown versions
    md = exporter.experiment_card_markdown("exp_001")
    md = exporter.lineage_report_markdown("exp_001")
    md = exporter.research_report_markdown()
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# ResearchExporter
# ---------------------------------------------------------------------------


class ResearchExporter:
    """Unified export interface for all Phase 19.1 research artifacts."""

    def __init__(
        self,
        lineage_tree: Any,
        timeline: Any,
        cost_tracker: Any,
        knowledge_graph: Any,
        differ: Any | None = None,
    ) -> None:
        self._tree = lineage_tree
        self._tl = timeline
        self._costs = cost_tracker
        self._kg = knowledge_graph
        self._differ = differ

    # ------------------------------------------------------------------
    # Experiment Card
    # ------------------------------------------------------------------

    def experiment_card(self, node_id: str) -> dict[str, Any]:
        """Return a one-page experiment card for a lineage node."""
        node = self._tree.get(node_id)
        if node is None:
            return {"error": f"Node not found: {node_id}"}

        lineage_report = self._tree.lineage_report(node_id)
        cost = self._costs.get(node_id)
        events = self._tl.by_entity(node_id)

        env_summary: dict[str, Any] = {}
        if node.environment:
            env = node.environment
            env_summary = {
                "git_commit": env.git_commit_hash[:12] if env.git_commit_hash else "",
                "git_branch": env.git_branch,
                "python_version": env.python_version.split()[0] if env.python_version else "",
                "pytorch_version": env.pytorch_version,
                "cuda_version": env.cuda_version,
                "gpu_model": env.gpu_model,
                "random_seed": env.random_seed,
                "training_duration_seconds": env.training_duration_seconds,
            }

        return {
            "card_type": "experiment_card",
            "exported_at": datetime.now(UTC).isoformat(),
            "node_id": node_id,
            "name": node.name,
            "researcher": node.researcher,
            "purpose": node.purpose,
            "branch": node.branch,
            "project": node.project,
            "status": node.status,
            "created_at": node.created_at,
            "completed_at": node.completed_at,
            "notes": node.notes,
            "tags": node.tags,
            "reproducibility_score": lineage_report["reproducibility_score"],
            "reproducibility_grade": lineage_report["reproducibility_grade"],
            "lineage_depth": lineage_report["lineage_depth"],
            "ancestor_count": lineage_report["ancestor_count"],
            "descendant_count": lineage_report["descendant_count"],
            "environment": env_summary,
            "dataset": node.dataset_snapshot.to_dict() if node.dataset_snapshot else None,
            "features": (
                {
                    "feature_count": node.feature_snapshot.feature_count,
                    "top_features": node.feature_snapshot.top_features(5),
                    "feature_set_hash": node.feature_snapshot.feature_set_hash,
                }
                if node.feature_snapshot
                else None
            ),
            "model": node.model_snapshot.to_dict() if node.model_snapshot else None,
            "validation": (
                node.validation_snapshot.summary() if node.validation_snapshot else None
            ),
            "cost": cost.to_dict() if cost else None,
            "timeline_events": [e.to_dict() for e in events],
        }

    def experiment_card_markdown(self, node_id: str) -> str:
        """Render an experiment card as Markdown."""
        card = self.experiment_card(node_id)
        if "error" in card:
            return f"# Error\n{card['error']}"

        lines = [
            f"# Experiment Card: {card['name']}",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| ID | `{card['node_id']}` |",
            f"| Researcher | {card['researcher']} |",
            f"| Branch | {card['branch']} |",
            f"| Project | {card['project']} |",
            f"| Status | {card['status']} |",
            f"| Created | {card['created_at'][:19]} |",
            f"| Completed | {(card['completed_at'] or '')[:19]} |",
            f"| Repro Score | {card['reproducibility_score']:.0%} ({card['reproducibility_grade']}) |",
            f"| Lineage Depth | {card['lineage_depth']} |",
            "",
            "## Purpose",
            f"{card['purpose'] or '_not specified_'}",
            "",
        ]

        env = card.get("environment", {})
        if env:
            lines += [
                "## Environment",
                "| Field | Value |",
                "|-------|-------|",
                f"| Git Commit | `{env.get('git_commit', 'N/A')}` |",
                f"| Git Branch | {env.get('git_branch', 'N/A')} |",
                f"| Python | {env.get('python_version', 'N/A')} |",
                f"| PyTorch | {env.get('pytorch_version', 'N/A')} |",
                f"| CUDA | {env.get('cuda_version', 'N/A')} |",
                f"| GPU | {env.get('gpu_model', 'N/A')} |",
                f"| Random Seed | {env.get('random_seed', 'N/A')} |",
                f"| Training Duration | {env.get('training_duration_seconds', 0):.0f}s |",
                "",
            ]

        val = card.get("validation")
        if val:
            lines += [
                "## Validation Results",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Walk-Forward Accuracy | {val.get('walk_forward_accuracy', 0):.3f} |",
                f"| Walk-Forward F1 Macro | {val.get('walk_forward_f1_macro', 0):.3f} |",
                f"| Calibration ECE | {val.get('calibration_ece', 0):.4f} |",
                f"| Sharpe Ratio | {val.get('sharpe_ratio', 0):.2f} |",
                f"| Max Drawdown | {val.get('max_drawdown', 0):.3f} |",
                f"| Win Rate | {val.get('win_rate', 0):.3f} |",
                f"| Promotion | {val.get('promotion_decision', 'N/A')} |",
                "",
            ]

        cost = card.get("cost")
        if cost:
            lines += [
                "## Cost Estimate",
                "| Item | Value |",
                "|------|-------|",
                f"| Total Cost | ${cost.get('total_cost_usd', 0):.4f} |",
                f"| GPU Hours | {cost.get('gpu_hours', 0):.2f} |",
                f"| CPU Hours | {cost.get('cpu_hours', 0):.2f} |",
                "",
            ]

        if card.get("notes"):
            lines += ["## Notes", card["notes"], ""]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Lineage Report
    # ------------------------------------------------------------------

    def lineage_report(self, node_id: str) -> dict[str, Any]:
        """Return a complete lineage report for one experiment."""
        node = self._tree.get(node_id)
        if node is None:
            return {"error": f"Node not found: {node_id}"}

        lineage = self._tree.lineage_report(node_id)
        ancestors = self._tree.ancestors(node_id)
        descendants = self._tree.descendants(node_id)

        return {
            "report_type": "lineage_report",
            "exported_at": datetime.now(UTC).isoformat(),
            "subject": {
                "node_id": node_id,
                "name": node.name,
                "status": node.status,
                "branch": node.branch,
                "project": node.project,
                "reproducibility_score": lineage["reproducibility_score"],
                "reproducibility_grade": lineage["reproducibility_grade"],
            },
            "ancestors": [
                {
                    "node_id": a.node_id,
                    "name": a.name,
                    "status": a.status,
                    "created_at": a.created_at,
                }
                for a in ancestors
            ],
            "descendants": [
                {
                    "node_id": d.node_id,
                    "name": d.name,
                    "status": d.status,
                    "created_at": d.created_at,
                }
                for d in descendants
            ],
            "full_lineage_report": lineage,
            "experiment_card": self.experiment_card(node_id),
        }

    def lineage_report_markdown(self, node_id: str) -> str:
        report = self.lineage_report(node_id)
        if "error" in report:
            return f"# Error\n{report['error']}"
        subj = report["subject"]
        lines = [
            f"# Lineage Report: {subj['name']}",
            "",
            f"- **Node ID**: `{subj['node_id']}`",
            f"- **Branch**: {subj['branch']}",
            f"- **Project**: {subj['project']}",
            f"- **Status**: {subj['status']}",
            f"- **Reproducibility**: {subj['reproducibility_score']:.0%} "
            f"(Grade {subj['reproducibility_grade']})",
            "",
            f"## Ancestors ({len(report['ancestors'])})",
        ]
        for a in report["ancestors"]:
            lines.append(f"- `{a['node_id'][:8]}` **{a['name']}** [{a['status']}]")
        lines.append("")
        lines.append(f"## Descendants ({len(report['descendants'])})")
        for d in report["descendants"]:
            lines.append(f"- `{d['node_id'][:8]}` **{d['name']}** [{d['status']}]")
        lines.append("")
        lines.append("## Experiment Card")
        lines.append(self.experiment_card_markdown(node_id))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Research Report
    # ------------------------------------------------------------------

    def research_report(self) -> dict[str, Any]:
        """Return a comprehensive platform-wide research report."""
        tree_summary = self._tree.tree_summary()
        cost_report = self._costs.cost_report()
        tl_report = self._tl.timeline_report()
        kg_report = self._kg.graph_report()

        nodes = self._tree.all_nodes()
        repro_scores = [n.reproducibility_score() for n in nodes]
        avg_repro = sum(repro_scores) / len(repro_scores) if repro_scores else 0.0

        return {
            "report_type": "research_report",
            "exported_at": datetime.now(UTC).isoformat(),
            "platform": "Signalium Quant Research Intelligence Platform v19.1",
            "lineage_tree": tree_summary,
            "timeline": {
                "total_events": tl_report["total_events"],
                "event_type_distribution": tl_report["event_type_distribution"],
            },
            "cost": {
                "total_cost_usd": cost_report["total_cost_usd"],
                "total_gpu_hours": cost_report["total_gpu_hours"],
                "cost_by_category": cost_report["cost_by_category"],
            },
            "knowledge_graph": kg_report,
            "reproducibility": {
                "average_score": round(avg_repro, 4),
                "distribution": _repro_distribution(repro_scores),
            },
            "experiment_history": tl_report.get("experiment_history", []),
            "champion_history": tl_report.get("champion_history", []),
        }

    def research_report_markdown(self) -> str:
        report = self.research_report()
        lt = report["lineage_tree"]
        cost = report["cost"]
        kg = report["knowledge_graph"]
        repro = report["reproducibility"]

        lines = [
            "# Research Intelligence Report",
            "",
            f"Generated: {report['exported_at'][:19]}",
            f"Platform:  {report['platform']}",
            "",
            "## Experiment Lineage Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Experiments | {lt['total_experiments']} |",
            f"| Root Experiments | {lt['root_experiments']} |",
            f"| Projects | {len(lt['projects'])} |",
            f"| Branches | {len(lt['branches'])} |",
            f"| Researchers | {len(lt['researchers'])} |",
            f"| Avg Reproducibility | {lt['average_reproducibility_score']:.0%} (Grade {lt['reproducibility_grade']}) |",
            "",
            "## Cost Summary",
            "",
            "| Category | Cost (USD) |",
            "|----------|-----------|",
        ]
        for cat, usd in cost["cost_by_category"].items():
            lines.append(f"| {cat.title()} | ${usd:.4f} |")
        lines += [
            f"| **Total** | **${cost['total_cost_usd']:.4f}** |",
            "",
            f"Total GPU Hours: {cost['total_gpu_hours']:.2f}",
            "",
            "## Knowledge Graph",
            "",
            f"- Nodes: {kg['node_count']}",
            f"- Edges: {kg['edge_count']}",
            f"- Champion Models: {kg['champion_models']}",
            f"- Retired Models: {kg['retired_models']}",
            f"- Deployments: {kg['deployments']}",
            "",
            "## Reproducibility",
            "",
            f"- Average Score: {repro['average_score']:.0%}",
            f"- Distribution: {repro['distribution']}",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Experiment Tree Export
    # ------------------------------------------------------------------

    def experiment_tree(self) -> dict[str, Any]:
        """Export the full experiment tree as a nested structure."""
        summary = self._tree.tree_summary()
        nodes = self._tree.all_nodes()

        node_map = {n.node_id: n.to_dict() for n in nodes}
        roots = [n.node_id for n in self._tree.roots()]

        return {
            "export_type": "experiment_tree",
            "exported_at": datetime.now(UTC).isoformat(),
            "summary": summary,
            "root_ids": roots,
            "nodes": node_map,
        }

    # ------------------------------------------------------------------
    # Knowledge Graph Export
    # ------------------------------------------------------------------

    def knowledge_graph_export(self) -> dict[str, Any]:
        """Export the full knowledge graph."""
        return {
            "export_type": "knowledge_graph",
            "exported_at": datetime.now(UTC).isoformat(),
            "graph": self._kg.to_dict(),
            "report": self._kg.graph_report(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repro_distribution(scores: list[float]) -> dict[str, int]:
    dist: dict[str, int] = {
        "A (≥90%)": 0,
        "B (75-89%)": 0,
        "C (55-74%)": 0,
        "D (35-54%)": 0,
        "F (<35%)": 0,
    }
    for s in scores:
        if s >= 0.90:
            dist["A (≥90%)"] += 1
        elif s >= 0.75:
            dist["B (75-89%)"] += 1
        elif s >= 0.55:
            dist["C (55-74%)"] += 1
        elif s >= 0.35:
            dist["D (35-54%)"] += 1
        else:
            dist["F (<35%)"] += 1
    return dist
