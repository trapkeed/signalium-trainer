"""Phase 23 — Report Generator.

Generates professional research reports:
Daily Summary, Weekly Report, Monthly Report, Quarterly Review,
Annual Report, Executive Summary, Model Health, Experiment Review,
Performance Review, Deployment Readiness, Feature Analysis, Research Progress.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_types import ReportType, Permission, Role, has_permission, QueryType
from trainer.analyst.analyst_models import ResearchReport, ResearchQuery, Citation
from trainer.analyst.knowledge_index  import KnowledgeIndex
from trainer.analyst.retrieval_engine import RetrievalEngine
from trainer.analyst.prompt_engine    import PromptEngine
from trainer.analyst.citation_engine  import CitationEngine
from trainer.analyst.llm_provider     import LLMProvider, MockLLMProvider


# Report title templates
_REPORT_TITLES = {
    ReportType.DAILY_SUMMARY:        "Daily Research Summary — {period}",
    ReportType.WEEKLY_REPORT:        "Weekly Research Report — {period}",
    ReportType.MONTHLY_REPORT:       "Monthly Research Report — {period}",
    ReportType.QUARTERLY_REVIEW:     "Quarterly Research Review — {period}",
    ReportType.ANNUAL_REPORT:        "Annual Research Report — {period}",
    ReportType.EXECUTIVE_SUMMARY:    "Executive Research Summary — {period}",
    ReportType.MODEL_HEALTH:         "Model Health Report — {period}",
    ReportType.EXPERIMENT_REVIEW:    "Experiment Review — {period}",
    ReportType.PERFORMANCE_REVIEW:   "Performance Review — {period}",
    ReportType.DEPLOYMENT_READINESS: "Deployment Readiness Report — {period}",
    ReportType.FEATURE_ANALYSIS:     "Feature Analysis Report — {period}",
    ReportType.RESEARCH_PROGRESS:    "Research Progress Report — {period}",
}

# Section structure per report type
_REPORT_SECTIONS = {
    ReportType.DAILY_SUMMARY:     ["Executive Highlights", "Experiments", "Model Status", "Issues"],
    ReportType.WEEKLY_REPORT:     ["Executive Highlights", "Experiments", "Model Health", "Features", "Recommendations"],
    ReportType.MONTHLY_REPORT:    ["Executive Summary", "Experiment Results", "Model Health", "Dataset Status", "Feature Analysis", "Recommendations"],
    ReportType.QUARTERLY_REVIEW:  ["Executive Summary", "Quarter Highlights", "Model Performance", "Dataset Diversity", "Research Velocity", "Strategic Recommendations"],
    ReportType.ANNUAL_REPORT:     ["Year in Review", "Model Champion History", "Research Milestones", "Dataset Evolution", "Feature Stability", "Future Directions"],
    ReportType.EXECUTIVE_SUMMARY: ["Key Findings", "Status", "Recommendations"],
    ReportType.MODEL_HEALTH:      ["Champion Status", "Challenger Analysis", "Calibration Health", "Drift Status", "Recommendations"],
    ReportType.EXPERIMENT_REVIEW: ["Active Experiments", "Completed Experiments", "Failed Experiments", "Key Learnings"],
    ReportType.PERFORMANCE_REVIEW: ["Forecast Accuracy", "Calibration Quality", "Sharpe Analysis", "Drawdown Analysis"],
    ReportType.DEPLOYMENT_READINESS: ["Candidate Models", "Validation Status", "Calibration Status", "Risk Assessment", "Go / No-Go Recommendation"],
    ReportType.FEATURE_ANALYSIS:  ["Feature Importance", "Feature Stability", "Redundant Features", "Missing Regimes"],
    ReportType.RESEARCH_PROGRESS: ["Milestones", "Velocity", "Blockers", "Next Steps"],
}


class ReportGenerator:
    """Generates professional research reports from indexed knowledge."""

    def __init__(
        self,
        index:    KnowledgeIndex,
        provider: Optional[LLMProvider] = None,
        top_k:    int = 8,
    ):
        self._index     = index
        self._provider  = provider or MockLLMProvider()
        self._retriever = RetrievalEngine(index, top_k=top_k)
        self._prompt    = PromptEngine()
        self._citation  = CitationEngine()
        self._reports:  List[ResearchReport] = []

    def generate(
        self,
        query:       ResearchQuery,
        report_type: ReportType,
        period:      str = "",
    ) -> ResearchReport:
        """Generate a research report of the specified type."""
        if not has_permission(query.role, Permission.SUMMARIZE):
            report = ResearchReport(
                report_type = report_type,
                title       = "Access Denied",
                content     = "SUMMARIZE permission required.",
                grounded    = False,
            )
            return report

        period = period or _current_period(report_type)

        # Retrieve relevant context
        search_query = f"{report_type.value.lower().replace('_', ' ')} {query.text}"
        chunks, context = self._retriever.retrieve_and_build(search_query, top_k=8)

        # Build prompt
        prompt = self._prompt.build(
            QueryType.REPORT,
            query.text or f"Generate {report_type.value}",
            context,
            period      = period,
            report_type = report_type,
        )

        # Generate
        response = self._provider.generate(prompt, context)
        response.retrieved_chunks = chunks

        # Citations
        citations = self._citation.extract(response.content)
        citations = self._citation.add_from_chunks(citations, chunks)

        # Section structure
        sections = self._build_sections(response.content, report_type)

        title = _REPORT_TITLES.get(report_type, "Research Report").format(period=period)

        report = ResearchReport(
            report_type  = report_type,
            title        = title,
            content      = response.content,
            sections     = sections,
            citations    = citations,
            period       = period,
            word_count   = len(response.content.split()),
            grounded     = response.grounded,
        )

        self._reports.append(report)

        logger.info(f"ReportGenerator: generated {report_type.value} report '{title}'")
        return report

    def _build_sections(self, content: str, report_type: ReportType) -> List[Dict]:
        """Build section metadata from report content."""
        section_names = _REPORT_SECTIONS.get(report_type, ["Content"])
        sections = []
        lines = content.split("\n")

        current_section = None
        current_lines: List[str] = []

        for line in lines:
            if line.startswith("## "):
                if current_section:
                    sections.append({
                        "title":    current_section,
                        "content":  "\n".join(current_lines).strip(),
                    })
                current_section = line[3:].strip()
                current_lines   = []
            else:
                current_lines.append(line)

        if current_section:
            sections.append({
                "title":   current_section,
                "content": "\n".join(current_lines).strip(),
            })

        # If no sections parsed, build skeleton
        if not sections:
            for name in section_names:
                sections.append({"title": name, "content": ""})

        return sections

    def list_reports(self) -> List[Dict]:
        return [r.to_dict() for r in self._reports]

    def get_report(self, report_id: str) -> Optional[ResearchReport]:
        return next((r for r in self._reports if r.report_id == report_id), None)

    def report_types(self) -> List[str]:
        return [r.value for r in ReportType]


def _current_period(report_type: ReportType) -> str:
    t = time.gmtime()
    if report_type in (ReportType.DAILY_SUMMARY,):
        return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"
    if report_type in (ReportType.WEEKLY_REPORT,):
        return f"{t.tm_year}-W{t.tm_yday // 7:02d}"
    if report_type in (ReportType.MONTHLY_REPORT, ReportType.EXECUTIVE_SUMMARY):
        return f"{t.tm_year}-{t.tm_mon:02d}"
    if report_type in (ReportType.QUARTERLY_REVIEW,):
        q = (t.tm_mon - 1) // 3 + 1
        return f"{t.tm_year}-Q{q}"
    if report_type in (ReportType.ANNUAL_REPORT,):
        return str(t.tm_year)
    return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"
