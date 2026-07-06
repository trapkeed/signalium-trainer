"""Phase 23 — Prompt Engine.

Builds structured prompts for the LLM including:
- System prompt (SAQRA identity + guardrails)
- Context injection (retrieved chunks)
- User query formatting
- Prompt templates for each query type
"""

from __future__ import annotations

from typing import Dict, List, Optional

from trainer.analyst.analyst_types import QueryType, ReportType
from trainer.analyst.analyst_models import KnowledgeChunk, ChatMessage

_SYSTEM_PROMPT = """You are SAQRA — Signalium's AI Quant Research Analyst.

ROLE: Institutional research assistant for the Signalium quantitative platform.

MISSION:
- Help researchers understand Signalium's quantitative platform behaviour.
- Explain model performance, feature importance, and experiment results.
- Ground every response in retrieved research evidence.
- Cite experiment IDs, model IDs, dataset versions, and report IDs.

ABSOLUTE CONSTRAINTS:
- NEVER generate BUY or SELL signals.
- NEVER recommend entering or exiting live trades.
- NEVER override production forecasting logic.
- NEVER make predictions about future price movements.
- NEVER hallucinate metrics, experiment IDs, or model names.

CITATION FORMAT: Reference artifacts as [TYPE:ID], e.g. [EXP:001], [MODEL:42], [DATASET:v18].

RESPONSE FORMAT:
- Use Markdown headers for structure.
- Begin with a concise executive summary.
- Support each claim with a citation.
- End with a one-line disclaimer confirming no trading signals are implied.
"""

_TEMPLATES: Dict[str, str] = {
    QueryType.CHAT.value: """{system}

RESEARCH CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

RESEARCHER QUERY:
{query}

Provide a grounded, evidence-based response. Cite all referenced artifacts.""",
    QueryType.COMPARISON.value: """{system}

RESEARCH CONTEXT:
{context}

COMPARISON REQUEST:
{query}

Generate a structured comparative analysis. Include:
1. Side-by-side metrics
2. Key differences
3. Root cause analysis
4. Evidence citations""",
    QueryType.ANALYSIS.value: """{system}

RESEARCH CONTEXT:
{context}

ANALYSIS REQUEST:
{query}

Provide a detailed research analysis. Include:
1. Executive summary
2. Evidence from retrieved artifacts
3. Contributing factors
4. Citations for all claims""",
    QueryType.SEARCH.value: """{system}

RESEARCH CONTEXT:
{context}

SEARCH REQUEST:
{query}

List all matching artifacts with:
1. Artifact ID and type
2. Relevance summary
3. Key metrics""",
    QueryType.SUMMARY.value: """{system}

RESEARCH CONTEXT:
{context}

SUMMARY REQUEST: {query}
PERIOD: {period}

Generate a professional research summary. Include:
1. Executive highlights
2. Experiment status
3. Model health
4. Key findings
5. Recommendations""",
    QueryType.RECOMMENDATION.value: """{system}

RESEARCH CONTEXT:
{context}

RECOMMENDATION REQUEST:
{query}

Generate evidence-based research recommendations. For each recommendation:
1. State the recommendation clearly
2. Provide evidence from retrieved artifacts
3. Assign priority (HIGH/MEDIUM/LOW)
4. Never recommend live trading actions""",
    QueryType.REPORT.value: """{system}

RESEARCH CONTEXT:
{context}

REPORT TYPE: {report_type}
REPORT REQUEST: {query}

Generate a professional institutional research report. Structure:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Citations
5. Appendix""",
}


class PromptEngine:
    """Builds structured prompts for the LLM."""

    def __init__(self, system_prompt: Optional[str] = None):
        self._system = system_prompt or _SYSTEM_PROMPT

    def build(
        self,
        query_type: QueryType,
        query: str,
        context: str,
        history: Optional[List[ChatMessage]] = None,
        period: str = "",
        report_type: Optional[ReportType] = None,
    ) -> str:
        """Build a complete prompt for the given query type."""
        template = _TEMPLATES.get(query_type.value, _TEMPLATES[QueryType.CHAT.value])

        history_text = ""
        if history:
            history_text = "\n".join(f"{m.role.upper()}: {m.content[:200]}" for m in history[-6:])

        try:
            prompt = template.format(
                system=self._system,
                context=context
                or "(No relevant artifacts retrieved. Indicate this in your response.)",
                query=query,
                history=history_text,
                period=period,
                report_type=report_type.value if report_type else "GENERAL",
            )
        except KeyError:
            # Fallback: use chat template
            prompt = _TEMPLATES[QueryType.CHAT.value].format(
                system=self._system,
                context=context or "",
                query=query,
                history=history_text,
            )

        return prompt

    def get_template(self, query_type: QueryType) -> str:
        return _TEMPLATES.get(query_type.value, _TEMPLATES[QueryType.CHAT.value])

    def list_templates(self) -> List[str]:
        return list(_TEMPLATES.keys())

    @property
    def system_prompt(self) -> str:
        return self._system

    def update_system_prompt(self, new_prompt: str) -> None:
        self._system = new_prompt
