"""Report Writer agent — synthesizes insights into a structured report."""

from __future__ import annotations

from typing import Any

from src.guardrails.output_validator import score_report_quality, strip_markdown_fences
from src.state import ErrorRecord, Report, ResearchState

_WRITER_PROMPT = """Write a comprehensive research report based on the following analysis.

Research query: {query}

Insights:
{insights}

Sources:
{sources}

Produce the report as JSON:
{{
    "title": "Report Title",
    "executive_summary": "2-3 sentence summary",
    "sections": [
        {{
            "heading": "Section Title",
            "content": "Detailed section content",
            "key_points": ["point 1", "point 2"]
        }}
    ],
    "insights": [
        {{
            "claim": "key claim",
            "evidence": "supporting data",
            "confidence": 0.0-1.0,
            "sources": ["url"]
        }}
    ],
    "sources": ["all source urls"],
    "quality_score": 0.0-1.0
}}

Include at least 3 sections. Be thorough and cite sources.
Return ONLY the raw JSON, no markdown fences, no explanation.
"""


async def writer_node(state: ResearchState, *, llm: Any) -> dict[str, Any]:
    """Generate a structured research report from analyzed insights.

    Args:
        state: Current research state with analysis insights.
        llm: LLM instance with an async ainvoke method.

    Returns:
        State updates with the generated report and status.
    """
    if not state["analysis"]:
        return {"status": "analyzing"}

    insights_text = "\n".join(
        f"- {insight.claim} (confidence: {insight.confidence})\n  Evidence: {insight.evidence}"
        for insight in state["analysis"]
    )

    sources_text = "\n".join(
        set(
            source
            for insight in state["analysis"]
            for source in insight.sources
        )
    )

    prompt = _WRITER_PROMPT.format(
        query=state["query"],
        insights=insights_text,
        sources=sources_text,
    )

    try:
        response = await llm.ainvoke(prompt)
        report = Report.model_validate_json(strip_markdown_fences(response.content))
        computed_score = score_report_quality(report)
        report = report.model_copy(update={"quality_score": computed_score})
    except Exception as e:
        return {
            "errors": state["errors"]
            + [
                ErrorRecord(
                    agent="writer",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=True,
                )
            ],
        }

    return {
        "report": report,
        "status": "review",
    }
