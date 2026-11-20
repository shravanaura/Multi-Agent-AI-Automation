"""Data Analyst agent — processes scraped data and extracts insights."""

from __future__ import annotations

import json
from typing import Any

from src.guardrails.output_validator import strip_markdown_fences
from src.state import ErrorRecord, Insight, ResearchState

_ANALYST_PROMPT = """Analyze the following scraped data and extract key insights.

Research query: {query}

Scraped data:
{data}

Extract insights as a JSON array:
[
  {{
    "claim": "specific factual claim",
    "evidence": "supporting evidence from the data",
    "confidence": 0.0-1.0,
    "sources": ["source_url"]
  }}
]

Focus on:
- Quantitative claims (market sizes, percentages, costs)
- Trends and patterns
- Comparisons between technologies
- Practical implications

Return ONLY the raw JSON array, no markdown fences, no explanation.
"""


async def analyst_node(state: ResearchState, *, llm: Any) -> dict[str, Any]:
    """Analyze scraped data and extract structured insights.

    Args:
        state: Current research state with scraped data.
        llm: LLM instance with an async ainvoke method.

    Returns:
        State updates with extracted insights and status.
    """
    if not state["scraped_data"]:
        return {"status": "scraping", "analysis": []}

    data_text = "\n\n".join(
        f"[{item.source}] {item.title}\nURL: {item.url}\n{item.content[:2000]}"
        for item in state["scraped_data"]
    )

    prompt = _ANALYST_PROMPT.format(query=state["query"], data=data_text)

    try:
        response = await llm.ainvoke(prompt)
        raw_insights = json.loads(strip_markdown_fences(response.content))
        new_insights = [Insight.model_validate(item) for item in raw_insights]
    except Exception as e:
        return {
            "errors": state["errors"]
            + [
                ErrorRecord(
                    agent="analyst",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=True,
                )
            ],
        }

    return {
        "analysis": list(state["analysis"]) + new_insights,
        "status": "writing",
    }
