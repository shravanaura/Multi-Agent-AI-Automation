"""Supervisor agent — plans, dispatches, and aggregates."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.guardrails.output_validator import strip_markdown_fences
from src.state import ErrorRecord, ResearchState

_SUPERVISOR_PROMPT = """You are a research supervisor.
Analyze the current state and decide what to do next.

Current query: {query}
Current status: {status}
Iteration: {iteration}
Data collected: {data_count} items
Insights extracted: {insight_count}
Errors: {error_count}

Respond with ONLY raw JSON, no markdown fences, no explanation:
{{
    "next_agent": "scraper" | "analyst" | "writer" | "none",
    "tasks": ["task1", "task2"],
    "reasoning": "why this decision",
    "done": true/false
}}
"""


_STATUS_MAP = {
    "scraper": "scraping",
    "analyst": "analyzing",
    "writer": "writing",
    "none": "done",
}


class SupervisorDecision(BaseModel):
    """The supervisor's routing decision."""

    next_agent: str
    tasks: list[str]
    reasoning: str
    done: bool


async def supervisor_node(state: ResearchState, *, llm: Any) -> dict[str, Any]:
    """Process the current state and decide the next action.

    Args:
        state: Current research state.
        llm: LLM instance with an async ainvoke method.

    Returns:
        State updates including plan, status, and iteration.
    """
    prompt = _SUPERVISOR_PROMPT.format(
        query=state["query"],
        status=state["status"],
        iteration=state["iteration"],
        data_count=len(state["scraped_data"]),
        insight_count=len(state["analysis"]),
        error_count=len(state["errors"]),
    )

    try:
        response = await llm.ainvoke(prompt)
        raw = strip_markdown_fences(response.content)
        decision = SupervisorDecision.model_validate_json(raw)
    except Exception as e:
        return {
            "errors": state["errors"]
            + [
                ErrorRecord(
                    agent="supervisor",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=True,
                )
            ],
            "iteration": state["iteration"] + 1,
        }

    updates: dict[str, Any] = {
        "iteration": state["iteration"] + 1,
    }

    if decision.tasks:
        updates["plan"] = decision.tasks

    if decision.done:
        updates["status"] = "done"
    else:
        updates["status"] = _STATUS_MAP.get(decision.next_agent, "planning")

    return updates


def route_next(state: ResearchState) -> str:
    """Determine the next node based on current status.

    Args:
        state: Current research state.

    Returns:
        Name of the next node to execute.
    """
    status = state["status"]
    route_map = {
        "planning": "supervisor",
        "scraping": "scraper",
        "analyzing": "analyst",
        "writing": "writer",
        "review": "supervisor",
        "done": "__end__",
    }
    return route_map.get(status, "supervisor")
