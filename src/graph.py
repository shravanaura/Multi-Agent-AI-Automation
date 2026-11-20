"""LangGraph StateGraph definition for the research pipeline."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.analyst import analyst_node
from src.agents.scraper import scraper_node
from src.agents.supervisor import route_next, supervisor_node
from src.agents.writer import writer_node
from src.state import ResearchState

_DEFAULT_MAX_ITERATIONS = 10


def create_initial_state(query: str) -> ResearchState:
    """Create a fresh initial state for a research query."""
    return ResearchState(
        query=query,
        plan=[],
        scraped_data=[],
        analysis=[],
        report=None,
        errors=[],
        iteration=0,
        status="planning",
    )


async def _supervisor_wrapper(
    state: ResearchState, *, llm: Any, max_iter: int
) -> dict[str, Any]:
    if state["iteration"] >= max_iter:
        return {"status": "done"}
    return await supervisor_node(state, llm=llm)


async def _scraper_wrapper(
    state: ResearchState, *, search_client: Any, http_client: Any
) -> dict[str, Any]:
    return await scraper_node(state, search_client=search_client, http_client=http_client)


async def _analyst_wrapper(state: ResearchState, *, llm: Any) -> dict[str, Any]:
    return await analyst_node(state, llm=llm)


async def _writer_wrapper(state: ResearchState, *, llm: Any) -> dict[str, Any]:
    return await writer_node(state, llm=llm)


def build_research_graph(
    *,
    supervisor_llm: Any,
    worker_llm: Any,
    search_client: Any,
    http_client: Any,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> Any:
    """Build the LangGraph research pipeline.

    Graph structure:
        START -> supervisor -> {scraper, analyst, writer} -> supervisor -> END
    """
    graph = StateGraph(ResearchState)

    graph.add_node(
        "supervisor",
        partial(_supervisor_wrapper, llm=supervisor_llm, max_iter=max_iterations),
    )
    graph.add_node(
        "scraper",
        partial(_scraper_wrapper, search_client=search_client, http_client=http_client),
    )
    graph.add_node("analyst", partial(_analyst_wrapper, llm=worker_llm))
    graph.add_node("writer", partial(_writer_wrapper, llm=worker_llm))

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "scraper": "scraper",
            "analyst": "analyst",
            "writer": "writer",
            "supervisor": "supervisor",
            "__end__": END,
        },
    )

    graph.add_edge("scraper", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()
