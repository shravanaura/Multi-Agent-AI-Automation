"""FastAPI server for the multi-agent research system."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv()

from src.graph import build_research_graph, create_initial_state

app = FastAPI(
    title="Multi-Agent Research System",
    description="Production-grade multi-agent research using LangGraph",
    version="0.1.0",
)


class ResearchRequest(BaseModel):
    """Request body for the research endpoint."""

    query: str = Field(min_length=1, description="Research topic to investigate")


class ResearchResponse(BaseModel):
    """Response body for the research endpoint."""

    status: str
    iterations: int
    report: dict[str, Any] | None
    errors: list[dict[str, Any]]


def _get_graph() -> Any:
    """Get or create the research graph with real dependencies.

    This is patched in tests to inject mocks.
    """
    import os

    import httpx
    from langchain_anthropic import ChatAnthropic
    from tavily import AsyncTavilyClient

    supervisor_llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    worker_llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    search_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    http_client = httpx.AsyncClient()

    return build_research_graph(
        supervisor_llm=supervisor_llm,
        worker_llm=worker_llm,
        search_client=search_client,
        http_client=http_client,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run a research query through the multi-agent pipeline."""
    graph = _get_graph()
    initial_state = create_initial_state(request.query)

    result = await graph.ainvoke(initial_state)

    report_dict = None
    if result.get("report") is not None:
        report_dict = result["report"].model_dump()

    error_dicts = [e.model_dump() for e in result.get("errors", [])]

    return ResearchResponse(
        status=result["status"],
        iterations=result["iteration"],
        report=report_dict,
        errors=error_dicts,
    )
