"""Integration tests for the full LangGraph research pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph import build_research_graph, create_initial_state


@pytest.fixture
def mock_dependencies() -> dict:
    """Full set of mocked dependencies for the graph."""
    supervisor_llm = MagicMock()
    worker_llm = MagicMock()
    search_client = MagicMock()
    http_client = AsyncMock()

    # Supervisor: first call plans, second call routes to analyst, third call marks done
    supervisor_llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content=json.dumps({
                "next_agent": "scraper",
                "tasks": ["Search for AI agent data"],
                "reasoning": "Need data first",
                "done": False,
            })),
            MagicMock(content=json.dumps({
                "next_agent": "analyst",
                "tasks": [],
                "reasoning": "Data collected",
                "done": False,
            })),
            MagicMock(content=json.dumps({
                "next_agent": "writer",
                "tasks": [],
                "reasoning": "Analysis complete",
                "done": False,
            })),
            MagicMock(content=json.dumps({
                "next_agent": "none",
                "tasks": [],
                "reasoning": "Report is ready",
                "done": True,
            })),
        ]
    )

    # Worker LLM for analyst and writer
    worker_llm.ainvoke = AsyncMock(
        side_effect=[
            # Analyst response
            MagicMock(content=json.dumps([{
                "claim": "AI market growing 40% YoY",
                "evidence": "Multiple reports",
                "confidence": 0.9,
                "sources": ["https://example.com"],
            }])),
            # Writer response
            MagicMock(content=json.dumps({
                "title": "AI Agent Analysis",
                "executive_summary": "AI agents transform workflows.",
                "sections": [
                    {
                        "heading": "Market",
                        "content": "Growing rapidly.",
                        "key_points": ["40% growth"],
                    },
                    {
                        "heading": "Tech",
                        "content": "LangGraph leads.",
                        "key_points": ["LangGraph dominant"],
                    },
                    {
                        "heading": "Cost",
                        "content": "$0.12 per run.",
                        "key_points": ["$0.12/run"],
                    },
                ],
                "insights": [
                    {
                        "claim": "Growth",
                        "evidence": "Data",
                        "confidence": 0.9,
                        "sources": ["url"],
                    },
                ],
                "sources": ["https://example.com"],
                "quality_score": 0.85,
            })),
        ]
    )

    # Search client
    search_client.search = AsyncMock(return_value={
        "results": [{
            "title": "AI Agents Guide",
            "url": "https://example.com/guide",
            "content": "Comprehensive AI agent guide content.",
            "score": 0.92,
        }]
    })

    # HTTP client for scraping
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><head><title>T</title></head><body><p>Content</p></body></html>"
    http_client.get = AsyncMock(return_value=mock_response)

    return {
        "supervisor_llm": supervisor_llm,
        "worker_llm": worker_llm,
        "search_client": search_client,
        "http_client": http_client,
    }


class TestCreateInitialState:
    def test_creates_valid_state(self) -> None:
        state = create_initial_state("test query")
        assert state["query"] == "test query"
        assert state["status"] == "planning"
        assert state["iteration"] == 0
        assert state["report"] is None
        assert state["scraped_data"] == []
        assert state["analysis"] == []
        assert state["errors"] == []
        assert state["plan"] == []


class TestBuildResearchGraph:
    def test_builds_graph(self, mock_dependencies: dict) -> None:
        graph = build_research_graph(**mock_dependencies)
        assert graph is not None

    async def test_full_pipeline_produces_report(self, mock_dependencies: dict) -> None:
        graph = build_research_graph(**mock_dependencies)
        initial = create_initial_state("AI agents market analysis")

        result = await graph.ainvoke(initial)

        assert result["status"] == "done"
        assert result["report"] is not None
        assert result["report"].title == "AI Agent Analysis"
        assert len(result["scraped_data"]) > 0
        assert len(result["analysis"]) > 0

    async def test_pipeline_stops_at_max_iterations(self, mock_dependencies: dict) -> None:
        # Override supervisor to never say done
        mock_dependencies["supervisor_llm"].ainvoke = AsyncMock(
            return_value=MagicMock(content=json.dumps({
                "next_agent": "scraper",
                "tasks": ["Search more"],
                "reasoning": "Need more data",
                "done": False,
            }))
        )

        graph = build_research_graph(**mock_dependencies, max_iterations=3)
        initial = create_initial_state("test")

        result = await graph.ainvoke(initial)
        assert result["iteration"] <= 3
