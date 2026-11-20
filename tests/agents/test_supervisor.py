"""Tests for the Supervisor agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.supervisor import route_next, supervisor_node
from src.state import ResearchState


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM that returns a SupervisorDecision."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=json.dumps(
                {
                    "next_agent": "scraper",
                    "tasks": ["Search for AI agent market data", "Find LangGraph tutorials"],
                    "reasoning": "Need web data before analysis",
                    "done": False,
                }
            )
        )
    )
    return llm


class TestSupervisorNode:
    async def test_creates_plan_on_planning_state(
        self, empty_research_state: ResearchState, mock_llm: MagicMock
    ) -> None:
        result = await supervisor_node(empty_research_state, llm=mock_llm)
        assert "plan" in result
        assert len(result["plan"]) == 2
        assert result["status"] == "scraping"

    async def test_routes_to_analyst_after_scraping(
        self, scraped_research_state: ResearchState, mock_llm: MagicMock
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content=json.dumps(
                    {
                        "next_agent": "analyst",
                        "tasks": [],
                        "reasoning": "Data collected, time to analyze",
                        "done": False,
                    }
                )
            )
        )
        result = await supervisor_node(scraped_research_state, llm=mock_llm)
        assert result["status"] == "analyzing"

    async def test_marks_done_when_report_ready(
        self, analyzed_research_state: ResearchState, mock_llm: MagicMock
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content=json.dumps(
                    {
                        "next_agent": "none",
                        "tasks": [],
                        "reasoning": "Report complete",
                        "done": True,
                    }
                )
            )
        )
        result = await supervisor_node(analyzed_research_state, llm=mock_llm)
        assert result["status"] == "done"

    async def test_increments_iteration(
        self, empty_research_state: ResearchState, mock_llm: MagicMock
    ) -> None:
        result = await supervisor_node(empty_research_state, llm=mock_llm)
        assert result["iteration"] == 1

    async def test_handles_malformed_llm_output(
        self, empty_research_state: ResearchState, mock_llm: MagicMock
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="not valid json")
        )
        result = await supervisor_node(empty_research_state, llm=mock_llm)
        assert len(result.get("errors", [])) > 0


class TestRouteNext:
    def test_routes_to_scraper(self) -> None:
        state: ResearchState = {
            "query": "test", "plan": [], "scraped_data": [], "analysis": [],
            "report": None, "errors": [], "iteration": 0, "status": "scraping",
        }
        assert route_next(state) == "scraper"

    def test_routes_to_analyst(self) -> None:
        state: ResearchState = {
            "query": "test", "plan": [], "scraped_data": [], "analysis": [],
            "report": None, "errors": [], "iteration": 0, "status": "analyzing",
        }
        assert route_next(state) == "analyst"

    def test_routes_to_writer(self) -> None:
        state: ResearchState = {
            "query": "test", "plan": [], "scraped_data": [], "analysis": [],
            "report": None, "errors": [], "iteration": 0, "status": "writing",
        }
        assert route_next(state) == "writer"

    def test_routes_to_end(self) -> None:
        state: ResearchState = {
            "query": "test", "plan": [], "scraped_data": [], "analysis": [],
            "report": None, "errors": [], "iteration": 0, "status": "done",
        }
        assert route_next(state) == "__end__"

    def test_routes_back_to_supervisor_on_review(self) -> None:
        state: ResearchState = {
            "query": "test", "plan": [], "scraped_data": [], "analysis": [],
            "report": None, "errors": [], "iteration": 0, "status": "review",
        }
        assert route_next(state) == "supervisor"
