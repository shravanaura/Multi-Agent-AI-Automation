"""Tests for the Data Analyst agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.analyst import analyst_node
from src.state import Insight, ResearchState


@pytest.fixture
def mock_llm_analyst() -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=json.dumps(
                [
                    {
                        "claim": "AI agent market will reach $47B by 2030",
                        "evidence": "Industry reports from multiple sources",
                        "confidence": 0.9,
                        "sources": ["https://example.com/report"],
                    },
                    {
                        "claim": "73% of enterprise AI projects fail",
                        "evidence": "Gartner survey data 2025",
                        "confidence": 0.85,
                        "sources": ["https://example.com/gartner"],
                    },
                ]
            )
        )
    )
    return llm


class TestAnalystNode:
    async def test_extracts_insights(
        self, scraped_research_state: ResearchState, mock_llm_analyst: MagicMock
    ) -> None:
        result = await analyst_node(scraped_research_state, llm=mock_llm_analyst)
        assert "analysis" in result
        assert len(result["analysis"]) == 2
        assert all(isinstance(i, Insight) for i in result["analysis"])

    async def test_sets_status_to_writing(
        self, scraped_research_state: ResearchState, mock_llm_analyst: MagicMock
    ) -> None:
        result = await analyst_node(scraped_research_state, llm=mock_llm_analyst)
        assert result["status"] == "writing"

    async def test_handles_llm_error(
        self, scraped_research_state: ResearchState
    ) -> None:
        bad_llm = MagicMock()
        bad_llm.ainvoke = AsyncMock(return_value=MagicMock(content="not json"))
        result = await analyst_node(scraped_research_state, llm=bad_llm)
        assert len(result.get("errors", [])) > 0

    async def test_skips_when_no_data(
        self, empty_research_state: ResearchState, mock_llm_analyst: MagicMock
    ) -> None:
        result = await analyst_node(empty_research_state, llm=mock_llm_analyst)
        assert result.get("status") == "scraping"
        assert len(result.get("analysis", [])) == 0

    async def test_appends_to_existing_insights(
        self,
        scraped_research_state: ResearchState,
        mock_llm_analyst: MagicMock,
        sample_insight: Insight,
    ) -> None:
        state = {**scraped_research_state, "analysis": [sample_insight]}
        result = await analyst_node(state, llm=mock_llm_analyst)
        assert len(result["analysis"]) == 3
