"""Tests for the Report Writer agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.writer import writer_node
from src.state import Report, ResearchState


@pytest.fixture
def mock_llm_writer() -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=json.dumps(
                {
                    "title": "AI Agents Market Analysis 2026",
                    "executive_summary": (
                        "The multi-agent AI market is projected to reach $47B by 2030."
                    ),
                    "sections": [
                        {
                            "heading": "Market Overview",
                            "content": "The AI agent market is experiencing rapid growth.",
                            "key_points": ["$47B by 2030", "73% enterprise adoption"],
                        },
                        {
                            "heading": "Technology Trends",
                            "content": "LangGraph and similar frameworks are gaining traction.",
                            "key_points": [
                                "LangGraph leads in enterprise",
                                "Supervisor pattern dominant",
                            ],
                        },
                    ],
                    "insights": [
                        {
                            "claim": "Market growth is accelerating",
                            "evidence": "YoY growth of 40%",
                            "confidence": 0.9,
                            "sources": ["https://example.com"],
                        }
                    ],
                    "sources": ["https://example.com/report", "https://example.com/data"],
                    "quality_score": 0.85,
                }
            )
        )
    )
    return llm


class TestWriterNode:
    async def test_produces_report(
        self, analyzed_research_state: ResearchState, mock_llm_writer: MagicMock
    ) -> None:
        result = await writer_node(analyzed_research_state, llm=mock_llm_writer)
        assert "report" in result
        assert isinstance(result["report"], Report)
        assert result["report"].title == "AI Agents Market Analysis 2026"
        assert len(result["report"].sections) == 2

    async def test_sets_status_to_review(
        self, analyzed_research_state: ResearchState, mock_llm_writer: MagicMock
    ) -> None:
        result = await writer_node(analyzed_research_state, llm=mock_llm_writer)
        assert result["status"] == "review"

    async def test_quality_score_below_threshold_triggers_review(
        self, analyzed_research_state: ResearchState
    ) -> None:
        low_quality_llm = MagicMock()
        low_quality_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content=json.dumps(
                    {
                        "title": "Short",
                        "executive_summary": "Brief.",
                        "sections": [],
                        "insights": [],
                        "sources": [],
                        "quality_score": 0.2,
                    }
                )
            )
        )
        result = await writer_node(analyzed_research_state, llm=low_quality_llm)
        assert result["status"] == "review"

    async def test_handles_llm_error(
        self, analyzed_research_state: ResearchState
    ) -> None:
        bad_llm = MagicMock()
        bad_llm.ainvoke = AsyncMock(return_value=MagicMock(content="broken json"))
        result = await writer_node(analyzed_research_state, llm=bad_llm)
        assert len(result.get("errors", [])) > 0

    async def test_skips_when_no_insights(
        self, empty_research_state: ResearchState, mock_llm_writer: MagicMock
    ) -> None:
        result = await writer_node(empty_research_state, llm=mock_llm_writer)
        assert result.get("status") == "analyzing"
