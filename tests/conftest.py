"""Shared test fixtures for multi-agent research tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.state import (
    Insight,
    Report,
    ReportSection,
    ResearchState,
    ScrapedItem,
)


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Mock Anthropic client that returns structured responses."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture
def mock_tavily_client() -> MagicMock:
    """Mock Tavily client for web search."""
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com/test",
                    "content": "Test content about AI agents.",
                    "score": 0.95,
                }
            ]
        }
    )
    return client


@pytest.fixture
def sample_scraped_item() -> ScrapedItem:
    """A sample scraped data item."""
    return ScrapedItem(
        url="https://example.com/test",
        title="Test Article",
        content=(
            "AI agents are transforming automation. "
            "Multi-agent systems enable complex workflows."
        ),
        source="tavily",
    )


@pytest.fixture
def sample_insight() -> Insight:
    """A sample analysis insight."""
    return Insight(
        claim="Multi-agent systems reduce processing time by 80%",
        evidence="Based on benchmark data from enterprise deployments",
        confidence=0.85,
        sources=["https://example.com/test"],
    )


@pytest.fixture
def sample_report(sample_insight: Insight) -> Report:
    """A sample research report."""
    return Report(
        title="AI Agents Market Analysis",
        executive_summary="Multi-agent systems are transforming enterprise automation.",
        sections=[
            ReportSection(
                heading="Market Overview",
                content="The AI agent market is projected to reach $47B by 2030.",
                key_points=["$47B market by 2030", "73% enterprise adoption"],
            ),
        ],
        insights=[sample_insight],
        sources=["https://example.com/test"],
        quality_score=0.88,
    )


@pytest.fixture
def empty_research_state() -> ResearchState:
    """An empty initial research state."""
    return ResearchState(
        query="AI agents market analysis 2026",
        plan=[],
        scraped_data=[],
        analysis=[],
        report=None,
        errors=[],
        iteration=0,
        status="planning",
    )


@pytest.fixture
def scraped_research_state(sample_scraped_item: ScrapedItem) -> ResearchState:
    """Research state after scraping is complete."""
    return ResearchState(
        query="AI agents market analysis 2026",
        plan=["Search for market data", "Find enterprise adoption stats"],
        scraped_data=[sample_scraped_item],
        analysis=[],
        report=None,
        errors=[],
        iteration=1,
        status="analyzing",
    )


@pytest.fixture
def analyzed_research_state(
    sample_scraped_item: ScrapedItem, sample_insight: Insight
) -> ResearchState:
    """Research state after analysis is complete."""
    return ResearchState(
        query="AI agents market analysis 2026",
        plan=["Search for market data", "Find enterprise adoption stats"],
        scraped_data=[sample_scraped_item],
        analysis=[sample_insight],
        report=None,
        errors=[],
        iteration=1,
        status="writing",
    )
