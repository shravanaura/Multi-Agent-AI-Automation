"""Tests for the Scraper agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.scraper import scraper_node
from src.state import ResearchState, ScrapedItem


@pytest.fixture
def mock_search_client() -> MagicMock:
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "AI Agent Guide",
                    "url": "https://example.com/ai-agents",
                    "content": "Comprehensive guide to building AI agents in production.",
                    "score": 0.92,
                },
                {
                    "title": "LangGraph Tutorial",
                    "url": "https://example.com/langgraph",
                    "content": "Step by step tutorial for LangGraph multi-agent systems.",
                    "score": 0.88,
                },
            ]
        }
    )
    return client


@pytest.fixture
def mock_http_client() -> AsyncMock:
    response = MagicMock()
    response.status_code = 200
    response.text = (
        "<html><head><title>Test</title></head>"
        "<body><p>Scraped content here.</p></body></html>"
    )
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    return client


class TestScraperNode:
    async def test_searches_and_scrapes(
        self,
        empty_research_state: ResearchState,
        mock_search_client: MagicMock,
        mock_http_client: AsyncMock,
    ) -> None:
        state = {**empty_research_state, "plan": ["Search for AI agents"], "status": "scraping"}
        result = await scraper_node(
            state,
            search_client=mock_search_client,
            http_client=mock_http_client,
        )
        assert "scraped_data" in result
        assert len(result["scraped_data"]) > 0
        assert all(isinstance(item, ScrapedItem) for item in result["scraped_data"])

    async def test_sets_status_to_analyzing(
        self,
        empty_research_state: ResearchState,
        mock_search_client: MagicMock,
        mock_http_client: AsyncMock,
    ) -> None:
        state = {**empty_research_state, "plan": ["Search"], "status": "scraping"}
        result = await scraper_node(
            state,
            search_client=mock_search_client,
            http_client=mock_http_client,
        )
        assert result["status"] == "analyzing"

    async def test_handles_search_failure(
        self,
        empty_research_state: ResearchState,
        mock_http_client: AsyncMock,
    ) -> None:
        failing_client = MagicMock()
        failing_client.search = AsyncMock(side_effect=ConnectionError("API down"))
        state = {**empty_research_state, "plan": ["Search"], "status": "scraping"}
        result = await scraper_node(
            state,
            search_client=failing_client,
            http_client=mock_http_client,
        )
        assert len(result.get("errors", [])) > 0
        assert result["errors"][-1].agent == "scraper"

    async def test_appends_to_existing_data(
        self,
        sample_scraped_item: ScrapedItem,
        mock_search_client: MagicMock,
        mock_http_client: AsyncMock,
    ) -> None:
        state: ResearchState = {
            "query": "test", "plan": ["Search more"],
            "scraped_data": [sample_scraped_item], "analysis": [],
            "report": None, "errors": [], "iteration": 1, "status": "scraping",
        }
        result = await scraper_node(
            state,
            search_client=mock_search_client,
            http_client=mock_http_client,
        )
        assert len(result["scraped_data"]) > 1

    async def test_replaces_with_longer_scraped_content(
        self,
        empty_research_state: ResearchState,
    ) -> None:
        """When direct scrape returns longer content than tavily, it replaces the item."""
        search_client = MagicMock()
        search_client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "title": "Short",
                        "url": "https://example.com/a",
                        "content": "Short.",
                        "score": 0.9,
                    },
                ]
            }
        )
        long_body = "A" * 500
        response = MagicMock()
        response.status_code = 200
        response.text = (
            f"<html><head><title>Full</title></head>"
            f"<body><p>{long_body}</p></body></html>"
        )
        http_client = AsyncMock()
        http_client.get = AsyncMock(return_value=response)

        state = {**empty_research_state, "plan": [], "status": "scraping"}
        result = await scraper_node(
            state, search_client=search_client, http_client=http_client,
        )
        assert result["scraped_data"][0].source == "direct_scrape"
        assert result["scraped_data"][0].title == "Full"

    async def test_scrape_failure_keeps_tavily_content(
        self,
        empty_research_state: ResearchState,
    ) -> None:
        """When direct scrape raises, the tavily result is kept."""
        search_client = MagicMock()
        search_client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "title": "Tavily Result",
                        "url": "https://example.com/b",
                        "content": "Tavily content here.",
                        "score": 0.9,
                    },
                ]
            }
        )
        http_client = AsyncMock()
        http_client.get = AsyncMock(side_effect=ConnectionError("timeout"))

        state = {**empty_research_state, "plan": [], "status": "scraping"}
        result = await scraper_node(
            state, search_client=search_client, http_client=http_client,
        )
        assert result["scraped_data"][0].source == "tavily"
        assert result["scraped_data"][0].title == "Tavily Result"

    async def test_scrape_url_exception_is_silently_caught(
        self,
        empty_research_state: ResearchState,
    ) -> None:
        """When scrape_url itself raises unexpectedly, the exception is caught."""
        search_client = MagicMock()
        search_client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com/c",
                        "content": "Content from search.",
                        "score": 0.9,
                    },
                ]
            }
        )
        http_client = AsyncMock()

        state = {**empty_research_state, "plan": [], "status": "scraping"}
        with patch(
            "src.agents.scraper.scrape_url",
            AsyncMock(side_effect=RuntimeError("unexpected")),
        ):
            result = await scraper_node(
                state, search_client=search_client, http_client=http_client,
            )
        # Original tavily item kept despite scrape_url failure
        assert result["scraped_data"][0].source == "tavily"
        assert result["scraped_data"][0].title == "Result"
