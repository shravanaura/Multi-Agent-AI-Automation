"""Tests for Tavily web search tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.schemas import SearchQuery, SearchResponse
from src.tools.web_search import tavily_search


class TestTavilySearch:
    """Tests for tavily_search function."""

    async def test_returns_search_response(self, mock_tavily_client: MagicMock) -> None:
        query = SearchQuery(query="AI agents")
        result = await tavily_search(query, client=mock_tavily_client)
        assert isinstance(result, SearchResponse)
        assert result.query == "AI agents"
        assert len(result.results) == 1
        assert result.results[0].title == "Test Result"
        assert result.results[0].relevance_score == 0.95

    async def test_handles_empty_results(self, mock_tavily_client: MagicMock) -> None:
        mock_tavily_client.search = AsyncMock(return_value={"results": []})
        query = SearchQuery(query="nonexistent topic")
        result = await tavily_search(query, client=mock_tavily_client)
        assert result.results == []

    async def test_passes_max_results(self, mock_tavily_client: MagicMock) -> None:
        query = SearchQuery(query="test", max_results=10)
        await tavily_search(query, client=mock_tavily_client)
        mock_tavily_client.search.assert_called_once_with(
            query="test", max_results=10, search_depth="advanced"
        )

    async def test_handles_missing_fields(self, mock_tavily_client: MagicMock) -> None:
        mock_tavily_client.search = AsyncMock(
            return_value={
                "results": [
                    {"title": "Partial", "url": "https://example.com", "content": "content"}
                ]
            }
        )
        query = SearchQuery(query="test")
        result = await tavily_search(query, client=mock_tavily_client)
        assert len(result.results) == 1
        assert result.results[0].relevance_score == 0.0

    async def test_propagates_client_errors(self, mock_tavily_client: MagicMock) -> None:
        mock_tavily_client.search = AsyncMock(side_effect=RuntimeError("API down"))
        query = SearchQuery(query="test")
        with pytest.raises(RuntimeError, match="API down"):
            await tavily_search(query, client=mock_tavily_client)
