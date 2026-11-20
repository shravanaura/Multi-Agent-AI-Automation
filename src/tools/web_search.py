"""Tavily web search tool with Pydantic validation."""

from __future__ import annotations

from typing import Any

from src.tools.schemas import SearchQuery, SearchResponse, SearchResult


async def tavily_search(query: SearchQuery, *, client: Any) -> SearchResponse:
    """Execute a web search using the Tavily API.

    Args:
        query: Search query with parameters.
        client: Tavily client instance (with async .search method).

    Returns:
        SearchResponse with validated results.

    Raises:
        Any exception from the client is propagated.
    """
    raw = await client.search(
        query=query.query,
        max_results=query.max_results,
        search_depth="advanced",
    )

    results = [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            relevance_score=r.get("score", 0.0),
        )
        for r in raw.get("results", [])
    ]

    return SearchResponse(results=results, query=query.query)
