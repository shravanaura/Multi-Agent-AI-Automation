"""Scraper agent — web search and content extraction."""

from __future__ import annotations

from typing import Any

from src.state import ErrorRecord, ResearchState, ScrapedItem
from src.tools.schemas import ScrapeRequest, SearchQuery
from src.tools.web_scrape import scrape_url
from src.tools.web_search import tavily_search


async def scraper_node(
    state: ResearchState,
    *,
    search_client: Any,
    http_client: Any,
) -> dict[str, Any]:
    """Search the web and scrape content for the research query.

    Args:
        state: Current research state.
        search_client: Tavily search client.
        http_client: HTTP client for direct scraping.

    Returns:
        State updates with scraped data and status.
    """
    query_text = state["query"]
    if state["plan"]:
        query_text = f"{state['query']} {state['plan'][0]}"

    try:
        search_query = SearchQuery(query=query_text, max_results=5)
        search_response = await tavily_search(search_query, client=search_client)
    except Exception as e:
        return {
            "errors": state["errors"]
            + [
                ErrorRecord(
                    agent="scraper",
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=True,
                )
            ],
        }

    new_items: list[ScrapedItem] = []

    for result in search_response.results:
        new_items.append(
            ScrapedItem(
                url=result.url,
                title=result.title,
                content=result.content,
                source="tavily",
            )
        )

        try:
            scrape_req = ScrapeRequest(url=result.url, timeout_seconds=15)
            scrape_resp = await scrape_url(scrape_req, http_client=http_client)
            if scrape_resp.success and len(scrape_resp.content) > len(result.content):
                new_items[-1] = ScrapedItem(
                    url=result.url,
                    title=scrape_resp.title or result.title,
                    content=scrape_resp.content,
                    source="direct_scrape",
                )
        except Exception:
            pass

    return {
        "scraped_data": list(state["scraped_data"]) + new_items,
        "status": "analyzing",
    }
