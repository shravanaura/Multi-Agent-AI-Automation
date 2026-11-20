"""Web scrape tool with HTML text extraction."""

from __future__ import annotations

import re
from typing import Any

from src.tools.schemas import ScrapeRequest, ScrapeResponse

_MAX_CONTENT_LENGTH = 10_000


async def scrape_url(request: ScrapeRequest, *, http_client: Any) -> ScrapeResponse:
    """Scrape a URL and extract text content.

    Args:
        request: Scrape request with URL and timeout.
        http_client: An httpx-compatible async client.

    Returns:
        ScrapeResponse with extracted content or error details.
    """
    try:
        response = await http_client.get(request.url, timeout=request.timeout_seconds)
    except Exception as exc:
        return ScrapeResponse(
            url=request.url,
            title="",
            content="",
            success=False,
            error=str(exc),
        )

    if response.status_code != 200:
        return ScrapeResponse(
            url=request.url,
            title="",
            content="",
            success=False,
            error=f"HTTP {response.status_code}",
        )

    html = response.text
    title = _extract_title(html)
    content = _extract_text(html)

    return ScrapeResponse(
        url=request.url,
        title=title,
        content=content,
        success=True,
    )


def _extract_title(html: str) -> str:
    """Extract title from HTML title tag.

    Args:
        html: Raw HTML string.

    Returns:
        Title text or empty string if not found.
    """
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_text(html: str) -> str:
    """Extract readable text from HTML.

    Removes script/style elements, strips tags, collapses whitespace,
    and truncates to 10k characters.

    Args:
        html: Raw HTML string.

    Returns:
        Clean text content, truncated to 10k chars.
    """
    # Remove script and style elements
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate
    return text[:_MAX_CONTENT_LENGTH]
