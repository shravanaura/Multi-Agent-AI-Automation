"""Tests for web scrape tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.schemas import ScrapeRequest, ScrapeResponse
from src.tools.web_scrape import _extract_text, _extract_title, scrape_url


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Mock httpx async client."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.text = (
        "<html><head><title>Test Page</title></head>"
        "<body><p>Hello world</p></body></html>"
    )
    client.get = AsyncMock(return_value=response)
    return client


class TestScrapeUrl:
    """Tests for scrape_url function."""

    async def test_successful_scrape(self, mock_http_client: MagicMock) -> None:
        request = ScrapeRequest(url="https://example.com")
        result = await scrape_url(request, http_client=mock_http_client)
        assert isinstance(result, ScrapeResponse)
        assert result.success is True
        assert result.title == "Test Page"
        assert "Hello world" in result.content
        assert result.error is None

    async def test_failed_scrape_returns_error(self, mock_http_client: MagicMock) -> None:
        mock_http_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        request = ScrapeRequest(url="https://example.com")
        result = await scrape_url(request, http_client=mock_http_client)
        assert result.success is False
        assert result.error is not None
        assert "Connection refused" in result.error

    async def test_non_200_returns_error(self, mock_http_client: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 404
        response.text = "Not found"
        mock_http_client.get = AsyncMock(return_value=response)
        request = ScrapeRequest(url="https://example.com/missing")
        result = await scrape_url(request, http_client=mock_http_client)
        assert result.success is False
        assert result.error is not None

    async def test_truncates_long_content(self, mock_http_client: MagicMock) -> None:
        long_text = "a" * 20_000
        response = MagicMock()
        response.status_code = 200
        response.text = (
            f"<html><head><title>Big</title></head>"
            f"<body><p>{long_text}</p></body></html>"
        )
        mock_http_client.get = AsyncMock(return_value=response)
        request = ScrapeRequest(url="https://example.com")
        result = await scrape_url(request, http_client=mock_http_client)
        assert len(result.content) <= 10_000


class TestExtractTitle:
    """Tests for _extract_title helper."""

    def test_extracts_title(self) -> None:
        assert _extract_title("<title>My Page</title>") == "My Page"

    def test_no_title_returns_empty(self) -> None:
        assert _extract_title("<html><body>No title</body></html>") == ""


class TestExtractText:
    """Tests for _extract_text helper."""

    def test_strips_html_tags(self) -> None:
        html = "<p>Hello <b>world</b></p>"
        text = _extract_text(html)
        assert "Hello" in text
        assert "world" in text
        assert "<" not in text

    def test_removes_script_and_style(self) -> None:
        html = "<script>var x=1;</script><style>.a{}</style><p>Content</p>"
        text = _extract_text(html)
        assert "var x" not in text
        assert "Content" in text

    def test_collapses_whitespace(self) -> None:
        html = "<p>Hello     world</p>"
        text = _extract_text(html)
        assert "Hello world" in text

    def test_truncates_to_10k(self) -> None:
        html = "<p>" + "x" * 20_000 + "</p>"
        text = _extract_text(html)
        assert len(text) <= 10_000
