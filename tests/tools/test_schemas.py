"""Tests for tool input/output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.tools.schemas import (
    ScrapeRequest,
    ScrapeResponse,
    SearchQuery,
    SearchResponse,
    SearchResult,
)


class TestSearchQuery:
    """Tests for SearchQuery schema."""

    def test_valid_creation(self) -> None:
        query = SearchQuery(query="AI agents")
        assert query.query == "AI agents"
        assert query.max_results == 5

    def test_default_max_results(self) -> None:
        query = SearchQuery(query="test")
        assert query.max_results == 5

    def test_custom_max_results(self) -> None:
        query = SearchQuery(query="test", max_results=10)
        assert query.max_results == 10

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchQuery(query="")

    def test_whitespace_only_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchQuery(query="   ")

    def test_whitespace_stripped(self) -> None:
        query = SearchQuery(query="  AI agents  ")
        assert query.query == "AI agents"

    def test_max_results_too_low(self) -> None:
        with pytest.raises(ValidationError):
            SearchQuery(query="test", max_results=0)

    def test_max_results_too_high(self) -> None:
        with pytest.raises(ValidationError):
            SearchQuery(query="test", max_results=21)


class TestSearchResult:
    """Tests for SearchResult schema."""

    def test_valid_creation(self) -> None:
        result = SearchResult(
            title="Test",
            url="https://example.com",
            content="Some content",
            relevance_score=0.95,
        )
        assert result.title == "Test"
        assert result.relevance_score == 0.95

    def test_score_at_boundaries(self) -> None:
        low = SearchResult(title="T", url="u", content="c", relevance_score=0.0)
        high = SearchResult(title="T", url="u", content="c", relevance_score=1.0)
        assert low.relevance_score == 0.0
        assert high.relevance_score == 1.0

    def test_score_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            SearchResult(title="T", url="u", content="c", relevance_score=1.5)


class TestSearchResponse:
    """Tests for SearchResponse schema."""

    def test_valid_creation(self) -> None:
        result = SearchResult(title="T", url="u", content="c", relevance_score=0.5)
        response = SearchResponse(results=[result], query="test")
        assert len(response.results) == 1
        assert response.query == "test"

    def test_empty_results(self) -> None:
        response = SearchResponse(results=[], query="test")
        assert response.results == []


class TestScrapeRequest:
    """Tests for ScrapeRequest schema."""

    def test_valid_creation(self) -> None:
        req = ScrapeRequest(url="https://example.com")
        assert req.url == "https://example.com"
        assert req.timeout_seconds == 30

    def test_custom_timeout(self) -> None:
        req = ScrapeRequest(url="https://example.com", timeout_seconds=60)
        assert req.timeout_seconds == 60

    def test_timeout_too_low(self) -> None:
        with pytest.raises(ValidationError):
            ScrapeRequest(url="https://example.com", timeout_seconds=2)

    def test_timeout_too_high(self) -> None:
        with pytest.raises(ValidationError):
            ScrapeRequest(url="https://example.com", timeout_seconds=200)


class TestScrapeResponse:
    """Tests for ScrapeResponse schema."""

    def test_successful_scrape(self) -> None:
        resp = ScrapeResponse(
            url="https://example.com",
            title="Test",
            content="Content",
            success=True,
        )
        assert resp.success is True
        assert resp.error is None

    def test_failed_scrape_with_error(self) -> None:
        resp = ScrapeResponse(
            url="https://example.com",
            title="",
            content="",
            success=False,
            error="Connection timeout",
        )
        assert resp.success is False
        assert resp.error == "Connection timeout"
