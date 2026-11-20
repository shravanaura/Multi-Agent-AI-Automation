"""Pydantic schemas for tool inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SearchQuery(BaseModel):
    """Input schema for web search queries."""

    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=20)

    @field_validator("query", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class SearchResult(BaseModel):
    """A single search result."""

    title: str
    url: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    """Response containing search results."""

    results: list[SearchResult]
    query: str


class ScrapeRequest(BaseModel):
    """Input schema for web scraping requests."""

    url: str
    timeout_seconds: int = Field(default=30, ge=5, le=120)


class ScrapeResponse(BaseModel):
    """Response from a web scraping operation."""

    url: str
    title: str
    content: str
    success: bool
    error: str | None = None
