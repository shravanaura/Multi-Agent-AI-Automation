"""Shared state schema for the multi-agent research system."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


class ScrapedItem(BaseModel):
    """A piece of scraped web content."""

    url: str
    title: str
    content: str = Field(max_length=10_000)
    source: Literal["tavily", "direct_scrape"]

    @field_validator("content", mode="before")
    @classmethod
    def truncate_content(cls, v: str) -> str:
        return v[:10_000] if len(v) > 10_000 else v


class Insight(BaseModel):
    """An extracted insight from analyzed data."""

    claim: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]


class ReportSection(BaseModel):
    """A section of the final research report."""

    heading: str
    content: str
    key_points: list[str]


class Report(BaseModel):
    """The final structured research report."""

    title: str
    executive_summary: str
    sections: list[ReportSection]
    insights: list[Insight]
    sources: list[str]
    quality_score: float = Field(ge=0.0, le=1.0)


class ErrorRecord(BaseModel):
    """A record of an error encountered during research."""

    agent: str
    error_type: str
    message: str
    recoverable: bool


class ResearchState(TypedDict):
    """LangGraph state schema for the research pipeline."""

    query: str
    plan: list[str]
    scraped_data: list[ScrapedItem]
    analysis: list[Insight]
    report: Report | None
    errors: list[ErrorRecord]
    iteration: int
    status: Literal["planning", "scraping", "analyzing", "writing", "review", "done"]
