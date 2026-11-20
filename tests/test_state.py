"""Tests for the shared research state schema."""

from __future__ import annotations

from src.state import (
    ErrorRecord,
    Insight,
    Report,
    ReportSection,
    ResearchState,
    ScrapedItem,
)


class TestScrapedItem:
    def test_create_valid(self) -> None:
        item = ScrapedItem(
            url="https://example.com",
            title="Test",
            content="Some content",
            source="tavily",
        )
        assert item.url == "https://example.com"
        assert item.source == "tavily"

    def test_content_truncation(self) -> None:
        long_content = "x" * 15_000
        item = ScrapedItem(
            url="https://example.com",
            title="Test",
            content=long_content,
            source="tavily",
        )
        assert len(item.content) <= 10_000

    def test_invalid_source_rejected(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            ScrapedItem(
                url="https://example.com",
                title="Test",
                content="content",
                source="invalid_source",
            )


class TestInsight:
    def test_create_valid(self) -> None:
        insight = Insight(
            claim="AI is growing",
            evidence="Market reports show 40% YoY growth",
            confidence=0.9,
            sources=["https://example.com"],
        )
        assert insight.confidence == 0.9

    def test_confidence_clamped_to_range(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            Insight(
                claim="Test",
                evidence="Test",
                confidence=1.5,
                sources=[],
            )

    def test_confidence_below_zero_rejected(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            Insight(
                claim="Test",
                evidence="Test",
                confidence=-0.1,
                sources=[],
            )


class TestReportSection:
    def test_create_valid(self) -> None:
        section = ReportSection(
            heading="Overview",
            content="Market is growing.",
            key_points=["Point 1", "Point 2"],
        )
        assert section.heading == "Overview"
        assert len(section.key_points) == 2


class TestReport:
    def test_create_valid(self) -> None:
        report = Report(
            title="Test Report",
            executive_summary="Summary here.",
            sections=[
                ReportSection(
                    heading="S1",
                    content="Content",
                    key_points=["P1"],
                )
            ],
            insights=[Insight(claim="C", evidence="E", confidence=0.8, sources=["url"])],
            sources=["https://example.com"],
            quality_score=0.85,
        )
        assert report.quality_score == 0.85

    def test_quality_score_range(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            Report(
                title="T",
                executive_summary="S",
                sections=[],
                insights=[],
                sources=[],
                quality_score=1.5,
            )


class TestErrorRecord:
    def test_create_valid(self) -> None:
        error = ErrorRecord(
            agent="scraper",
            error_type="TimeoutError",
            message="Request timed out after 30s",
            recoverable=True,
        )
        assert error.recoverable is True


class TestResearchState:
    def test_initial_state(self) -> None:
        state = ResearchState(
            query="AI agents market analysis 2026",
            plan=[],
            scraped_data=[],
            analysis=[],
            report=None,
            errors=[],
            iteration=0,
            status="planning",
        )
        assert state["status"] == "planning"
        assert state["iteration"] == 0
        assert state["report"] is None

    def test_state_is_typed_dict(self) -> None:
        """ResearchState must be a TypedDict for LangGraph compatibility."""
        assert hasattr(ResearchState, "__annotations__")
        assert hasattr(ResearchState, "__required_keys__") or hasattr(
            ResearchState, "__optional_keys__"
        )
