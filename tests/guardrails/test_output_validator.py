"""Tests for output validator guardrail."""

from __future__ import annotations

import json

import pytest

from src.guardrails.output_validator import ValidationError, score_report_quality, validate_output
from src.state import Insight, Report, ReportSection


class TestValidateOutput:
    """Tests for validate_output function."""

    def test_valid_json_parses(self) -> None:
        raw = json.dumps({"claim": "test", "evidence": "data", "confidence": 0.9, "sources": ["a"]})
        result = validate_output(raw, Insight)
        assert result.claim == "test"
        assert result.confidence == 0.9

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_output("not json at all", Insight)

    def test_missing_field_raises(self) -> None:
        raw = json.dumps({"claim": "test"})
        with pytest.raises(ValidationError):
            validate_output(raw, Insight)

    def test_wrong_type_raises(self) -> None:
        raw = json.dumps(
            {
                "claim": "test",
                "evidence": "data",
                "confidence": "not_a_float",
                "sources": ["a"],
            }
        )
        with pytest.raises(ValidationError):
            validate_output(raw, Insight)


class TestScoreReportQuality:
    """Tests for score_report_quality function."""

    def test_high_quality_report_scores_above_half(self, sample_report: Report) -> None:
        score = score_report_quality(sample_report)
        assert score > 0.5

    def test_empty_report_scores_below_third(self) -> None:
        empty_report = Report(
            title="",
            executive_summary="",
            sections=[],
            insights=[],
            sources=[],
            quality_score=0.0,
        )
        score = score_report_quality(empty_report)
        assert score < 0.3

    def test_more_sections_higher_score(self) -> None:
        insight = Insight(claim="c", evidence="e", confidence=0.5, sources=["s"])
        one_section = Report(
            title="A decent title here",
            executive_summary="A summary that is long enough to score well in our quality check.",
            sections=[ReportSection(heading="H", content="C", key_points=["p"])],
            insights=[insight],
            sources=["s1"],
            quality_score=0.5,
        )
        three_sections = Report(
            title="A decent title here",
            executive_summary="A summary that is long enough to score well in our quality check.",
            sections=[
                ReportSection(heading="H1", content="C1", key_points=["p1"]),
                ReportSection(heading="H2", content="C2", key_points=["p2"]),
                ReportSection(heading="H3", content="C3", key_points=["p3"]),
            ],
            insights=[insight],
            sources=["s1"],
            quality_score=0.5,
        )
        assert score_report_quality(three_sections) > score_report_quality(one_section)

    def test_score_in_range(self, sample_report: Report) -> None:
        score = score_report_quality(sample_report)
        assert 0.0 <= score <= 1.0
