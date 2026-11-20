"""Output validator with report quality scoring."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from src.state import Report

T = TypeVar("T", bound=BaseModel)


class ValidationError(Exception):
    """Raised when output validation fails."""


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM output."""
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


def validate_output(raw: str, model: type[T]) -> T:
    """Parse a JSON string and validate it against a Pydantic model.

    Args:
        raw: Raw JSON string.
        model: Pydantic model class to validate against.

    Returns:
        Validated model instance.

    Raises:
        ValidationError: If JSON is invalid or validation fails.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON: {exc}") from exc

    try:
        return model.model_validate(data)
    except PydanticValidationError as exc:
        raise ValidationError(f"Validation failed: {exc}") from exc


def score_report_quality(report: Report) -> float:
    """Score a report's quality on a 0.0-1.0 scale.

    Scoring weights:
        - title length: 10%
        - executive_summary length: 20%
        - sections count: 25%
        - insights count: 20%
        - sources count: 15%
        - key_points coverage: 10%

    Args:
        report: The report to score.

    Returns:
        Quality score between 0.0 and 1.0.
    """
    score = 0.0

    # Title length (10%) - good title is 10+ chars
    title_score = min(len(report.title) / 20.0, 1.0)
    score += title_score * 0.10

    # Executive summary length (20%) - good summary is 50+ chars
    summary_score = min(len(report.executive_summary) / 100.0, 1.0)
    score += summary_score * 0.20

    # Sections count (25%) - target 3+ sections
    sections_score = min(len(report.sections) / 3.0, 1.0)
    score += sections_score * 0.25

    # Insights count (20%) - target 3+ insights
    insights_score = min(len(report.insights) / 3.0, 1.0)
    score += insights_score * 0.20

    # Sources count (15%) - target 3+ sources
    sources_score = min(len(report.sources) / 3.0, 1.0)
    score += sources_score * 0.15

    # Key points coverage (10%) - percentage of sections with key points
    if report.sections:
        sections_with_points = sum(1 for s in report.sections if s.key_points)
        key_points_score = sections_with_points / len(report.sections)
    else:
        key_points_score = 0.0
    score += key_points_score * 0.10

    return min(score, 1.0)
