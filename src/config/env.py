"""Environment configuration with validation."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    anthropic_api_key: str
    tavily_api_key: str
    openai_api_key: str | None
    port: int
    log_level: str


def load_config() -> Config:
    """Load and validate configuration from environment variables.

    Raises:
        ValueError: If required environment variables are missing.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")

    return Config(
        anthropic_api_key=anthropic_key,
        tavily_api_key=tavily_key,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        port=int(os.environ.get("PORT", "8005")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
