"""Tests for the FastAPI server."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.state import Insight, Report, ReportSection


@pytest.fixture
def mock_graph() -> MagicMock:
    report = Report(
        title="Test Report",
        executive_summary="Test summary with enough content here.",
        sections=[
            ReportSection(heading="S1", content="Content", key_points=["P1", "P2"]),
        ],
        insights=[
            Insight(claim="C", evidence="E", confidence=0.9, sources=["url"]),
        ],
        sources=["url1"],
        quality_score=0.8,
    )
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "query": "test",
            "plan": ["task1"],
            "scraped_data": [],
            "analysis": [],
            "report": report,
            "errors": [],
            "iteration": 3,
            "status": "done",
        }
    )
    return graph


@pytest.fixture
def client(mock_graph: MagicMock) -> TestClient:
    with patch("src.server._get_graph", return_value=mock_graph):
        from src.server import app

        with TestClient(app) as tc:
            yield tc


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestResearchEndpoint:
    def test_successful_research(self, client: TestClient) -> None:
        response = client.post("/research", json={"query": "AI agents 2026"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "done"
        assert data["report"]["title"] == "Test Report"

    def test_empty_query_rejected(self, client: TestClient) -> None:
        response = client.post("/research", json={"query": ""})
        assert response.status_code == 422

    def test_missing_query_rejected(self, client: TestClient) -> None:
        response = client.post("/research", json={})
        assert response.status_code == 422


class TestGetGraph:
    def test_get_graph_creates_graph_with_real_deps(self) -> None:
        """Test _get_graph wires up real dependencies (with mocked imports)."""
        mock_graph = MagicMock()

        with (
            patch.dict(
                "os.environ",
                {"ANTHROPIC_API_KEY": "test-key", "TAVILY_API_KEY": "test-tavily"},
            ),
            patch("src.server.build_research_graph", return_value=mock_graph),
        ):
            from src.server import _get_graph

            with (
                patch("src.server.build_research_graph", return_value=mock_graph),
            ):
                # We need to mock the lazy imports inside _get_graph
                # Patch them as builtins that _get_graph will import
                mock_anthropic_mod = MagicMock()
                mock_tavily_mod = MagicMock()
                mock_httpx_mod = MagicMock()

                import builtins

                original_import = builtins.__import__

                def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
                    if name == "langchain_anthropic":
                        return mock_anthropic_mod
                    if name == "tavily":
                        return mock_tavily_mod
                    if name == "httpx":
                        return mock_httpx_mod
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = _get_graph()
                assert result is mock_graph
