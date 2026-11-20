"""Tests for circuit breaker guardrail."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.guardrails.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)


@pytest.fixture
def fast_breaker() -> CircuitBreaker:
    """Circuit breaker with fast reset for testing."""
    config = CircuitBreakerConfig(failure_threshold=3, reset_timeout=0.01, half_open_max_calls=1)
    return CircuitBreaker(config)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_states_exist(self) -> None:
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self, fast_breaker: CircuitBreaker) -> None:
        assert fast_breaker.state == CircuitState.CLOSED

    async def test_passes_through_when_closed(self, fast_breaker: CircuitBreaker) -> None:
        func = AsyncMock(return_value="ok")
        result = await fast_breaker.call(func)
        assert result == "ok"
        func.assert_called_once()

    async def test_opens_after_threshold(self, fast_breaker: CircuitBreaker) -> None:
        func = AsyncMock(side_effect=ValueError("fail"))
        for _ in range(3):
            with pytest.raises(ValueError):
                await fast_breaker.call(func)
        assert fast_breaker.state == CircuitState.OPEN

    async def test_rejects_when_open(self, fast_breaker: CircuitBreaker) -> None:
        func = AsyncMock(side_effect=ValueError("fail"))
        for _ in range(3):
            with pytest.raises(ValueError):
                await fast_breaker.call(func)

        with pytest.raises(CircuitOpenError):
            await fast_breaker.call(AsyncMock())

    async def test_fallback_called_when_open(self, fast_breaker: CircuitBreaker) -> None:
        func = AsyncMock(side_effect=ValueError("fail"))
        for _ in range(3):
            with pytest.raises(ValueError):
                await fast_breaker.call(func)

        fallback = AsyncMock(return_value="fallback_value")
        result = await fast_breaker.call(AsyncMock(), fallback=fallback)
        assert result == "fallback_value"

    async def test_half_open_after_timeout(self, fast_breaker: CircuitBreaker) -> None:
        func = AsyncMock(side_effect=ValueError("fail"))
        for _ in range(3):
            with pytest.raises(ValueError):
                await fast_breaker.call(func)

        await asyncio.sleep(0.02)
        assert fast_breaker.state == CircuitState.HALF_OPEN

    async def test_half_open_success_closes(self, fast_breaker: CircuitBreaker) -> None:
        fail_func = AsyncMock(side_effect=ValueError("fail"))
        for _ in range(3):
            with pytest.raises(ValueError):
                await fast_breaker.call(fail_func)

        await asyncio.sleep(0.02)
        success_func = AsyncMock(return_value="recovered")
        result = await fast_breaker.call(success_func)
        assert result == "recovered"
        assert fast_breaker.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens(self, fast_breaker: CircuitBreaker) -> None:
        fail_func = AsyncMock(side_effect=ValueError("fail"))
        for _ in range(3):
            with pytest.raises(ValueError):
                await fast_breaker.call(fail_func)

        await asyncio.sleep(0.02)
        with pytest.raises(ValueError):
            await fast_breaker.call(fail_func)
        assert fast_breaker.state == CircuitState.OPEN

    def test_reset(self, fast_breaker: CircuitBreaker) -> None:
        # Manually record failures
        for _ in range(3):
            fast_breaker._record_failure()
        assert fast_breaker.state == CircuitState.OPEN
        fast_breaker.reset()
        assert fast_breaker.state == CircuitState.CLOSED

    async def test_passes_args(self, fast_breaker: CircuitBreaker) -> None:
        func = AsyncMock(return_value="done")
        result = await fast_breaker.call(func, 1, 2, key="val")
        assert result == "done"
        func.assert_called_once_with(1, 2, key="val")
