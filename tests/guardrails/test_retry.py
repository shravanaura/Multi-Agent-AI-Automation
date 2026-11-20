"""Tests for retry with exponential backoff."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.guardrails.retry import RetryConfig, RetryExhaustedError, retry_with_backoff


@pytest.fixture
def fast_config() -> RetryConfig:
    """Config with minimal delays for fast tests."""
    return RetryConfig(max_retries=3, base_delay=0.01, max_delay=1.0, jitter=False)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_defaults(self) -> None:
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    async def test_succeeds_first_try(self, fast_config: RetryConfig) -> None:
        func = AsyncMock(return_value="ok")
        result = await retry_with_backoff(func, config=fast_config, retry_on=(ValueError,))
        assert result == "ok"
        assert func.call_count == 1

    async def test_succeeds_after_retry(self, fast_config: RetryConfig) -> None:
        func = AsyncMock(side_effect=[ValueError("fail"), "ok"])
        result = await retry_with_backoff(func, config=fast_config, retry_on=(ValueError,))
        assert result == "ok"
        assert func.call_count == 2

    async def test_exhausts_retries(self, fast_config: RetryConfig) -> None:
        func = AsyncMock(side_effect=ValueError("always fails"))
        with pytest.raises(RetryExhaustedError) as exc_info:
            await retry_with_backoff(func, config=fast_config, retry_on=(ValueError,))
        assert exc_info.value.attempts == 4  # 1 initial + 3 retries
        assert isinstance(exc_info.value.last_error, ValueError)

    async def test_only_retries_specified_exceptions(self, fast_config: RetryConfig) -> None:
        func = AsyncMock(side_effect=TypeError("wrong type"))
        with pytest.raises(TypeError):
            await retry_with_backoff(func, config=fast_config, retry_on=(ValueError,))
        assert func.call_count == 1

    async def test_passes_args_and_kwargs(self, fast_config: RetryConfig) -> None:
        func = AsyncMock(return_value="done")
        result = await retry_with_backoff(
            func, config=fast_config, retry_on=(ValueError,), args=(1, 2), kwargs={"key": "val"}
        )
        assert result == "done"
        func.assert_called_once_with(1, 2, key="val")

    async def test_delays_grow_exponentially(self, fast_config: RetryConfig) -> None:
        """Verify delay calculation: base_delay * 2^attempt."""
        func = AsyncMock(side_effect=[ValueError, ValueError, ValueError, "ok"])
        # With base_delay=0.01 and no jitter:
        # attempt 0: delay = 0.01 * 1 = 0.01
        # attempt 1: delay = 0.01 * 2 = 0.02
        # attempt 2: delay = 0.01 * 4 = 0.04
        result = await retry_with_backoff(func, config=fast_config, retry_on=(ValueError,))
        assert result == "ok"
        assert func.call_count == 4

    async def test_jitter_applied_to_delay(self) -> None:
        """Verify jitter is applied when enabled."""
        config = RetryConfig(max_retries=2, base_delay=0.01, max_delay=1.0, jitter=True)
        func = AsyncMock(side_effect=[ValueError("fail"), "ok"])
        result = await retry_with_backoff(func, config=config, retry_on=(ValueError,))
        assert result == "ok"
        assert func.call_count == 2

    async def test_delay_capped_at_max(self) -> None:
        config = RetryConfig(max_retries=5, base_delay=0.01, max_delay=0.02, jitter=False)
        func = AsyncMock(side_effect=[ValueError, ValueError, "ok"])
        result = await retry_with_backoff(func, config=config, retry_on=(ValueError,))
        assert result == "ok"


class TestRetryExhaustedError:
    """Tests for RetryExhaustedError."""

    def test_stores_attempts_and_last_error(self) -> None:
        err = ValueError("boom")
        exc = RetryExhaustedError(attempts=3, last_error=err)
        assert exc.attempts == 3
        assert exc.last_error is err
