"""Circuit breaker guardrail with fallback support."""

from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted on an open circuit."""


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    reset_timeout: float = 60.0
    half_open_max_calls: int = 1


class CircuitBreaker:
    """Circuit breaker that protects against cascading failures."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._failure_count: int = 0
        self._last_failure_time: float | None = None
        self._state: CircuitState = CircuitState.CLOSED
        self._half_open_calls: int = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, auto-transitioning to HALF_OPEN if timeout elapsed."""
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._config.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        fallback: Callable[..., Coroutine[Any, Any, T]] | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute func through the circuit breaker.

        Args:
            func: Async function to call.
            *args: Positional arguments for func.
            fallback: Optional fallback function called when circuit is open.
            **kwargs: Keyword arguments for func.

        Returns:
            The return value of func or fallback.

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            if fallback is not None:
                return await fallback()
            raise CircuitOpenError("Circuit is open; call rejected.")

        try:
            result = await func(*args, **kwargs)
            if current_state == CircuitState.HALF_OPEN:
                self.reset()
            return result
        except Exception:
            self._record_failure()
            if current_state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.monotonic()
            raise

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        self._half_open_calls = 0

    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._config.failure_threshold:
            self._state = CircuitState.OPEN
