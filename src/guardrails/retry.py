"""Retry with exponential backoff guardrail."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Exhausted {attempts} attempts. Last error: {last_error}")


async def retry_with_backoff(
    func: Callable[..., Coroutine[Any, Any, T]],
    *,
    config: RetryConfig | None = None,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
) -> T:
    """Execute an async function with exponential backoff retry.

    Args:
        func: Async function to call.
        config: Retry configuration.
        retry_on: Tuple of exception types to retry on.
        args: Positional arguments to pass to func.
        kwargs: Keyword arguments to pass to func.

    Returns:
        The return value of func.

    Raises:
        RetryExhaustedError: If all retries are exhausted.
        Exception: If a non-retryable exception is raised.
    """
    cfg = config or RetryConfig()
    kw = kwargs or {}
    last_error: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            return await func(*args, **kw)
        except retry_on as exc:
            last_error = exc
            if attempt < cfg.max_retries:
                delay = min(cfg.base_delay * (2**attempt), cfg.max_delay)
                if cfg.jitter:
                    delay = delay * random.uniform(0.5, 1.5)
                await asyncio.sleep(delay)

    raise RetryExhaustedError(attempts=cfg.max_retries + 1, last_error=last_error)  # type: ignore[arg-type]
