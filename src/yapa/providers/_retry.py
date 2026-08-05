"""Minimal async retry wrapper for providers (REQ-PROV-26)."""


async def retry_async(coro_factory, max_attempts: int, retryable):
    """
    Call ``coro_factory()`` up to ``max_attempts`` times.

    Only retryable failures are retried.

    Args:
        coro_factory: A zero-arg callable returning an awaitable.
        max_attempts: The maximum number of attempts to make.
        retryable: A predicate deciding whether a raised exception can be retried.

    Returns:
        The result of the first successful call.

    Raises:
        The last exception raised if all attempts fail or the failure is not
        retryable.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not retryable(exc) or attempt == max_attempts - 1:
                raise
    assert last_exc is not None
    raise last_exc
