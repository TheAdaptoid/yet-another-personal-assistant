"""AsyncOpenAI client construction honouring optional auth."""

import httpx
from openai import AsyncOpenAI

_SENTINEL_KEY = "no-key-provider"


class _StripAuthClient(httpx.AsyncClient):
    """httpx client that strips the Authorization header (no-auth providers)."""

    async def send(self, request, **kwargs):
        request.headers.pop("Authorization", None)
        return await super().send(request, **kwargs)


def build_openai_client(
    api_key: str | None,
    base_url: str | None,
    timeout: int,
    max_retries: int,
) -> AsyncOpenAI:
    key = (api_key or "").strip() or None
    if key is not None:
        return AsyncOpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    return AsyncOpenAI(
        api_key=_SENTINEL_KEY,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        http_client=_StripAuthClient(),
    )
