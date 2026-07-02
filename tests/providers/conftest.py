"""Test fixtures for provider tests."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    ModelData,
    ModelType,
    StreamDelta,
)
from yapa.models.message import UserMessage
from yapa.providers.base import InferenceProvider


@pytest.fixture(autouse=True)
def mock_logger() -> Generator[MagicMock, None, None]:
    with patch("yapa.providers.base.get_logger") as mock:
        mock.return_value = MagicMock(spec=logging.Logger)
        yield mock


@pytest.fixture
def mock_model_fetcher() -> MagicMock:
    fetcher = MagicMock()
    fetcher.list_models = AsyncMock(return_value=[])
    fetcher.get_model = AsyncMock()
    return fetcher


@pytest.fixture
def mock_llm_invoker() -> MagicMock:
    invoker = MagicMock()

    async def _stream(
        model_id: str,
        messages: list[Any],
        tools: Any = None,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        yield StreamDelta(content="Hello", reasoning_content=None, done=False)
        yield StreamDelta(content=None, reasoning_content=None, done=True)

    invoker.stream_invoke = _stream
    invoker.static_invoke = AsyncMock(
        return_value=AssistantMessage(content="Hello", role="assistant")
    )
    return invoker


@pytest.fixture
def provider(
    mock_model_fetcher: MagicMock, mock_llm_invoker: MagicMock
) -> InferenceProvider:
    return InferenceProvider(
        identifier="test_prov",
        name="Test Provider",
        model_fetcher=mock_model_fetcher,
        llm_invoker=mock_llm_invoker,
    )


@pytest.fixture
def sample_model() -> ModelData:
    return ModelData(id="gpt-4", provider_id="test_prov", type=ModelType.LLM)


@pytest.fixture
def sample_messages() -> list[UserMessage]:
    return [UserMessage(content="hello")]
