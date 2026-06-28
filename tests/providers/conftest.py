"""Test fixtures for provider tests."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import ModelData, ModelType, StreamDelta
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
def mock_model_invoker() -> MagicMock:
    invoker = MagicMock()

    async def _invoke(
        model_id: str, messages: list, params=None
    ) -> AsyncGenerator[StreamDelta, None]:
        yield StreamDelta(content="Hello", reasoning_content=None, done=False)
        yield StreamDelta(content=None, reasoning_content=None, done=True)

    invoker.invoke_llm_stream = _invoke
    return invoker


@pytest.fixture
def provider(
    mock_model_fetcher: MagicMock, mock_model_invoker: MagicMock
) -> InferenceProvider:
    return InferenceProvider(
        identifier="test_prov",
        name="Test Provider",
        model_fetcher=mock_model_fetcher,
        model_invoker=mock_model_invoker,
    )


@pytest.fixture
def sample_model() -> ModelData:
    return ModelData(id="gpt-4", provider_id="test_prov", type=ModelType.LLM)


@pytest.fixture
def sample_messages() -> list[UserMessage]:
    return [UserMessage(content="hello")]
