"""Test fixtures for service-layer tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from yapa.config import Config
from yapa.database.repositories import SessionRepository
from yapa.models import ModelData, ModelType, StreamDelta


@pytest.fixture
def engine():
    e = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(e)
    yield e
    e.dispose()


@pytest.fixture
def repo(engine):
    return SessionRepository(engine=engine)


@pytest.fixture
def config():
    return Config(default_model="test:test-default-model")


@pytest.fixture
def mock_provider():
    provider = MagicMock()

    async def _invoke(model, messages, params=None):
        yield StreamDelta(content="Hi!", reasoning_content=None, done=False)
        yield StreamDelta(content=None, reasoning_content=None, done=True)

    provider.invoke_llm = _invoke
    provider.get_model = AsyncMock()
    return provider


@pytest.fixture
def mock_provider_service(mock_provider):
    ps = MagicMock()
    ps.get_provider_by_model.return_value = mock_provider
    ps.get_provider_by_model_full_id.return_value = mock_provider
    ps.get_model = AsyncMock(
        return_value=ModelData(
            id="test-default-model", provider_id="test", type=ModelType.LLM
        )
    )
    return ps
