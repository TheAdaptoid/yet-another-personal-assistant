"""Test fixtures for service-layer tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from yapa.config import Config
from yapa.database.repositories import SessionRepository
from yapa.models import StreamDelta


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
    return Config(default_model_id="test-default-model", default_provider_id="test")


@pytest.fixture
def mock_provider():
    provider = MagicMock()

    async def _invoke(model, messages):
        yield StreamDelta(content="Hi!", done=False)
        yield StreamDelta(content=None, done=True)

    provider.invoke_model = _invoke
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def mock_provider_service(mock_provider):
    ps = MagicMock()
    ps.get_provider_by_model.return_value = mock_provider
    return ps
