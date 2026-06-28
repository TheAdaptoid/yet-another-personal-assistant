"""Test fixtures for service-layer tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from yapa.config import Config
from yapa.models import ModelData, ModelType, Session, StreamDelta
from yapa.storage import GenericStore


@pytest.fixture
def config(tmp_path):
    return Config(default_model="test:test-default-model", storage_dir=tmp_path)


@pytest.fixture
def store(config):
    return GenericStore[Session](
        storage_dir=config.storage_dir / "sessions",
        entity_type=Session,
    )


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
