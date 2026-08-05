from __future__ import annotations

import json
import logging
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import ModelData, ModelType
from yapa.models.message import UserMessage
from yapa.services.config import Config, ProviderConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict | list:
    """Load a recorded-response fixture by filename."""
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def mock_logger() -> Generator[MagicMock, None, None]:
    with patch("yapa.providers.base.get_logger") as mock:
        mock.return_value = MagicMock(spec=logging.Logger)
        yield mock


@pytest.fixture
def sample_config() -> Config:
    return Config(
        provider_configs={
            "openai": ProviderConfig(api_key="sk-test"),
            "lmstudio": ProviderConfig(api_key="test-key"),
            "ollama": ProviderConfig(api_key="test-key"),
            "openrouter": ProviderConfig(api_key="sk-or-test"),
        },
    )


@pytest.fixture
def mock_openai_client() -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.models.list = AsyncMock()
    client.models.retrieve = AsyncMock()
    return client


@pytest.fixture
def sample_llm_model() -> ModelData:
    return ModelData(id="gpt-4", provider_id="test", type=ModelType.LLM)


@pytest.fixture
def sample_other_model() -> ModelData:
    return ModelData(id="embed-3", provider_id="test", type=ModelType.OTHER)


@pytest.fixture
def sample_messages() -> list[UserMessage]:
    return [UserMessage(content="hello")]
