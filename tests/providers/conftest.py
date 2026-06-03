from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from yapa.models.message import UserMessage
from yapa.providers.base import InferenceProvider


@pytest.fixture(autouse=True)
def mock_logger() -> MagicMock:
    with patch("yapa.providers.base.get_logger") as mock:
        mock.return_value = MagicMock(spec=logging.Logger)
        yield mock


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def provider(mock_client: MagicMock) -> InferenceProvider:
    return InferenceProvider(identifier="test_prov", name="Test", client=mock_client)


@pytest.fixture
def sample_messages() -> list[UserMessage]:
    return [UserMessage(content="hello")]
