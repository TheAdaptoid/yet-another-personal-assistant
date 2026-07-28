"""Fixtures for CLI tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from yapa.services import ModelService


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_config_store():
    with patch("yapa.cli.app.JsonConfigStore") as mock:
        store = MagicMock()
        mock.return_value = store
        yield store


@pytest.fixture
def mock_model_service():
    svc = MagicMock(spec=ModelService)
    svc.list_models = AsyncMock(return_value=[])
    return svc


@pytest.fixture
def mock_session_service():
    return MagicMock()


@pytest.fixture
def mock_store():
    with patch("yapa.cli.app.JsonSessionStore") as mock:
        store = MagicMock()
        mock.return_value = store
        yield store
