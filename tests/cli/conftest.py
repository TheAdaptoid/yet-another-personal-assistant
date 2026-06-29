"""Test fixtures for CLI tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from yapa.models import AssistantMessage, UserMessage
from yapa.services import SessionService


@pytest.fixture(autouse=True)
def patch_cli_session_service(tmp_path):
    """Redirect _get_session_service() to use tmp_path storage."""
    svc = SessionService(storage_dir=tmp_path)
    with patch("yapa.cli.sessions._get_session_service", return_value=svc):
        yield


@pytest.fixture(autouse=True)
def patch_save_config():
    """Prevent chat tests from writing to the real config file."""
    with patch("yapa.cli.chat.save_config"):
        yield


@pytest.fixture
def seeded_session(tmp_path):
    """Create a session with one user and one assistant message."""
    svc = SessionService(storage_dir=tmp_path)
    session = svc.create(title="Test Session")
    session.messages = [
        UserMessage(content="hello"),
        AssistantMessage(content="hi there", model="test-model"),
    ]
    svc._store.save(session, overwrite=True)
    return session
