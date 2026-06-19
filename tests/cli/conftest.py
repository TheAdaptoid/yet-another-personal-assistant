"""Test fixtures for CLI tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from yapa.database.repositories import SessionRepository
from yapa.models import AssistantMessage, UserMessage

_session_repo = SessionRepository()


@pytest.fixture(autouse=True)
def patch_get_engine():
    """Redirect all DB operations to a fresh in-memory SQLite database."""
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(engine)
    with patch("yapa.database.engine.get_engine", return_value=engine):
        yield
    engine.dispose()


@pytest.fixture(autouse=True)
def patch_save_config():
    """Prevent chat tests from writing to the real config file."""
    with patch("yapa.cli.chat.save_config"):
        yield


@pytest.fixture
def seeded_session():
    """Create a session with one user and one assistant message."""
    session = _session_repo.create()
    _session_repo.add_message(session.id, UserMessage(content="hello"))
    _session_repo.add_message(
        session.id, AssistantMessage(content="hi there", model="test-model")
    )
    return session
