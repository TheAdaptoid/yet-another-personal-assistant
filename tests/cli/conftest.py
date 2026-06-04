"""Test fixtures for CLI tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from yapa.database import SessionRepository
from yapa.models import AssistantMessage, UserMessage


@pytest.fixture(autouse=True)
def patch_get_engine():
    """Redirect all DB operations to a fresh in-memory SQLite database."""
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(engine)
    with patch("yapa.database.engine.get_engine", return_value=engine):
        yield
    engine.dispose()


@pytest.fixture
def seeded_session():
    """Create a session with one user and one assistant message."""
    session = SessionRepository.create()
    SessionRepository.add_message(session.id, UserMessage(content="hello"))
    SessionRepository.add_message(
        session.id, AssistantMessage(content="hi there", model="test-model")
    )
    return session
