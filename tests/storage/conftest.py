"""Test fixtures for storage module tests."""

from __future__ import annotations

import pytest

from yapa.models import Session
from yapa.storage import GenericStore


@pytest.fixture
def store_dir(tmp_path):
    """Isolated temp directory for entity storage."""
    return tmp_path / "entities"


@pytest.fixture
def store(store_dir):
    """Return a GenericStore backed by the temp directory."""
    return GenericStore(store_dir, Session)


@pytest.fixture
def make_session(store):
    """Return a factory for building and persisting Session entities."""

    def _make(*, title: str = "Test Session", messages=None):
        session = Session(title=title, messages=messages or [])
        store.save(session)
        return session

    return _make
