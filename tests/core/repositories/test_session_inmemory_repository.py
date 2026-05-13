"""Tests for the in-memory session repository."""

import logging
import pytest

from yapa.core.repositories import SessionInMemoryRepository
from yapa.shared import Config
from yapa.shared.models import Session


@pytest.fixture
def dummy_logger():
    """Return a mock logger with the methods used in the repo."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.NullHandler()
        logger.addHandler(handler)
    return logger


@pytest.fixture
def test_config() -> Config:
    """Config object (value not used by in-memory repo but required)."""
    return Config()  # defaults are fine


@pytest.fixture
def inmemory_repo(test_config, dummy_logger):
    return SessionInMemoryRepository(test_config, dummy_logger)


@pytest.mark.asyncio
async def test_initial_state_empty(inmemory_repo):
    assert await inmemory_repo.load_all() == []


@pytest.mark.asyncio
async def test_save_and_load(inmemory_repo):
    session = Session()
    await inmemory_repo.save(session)

    loaded = await inmemory_repo.load(session.id)
    assert loaded is not None
    assert loaded.id == session.id
    assert loaded.title == session.title


@pytest.mark.asyncio
async def test_load_all_order_preserved(inmemory_repo):
    s1 = Session()
    s2 = Session()
    s3 = Session()
    await inmemory_repo.save(s1)
    await inmemory_repo.save(s2)
    await inmemory_repo.save(s3)

    all_sessions = await inmemory_repo.load_all()
    # Preserve insertion order
    assert [s.id for s in all_sessions] == [s1.id, s2.id, s3.id]


@pytest.mark.asyncio
async def test_delete_existing(inmemory_repo):
    session = Session()
    await inmemory_repo.save(session)

    assert await inmemory_repo.delete(session.id) is True
    assert await inmemory_repo.load(session.id) is None


@pytest.mark.asyncio
async def test_delete_missing_returns_false(inmemory_repo):
    assert await inmemory_repo.delete("nonexistent") is False


@pytest.mark.asyncio
async def test_save_overwrites(inmemory_repo):
    session = Session(title="first")
    await inmemory_repo.save(session)

    session.title = "second"
    await inmemory_repo.save(session)

    loaded = await inmemory_repo.load(session.id)
    assert loaded.title == "second"