"""Tests for the session service."""

import logging
from unittest.mock import AsyncMock, create_autospec

import pytest

from yapa.core.repositories import (
    SessionDeleteError,
    SessionLoadError,
    SessionNotFoundError,
    SessionRepository,
    SessionSaveError,
)
from yapa.core.services.session_service import SessionService
from yapa.shared import Config
from yapa.shared.models import Session


# --- Fixtures ------------------------------------------------------------------


@pytest.fixture
def dummy_logger():
    """Return a mock logger with the methods used in the service."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.NullHandler()
        logger.addHandler(handler)
    return logger


@pytest.fixture
def test_config() -> Config:
    """Default config (not used when the repo is mocked)."""
    return Config()


@pytest.fixture
def mock_repo():
    """Mock SessionRepository with AsyncMock for every async method."""
    repo = create_autospec(SessionRepository, instance=True)
    repo.save = AsyncMock()
    repo.load = AsyncMock()
    repo.load_all = AsyncMock()
    repo.delete = AsyncMock()
    return repo


@pytest.fixture
def service(test_config, dummy_logger, mock_repo):
    """SessionService backed by a mocked repository."""
    return SessionService(test_config, dummy_logger, mock_repo)


@pytest.fixture
def sample_session() -> Session:
    """Pre-built Session for reuse in happy-path tests."""
    return Session(title="test")


# --- create_session ------------------------------------------------------------


class TestCreateSession:
    """Tests for SessionService.create_session."""

    @pytest.mark.asyncio
    async def test_create_default(self, service, mock_repo):
        """Creating without arguments gives a session with default title."""
        session = await service.create_session()

        mock_repo.save.assert_awaited_once_with(session)
        assert session.title == "New Session"

    @pytest.mark.asyncio
    async def test_create_with_title(self, service, mock_repo):
        """Passing a title sets it on the session."""
        session = await service.create_session(title="My Chat")

        mock_repo.save.assert_awaited_once_with(session)
        assert session.title == "My Chat"

    @pytest.mark.asyncio
    async def test_create_empty_title(self, service, mock_repo):
        """Empty title string is treated as no title — uses default."""
        session = await service.create_session(title="")

        assert session.title == "New Session"

    @pytest.mark.asyncio
    async def test_create_whitespace_title(self, service, mock_repo):
        """Whitespace-only title is treated as no title — uses default."""
        session = await service.create_session(title="   ")

        assert session.title == "New Session"

    @pytest.mark.asyncio
    async def test_create_propagates_save_error(self, service, mock_repo):
        """SessionSaveError is propagated — not caught by the service."""
        mock_repo.save.side_effect = SessionSaveError("disk full")

        with pytest.raises(SessionSaveError):
            await service.create_session()


# --- list_sessions -------------------------------------------------------------


class TestListSessions:
    """Tests for SessionService.list_sessions."""

    @pytest.mark.asyncio
    async def test_returns_sessions_newest_first(self, service, mock_repo):
        """Sessions are sorted by created_at descending."""
        s1 = Session()
        s1.created_at = 100
        s2 = Session()
        s2.created_at = 200
        s3 = Session()
        s3.created_at = 150
        mock_repo.load_all.return_value = [s1, s2, s3]

        result = await service.list_sessions()

        assert [s.created_at for s in result] == [200, 150, 100]

    @pytest.mark.asyncio
    async def test_empty(self, service, mock_repo):
        """No sessions returns an empty list."""
        mock_repo.load_all.return_value = []

        assert await service.list_sessions() == []

    @pytest.mark.asyncio
    async def test_propagates_load_error(self, service, mock_repo):
        """SessionLoadError is propagated."""
        mock_repo.load_all.side_effect = SessionLoadError("storage error")

        with pytest.raises(SessionLoadError):
            await service.list_sessions()


# --- get_session ---------------------------------------------------------------


class TestGetSession:
    """Tests for SessionService.get_session."""

    @pytest.mark.asyncio
    async def test_found(self, service, mock_repo, sample_session):
        """Found session is returned as-is."""
        mock_repo.load.return_value = sample_session

        result = await service.get_session(sample_session.id)

        assert result is sample_session

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self, service, mock_repo):
        """SessionNotFoundError is caught and translated to None."""
        mock_repo.load.side_effect = SessionNotFoundError

        result = await service.get_session("missing")

        assert result is None

    @pytest.mark.asyncio
    async def test_propagates_load_error(self, service, mock_repo):
        """SessionLoadError is propagated (infrastructure failure)."""
        mock_repo.load.side_effect = SessionLoadError("corrupt file")

        with pytest.raises(SessionLoadError):
            await service.get_session("broken")


# --- rename_session ------------------------------------------------------------


class TestRenameSession:
    """Tests for SessionService.rename_session."""

    @pytest.mark.asyncio
    async def test_success(self, service, mock_repo, sample_session):
        """Happy path: loads session, sets title, saves, returns True."""
        mock_repo.load.return_value = sample_session

        result = await service.rename_session(sample_session.id, "new title")

        assert result is True
        assert sample_session.title == "new title"
        mock_repo.save.assert_awaited_once_with(sample_session)

    @pytest.mark.asyncio
    async def test_not_found_returns_false(self, service, mock_repo):
        """SessionNotFoundError is caught and translated to False."""
        mock_repo.load.side_effect = SessionNotFoundError

        result = await service.rename_session("missing", "new title")

        assert result is False
        mock_repo.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_title_returns_false(self, service, mock_repo):
        """Empty title short-circuits before any repo call."""
        result = await service.rename_session("any", "")

        assert result is False
        mock_repo.load.assert_not_called()
        mock_repo.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_title_returns_false(self, service, mock_repo):
        """Whitespace-only title short-circuits before any repo call."""
        result = await service.rename_session("any", "   ")

        assert result is False
        mock_repo.load.assert_not_called()

    @pytest.mark.asyncio
    async def test_propagates_load_error(self, service, mock_repo):
        """SessionLoadError from load is propagated."""
        mock_repo.load.side_effect = SessionLoadError("corrupt file")

        with pytest.raises(SessionLoadError):
            await service.rename_session("any", "title")

    @pytest.mark.asyncio
    async def test_propagates_save_error(self, service, mock_repo, sample_session):
        """SessionSaveError from save is propagated."""
        mock_repo.load.return_value = sample_session
        mock_repo.save.side_effect = SessionSaveError("disk full")

        with pytest.raises(SessionSaveError):
            await service.rename_session(sample_session.id, "title")


# --- delete_session ------------------------------------------------------------


class TestDeleteSession:
    """Tests for SessionService.delete_session."""

    @pytest.mark.asyncio
    async def test_success(self, service, mock_repo):
        """Successful deletion returns True."""
        mock_repo.delete.return_value = None

        result = await service.delete_session("sess-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_not_found_returns_false(self, service, mock_repo):
        """SessionNotFoundError is caught and translated to False."""
        mock_repo.delete.side_effect = SessionNotFoundError

        result = await service.delete_session("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_propagates_delete_error(self, service, mock_repo):
        """SessionDeleteError is propagated (infrastructure failure)."""
        mock_repo.delete.side_effect = SessionDeleteError("permission denied")

        with pytest.raises(SessionDeleteError):
            await service.delete_session("any")


# --- Factory methods -----------------------------------------------------------


class TestFactoryMethods:
    """Tests for SessionService factory classmethods."""

    def test_with_file_repository(self, dummy_logger, tmp_path):
        """with_file_repository creates a service backed by SessionFileRepository."""
        config = Config(data_dir=tmp_path)
        service = SessionService.with_file_repository(config, dummy_logger)

        assert isinstance(service, SessionService)
        from yapa.core.repositories import SessionFileRepository

        assert isinstance(service._repository, SessionFileRepository)

    def test_with_in_memory_repository(self, dummy_logger, tmp_path):
        """with_in_memory_repository creates a service backed by SessionInMemoryRepository."""
        config = Config(data_dir=tmp_path)
        service = SessionService.with_in_memory_repository(config, dummy_logger)

        assert isinstance(service, SessionService)
        from yapa.core.repositories import SessionInMemoryRepository

        assert isinstance(service._repository, SessionInMemoryRepository)
