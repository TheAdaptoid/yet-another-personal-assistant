"""Session service for managing chat sessions."""

from yapa.shared.models.session import SessionData

import logging
from typing import Optional

from yapa.core.repositories import SessionNotFoundError, SessionRepository
from yapa.shared import Config, Message
from yapa.shared.models import Session, SessionData


class SessionService:
    """
    Business logic for session CRUD operations.

    Provides an abstraction layer between HTTP handlers and the
    persistence repository, handling domain logic such as timestamp
    updates and logging.
    """

    def __init__(
        self, config: Config, logger: logging.Logger, repository: SessionRepository
    ) -> None:
        """
        Initialize the session service.

        Args:
            config (Config): Application configuration.
            logger (logging.Logger): Logger instance for recording operations.
            repository (SessionRepository): The session repository to use.
        """
        self._config = config
        self._logger = logger
        self._repository = repository

    async def create_session(self, title: str | None = None) -> Session:
        """
        Create a new session with optional title.

        Args:
            title (str | None): Optional title for the session.

        Returns:
            Session: The created session.
        """
        session = Session.create(title=title)
        await self._repository.save(session)
        self._logger.debug("Created session %s", session.id)
        return session

    async def list_sessions(self) -> list[SessionData]:
        """
        List all stored sessions.

        Returns:
            list[SessionData]: List of all sessions ordered by creation time (newest first).
        """
        sessions = await self._repository.load_all()
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        self._logger.debug("Listed %d sessions", len(sessions))
        return [session.data for session in sessions]

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by its ID.

        Args:
            session_id (str): The unique session identifier.

        Returns:
            Session | None: The session data if found, otherwise None.
        """
        try:
            session = await self._repository.load(session_id)
        except SessionNotFoundError:
            self._logger.debug("Session not found: %s", session_id)
            return None
        self._logger.debug("Retrieved session %s", session_id)
        return session

    async def rename_session(self, session_id: str, new_title: str) -> bool:
        """
        Rename an existing session.

        Args:
            session_id (str): The unique session identifier.
            new_title (str): The new title for the session.

        Returns:
            bool: True if the session was renamed, False if not found.
        """
        if not new_title.strip():
            self._logger.debug(
                "Cannot rename session %s: new title is empty", session_id
            )
            return False

        try:
            session = await self._repository.load(session_id)
        except SessionNotFoundError:
            self._logger.debug("Cannot rename: session %s not found", session_id)
            return False
        updated_session = session.update_title(new_title)
        await self._repository.save(updated_session)
        self._logger.debug("Renamed session %s to %s", session_id, new_title)
        return True

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session by its ID.

        Args:
            session_id (str): The unique session identifier.

        Returns:
            bool: True if the session was deleted, False if not found.
        """
        try:
            await self._repository.delete(session_id)
        except SessionNotFoundError:
            self._logger.debug("Session %s not found for deletion", session_id)
            return False
        self._logger.debug("Deleted session %s", session_id)
        return True

    @classmethod
    def with_file_repository(
        cls, config: Config, logger: logging.Logger
    ) -> "SessionService":
        """
        Create a SessionService with a file-based repository.

        Args:
            config (Config): Application configuration.
            logger (logging.Logger): Logger instance for recording operations.

        Returns:
            SessionService: An instance of SessionService using a file repository.
        """
        from yapa.core.repositories.session_file_repository import SessionFileRepository

        repository = SessionFileRepository(config, logger)
        return cls(config, logger, repository)

    @classmethod
    def with_in_memory_repository(
        cls, config: Config, logger: logging.Logger
    ) -> "SessionService":
        """
        Create a SessionService with an in-memory repository.

        Args:
            config (Config): Application configuration.
            logger (logging.Logger): Logger instance for recording operations.

        Returns:
            SessionService: An instance of SessionService using an in-memory repository.
        """
        from yapa.core.repositories.session_inmemory_repository import (
            SessionInMemoryRepository,
        )

        repository = SessionInMemoryRepository(config, logger)
        return cls(config, logger, repository)
