"""In-memory implementation of the SessionRepository interface."""

import logging

from yapa.core.repositories.session_repository import (
    SessionNotFoundError,
    SessionRepository,
)
from yapa.shared import Config
from yapa.shared.models import Session


class SessionInMemoryRepository(SessionRepository):
    """
    In-memory implementation of the SessionRepository interface.

    Designed for testing and development, this repository stores sessions in a simple
    dictionary. Data is lost when the instance is destroyed.

    Example:
        >>> config = Config()
        >>> logger = get_logger("core")
        >>> repo = SessionInMemoryRepository(config, logger)
    """

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        """
        Initialize the in-memory session repository.

        Args:
            config (Config): Configuration object for the repository.
            logger (logging.Logger): Logger instance for the repository.
        """
        self._sessions: dict[str, Session] = {}
        super().__init__(config, logger)
        logger.debug("Initialized in-memory repository")

    async def save(self, session: Session) -> None:
        """
        Save a session to the in-memory repository.

        Args:
            session (Session): The session object to persist.

        Notes:
            Overwrites any existing session with the same ID.
        """
        self._sessions[session.id] = session
        self._logger.debug("Saved session %s in memory", session.id)

    async def load(self, session_id: str) -> Session:
        """
        Load a session from the in-memory repository.

        Args:
            session_id (str): The unique identifier of the session.

        Returns:
            Session: The session object if found.

        Raises:
            SessionNotFoundError: If the session was not found.
        """
        session = self._sessions.get(session_id)

        if not session:
            err_msg = f"Session {session_id} not found in memory"
            self._logger.debug(err_msg)
            raise SessionNotFoundError(err_msg)

        self._logger.debug("Loaded session %s from memory", session_id)
        return session

    async def load_all(self) -> list[Session]:
        """
        Load all sessions from the in-memory repository.

        Returns:
            list[Session]: A list of all session objects currently stored.

        Notes:
            The sessions are returned in the order they were added to the repository.
        """
        sessions = list(self._sessions.values())
        self._logger.debug(
            "Loaded all sessions from memory: %d sessions", len(sessions)
        )
        return sessions

    async def delete(self, session_id: str) -> None:
        """
        Delete a session from the in-memory repository.

        Args:
            session_id (str): The unique identifier of the session.

        Raises:
            SessionNotFoundError: If the session was not found.
        """
        if session_id not in self._sessions:
            err_msg = f"Session {session_id} not found in memory for deletion"
            self._logger.debug(err_msg)
            raise SessionNotFoundError(err_msg)

        del self._sessions[session_id]
        self._logger.debug("Deleted session %s from memory", session_id)
