"""In-memory implementation of the SessionRepository interface."""

import logging
from typing import Optional

from yapa.core.repositories.session_repository import SessionRepository
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

    async def save(self, session: Session) -> bool:
        """
        Save a session to the in-memory repository.

        Args:
            session (Session): The session object to persist.

        Returns:
            bool: True if the session was saved successfully, otherwise False.

        Notes:
            Overwrites any existing session with the same ID.
        """
        self._sessions[session.id] = session
        self._logger.debug("Saved session %s in memory", session.id)
        return True

    async def load(self, session_id: str) -> Optional[Session]:
        """
        Load a session from the in-memory repository.

        Args:
            session_id (str): The unique identifier of the session.

        Returns:
            Session | None: The session object if found, otherwise None.
        """
        session = self._sessions.get(session_id)
        if session:
            self._logger.debug("Loaded session %s from memory", session_id)
        else:
            self._logger.debug("Session %s not found in memory", session_id)
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

    async def delete(self, session_id: str) -> bool:
        """
        Delete a session from the in-memory repository.

        Args:
            session_id (str): The unique identifier of the session.

        Returns:
            bool: True if the session was deleted successfully, otherwise False.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._logger.debug("Deleted session %s from memory", session_id)
            return True
        else:
            self._logger.debug(
                "Session %s not found in memory for deletion", session_id
            )
            return False
