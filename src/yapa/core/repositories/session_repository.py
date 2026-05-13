"""Session repository interface and implementation for file-based storage."""

import logging
from abc import ABC, abstractmethod

from yapa.shared import Config
from yapa.shared.models import Session


class SessionSaveError(Exception):
    """Custom exception for session save failures."""


class SessionLoadError(Exception):
    """Custom exception for session load failures."""


class SessionNotFoundError(Exception):
    """Custom exception for session not found."""


class SessionDeleteError(Exception):
    """Custom exception for session delete failures."""


class SessionRepository(ABC):
    """Abstract base class for session repositories."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        """
        Initialize the session repository.

        Repositories inherit the application configuration and logger from the caller,
        ensuring consistent access to settings and logging across different
        implementations.

        Args:
            config (Config): Configuration object for the repository.
            logger (logging.Logger): Logger instance for the repository.
        """

        self._config = config
        self._logger = logger

    @abstractmethod
    async def save(self, session: Session) -> None:
        """
        Save a session to the repository.

        Args:
            session (Session): The session object to persist.

        Raises:
            SessionSaveError: If the session could not be saved.
        """

    @abstractmethod
    async def load(self, session_id: str) -> Session:
        """
        Load a session from the repository.

        Args:
            session_id (str): The unique identifier of the session.

        Returns:
            Session: The session object if found.

        Raises:
            SessionLoadError: If there was an error loading the session.
            SessionNotFoundError: If the session was not found.
        """

    @abstractmethod
    async def load_all(self) -> list[Session]:
        """
        Load all sessions from the repository.

        Returns:
            list[Session]: A list of all session objects in the repository.

        Raises:
            SessionLoadError: If there was an error loading the sessions.
        """

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """
        Delete a session from the repository.

        Args:
            session_id (str): The unique identifier of the session.

        Raises:
            SessionDeleteError: If there was an error deleting the session.
            SessionNotFoundError: If the session was not found.
        """
