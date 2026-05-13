"""Session repository interface and implementation for file-based storage."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from yapa.shared import Config
from yapa.shared.models import Session


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
    async def save(self, session: Session) -> bool:
        """
        Save a session to the repository.

        Args:
            session (Session): The session object to persist.

        Returns:
            bool: True if the session was saved successfully, otherwise False.
        """

    @abstractmethod
    async def load(self, session_id: str) -> Optional[Session]:
        """
        Load a session from the repository.

        Args:
            session_id (str): The unique identifier of the session.

        Returns:
            Optional[Session]: The session object if found, otherwise None.
        """

    @abstractmethod
    async def load_all(self) -> list[Session]:
        """
        Load all sessions from the repository.

        Returns:
            list[Session]: A list of all session objects in the repository.
        """

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """
        Delete a session from the repository.

        Args:
            session_id (str): The unique identifier of the session.

        Returns:
            bool: True if the session was deleted successfully, otherwise False.
        """
