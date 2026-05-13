"""File-based implementation of the SessionRepository interface."""

import asyncio
import logging
from pathlib import Path
from typing import cast

from yapa.core.repositories.session_repository import SessionRepository
from yapa.shared import Config
from yapa.shared.models import Session


class SessionFileRepository(SessionRepository):
    """
    Async repository that stores each Session as a JSON file.

    Designed for production use, this repository persists sessions as individual
    JSON files in a specified directory. It supports asynchronous file operations to
    avoid blocking the event loop.

    Example:
        >>> config = Config()
        >>> logger = get_logger("core")
        >>> repo = SessionFileRepository(config, logger)
        >>> success: bool = await repo.save(session)
    """

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        """
        Initialize the JSON file-based session repository.

        Args:
            config (Config): Configuration object for the repository.
            logger (logging.Logger): Logger instance for the repository.
        """
        super().__init__(config, logger)
        self._directory = self._config.data_dir / "sessions"
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        return self._directory / f"{session_id}.json"

    async def save(self, session: Session) -> bool:
        """
        Save a session to a JSON file.

        Args:
            session (Session): The session object to persist.

        Returns:
            bool: True if the session was saved successfully, otherwise False.
        """
        path = self._path_for(session.id)
        try:
            content = session.model_dump_json(indent=2)
            await asyncio.to_thread(path.write_text, content, encoding="utf-8")
            self._logger.debug("Saved session %s to %s", session.id, path)
            return True
        except OSError:
            self._logger.exception("Failed to save session %s", session.id)
            return False

    async def load(self, session_id: str) -> Session | None:
        """
        Load a session from a JSON file.

        Args:
            session_id (str): The unique identifier of the session.

        Returns:
            Session | None: The session object if found, otherwise None.
        """
        path = self._path_for(session_id)
        exists = await asyncio.to_thread(path.exists)
        if not exists:
            self._logger.debug("Session file not found: %s", path)
            return None
        try:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            session = Session.model_validate_json(content)
            self._logger.debug("Loaded session %s from %s", session_id, path)
            return session
        except (OSError, ValueError):
            self._logger.exception("Failed to load session %s", session_id)
            return None

    async def load_all(self) -> list[Session]:
        """
        Load all sessions from the repository.

        Returns:
            list[Session]: A list of all session objects in the repository.
        """
        sessions: list[Session] = []
        try:
            file_paths = await asyncio.to_thread(list, self._directory.glob("*.json"))
            paths: list[Path] = sorted(cast(list[Path], file_paths))
        except OSError:
            self._logger.exception("Failed to list session files")
            return sessions
        for path in paths:
            try:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8")
                session = Session.model_validate_json(content)
                sessions.append(session)
            except (OSError, ValueError):
                self._logger.warning("Skipping invalid session file: %s", path)
        return sessions

    async def delete(self, session_id: str) -> bool:
        """
        Delete a session file.

        Args:
            session_id (str): The unique identifier of the session.

        Returns:
            bool: True if the session was deleted successfully, otherwise False.
        """
        path = self._path_for(session_id)
        exists = await asyncio.to_thread(path.exists)
        if not exists:
            self._logger.debug("Session file not found for deletion: %s", path)
            return False
        try:
            await asyncio.to_thread(path.unlink)
            self._logger.debug("Deleted session %s", session_id)
            return True
        except OSError:
            self._logger.exception("Failed to delete session %s", session_id)
            return False
