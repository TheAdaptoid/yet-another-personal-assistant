"""Session persistence — SessionStore protocol and JsonSessionStore."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from yapa.models import Session
from yapa.storage import GenericStore


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for session persistence."""

    def load(self, id: str) -> Session:
        """Load a session by ID."""

    def save(self, session: Session, *, overwrite: bool = False) -> None:
        """Save a session."""

    def list(self) -> list[Session]:
        """List all sessions."""

    def delete(self, id: str) -> None:
        """Delete a session by ID."""

    def exists(self, id: str) -> bool:
        """Check if a session exists."""

    def count(self) -> int:
        """Return the total number of sessions."""


class JsonSessionStore:
    """JSON-file-backed session store wrapping GenericStore."""

    def __init__(self, storage_dir: Path) -> None:
        """Initialize the JSON session store."""
        self._store = GenericStore[Session](
            storage_dir=Path(storage_dir),
            entity_type=Session,
        )

    def load(self, id: str) -> Session:
        """Load a session by ID."""
        return self._store.load(id)

    def save(self, session: Session, *, overwrite: bool = False) -> None:
        """Save a session."""
        self._store.save(session, overwrite=overwrite)

    def list(self) -> list[Session]:
        """List all sessions."""
        return self._store.list()

    def delete(self, id: str) -> None:
        """Delete a session by ID."""
        self._store.delete(id)

    def exists(self, id: str) -> bool:
        """Check if a session exists."""
        return self._store.exists(id)

    def count(self) -> int:
        """Return the total number of sessions."""
        return self._store.count()
