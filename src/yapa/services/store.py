"""Session persistence — SessionStore protocol and JsonSessionStore."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from yapa.models import Session
from yapa.storage import GenericStore


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for session persistence."""

    def load(self, id: str) -> Session: ...
    def save(self, session: Session, *, overwrite: bool = False) -> None: ...
    def list(self) -> list[Session]: ...
    def delete(self, id: str) -> None: ...


class JsonSessionStore:
    """JSON-file-backed session store wrapping GenericStore."""

    def __init__(self, storage_dir: Path) -> None:
        self._store = GenericStore[Session](
            storage_dir=Path(storage_dir),
            entity_type=Session,
        )

    def load(self, id: str) -> Session:
        return self._store.load(id)

    def save(self, session: Session, *, overwrite: bool = False) -> None:
        self._store.save(session, overwrite=overwrite)

    def list(self) -> list[Session]:
        return self._store.list()

    def delete(self, id: str) -> None:
        self._store.delete(id)
