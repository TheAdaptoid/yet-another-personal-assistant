"""Database engine and session management."""

from sqlalchemy import Engine, create_engine
from sqlmodel import Session, SQLModel

from yapa.config import get_config

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return cached engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"sqlite:///{get_config().database_path}",
            echo=False,
        )
    return _engine


def get_session() -> Session:
    """Return a new DB session."""
    return Session(get_engine())


def init_db() -> None:
    """Create all tables defined in SQLModel metadata."""
    SQLModel.metadata.create_all(get_engine())
