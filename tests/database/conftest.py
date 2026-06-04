"""Test fixtures for database tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel


@pytest.fixture(autouse=True)
def patch_get_engine():
    """Redirect all DB operations to a fresh in-memory SQLite database."""
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(engine)
    with patch("yapa.database.engine.get_engine", return_value=engine):
        yield
