"""YAPA CLI - Typer app definition."""

import asyncio

from typer import Argument, Option, Typer

from .chat import run_conversation
from .models import list_models
from .sessions import delete_session, list_sessions, rename_session

cli = Typer()
sessions_app = Typer()
cli.add_typer(sessions_app, name="sessions", help="Manage conversation sessions")

# --- Models command ---


@cli.command()
def models(provider: str | None = None) -> None:
    """List available models for a provider."""
    try:
        asyncio.run(list_models(provider))
    except KeyboardInterrupt:
        pass


# --- Chat command ---


@cli.command(name="chat")
def chat(
    model: str | None = None,
    session: str | None = Option(None, "--session", "-s", help="Session ID to resume"),
) -> None:
    """Start an interactive chat session."""
    try:
        asyncio.run(run_conversation(model=model, session_id=session))
    except KeyboardInterrupt:
        pass


# --- Session management commands ---


@sessions_app.command(name="list")
def sessions_list() -> None:
    """List all conversation sessions."""
    list_sessions()


@sessions_app.command(name="rename")
def sessions_rename(
    session_id: str = Argument(..., help="Session ID to rename"),
    title: str = Argument(..., help="New title for the session"),
) -> None:
    """Rename a conversation session."""
    rename_session(session_id, title)


@sessions_app.command(name="delete")
def sessions_delete(
    session_id: str = Argument(..., help="Session ID to delete"),
) -> None:
    """Delete a conversation session and its messages."""
    delete_session(session_id)
