"""YAPA CLI - Typer app definition."""

import asyncio

from rich.console import Console
from typer import Argument, Exit, Option, Typer

from .chat import run_conversation
from .models import list_models, set_default_model
from .sessions import (
    delete_session,
    list_sessions,
    rename_session,
)

cli = Typer()
sessions_app = Typer()
cli.add_typer(sessions_app, name="sessions", help="Manage conversation sessions")


@cli.command()
def models(
    provider: str | None = Option(
        None,
        "--provider",
        help="Provider to list or scope --set lookup",
    ),
    set_model: str | None = Option(None, "--set", help="Set default model ID"),
) -> None:
    """List available models or set the default model."""
    try:
        if set_model:
            asyncio.run(set_default_model(set_model, provider))
        else:
            asyncio.run(list_models(provider))
    except KeyboardInterrupt:
        pass


@cli.command(name="chat")
def chat(
    model: str | None = None,
    session: str | None = Option(None, "--session", "-s", help="Session ID to resume"),
    continue_last: bool = Option(
        False, "--continue", "-c", help="Continue the most recent session"
    ),
) -> None:
    """Start an interactive chat session."""
    try:
        session_id = session
        if continue_last:
            from yapa.services import SessionService

            sessions = SessionService().list_sessions()
            if sessions:
                session_id = str(sessions[0].id)
        asyncio.run(run_conversation(model_id=model, session_id=session_id))
    except KeyboardInterrupt:
        pass


@sessions_app.command(name="list")
def sessions_list() -> None:
    """List all conversation sessions."""
    list_sessions()


@sessions_app.command(name="rename")
def sessions_rename(
    session_id: str = Argument(..., help="Session ID to rename"),
    title: str | None = Argument(None, help="New title for the session"),
) -> None:
    """Rename a conversation session."""
    if title:
        rename_session(session_id, title)
    else:
        Console().print("[red]Error:[/red] Provide a title or use --auto")
        raise Exit(1)


@sessions_app.command(name="delete")
def sessions_delete(
    session_id: str = Argument(..., help="Session ID to delete"),
) -> None:
    """Delete a conversation session and its messages."""
    delete_session(session_id)
