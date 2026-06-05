"""YAPA CLI - Typer app definition."""

import asyncio

from rich.console import Console
from typer import Argument, Option, Typer

from .chat import run_conversation
from .models import list_models, set_default_model
from .sessions import delete_session, list_sessions, purge_sessions, rename_session

cli = Typer()
sessions_app = Typer()
cli.add_typer(sessions_app, name="sessions", help="Manage conversation sessions")

# --- Models command ---


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


# --- Chat command ---


@cli.command(name="chat")
def chat(
    model: str | None = None,
    session: str | None = Option(None, "--session", "-s", help="Session ID to resume"),
) -> None:
    """Start an interactive chat session."""
    try:
        asyncio.run(run_conversation(model_id=model, session_id=session))
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
    session_id: str | None = Argument(None, help="Session ID to delete"),
    purge: bool = Option(
        False,
        "--purge",
        "-p",
        help="Delete all sessions with fewer than 2 messages",
    ),
) -> None:
    """Delete a conversation session and its messages, or purge empty sessions."""
    if purge:
        purge_sessions()
    elif session_id:
        delete_session(session_id)
    else:
        console = Console()
        console.print("[red]Error:[/red] Specify a session ID or use --purge")
