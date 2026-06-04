"""Session command handlers."""

from rich.console import Console
from rich.table import Table

from yapa.database import SessionRepository


def list_sessions() -> None:
    """List all conversation sessions."""
    console = Console()
    sessions = SessionRepository.list_all()

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Last Opened")

    # Sort sessions by updated_at descending
    sessions.sort(key=lambda s: s.updated_at, reverse=True)

    for session in sessions:
        table.add_row(
            session.id,
            session.title,
            session.updated_at.strftime("%Y-%m-%d %H:%M")
        )

    console.print(table)


def rename_session(session_id: str, title: str) -> None:
    """Rename a conversation session."""
    console = Console()
    try:
        SessionRepository.rename(session_id, title)
        short_id = session_id[:8]
        console.print(f"[green]Session '{short_id}' renamed to '{title}'.[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


def delete_session(session_id: str) -> None:
    """Delete a conversation session and its messages."""
    console = Console()
    try:
        SessionRepository.delete(session_id)
        console.print(f"[green]Session '{session_id[:8]}' deleted.[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
