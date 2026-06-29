"""Session command handlers."""

from datetime import date, datetime, timedelta

from rich.console import Console
from rich.rule import Rule

from yapa.models import Session
from yapa.services import SessionService

_session_service: SessionService | None = None


def _get_session_service() -> SessionService:
    """Get or create the session service singleton."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service


def _format_time(session: Session) -> str:
    """Format the time string based on recency."""
    local = session.updated_at.astimezone()
    if local.date() == date.today():
        return local.strftime("%H:%M")
    if local.date() == date.today() - timedelta(days=1):
        return local.strftime("%H:%M")
    return local.strftime("%b %d")


def _truncate(text: str, max_len: int = 28) -> str:
    """Truncate with ellipsis if too long."""
    return text if len(text) <= max_len else text[: max_len - 1] + "\u2026"


def _group_by_date(
    sessions: list[Session],
) -> list[tuple[str, list[Session]]]:
    """Group sessions by relative date bucket, preserving input order."""
    today = date.today()
    groups: dict[str, list[Session]] = {
        "Today": [],
        "Yesterday": [],
        "Older": [],
    }
    for s in sessions:
        s_date = s.updated_at.astimezone().date()
        if s_date == today:
            groups["Today"].append(s)
        elif s_date == today - timedelta(days=1):
            groups["Yesterday"].append(s)
        else:
            groups["Older"].append(s)
    return [(k, groups[k]) for k in ["Today", "Yesterday", "Older"] if groups[k]]


def _timestamp() -> str:
    """Return the current time as HH:MM:SS."""
    return datetime.now().astimezone().strftime("%H:%M:%S")


def list_sessions() -> None:
    """List all conversation sessions."""
    console = Console()
    sessions = _get_session_service().list_sessions()

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    for group_name, group in _group_by_date(sessions):
        console.print(f"\n[bold]{group_name}[/bold]")
        console.print(Rule(style="dim"))
        for s in group:
            title = _truncate(s.title)
            time_str = _format_time(s)
            console.print(
                f"\u2502  [blue]{str(s.id)}[/blue]  "
                f"{title:<28}  "
                f"[cyan]{len(s.messages):>3}[/cyan] msgs  "
                f"[dim]{time_str}[/dim]"
            )


def rename_session(session_id: str, title: str) -> None:
    """Rename a conversation session."""
    console = Console()
    try:
        _get_session_service().rename(session_id, title)
        console.print(
            f"[dim]{_timestamp()}[/dim] \u2192 "
            f"Renamed [blue]{session_id}[/blue] to '{title}'"
        )
    except ValueError as e:
        console.print(f"[dim]{_timestamp()}[/dim] \u2717 [red]{e}[/red]")


def delete_session(session_id: str) -> None:
    """Delete a conversation session and its messages."""
    console = Console()
    try:
        _get_session_service().delete(session_id)
        console.print(
            f"[dim]{_timestamp()}[/dim] \u2192 "
            f"Deleted session [blue]{session_id}[/blue]"
        )
    except ValueError as e:
        console.print(f"[dim]{_timestamp()}[/dim] \u2717 [red]{e}[/red]")
