"""Session command handlers."""

from datetime import date, datetime, timedelta

from rich.console import Console
from rich.prompt import Confirm
from rich.rule import Rule

from yapa.models import SessionSummary
from yapa.services import ConversationService, SessionService

_session_service: SessionService | None = None


def _get_session_service() -> SessionService:
    """Get or create the session service singleton."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service


def _format_time(summary: SessionSummary) -> str:
    """Format the time string based on recency."""
    local = summary.updated_at.astimezone()
    if local.date() == date.today():
        return local.strftime("%H:%M")
    if local.date() == date.today() - timedelta(days=1):
        return local.strftime("%H:%M")
    return local.strftime("%b %d")


def _truncate(text: str, max_len: int = 28) -> str:
    """Truncate with ellipsis if too long."""
    return text if len(text) <= max_len else text[: max_len - 1] + "\u2026"


def _group_by_date(
    summaries: list[SessionSummary],
) -> list[tuple[str, list[SessionSummary]]]:
    """Group sessions by relative date bucket, preserving input order."""
    today = date.today()
    groups: dict[str, list[SessionSummary]] = {
        "Today": [],
        "Yesterday": [],
        "Older": [],
    }
    for s in summaries:
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
    summaries = _get_session_service().list_all()

    if not summaries:
        console.print("[dim]No sessions found.[/dim]")
        return

    for group_name, group in _group_by_date(summaries):
        console.print(f"\n[bold]{group_name}[/bold]")
        console.print(Rule(style="dim"))
        for s in group:
            title = _truncate(s.title)
            time_str = _format_time(s)
            console.print(
                f"\u2502  [blue]{s.id}[/blue]  "
                f"{title:<28}  "
                f"[cyan]{s.message_count:>3}[/cyan] msgs  "
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


def purge_sessions() -> None:
    """Delete all sessions with fewer than 2 messages."""
    console = Console()
    summaries = _get_session_service().list_all()
    to_purge = [s for s in summaries if s.message_count < 2]

    if not to_purge:
        console.print("[dim]No empty sessions to purge.[/dim]")
        return

    console.print(
        f"[yellow]Sessions with fewer than 2 messages ({len(to_purge)}):[/yellow]"
    )
    for s in to_purge:
        count_str = "message" if s.message_count == 1 else "messages"
        console.print(
            f"  [blue]{s.id}[/blue]  {s.title:<28}  "
            f"({s.message_count} {count_str})"
        )

    if not Confirm.ask("Delete these sessions?"):
        console.print("[dim]Cancelled.[/dim]")
        return

    _get_session_service().purge()
    console.print(f"[green]Purged {len(to_purge)} session(s).[/green]")


async def _auto_rename_session(session_id: str) -> str | None:
    """Auto-rename a session using LLM title generation."""
    try:
        async with ConversationService() as svc:
            await svc.start(session_id=session_id)
            for msg in svc.messages:
                if msg.role == "user":
                    title = await svc.generate_title(msg.content)
                    if title:
                        _get_session_service().rename(session_id, title)
                        return title
                    return None
    except ValueError:
        return None
