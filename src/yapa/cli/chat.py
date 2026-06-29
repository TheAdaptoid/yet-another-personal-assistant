"""Interactive conversation handler — slash-command-aware chat loop."""

from uuid import UUID

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

from yapa.config import Config, get_config, save_config
from yapa.models import AssistantMessage, Session
from yapa.services import ConversationError, ConversationService

_HELP_TEXT = """\
[bold]Available commands:[/bold]
  /help                            Show this help
  /exit                            Exit the chat
  /model <model-id>                Switch to a different model
  /session <session-id>            Switch to a different session
  /sessions                        List all sessions
"""


async def _handle_slash_command(
    cmd: str,
    arg: str,
    svc: ConversationService,
    cfg: Config,
    con: Console,
) -> Session | None:
    """Handle a parsed slash command. Returns new Session if /session."""

    if cmd == "help":
        con.print(_HELP_TEXT)
    elif cmd == "sessions":
        from .sessions import list_sessions

        list_sessions()
    elif cmd == "model":
        if not arg:
            con.print("[red]Usage: /model <model-id>[/red]")
            return None
        try:
            parsed = await svc.resolve_model(arg)
        except ValueError as e:
            con.print(f"[red]{e}[/red]")
            return None
        svc.model = parsed
        cfg.default_model = parsed.full_id
        save_config(cfg)
        con.print(f"[dim]Switched to model '{arg}'[/dim]")
    elif cmd == "session":
        if not arg:
            con.print("[red]Usage: /session <session-id>[/red]")
            return None
        try:
            info = svc.switch_session(UUID(arg))
            con.print(
                f"[dim]Switched to session '{str(info.id)}'"
                f" ({len(info.messages)} messages)"
                "[/dim]"
            )
            return info
        except ValueError as e:
            con.print(f"[red]{e}[/red]")
    else:
        con.print(
            f"[red]Unknown command: /{cmd}. Type /help for available commands.[/red]"
        )
    return None


def _build_renderables(
    reasoning_content: str, content: str
) -> list:
    """Build Rich renderables for reasoning + content display."""
    renderables = []
    if reasoning_content:
        renderables.append(Text(reasoning_content, style="dim"))
    if content:
        renderables.append(Markdown(content))
    return renderables


async def _stream_response(
    svc: ConversationService,
    con: Console,
    prompt: str,
) -> None:
    """Stream an assistant response."""
    try:
        stream = svc.stream_response(prompt)
        model_label = svc.model.id if svc.model else "Assistant"
        con.print(
            Rule(
                f"[bold green]{model_label}[/bold green]",
                style="dim",
                align="left",
            )
        )

        content_buffer = ""
        reasoning_buffer = ""
        had_reasoning = False
        last_reasoning_chunk = ""

        with Live(
            console=con, refresh_per_second=10, vertical_overflow="visible"
        ) as live:
            live.update(Text("  Thinking...  ", style="dim"))

            async for chunk in stream:
                if chunk.reasoning_content:
                    reasoning_buffer += chunk.reasoning_content
                    had_reasoning = True
                    last_reasoning_chunk = chunk.reasoning_content

                if chunk.content:
                    content_buffer += chunk.content

                if chunk.done:
                    if had_reasoning and not last_reasoning_chunk.endswith("\n"):
                        reasoning_buffer += "\n"
                    renderables = _build_renderables(
                        reasoning_buffer, content_buffer
                    )
                    if renderables:
                        live.update(Group(*renderables))
                    break

                renderables = _build_renderables(
                    reasoning_buffer, content_buffer
                )
                if renderables:
                    live.update(Group(*renderables))
    except ConversationError as e:
        con.print(" " * 20, end="\r")
        con.print(f"[red]{e}[/red]")


def _start_session(
    svc: ConversationService,
    info: Session,
    session_id: str | None,
    con: Console,
) -> None:
    """Print the session header with resume/new info, model, and history."""
    if session_id:
        con.print(
            Rule(
                f"resumed session: {str(info.id)} ({len(info.messages)} messages)",
                style="dim",
                align="left",
            )
        )
    else:
        con.print(Rule(f"new session: {str(info.id)}", style="dim", align="left"))

    con.print(
        Rule(
            f"model: {svc.model.id if svc.model else 'N/A'}",
            style="dim",
            align="left",
        )
    )

    if session_id and svc.messages:
        for msg in svc.messages[-2:]:
            if msg.role == "user":
                con.print(
                    Rule(
                        "[bold blue]You[/bold blue]",
                        style="dim",
                        align="left",
                    )
                )
                con.print(f"> {msg.content}")
            elif isinstance(msg, AssistantMessage):
                model_str = msg.model or (svc.model.id if svc.model else "Assistant")
                con.print(
                    Rule(
                        f"[bold green]{model_str}[/bold green]",
                        style="dim",
                        align="left",
                    )
                )
                con.print(Markdown(msg.content))


async def run_conversation(
    model_id: str | None = None,
    session_id: str | None = None,
    *,
    service: ConversationService | None = None,
    console: Console | None = None,
    config: Config | None = None,
) -> None:
    """
    Run an interactive chat session via the CLI.

    Supports slash-commands: /help, /exit, /model, /session, /sessions.

    Args:
        model_id (str | None): Model identifier string. Falls back to
            config.default_model.
        session_id (str | None): Session to resume, if any.
        service (ConversationService | None): Injectable conversation service
            (default: fresh ConversationService).
        console (Console | None): Injectable Rich console (default: new Console()).
        config (Config | None): Injectable config (default: get_config()).
    """
    cfg = config or get_config()
    con = console or Console()
    svc = service or ConversationService(config=cfg)

    model = await svc.resolve_model(model_id) if model_id else None
    if model is not None:
        svc.model = model

    info = await svc.start(session_id=UUID(session_id) if session_id else None)
    is_new = len(info.messages) == 0
    _start_session(svc, info, session_id, con)

    try:
        while True:
            con.print(Rule("[bold blue]You[/bold blue]", style="dim", align="left"))
            try:
                prompt = con.input("[blue]> [/blue]")
            except EOFError:
                break

            if not prompt.strip():
                continue

            if prompt == "/exit":
                con.print(
                    Rule(
                        f"session saved ({str(info.id)})",
                        style="red",
                        align="left",
                    )
                )
                break

            if prompt.startswith("/"):
                parts = prompt[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                result = await _handle_slash_command(cmd, arg, svc, cfg, con)
                if result is not None:
                    info = result

                continue

            await _stream_response(svc, con, prompt)

            if is_new:
                title = await svc.auto_title()
                if title:
                    con.print(f"[dim]Session titled: '{title}'[/dim]")
                is_new = False
    finally:
        save_config(cfg)
        await svc.close()
