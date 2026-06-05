"""Interactive conversation handler — slash-command-aware chat loop."""

from rich.console import Console
from rich.rule import Rule

from yapa.config import Config, get_config, save_config
from yapa.models import AssistantMessage, ModelData, SessionSummary
from yapa.services import ConversationError, ConversationService


def _parse_model(model_id: str) -> ModelData:
    """Build a ModelData from a model-id string (e.g. 'openrouter/free')."""
    provider_id = model_id.split("/", 1)[0] if "/" in model_id else "other"
    return ModelData(id=model_id, provider_id=provider_id)


_HELP_TEXT = """\
[bold]Available commands:[/bold]
  /help                            Show this help
  /exit                            Exit the chat
  /model <model-id>                Switch to a different model
  /session <session-id>            Switch to a different session
  /sessions                        List all sessions
"""


def _handle_slash_command(
    cmd: str,
    arg: str,
    svc: ConversationService,
    cfg: Config,
    con: Console,
) -> None:
    """Handle a parsed slash command."""
    if cmd == "help":
        con.print(_HELP_TEXT)
    elif cmd == "sessions":
        from .sessions import list_sessions

        list_sessions()
    elif cmd == "model":
        if not arg:
            con.print("[red]Usage: /model <model-id>[/red]")
            return
        parsed = _parse_model(arg)
        svc.model = parsed
        cfg.default_model_id = parsed.id
        cfg.default_provider_id = parsed.provider_id
        save_config(cfg)
        con.print(f"[dim]Switched to model '{arg}'[/dim]")
    else:
        con.print(
            f"[red]Unknown command: /{cmd}. Type /help for available commands.[/red]"
        )


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
        con.print("[dim]  Thinking...  [/dim]", end="\r")
        first_output = True
        async for chunk in stream:
            if first_output and (chunk.content or chunk.reasoning_content):
                con.print(" " * 20, end="\r")
                first_output = False
            if chunk.content:
                con.print(chunk.content, end="")
            if chunk.reasoning_content:
                con.print(f"[dim]{chunk.reasoning_content}[/dim]", end="")
            if chunk.done:
                con.print()
                break
    except ConversationError as e:
        con.print(" " * 20, end="\r")
        con.print(f"[red]{e}[/red]")


def _start_session(
    svc: ConversationService,
    info: SessionSummary,
    session_id: str | None,
    con: Console,
) -> None:
    """Print the session header with resume/new info, model, and history."""
    if session_id:
        con.print(
            Rule(
                f"resumed session: {info.id} ({info.message_count} messages)",
                style="dim",
                align="left",
            )
        )
    else:
        con.print(Rule(f"new session: {info.id}", style="dim", align="left"))

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
                con.print(msg.content)


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
            config.default_model_id.
        session_id (str | None): Session to resume, if any.
        service (ConversationService | None): Injectable conversation service
            (default: fresh ConversationService).
        console (Console | None): Injectable Rich console (default: new Console()).
        config (Config | None): Injectable config (default: get_config()).
    """
    cfg = config or get_config()
    con = console or Console()
    svc = service or ConversationService(config=cfg)

    model = _parse_model(model_id) if model_id else None
    if model is not None:
        svc.model = model

    info = svc.start(session_id=session_id)
    _start_session(svc, info, session_id, con)

    try:
        while True:
            con.print(Rule("[bold blue]You[/bold blue]", style="dim", align="left"))
            prompt = con.input("[blue]> [/blue]")

            if prompt.lower() in {"exit", "quit", "/exit"}:
                con.print(
                    Rule(
                        f"session saved ({info.id})",
                        style="red",
                        align="left",
                    )
                )
                break

            if prompt.startswith("/"):
                parts = prompt[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "session":
                    if not arg:
                        con.print("[red]Usage: /session <session-id>[/red]")
                        continue
                    try:
                        info = svc.switch_session(arg)
                        con.print(
                            f"[dim]Switched to session '{info.id}'"
                            f" ({info.message_count} messages)"
                            "[/dim]"
                        )
                    except ValueError as e:
                        con.print(f"[red]{e}[/red]")
                else:
                    _handle_slash_command(cmd, arg, svc, cfg, con)

                continue

            await _stream_response(svc, con, prompt)
    finally:
        save_config(cfg)
        await svc.close()
