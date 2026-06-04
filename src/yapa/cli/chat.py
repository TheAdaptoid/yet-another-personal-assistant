"""Interactive conversation handler."""

from rich.console import Console

from yapa.config import Config, get_config
from yapa.database import SessionRepository
from yapa.models import AssistantMessage, Message, UserMessage
from yapa.providers import ProviderManager

_default_provider_manager = ProviderManager()


async def run_conversation(
    model: str | None = None,
    session_id: str | None = None,
    *,
    provider_manager: ProviderManager | None = None,
    console: Console | None = None,
    config: Config | None = None,
) -> None:
    """
    Run the interactive conversation loop.

    Args:
        model (str | None): Model identifier. Falls back to config.default_model.
        session_id (str | None): Session to resume, if any.
        provider_manager (ProviderManager | None): Injectable provider manager
            (default: module-level singleton).
        console (Console | None): Injectable Rich console (default: new Console()).
        config (Config | None): Injectable config (default: get_config()).
    """
    pm = provider_manager or _default_provider_manager
    con = console or Console()
    cfg = config or get_config()

    if not model:
        model = cfg.default_model
        print(f"No model specified. Using default model: {model}")

    provider = await pm.get_provider_by_model(model)

    if session_id:
        session = SessionRepository.get(session_id)
        table_messages = SessionRepository.get_messages(session_id)
        messages: list[Message] = [m.to_pydantic() for m in table_messages]
        con.print(
            f"[dim]Resumed session '{session.id[:8]}' ({len(messages)} messages)[/dim]"
        )
    else:
        session = SessionRepository.create()
        messages: list[Message] = []
        con.print(f"[dim]Started new session '{session.id[:8]}'[/dim]")

    while True:
        prompt = con.input("[blue]You: [/blue]")
        if prompt.lower() in {"exit", "quit"}:
            con.print(f"[red]Exiting. Session '{session.id[:8]}' saved.[/red]")
            break

        user_msg = UserMessage(content=prompt)
        messages.append(user_msg)
        SessionRepository.add_message(session.id, user_msg)

        stream = provider.invoke_model(
            model=model,
            messages=messages,
        )
        con.print("[green]Assistant:[/green]", end=" ")
        buffer = ""
        async for chunk in stream:
            if chunk.content:
                buffer += chunk.content
                con.print(chunk.content, end="")
            if chunk.reasoning_content:
                con.print(f"[dim]{chunk.reasoning_content}[/dim]", end="")
            if chunk.done:
                con.print()
                break

        assistant_msg = AssistantMessage(content=buffer, model=model)
        messages.append(assistant_msg)
        SessionRepository.add_message(session.id, assistant_msg)
