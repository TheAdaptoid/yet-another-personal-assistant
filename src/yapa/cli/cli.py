"""YAPA CLI."""

import asyncio

from rich.console import Console
from rich.table import Table
from typer import Argument, Option, Typer

from yapa.config import get_config
from yapa.database import SessionRepository
from yapa.models import AssistantMessage, Message, ModelData, UserMessage
from yapa.providers import ProviderManager

cli = Typer()
sessions_app = Typer()
cli.add_typer(sessions_app, name="sessions", help="Manage conversation sessions")
provider_manager = ProviderManager()


def display_models(provider: str, models: list[ModelData]):
    """Display models for a provider."""
    console = Console()
    console.print(f"Models for provider '{provider}':")
    for model in models:
        console.print(f"- `{model.id}` [dim]({model.provider_id})[/dim]")


@cli.command()
def models(provider: str | None = None):
    """List available models for a provider."""
    try:
        asyncio.run(_models(provider))
    except KeyboardInterrupt:
        pass


async def _models(provider: str | None = None):

    if provider:
        provider_instance = provider_manager.get_provider(provider)
        models = await provider_instance.get_models()
        display_models(provider_instance.id, models)
    else:
        print("Available providers and their models:")
        for provider_instance in provider_manager.providers:
            models = await provider_instance.get_models()
            display_models(provider_instance.id, models)


@cli.command(name="invoke")
def invoke(
    model: str | None = None,
    session: str | None = Option(None, "--session", "-s", help="Session ID to resume"),
):
    """Invoke a model with a prompt."""
    try:
        asyncio.run(_invoke(model=model, session_id=session))
    except KeyboardInterrupt:
        pass


async def _invoke(
    model: str | None = None,
    session_id: str | None = None,
) -> None:
    console = Console()

    if not model:
        model = get_config().default_model
        print(f"No model specified. Using default model: {model}")

    provider = await provider_manager.get_provider_by_model(model)

    if session_id:
        session = SessionRepository.get(session_id)
        table_messages = SessionRepository.get_messages(session_id)
        messages: list[Message] = [m.to_pydantic() for m in table_messages]
        console.print(
            f"[dim]Resumed session '{session.id[:8]}' "
            f"({len(messages)} messages)[/dim]"
        )
    else:
        session = SessionRepository.create()
        messages: list[Message] = []
        console.print(f"[dim]Started new session '{session.id[:8]}'[/dim]")

    while True:
        prompt = console.input("[blue]You: [/blue]")
        if prompt.lower() in {"exit", "quit"}:
            console.print(
                f"[red]Exiting. Session '{session.id[:8]}' saved.[/red]"
            )
            break

        user_msg = UserMessage(content=prompt)
        messages.append(user_msg)
        SessionRepository.add_message(session.id, user_msg)

        stream = provider.invoke_model(
            model=model,
            messages=messages,
        )
        console.print("[green]Assistant:[/green]", end=" ")
        buffer = ""
        async for chunk in stream:
            if chunk.content:
                buffer += chunk.content
                console.print(chunk.content, end="")
            if chunk.reasoning_content:
                console.print(f"[dim]{chunk.reasoning_content}[/dim]", end="")
            if chunk.done:
                console.print()  # Newline at the end of the response
                break

        assistant_msg = AssistantMessage(content=buffer, model=model)
        messages.append(assistant_msg)
        SessionRepository.add_message(session.id, assistant_msg)


@sessions_app.command(name="list")
def sessions_list() -> None:
    """List all conversation sessions."""
    console = Console()
    sessions = SessionRepository.list_all()

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Created")

    for session in sessions:
        table.add_row(
            session.id,
            session.title,
            session.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@sessions_app.command(name="rename")
def sessions_rename(
    session_id: str = Argument(..., help="Session ID to rename"),
    title: str = Argument(..., help="New title for the session"),
) -> None:
    """Rename a conversation session."""
    console = Console()
    try:
        SessionRepository.rename(session_id, title)
        short_id = session_id[:8]
        console.print(f"[green]Session '{short_id}' renamed to '{title}'.[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


@sessions_app.command(name="delete")
def sessions_delete(
    session_id: str = Argument(..., help="Session ID to delete"),
) -> None:
    """Delete a conversation session and its messages."""
    console = Console()
    try:
        SessionRepository.delete(session_id)
        console.print(f"[green]Session '{session_id[:8]}' deleted.[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
