"""YAPA CLI."""

import asyncio

from rich.console import Console
from typer import Typer

from yapa.config import get_config
from yapa.models import AssistantMessage, Message, ModelData, UserMessage
from yapa.providers import ProviderManager

cli = Typer()
provider_manager = ProviderManager()


def display_modes(provider: str, models: list[ModelData]):
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
        display_modes(provider_instance.id, models)
    else:
        print("Available providers and their models:")
        for provider_instance in provider_manager.providers:
            models = await provider_instance.get_models()
            display_modes(provider_instance.id, models)


@cli.command(name="invoke")
def invoke(model: str | None = None):
    """Invoke a model with a prompt."""
    try:
        asyncio.run(_invoke(model))
    except KeyboardInterrupt:
        pass


async def _invoke(model: str | None = None) -> None:
    messages: list[Message] = []
    console = Console()
    if not model:
        model = get_config().default_model
        print(f"No model specified. Using default model: {model}")
    provider = await provider_manager.get_provider_by_model(model)

    while True:
        prompt = console.input("[blue]You: [/blue]")
        if prompt.lower() in {"exit", "quit"}:
            console.print("[red]Exiting.[/red]")
            break
        messages.append(UserMessage(content=prompt))

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
        messages.append(AssistantMessage(content=buffer))
