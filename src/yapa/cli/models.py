"""Models command handler."""

from collections import defaultdict

from rich.console import Console

from yapa.models import ModelData
from yapa.providers import ProviderManager

provider_manager = ProviderManager()


def _group_key(model_id: str) -> str:
    """
    Derive a display group label from a model ID.

    Models with a vendor prefix (e.g. 'openai/gpt-4') are grouped
    under that prefix. Models without a '/' are grouped as 'other'.
    """
    return model_id.split("/", 1)[0] if "/" in model_id else "other"


def _strip_group(model_id: str, group: str) -> str:
    """Remove the group prefix from a model ID for display."""
    prefix = f"{group}/"
    return model_id[len(prefix):] if model_id.startswith(prefix) else model_id


def display_models(provider: str, models: list[ModelData]) -> None:
    """Display models for a provider, grouped by vendor prefix."""
    console = Console()
    group_count = len(models)
    console.print(f"Models for provider '{provider}' ({group_count} total):")

    groups: dict[str, list[ModelData]] = defaultdict(list)
    for model in models:
        groups[_group_key(model.id)].append(model)

    for group in sorted(groups):
        group_models = sorted(groups[group], key=lambda m: m.id)
        console.print(f"\n  [bold]{group}[/bold] ({len(group_models)}):")
        for model in group_models:
            display_name = _strip_group(model.id, group)
            console.print(f"    {display_name}")


async def list_models(provider: str | None = None) -> None:
    """List available models."""
    if provider:
        provider_instance = provider_manager.get_provider(provider)
        models = await provider_instance.get_models()
        display_models(provider_instance.id, models)
    else:
        print("Available providers and their models:")
        for provider_instance in provider_manager.providers:
            models = await provider_instance.get_models()
            display_models(provider_instance.id, models)
