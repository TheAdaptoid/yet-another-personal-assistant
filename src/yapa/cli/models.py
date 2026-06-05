"""Models command handler — thin CLI adapter."""

from rich.console import Console
from rich.tree import Tree

from yapa.config import get_config, save_config
from yapa.models import ModelData
from yapa.services import ProviderService

_provider_service: ProviderService | None = None


def _get_provider_service() -> ProviderService:
    """Get or create the provider service singleton."""
    global _provider_service
    if _provider_service is None:
        _provider_service = ProviderService()
    return _provider_service


def _strip_group(model_id: str, group: str) -> str:
    """Remove the group prefix from a model ID for display."""
    prefix = f"{group}/"
    return model_id[len(prefix) :] if model_id.startswith(prefix) else model_id


def display_models(provider: str, models: list[ModelData]) -> None:
    """Display models for a provider, grouped by vendor prefix, as a tree."""
    console = Console()
    groups = ProviderService.group_models_by_vendor(models)
    tree = Tree(f"Models for [bold cyan]{provider}[/bold cyan] ({len(models)})")
    for group in sorted(groups):
        group_models = sorted(groups[group], key=lambda m: m.id)
        branch = tree.add(f"[bold blue]{group}[/bold blue] ({len(group_models)})")
        for model in group_models:
            display_name = _strip_group(model.id, group)
            branch.add(display_name)
    console.print(tree)


async def set_default_model(model_id: str, provider: str | None = None) -> None:
    """Set the default model and persist to config."""
    console = Console()

    svc = _get_provider_service()
    if provider:
        models = (await svc.get_models(provider)).get(provider, [])
    else:
        all_models = await svc.get_models()
        models = [m for ms in all_models.values() for m in ms]

    for m in models:
        if m.id == model_id:
            config = get_config()
            config.default_model_id = model_id
            config.default_provider_id = m.provider_id
            save_config(config)
            console.print(
                f"[dim]Default model set to[/dim] [bold]{model_id}[/bold]"
            )
            return

    console.print(
        f"[red]Error:[/red] Model '[bold]{model_id}[/bold]' not found."
    )


async def list_models(provider: str | None = None) -> None:
    """List available models."""
    svc = _get_provider_service()
    if provider:
        models = (await svc.get_models(provider)).get(provider, [])
        display_models(provider, models)
    else:
        Console().print("Available providers and their models:")
        all_models = await svc.get_models()
        for provider_id, models in all_models.items():
            display_models(provider_id, models)
