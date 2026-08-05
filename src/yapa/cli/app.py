"""Main CLI application — maintenance-only commands."""

import asyncio
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import typer
import uvicorn
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from yapa.services import ModelService, SessionService
from yapa.services.config import JsonConfigStore, ProviderConfig
from yapa.services.store import JsonSessionStore

cli = typer.Typer(name="yapa", help="YAPA — Your AI Personal Assistant")
console = Console()

ORANGE = "orange1"
LOGO = """\
  __  _____   ___  ___
  \\ \\/ / _ | / _ \\/ _ |
   \\  / __ |/ ___/ __ |
   /_/_/ |_/_/  /_/ |_|"""


def _style(key: str, value: str = "", end: str = "\n") -> None:
    msg = f"[{ORANGE}]{key}[/{ORANGE}] [bold white]{value}[/bold white]"
    console.print(msg, end=end)


def _success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def _error(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")


def _coerce_config_value(target: Any, key: str, value: str) -> Any:
    if isinstance(target, BaseModel):
        field = target.__class__.model_fields.get(key)
        if field is not None:
            annotation = field.annotation
            if annotation is Path:
                return Path(value)
            if annotation is int:
                return int(value)
            if annotation is float:
                return float(value)
            if annotation is bool:
                return value.lower() in {"1", "true", "yes", "on"}

    return value


def _set_nested_value(target: Any, parts: list[str], value: str) -> None:
    if len(parts) >= 3 and parts[0] == "provider_configs":
        provider_configs = getattr(target, "provider_configs", None)
        if provider_configs is None:
            provider_configs = {}
            setattr(target, "provider_configs", provider_configs)

        provider_id = parts[1]
        provider_config = provider_configs.get(provider_id)
        if provider_config is None:
            provider_config = ProviderConfig()
            provider_configs[provider_id] = provider_config

        if len(parts) == 3:
            setattr(provider_config, parts[2], value)
            return

        _set_nested_value(provider_config, parts[2:], value)
        return

    if len(parts) == 1:
        coerced_value = _coerce_config_value(target, parts[0], value)
        if isinstance(target, MutableMapping):
            target[parts[0]] = coerced_value
        else:
            setattr(target, parts[0], coerced_value)
        return

    head, *tail = parts
    if isinstance(target, MutableMapping):
        next_target = target.get(head)
        if next_target is None:
            next_target = {}
            target[head] = next_target
    else:
        next_target = getattr(target, head, None)
        if next_target is None:
            next_target = {}
            setattr(target, head, next_target)

    _set_nested_value(next_target, tail, value)


# ----- server -----


@cli.command()
def server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
):
    """Start the YAPA API server."""
    console.print(
        Panel(
            Text(LOGO, style=ORANGE),
            subtitle=(
                f"[dim]Listening on [white]http://{host}:{port}[/white]  "
                f"Docs at [white]http://{host}:{port}/docs[/white]  "
                f"Press Ctrl+C to stop[/dim]"
            ),
        )
    )
    uvicorn.run(
        "yapa.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


# ----- config -----


config_cli = typer.Typer(help="Manage YAPA configuration.")
cli.add_typer(config_cli, name="config")


@config_cli.command(name="show")
def config_show():
    """Show current configuration."""
    store = JsonConfigStore()
    cfg = store.load()

    if not cfg.provider_configs:
        console.print("[dim]No providers configured[/dim]")

    _style("Log Level", cfg.log_level)
    _style("Storage Dir", str(cfg.storage_dir))
    _style("Timeout", str(cfg.provider_timeout) + "s")
    _style("Max Retries", str(cfg.provider_max_retries))

    for provider_id, pconfig in cfg.provider_configs.items():
        _style(f"\n\\[{provider_id}\\]")
        key_preview = pconfig.api_key[:8] + "..." if pconfig.api_key else "not set"
        _style("  API Key", key_preview)
        _style("  Base URL", pconfig.base_url or "(default)")


@config_cli.command(name="set")
def config_set(
    key: str = typer.Argument(
        ..., help="Dot-separated config key, e.g. provider_configs.openai.api_key"
    ),
    value: str = typer.Argument(..., help="Value to set"),
):
    """Set a config value."""
    store = JsonConfigStore()
    cfg = store.load()

    _set_nested_value(cfg, key.split("."), value)

    store.save(cfg)
    _success(f"Config {key} set to {value}")


# ----- models -----


def _pricing_label(m) -> str:
    """Render a model's pricing as a CLI table cell, or '-' if absent."""
    p = getattr(m, "pricing", None)
    if p is None:
        return "-"
    parts = []
    if p.input is not None:
        parts.append(f"in ${p.input:g}/1M")
    if p.output is not None:
        parts.append(f"out ${p.output:g}/1M")
    if p.request is not None:
        parts.append(f"req ${p.request:g}")
    return " ".join(parts) or "-"


def _comma_formatting(value: str) -> str:
    """Format a number with commas as thousands separators."""
    try:
        value_f = int(value)
    except ValueError:
        return value
    return f"{value_f:,}"


@cli.command()
def models(
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="Filter by provider ID"
    ),
    model_type: str | None = typer.Option(
        None,
        "--model-type",
        "-t",
        help="Filter by model type: llm, embedding, or other",
    ),
):
    """List available models."""
    from yapa.models import ModelType

    model_type_enum = None
    if model_type:
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            available = ", ".join(m.value for m in ModelType)
            _error(f"Invalid model type '{model_type}'. Must be one of: {available}")
            raise typer.Exit(code=1)

    service = ModelService()
    results = asyncio.run(
        service.list_models(provider_id=provider, model_type=model_type_enum)
    )

    if not results:
        if provider:
            _error(f"{provider}: no models found")
        else:
            console.print("[dim]No models found[/dim]")
        raise typer.Exit(code=1)

    table = Table(header_style="blue", box=None)
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Type")
    table.add_column("Context")
    table.add_column("Output")
    table.add_column("Pricing")

    for m in results:
        table.add_row(
            m.provider_id,
            m.id,
            m.type if isinstance(m.type, str) else m.type.value,
            _comma_formatting(str(getattr(m, "context_length", None) or "-")),
            _comma_formatting(str(getattr(m, "max_output", None) or "-")),
            _pricing_label(m),
        )

    console.print(table)


# ----- sessions -----


sessions_cli = typer.Typer(help="Manage chat sessions.")
cli.add_typer(sessions_cli, name="sessions")


def _get_session_service() -> SessionService:
    config = JsonConfigStore().load()
    store = JsonSessionStore(config.storage_dir)
    return SessionService(store)


@sessions_cli.command(name="list")
def sessions_list():
    """List all sessions."""
    service = _get_session_service()
    all_sessions = service.list(newest_first=True)

    if not all_sessions:
        console.print("[dim]No sessions yet[/dim]")
        return

    table = Table(header_style="blue", box=None)
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Messages")
    table.add_column("Updated")

    for s in all_sessions:
        short_id = str(s.id)[:8]
        updated = s.updated_at.strftime("%Y-%m-%d %H:%M") if s.updated_at else "-"
        table.add_row(
            short_id,
            s.title,
            str(len(s.messages)),
            updated,
        )

    console.print(table)
    label = f"({len(all_sessions)} session{'s' if len(all_sessions) != 1 else ''})"
    console.print(f"[dim]{label}[/dim]")


@sessions_cli.command(name="get")
def sessions_get(
    session_id: str = typer.Argument(..., help="Session ID"),
):
    """Show session details."""
    service = _get_session_service()
    try:
        session = service.get(session_id)
    except ValueError as e:
        _error(str(e))
        raise typer.Exit(code=1)

    _style("ID", str(session.id))
    _style("Title", session.title)
    model_label = session.model.full_id if session.model else "(none)"
    _style("Model", model_label)
    _style("Messages", str(len(session.messages)))
    fmt = "%Y-%m-%d %H:%M:%S"
    created = session.created_at.strftime(fmt) if session.created_at else "-"
    _style("Created", created)
    updated = session.updated_at.strftime(fmt) if session.updated_at else "-"
    _style("Updated", updated)
    if session.system_prompt:
        _style("System Prompt", session.system_prompt)


@sessions_cli.command(name="delete")
def sessions_delete(
    session_id: str = typer.Argument(..., help="Session ID"),
):
    """Delete a session."""
    service = _get_session_service()
    try:
        service.delete(session_id)
    except ValueError as e:
        _error(str(e))
        raise typer.Exit(code=1)

    _success(f"Session {session_id} deleted")


@sessions_cli.command(name="rename")
def sessions_rename(
    session_id: str = typer.Argument(..., help="Session ID"),
    title: str = typer.Argument(..., help="New title"),
):
    """Rename a session."""
    service = _get_session_service()
    try:
        service.rename(session_id, title)
    except ValueError as e:
        _error(str(e))
        raise typer.Exit(code=1)

    _success(f"Session renamed to '{title}'")
