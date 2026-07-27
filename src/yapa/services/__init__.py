"""Service layer — UI-agnostic business logic."""

from __future__ import annotations

import typing as _t

__all__ = [
    "ChatError",
    "ChatService",
    "Config",
    "ConfigStore",
    "JsonConfigStore",
    "JsonSessionStore",
    "ModelService",
    "ProviderConfig",
    "SessionService",
    "SessionStore",
]

_modules: dict[str, tuple[str, str]] = {
    "ChatError": (".exceptions", "ChatError"),
    "ChatService": (".chat", "ChatService"),
    "Config": (".config", "Config"),
    "ConfigStore": (".config", "ConfigStore"),
    "JsonConfigStore": (".config", "JsonConfigStore"),
    "JsonSessionStore": (".store", "JsonSessionStore"),
    "ModelService": (".models", "ModelService"),
    "ProviderConfig": (".config", "ProviderConfig"),
    "SessionService": (".session", "SessionService"),
    "SessionStore": (".store", "SessionStore"),
}


def __getattr__(name: str) -> _t.Any:
    if name in _modules:
        mod, attr = _modules[name]
        import importlib as _il

        module = _il.import_module(mod, __package__)
        return getattr(module, attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
