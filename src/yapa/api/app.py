"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from yapa.providers import ProviderNotAvailableError
from yapa.services import ChatService, ModelService, SessionService
from yapa.services.config import Config, JsonConfigStore
from yapa.services.exceptions import ChatError
from yapa.services.store import JsonSessionStore
from yapa.tools import ToolRegistry
from yapa.tools.core import default_tools

from .routes import health, models, sessions
from .websocket import chat as chat_ws


def _build_services(config: Config):
    store = JsonSessionStore(config.storage_dir)
    session_service = SessionService(store)
    model_service = ModelService()
    tool_registry = ToolRegistry(default_tools())
    chat_service = ChatService(
        sessions=session_service,
        models=model_service,
        tools=tool_registry,
    )
    return session_service, model_service, chat_service, tool_registry


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if not all(
        hasattr(app.state, name)
        for name in (
            "session_service",
            "model_service",
            "chat_service",
            "tool_registry",
        )
    ):
        (
            app.state.session_service,
            app.state.model_service,
            app.state.chat_service,
            app.state.tool_registry,
        ) = _build_services(app.state.config)

    yield


def create_app(config: Config | None = None) -> FastAPI:
    """Create a configured FastAPI application."""
    if config is None:
        config = JsonConfigStore().load()

    app = FastAPI(title="YAPA", lifespan=_lifespan)
    app.state.config = config

    @app.exception_handler(ValueError)
    def _value_error(_request: Request, exc: ValueError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ProviderNotAvailableError)
    def _provider_error(_request: Request, exc: ProviderNotAvailableError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ChatError)
    def _chat_error(_request: Request, exc: ChatError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    def _generic_error(_request: Request, exc: Exception):
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )

    app.include_router(health.router, prefix=config.api_prefix)
    app.include_router(sessions.router, prefix=config.api_prefix)
    app.include_router(models.router, prefix=config.api_prefix)
    app.include_router(chat_ws.router, prefix=config.api_prefix)

    return app
