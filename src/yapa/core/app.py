"""FastAPI application factory."""

from fastapi import FastAPI

from yapa.core.routers import chat_router, sessions_router


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="YAPA Core API",
        description="Async JSON-backed chat session service.",
        version="0.1.0",
    )

    app.include_router(sessions_router)
    app.include_router(chat_router)

    return app


app = create_app()
