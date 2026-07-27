"""FastAPI dependency injection for service layer."""

from fastapi import Request

from yapa.services import ChatService, ModelService, SessionService


def get_session_service(request: Request) -> SessionService:
    """Return the session service from app state."""
    return request.app.state.session_service


def get_model_service(request: Request) -> ModelService:
    """Return the model service from app state."""
    return request.app.state.model_service


def get_chat_service(request: Request) -> ChatService:
    """Return the chat service from app state."""
    return request.app.state.chat_service
