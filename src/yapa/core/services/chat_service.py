"""Chat service for managing WebSocket chat interactions."""

import logging

from yapa.core.inference.exceptions import InferenceError, ModelNotFoundError
from yapa.core.services.inference_service import InferenceService
from yapa.core.services.session_service import SessionService
from yapa.shared.models import (
    ChatResponse,
    ModelData,
    UserMessage,
)


class ChatService:
    """Orchestrates chat message processing between sessions and inference."""

    def __init__(
        self,
        logger: logging.Logger,
        session_service: SessionService,
        inference_service: InferenceService,
    ) -> None:
        """
        Initialize the chat service.

        Args:
            logger (logging.Logger): Logger instance.
            session_service (SessionService): Service for session management.
            inference_service (InferenceService): Service for model inference.
        """
        self._logger = logger
        self._session_service = session_service
        self._inference_service = inference_service

    async def process_message(
        self,
        session_id: str,
        model: ModelData,
        message: str,
        retry: bool = False,
    ) -> ChatResponse:
        """
        Process a chat message within a session.

        Handles loading/appending to the session, running inference, and
        persisting the assistant response. On retry, re-uses the existing
        messages (the last one must be a UserMessage).

        Args:
            session_id (str): The session to operate on.
            model (ModelData): The model to use for inference.
            message (str): The user's message text.
            retry (bool): If True, skip appending the user message and
                re-run inference on the existing session.

        Returns:
            ChatResponse: The result of processing, with error details
                if anything went wrong.
        """
        if retry:
            session = await self._session_service.get_session(session_id)
            if session is None:
                return ChatResponse(
                    response="",
                    done=True,
                    error=f"Session '{session_id}' not found for retry",
                )
            if not session.messages or not isinstance(
                session.messages[-1], UserMessage
            ):
                self._logger.warning(
                    "Retry requested on session %s but last message is not a "
                    "UserMessage",
                    session_id,
                )
                return ChatResponse(
                    response="",
                    done=True,
                    error="Cannot retry: last message is not a user message",
                )
        else:
            session = await self._session_service.append_message(
                session_id, UserMessage(content=message)
            )
            if session is None:
                return ChatResponse(
                    response="",
                    done=True,
                    error=f"Session '{session_id}' not found",
                )

        try:
            assistant_msg = await self._inference_service.invoke_model(
                model=model, messages=session.messages
            )
        except ModelNotFoundError:
            self._logger.exception("Model not found during inference")
            return ChatResponse(
                response="",
                done=True,
                error=f"Model '{model.id}' is not available from any provider",
            )
        except InferenceError:
            self._logger.exception("Inference failed")
            return ChatResponse(
                response="",
                done=True,
                error="Model inference failed. You can retry.",
            )
        except Exception:
            self._logger.exception("Unexpected inference error")
            return ChatResponse(
                response="",
                done=True,
                error="An unexpected error occurred during inference. You can retry.",
            )

        session = await self._session_service.append_message(
            session_id, assistant_msg
        )
        if session is None:
            self._logger.error(
                "Session %s disappeared after inference", session_id
            )
            return ChatResponse(
                response=assistant_msg.content,
                done=True,
                error="Response was generated but could not be saved to session",
            )

        return ChatResponse(response=assistant_msg.content, done=True)
