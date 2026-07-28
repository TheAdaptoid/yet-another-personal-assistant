"""Stateless chat orchestrator — single model invocation per stream() call."""

from collections.abc import AsyncGenerator
from uuid import UUID

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    Message,
    ModelData,
    SystemMessage,
    UserMessage,
)
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    Event,
    ReasoningEvent,
    TextEvent,
)
from yapa.services.models import ModelService
from yapa.services.session import SessionService


class ChatService:
    """Stateless orchestrator for a single model invocation."""

    def __init__(
        self,
        *,
        sessions: SessionService,
        models: ModelService,
        tools: object | None = None,
    ) -> None:
        """Initialize the chat service."""
        self._sessions = sessions
        self._models = models
        self._tools = tools

    async def stream(
        self,
        session_id: UUID,
        prompt: str,
        model: ModelData | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Stream a model response for the given prompt."""
        session = self._sessions.get(str(session_id))

        if model is None:
            if session.model is not None:
                model = session.model
            else:
                raise ValueError("No model specified")

        yield AgentStartEvent(model_id=model.full_id)

        provider = self._models.get_provider_by_model(model)

        messages: list[Message] = []
        if session.system_prompt is not None:
            messages.append(SystemMessage(content=session.system_prompt))
        messages.extend(session.messages)
        messages.append(UserMessage(content=prompt))

        params = session.inference_params or InferenceParams()

        content_buffer = ""
        finish_reason: str | None = None

        try:
            async for delta in provider.stream_chat(
                model=model,
                messages=messages,
                params=params,
            ):
                if delta.reasoning_content:
                    yield ReasoningEvent(content=delta.reasoning_content)
                if delta.content:
                    content_buffer += delta.content
                    yield TextEvent(content=delta.content)
                if delta.finish_reason:
                    finish_reason = delta.finish_reason
                usage = delta.usage
        except Exception as e:
            yield AgentErrorEvent(message=str(e))
            return

        if not content_buffer:
            yield AgentErrorEvent(message="Model returned empty response")
            return

        assistant_msg = AssistantMessage(
            content=content_buffer,
            model=model.full_id,
        )

        self._sessions.add_messages(
            str(session_id),
            [UserMessage(content=prompt), assistant_msg],
            model=model,
        )

        yield AgentDoneEvent(
            content=content_buffer,
            finish_reason=finish_reason,
            usage=usage,
        )
