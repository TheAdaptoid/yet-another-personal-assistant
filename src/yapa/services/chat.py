"""Stateless chat orchestrator — agentic loop with tool execution and approval."""

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from uuid import UUID

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    LanguageModel,
    Message,
    ReasoningEffort,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    Event,
    ReasoningEvent,
    TextEvent,
    ToolApprovalRequestEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from yapa.models.stream import (
    ContentDelta,
    ReasoningDelta,
    StreamEndEvent,
    TokenUsage,
    ToolCallDeltaEvent,
)
from yapa.models.tool import ToolApprovalRequest, ToolApprovalResponse, ToolCall
from yapa.services.models import ModelService
from yapa.services.session import SessionService
from yapa.tools.registry import ToolRegistry

ToolApprovalGetter = Callable[
    [ToolApprovalRequest],
    Awaitable[ToolApprovalResponse],
]


def _assemble_tool_calls(raw_tool_calls):
    """Accumulate streamed tool-call deltas into ToolCalls (REQ-SERV-03)."""
    merged: dict[int, ToolCallDeltaEvent] = {}
    for tcd in raw_tool_calls:
        cur = merged.get(tcd.index)
        if cur is None:
            merged[tcd.index] = tcd
            continue
        merged[tcd.index] = ToolCallDeltaEvent(
            index=tcd.index,
            id=tcd.id or cur.id,
            name=tcd.name or cur.name,
            arguments=(cur.arguments or "") + (tcd.arguments or ""),
        )

    tool_calls: list[ToolCall] = []
    for idx in sorted(merged):
        tcd = merged[idx]
        if not (tcd.id and tcd.name):
            continue
        args = tcd.arguments or ""
        if args.strip() == "":
            parsed: dict = {}
        else:
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                parsed = {}
        tool_calls.append(ToolCall(id=tcd.id, tool_name=tcd.name, arguments=parsed))
    return tool_calls


class ChatService:
    """Stateless orchestrator for the agentic loop with tool execution."""

    MAX_ITERATIONS = 10

    def __init__(
        self,
        *,
        sessions: SessionService,
        models: ModelService,
        tools: ToolRegistry,
    ) -> None:
        """Initialize with session, model, and tool services."""
        self._sessions = sessions
        self._models = models
        self._tools = tools

    async def stream(
        self,
        session_id: UUID,
        prompt: str,
        model: LanguageModel | None = None,
        reasoning: ReasoningEffort | None = None,
        get_approval: ToolApprovalGetter | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Run the agentic loop: stream model response, execute tools, repeat."""
        session = self._sessions.get(str(session_id))
        model = self._resolve_model(session, model, reasoning)

        yield AgentStartEvent(model_id=model.full_id)

        try:
            provider = self._models.get_provider_by_model(model)
            messages, user_msg = self._build_initial_messages(session, prompt)
            start = len(messages)
            params = session.inference_params or InferenceParams()

            for _ in range(self.MAX_ITERATIONS):
                (
                    events,
                    content_buffer,
                    reasoning_buffer,
                    finish_reason,
                    usage,
                    raw_tool_calls,
                ) = await self._process_stream_deltas(
                    provider, model, messages, params, reasoning
                )
                for event in events:
                    yield event

                tool_calls = _assemble_tool_calls(raw_tool_calls)

                assistant_msg = AssistantMessage(
                    content=content_buffer or None,
                    reasoning_content=reasoning_buffer or None,
                    tool_calls=tool_calls,
                    model=model.full_id,
                    usage=usage,
                )
                messages.append(assistant_msg)

                if not tool_calls:
                    yield self._finalize_turn(
                        session_id=session_id,
                        user_msg=user_msg,
                        assistant_msg=assistant_msg,
                        model=model,
                        content_buffer=content_buffer,
                        finish_reason=finish_reason,
                        usage=usage,
                    )
                    return

                tool_events, messages = await self._execute_tool_calls(
                    tool_calls=tool_calls,
                    messages=messages,
                    get_approval=get_approval,
                )
                for event in tool_events:
                    yield event

            self._sessions.add_messages(
                str(session_id), [user_msg] + messages[start:], model=model
            )
            yield AgentErrorEvent(message="Max iterations reached")

        except Exception as e:
            yield AgentErrorEvent(message=str(e))

    def _resolve_model(
        self,
        session,
        model: LanguageModel | None,
        reasoning: ReasoningEffort | None,
    ) -> LanguageModel:
        """Resolve model from session fallback and validate reasoning support."""
        if model is None:
            if session.model is not None:
                model = session.model
            else:
                raise ValueError("No model specified")
        if reasoning is not None and reasoning != ReasoningEffort.OFF:
            if not model.supports_reasoning:
                raise ValueError(
                    f"Model '{model.full_id}' does not support reasoning. "
                    f"Use reasoning=off or choose a reasoning-capable model."
                )
        return model

    def _build_initial_messages(
        self,
        session,
        prompt: str,
    ) -> tuple[list[Message], Message]:
        """Build the message list and return it with the user prompt message."""
        messages: list[Message] = []
        if session.system_prompt is not None:
            messages.append(SystemMessage(content=session.system_prompt))
        messages.extend(session.messages)
        user_msg = UserMessage(content=prompt)
        messages.append(user_msg)
        return messages, user_msg

    def _finalize_turn(
        self,
        *,
        session_id: UUID,
        user_msg: Message,
        assistant_msg: AssistantMessage,
        model: LanguageModel,
        content_buffer: str,
        finish_reason: str | None,
        usage: TokenUsage | None,
    ) -> Event:
        """Persist final turn and return the completion event."""
        if not content_buffer:
            return AgentErrorEvent(message="Model returned empty response")
        self._sessions.add_messages(
            str(session_id),
            [user_msg, assistant_msg],
            model=model,
        )
        return AgentDoneEvent(
            content=content_buffer,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _process_stream_deltas(
        self,
        provider,
        model: LanguageModel,
        messages: list[Message],
        params: InferenceParams,
        reasoning: ReasoningEffort | None,
    ) -> tuple[
        list[Event],
        str,
        str,
        str | None,
        TokenUsage | None,
        list[ToolCallDeltaEvent],
    ]:
        """Stream events from provider, emit events, and return accumulators."""
        events: list[Event] = []
        content_buffer = ""
        reasoning_buffer = ""
        finish_reason: str | None = None
        usage = None
        raw_tool_calls: list[ToolCallDeltaEvent] = []

        async for event in provider.stream_chat(
            model=model,
            messages=messages,
            tools=self._tools.list_tools(),
            params=params,
            reasoning=reasoning,
        ):
            if isinstance(event, ReasoningDelta):
                reasoning_buffer += event.content
                events.append(ReasoningEvent(content=event.content))
            elif isinstance(event, ContentDelta):
                content_buffer += event.content
                events.append(TextEvent(content=event.content))
            elif isinstance(event, ToolCallDeltaEvent):
                raw_tool_calls.append(event)
            elif isinstance(event, StreamEndEvent):
                finish_reason = event.finish_reason or finish_reason
                if event.usage is not None:
                    usage = event.usage

        return (
            events,
            content_buffer,
            reasoning_buffer,
            finish_reason,
            usage,
            raw_tool_calls,
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        messages: list[Message],
        get_approval: ToolApprovalGetter | None,
    ) -> tuple[list[Event], list[Message]]:
        """Execute tool calls, collect events, return updated messages."""
        events: list[Event] = []
        for tc in tool_calls:
            tool = self._tools.get_tool(tc.tool_name)

            if tool is None:
                events.append(
                    ToolCallEvent(
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        call_id=tc.id,
                    )
                )
                messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {tc.tool_name}",
                        tool_call_id=tc.id,
                        tool_name=tc.tool_name,
                    )
                )
                continue

            events.append(
                ToolCallEvent(
                    tool_name=tc.tool_name, arguments=tc.arguments, call_id=tc.id
                )
            )

            if tool.needs_approval:
                events.append(
                    ToolApprovalRequestEvent(
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        call_id=tc.id,
                    )
                )
                if get_approval is None:
                    messages.append(
                        ToolMessage(
                            content="Tool call denied: no approval callback provided",
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        )
                    )
                    continue
                try:
                    response = await get_approval(
                        ToolApprovalRequest(
                            call_id=tc.id,
                            name=tc.tool_name,
                            arguments=tc.arguments,
                        )
                    )
                except asyncio.TimeoutError:
                    messages.append(
                        ToolMessage(
                            content="Error: approval timeout",
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        )
                    )
                    continue
                except Exception as e:
                    messages.append(
                        ToolMessage(
                            content=f"Tool call denied: approval failed ({e})",
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        )
                    )
                    continue
                if not response.approved:
                    reason = response.reason or "No reason given"
                    messages.append(
                        ToolMessage(
                            content=f"Tool call denied: {reason}",
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        )
                    )
                    continue

            try:
                result = await tool.execute(**tc.arguments)
                events.append(
                    ToolResultEvent(
                        tool_name=tc.tool_name, call_id=tc.id, result=result
                    )
                )
                messages.append(
                    ToolMessage(
                        content=_serialize_result(result),
                        tool_call_id=tc.id,
                        tool_name=tc.tool_name,
                    )
                )
            except Exception as e:
                events.append(
                    ToolResultEvent(
                        tool_name=tc.tool_name,
                        call_id=tc.id,
                        result={"error": str(e)},
                    )
                )
                messages.append(
                    ToolMessage(
                        content=f"Error: {e}",
                        tool_call_id=tc.id,
                        tool_name=tc.tool_name,
                    )
                )
        return events, messages


def _serialize_result(result: object) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)
