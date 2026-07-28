"""Stateless chat orchestrator — agentic loop with tool execution and approval."""

import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from uuid import UUID

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    Message,
    ModelData,
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
from yapa.models.inference import ToolCallDelta
from yapa.models.tool import ToolApprovalRequest, ToolApprovalResponse, ToolCall
from yapa.services.models import ModelService
from yapa.services.session import SessionService
from yapa.tools.registry import ToolRegistry


ToolApprovalGetter = Callable[
    [ToolApprovalRequest],
    Awaitable[ToolApprovalResponse],
]


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
        self._sessions = sessions
        self._models = models
        self._tools = tools

    async def stream(
        self,
        session_id: UUID,
        prompt: str,
        model: ModelData | None = None,
        get_approval: ToolApprovalGetter | None = None,
    ) -> AsyncGenerator[Event, None]:
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
        initial_prompt_msg = messages[-1]

        try:
            for _ in range(self.MAX_ITERATIONS):
                content_buffer = ""
                finish_reason: str | None = None
                usage = None
                raw_tool_calls: list[tuple[int, str | None, str | None, str | None]] = []

                async for delta in provider.stream_chat(
                    model=model,
                    messages=messages,
                    tools=self._tools.list_tools(),
                    params=params,
                ):
                    if delta.reasoning_content:
                        yield ReasoningEvent(content=delta.reasoning_content)
                    if delta.content:
                        content_buffer += delta.content
                        yield TextEvent(content=delta.content)
                    for tcd in delta.tool_calls:
                        _merge_tool_call_delta(raw_tool_calls, tcd)
                    if delta.finish_reason:
                        finish_reason = delta.finish_reason
                    if delta.usage:
                        usage = delta.usage

                # Assemble tool calls
                tool_calls = [
                    ToolCall(id=tc_id, tool_name=tc_name, arguments=json.loads(tc_args))
                    for _, tc_id, tc_name, tc_args in raw_tool_calls
                    if tc_id and tc_name and tc_args
                ]

                assistant_msg = AssistantMessage(
                    content=content_buffer or None,
                    tool_calls=tool_calls,
                    model=model.full_id,
                    usage=usage,
                )
                messages.append(assistant_msg)

                # No tool calls → done
                if not tool_calls:
                    if not content_buffer:
                        yield AgentErrorEvent(message="Model returned empty response")
                        return
                    self._sessions.add_messages(
                        str(session_id),
                        [initial_prompt_msg, assistant_msg],
                        model=model,
                    )
                    yield AgentDoneEvent(
                        content=content_buffer,
                        finish_reason=finish_reason,
                        usage=usage,
                    )
                    return

                # Execute tool calls
                for tc in tool_calls:
                    tool = self._tools.get_tool(tc.tool_name)

                    if tool is None:
                        yield ToolCallEvent(tool_name=tc.tool_name, arguments=tc.arguments, call_id=tc.id)
                        messages.append(ToolMessage(
                            content=f"Unknown tool: {tc.tool_name}",
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        ))
                        continue

                    yield ToolCallEvent(tool_name=tc.tool_name, arguments=tc.arguments, call_id=tc.id)

                    if tool.needs_approval:
                        yield ToolApprovalRequestEvent(
                            tool_name=tc.tool_name,
                            arguments=tc.arguments,
                            call_id=tc.id,
                        )
                        if get_approval is not None:
                            response = await get_approval(ToolApprovalRequest(
                                call_id=tc.id,
                                name=tc.tool_name,
                                arguments=tc.arguments,
                            ))
                            if not response.approved:
                                reason = response.reason or "No reason given"
                                messages.append(ToolMessage(
                                    content=f"Tool call denied: {reason}",
                                    tool_call_id=tc.id,
                                    tool_name=tc.tool_name,
                                ))
                                continue

                    try:
                        result = await tool.execute(**tc.arguments)
                        yield ToolResultEvent(tool_name=tc.tool_name, call_id=tc.id, result=result)
                        messages.append(ToolMessage(
                            content=_serialize_result(result),
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        ))
                    except Exception as e:
                        yield ToolResultEvent(tool_name=tc.tool_name, call_id=tc.id, result={"error": str(e)})
                        messages.append(ToolMessage(
                            content=f"Error: {e}",
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        ))

            # Exceeded max iterations
            yield AgentErrorEvent(message="Max iterations reached")

        except Exception as e:
            yield AgentErrorEvent(message=str(e))


def _merge_tool_call_delta(
    acc: list[tuple[int, str | None, str | None, str | None]],
    delta: ToolCallDelta,
) -> None:
    """Merge a tool call delta into the accumulator list."""
    while len(acc) <= delta.index:
        acc.append((len(acc), None, None, None))
    idx, tid, tname, targs = acc[delta.index]
    if delta.id:
        tid = delta.id
    if delta.name:
        tname = delta.name
    if delta.arguments:
        targs = (targs or "") + delta.arguments
    acc[delta.index] = (idx, tid, tname, targs)


def _serialize_result(result: object) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)