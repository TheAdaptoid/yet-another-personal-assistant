"""Tests for Phase 1 event model."""

from datetime import datetime

import pytest

from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    Event,
    EventSource,
    EventType,
    ReasoningEvent,
    TextEvent,
)
from yapa.models.inference import TokenUsage


class TestEventType:
    def test_text_chunk(self):
        assert EventType.TEXT_CHUNK == "text_chunk"

    def test_reasoning_chunk(self):
        assert EventType.REASONING_CHUNK == "reasoning_chunk"

    def test_agent_start(self):
        assert EventType.AGENT_START == "agent_start"

    def test_agent_done(self):
        assert EventType.AGENT_DONE == "agent_done"

    def test_agent_error(self):
        assert EventType.AGENT_ERROR == "agent_error"

    def test_tool_events_added(self):
        assert EventType.TOOL_CALL == "tool_call"
        assert EventType.TOOL_RESULT == "tool_result"
        assert EventType.TOOL_APPROVAL_REQUEST == "tool_approval_request"


class TestEventBase:
    def test_type_is_required(self):
        event = Event(type=EventType.AGENT_START)
        assert event.type == EventType.AGENT_START

    def test_source_defaults_to_system(self):
        event = Event(type=EventType.AGENT_START)
        assert event.source == EventSource.SYSTEM

    def test_timestamp_is_datetime_with_tz(self):
        event = Event(type=EventType.AGENT_START)
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo is not None


class TestTextEvent:
    def test_creates_with_content(self):
        event = TextEvent(content="Hello")
        assert event.content == "Hello"
        assert event.type == EventType.TEXT_CHUNK
        assert event.source == EventSource.AGENT

    def test_content_empty_string(self):
        event = TextEvent(content="")
        assert event.content == ""


class TestReasoningEvent:
    def test_creates_with_content(self):
        event = ReasoningEvent(content="thinking...")
        assert event.content == "thinking..."
        assert event.type == EventType.REASONING_CHUNK
        assert event.source == EventSource.AGENT


class TestAgentStartEvent:
    def test_creates_with_model_id(self):
        event = AgentStartEvent(model_id="openai:gpt-4")
        assert event.model_id == "openai:gpt-4"
        assert event.type == EventType.AGENT_START
        assert event.source == EventSource.AGENT


class TestAgentDoneEvent:
    def test_creates_with_content(self):
        event = AgentDoneEvent(content="Full response")
        assert event.content == "Full response"
        assert event.type == EventType.AGENT_DONE
        assert event.source == EventSource.AGENT
        assert event.finish_reason is None
        assert event.usage is None

    def test_with_finish_reason(self):
        event = AgentDoneEvent(content="Done", finish_reason="stop")
        assert event.finish_reason == "stop"

    def test_with_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        event = AgentDoneEvent(content="Done", usage=usage)
        assert event.usage == usage


class TestAgentErrorEvent:
    def test_creates_with_message(self):
        event = AgentErrorEvent(message="Something went wrong")
        assert event.message == "Something went wrong"
        assert event.type == EventType.AGENT_ERROR
        assert event.source == EventSource.AGENT
