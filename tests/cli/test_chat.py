"""Tests for chat conversation handler."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from yapa.cli.chat import run_conversation
from yapa.config import Config
from yapa.models import StreamDelta


class TestRunConversation:
    """Tests for run_conversation with injected mocks."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()

        async def _invoke(model, messages):
            yield StreamDelta(content="", done=True)

        provider.invoke_model = _invoke
        return provider

    @pytest.fixture
    def mock_provider_manager(self, mock_provider):
        pm = MagicMock()
        pm.get_provider_by_model = AsyncMock(return_value=mock_provider)
        return pm

    @pytest.fixture
    def mock_console(self):
        con = MagicMock(spec=Console)
        con.input.return_value = "exit"
        return con

    async def test_exits_immediately(self, mock_provider_manager, mock_console):
        """Typing 'exit' as the first prompt ends the loop."""
        await run_conversation(
            model="test-model",
            provider_manager=mock_provider_manager,
            console=mock_console,
        )
        mock_console.input.assert_called_once_with("[blue]You: [/blue]")

    async def test_resumes_session(
        self, mock_provider_manager, mock_console, seeded_session
    ):
        """Providing a session_id resumes the existing session."""
        mock_console.input.return_value = "exit"
        await run_conversation(
            model="test-model",
            session_id=seeded_session.id,
            provider_manager=mock_provider_manager,
            console=mock_console,
        )
        resume_calls = [
            c for c in mock_console.print.call_args_list if "Resumed session" in str(c)
        ]
        started_calls = [
            c
            for c in mock_console.print.call_args_list
            if "Started new session" in str(c)
        ]
        assert len(resume_calls) >= 1
        assert len(started_calls) == 0

    async def test_default_model_from_config(self, mock_provider_manager, mock_console):
        """When model is None, falls back to config.default_model."""
        mock_console.input.return_value = "exit"
        cfg = Config(default_model="cfg-default-model")
        await run_conversation(
            model=None,
            provider_manager=mock_provider_manager,
            console=mock_console,
            config=cfg,
        )
        mock_provider_manager.get_provider_by_model.assert_awaited_once_with(
            "cfg-default-model"
        )

    async def test_full_turn(self, mock_provider_manager, mock_console):
        """Sends a user message, receives an assistant reply, then exits."""
        mock_console.input.side_effect = ["hello", "exit"]

        provider = mock_provider_manager.get_provider_by_model.return_value

        async def _invoke(model, messages):
            yield StreamDelta(content="Hi!", done=True)

        provider.invoke_model = _invoke

        await run_conversation(
            model="test-model",
            provider_manager=mock_provider_manager,
            console=mock_console,
        )

        assert mock_console.input.call_count == 2
        assistant_calls = [
            c for c in mock_console.print.call_args_list if "Hi!" in str(c)
        ]
        assert len(assistant_calls) >= 1
