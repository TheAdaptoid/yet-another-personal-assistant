"""Tests for chat conversation handler."""

import io
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from rich.console import Console

from yapa.cli.chat import run_conversation
from yapa.config import Config
from yapa.models import (
    AssistantMessage,
    ModelData,
    ModelType,
    Session,
    StreamDelta,
    UserMessage,
)
from yapa.services import ConversationService
from yapa.storage import GenericStore


def _make_session(**overrides):
    s = MagicMock()
    s.id = overrides.get("id", "test-id-1234")
    s.title = overrides.get("title", "Test Session")
    s.messages = overrides.get("messages", [])
    return s


class TestRunConversation:
    """Tests for run_conversation with injected mocks."""

    @pytest.fixture
    def mock_service(self):
        svc = MagicMock()
        svc.start = AsyncMock(return_value=_make_session())
        svc.switch_session.return_value = _make_session(messages=[
            UserMessage(content="What is the capital of France?"),
            AssistantMessage(
                content="The capital of France is Paris.",
                model="test-model",
            ),
        ])
        svc.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        svc.messages = [
            UserMessage(content="What is the capital of France?"),
            AssistantMessage(
                content="The capital of France is Paris.",
                model="test-model",
            ),
        ]

        async def _stream(model, messages):
            yield StreamDelta(content=None, done=True)

        svc.stream_response = _stream
        svc.close = AsyncMock()
        svc.auto_title = AsyncMock(return_value=None)
        svc.resolve_model = AsyncMock(
            return_value=ModelData(
                id="test-model", provider_id="test", type=ModelType.LLM
            )
        )
        return svc

    @pytest.fixture
    def mock_console(self):
        con = Console(file=io.StringIO(), force_terminal=False, no_color=True)
        con.input = MagicMock(return_value="/exit")
        return con

    async def test_exits_immediately(self, mock_service, mock_console):
        """Typing 'exit' as the first prompt ends the loop."""
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        mock_console.input.assert_called_once_with("[blue]> [/blue]")

    async def test_resumes_session(
        self, mock_service, mock_console, seeded_session
    ):
        """Providing a session_id resumes the session and shows last 2 messages."""
        mock_console.input.return_value = "/exit"
        await run_conversation(
            session_id=str(seeded_session.id),
            service=mock_service,
            console=mock_console,
        )
        output = mock_console.file.getvalue()
        assert "resumed session" in output
        assert "new session" not in output
        assert "What is the capital of France?" in output
        assert "The capital of France is Paris." in output

    async def test_default_model_from_config(self, mock_console, tmp_path):
        """When model_id is None, falls back to config.default_model."""
        mock_console.input.return_value = "/exit"
        mock_provider = MagicMock()

        async def _invoke(model, messages, params=None):
            yield StreamDelta(content=None, reasoning_content=None, done=True)

        mock_provider.invoke_llm_stream = _invoke

        ps = MagicMock()
        ps.get_model = AsyncMock(
            return_value=ModelData(
                id="openrouter/free", provider_id="openrouter", type=ModelType.LLM
            )
        )
        ps.get_provider_by_model.return_value = mock_provider

        cfg = Config(
            default_model="openrouter:openrouter/free",
            storage_dir=tmp_path,
        )
        store = GenericStore[Session](
            storage_dir=tmp_path / "sessions",
            entity_type=Session,
        )
        svc = ConversationService(provider_service=ps, config=cfg, store=store)

        await run_conversation(
            model_id=None,
            service=svc,
            console=mock_console,
            config=cfg,
        )
        ps.get_model.assert_awaited_once_with("openrouter:openrouter/free")

    async def test_full_turn(self, mock_service, mock_console):
        """Sends a user message, receives an assistant reply, then exits."""
        mock_console.input.side_effect = ["hello", "/exit"]

        async def _stream(prompt, model=None):
            yield StreamDelta(content="Hi!", done=False)
            yield StreamDelta(content=None, done=True)

        mock_service.stream_response = _stream
        mock_service.auto_title = AsyncMock(return_value="My Title")

        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )

        assert mock_console.input.call_count == 2
        output = mock_console.file.getvalue()
        assert "Hi!" in output
        assert "Session titled: 'My Title'" in output
        mock_service.auto_title.assert_awaited_once()

    async def test_empty_input_skipped(self, mock_service, mock_console):
        """Empty input does not call stream_response."""
        mock_console.input.side_effect = ["", "/exit"]
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        mock_console.input.assert_called_with("[blue]> [/blue]")
        output = mock_console.file.getvalue()
        assert "Hi!" not in output

    async def test_whitespace_input_skipped(self, mock_service, mock_console):
        """Whitespace-only input does not call stream_response."""
        mock_console.input.side_effect = ["   ", "/exit"]
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        mock_console.input.assert_called_with("[blue]> [/blue]")
        output = mock_console.file.getvalue()
        assert "Hi!" not in output

    async def test_eof_exits_cleanly(self, mock_service, mock_console):
        """Ctrl+D (EOFError) exits the loop gracefully."""
        mock_console.input.side_effect = EOFError
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        mock_service.close.assert_awaited_once()

    async def test_bare_exit_does_not_exit(self, mock_service, mock_console):
        """Bare 'exit' (without slash) is sent to the LLM as a message."""
        mock_console.input.side_effect = ["exit", "/exit"]

        async def _stream(prompt, model=None):
            yield StreamDelta(content="echo", done=False)
            yield StreamDelta(content=None, done=True)

        mock_service.stream_response = _stream
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        output = mock_console.file.getvalue()
        assert "echo" in output


class TestSlashCommands:
    """Tests for slash commands in the chat loop."""

    @pytest.fixture
    def mock_service(self):
        svc = MagicMock()
        svc.start = AsyncMock(return_value=_make_session())
        svc.switch_session.return_value = _make_session(messages=[
            UserMessage(content="What is the capital of France?"),
            AssistantMessage(
                content="The capital of France is Paris.",
                model="test-model",
            ),
        ])
        svc.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        svc.close = AsyncMock()
        svc.auto_title = AsyncMock(return_value=None)

        async def _done(model, messages):
            yield StreamDelta(content=None, done=True)

        svc.stream_response = _done
        svc.resolve_model = AsyncMock(
            return_value=ModelData(
                id="test-model", provider_id="test", type=ModelType.LLM
            )
        )
        return svc

    @pytest.fixture
    def mock_console(self):
        return MagicMock(spec=Console)

    async def test_slash_exit(self, mock_service, mock_console):
        """'/exit' exits the loop."""
        mock_console.input.return_value = "/exit"
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        mock_console.input.assert_called_once_with("[blue]> [/blue]")

    async def test_slash_model(self, mock_service, mock_console, tmp_path):
        """'/model <id>' resolves the model and persists to config."""
        mock_service.resolve_model = AsyncMock(
            return_value=ModelData(
                id="new-provider/new-model",
                provider_id="real-provider",
                type=ModelType.LLM,
            )
        )
        mock_console.input.side_effect = ["/model new-provider/new-model", "/exit"]
        cfg = Config(default_model="old:old/model")
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
            config=cfg,
        )
        assert mock_service.model == ModelData(
            id="new-provider/new-model",
            provider_id="real-provider",
            type=ModelType.LLM,
        )
        assert cfg.default_model == "real-provider:new-provider/new-model"

    async def test_slash_model_missing_arg(self, mock_service, mock_console):
        """'/model' with no arg shows usage."""
        mock_console.input.side_effect = ["/model", "/exit"]
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        usage_calls = [
            c for c in mock_console.print.call_args_list if "Usage: /model" in str(c)
        ]
        assert len(usage_calls) >= 1

    async def test_slash_session(self, mock_service, mock_console):
        """'/session <id>' calls switch_session."""
        sid = "12345678-1234-5678-1234-567812345678"
        mock_console.input.side_effect = [f"/session {sid}", "/exit"]
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        mock_service.switch_session.assert_called_once_with(UUID(sid))

    async def test_slash_session_missing_arg(self, mock_service, mock_console):
        """'/session' with no arg shows usage."""
        mock_console.input.side_effect = ["/session", "/exit"]
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        usage_calls = [
            c for c in mock_console.print.call_args_list if "Usage: /session" in str(c)
        ]
        assert len(usage_calls) >= 1

    async def test_slash_help(self, mock_service, mock_console):
        """'/help' prints the command list."""
        mock_console.input.side_effect = ["/help", "/exit"]
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        help_calls = [
            c for c in mock_console.print.call_args_list
            if "Available commands" in str(c)
        ]
        assert len(help_calls) >= 1

    async def test_slash_unknown(self, mock_service, mock_console):
        """Unknown slash command shows error."""
        mock_console.input.side_effect = ["/bogus", "/exit"]
        await run_conversation(
            model_id="test-model",
            service=mock_service,
            console=mock_console,
        )
        err_calls = [
            c for c in mock_console.print.call_args_list
            if "Unknown command" in str(c)
        ]
        assert len(err_calls) >= 1


class TestSlashSessions:
    """Tests for the /sessions slash command."""

    @pytest.fixture
    def mock_service(self):
        svc = MagicMock()
        svc.start = AsyncMock(return_value=_make_session())
        svc.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        svc.close = AsyncMock()
        svc.auto_title = AsyncMock(return_value=None)
        svc.stream_response = AsyncMock()
        svc.resolve_model = AsyncMock(
            return_value=ModelData(
                id="test-model", provider_id="test", type=ModelType.LLM
            )
        )
        return svc

    @pytest.fixture
    def mock_console(self):
        return MagicMock(spec=Console)

    async def test_slash_sessions_lists_sessions(
        self, mock_service, mock_console
    ):
        """'/sessions' delegates to sessions.list_sessions()."""
        with patch("yapa.cli.sessions.list_sessions") as mock_list:
            mock_console.input.side_effect = ["/sessions", "/exit"]
            await run_conversation(
                model_id="test-model",
                service=mock_service,
                console=mock_console,
            )
            mock_list.assert_called_once()
