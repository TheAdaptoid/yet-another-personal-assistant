"""Tests for service-layer exception classes."""

import pytest

from yapa.services.exceptions import ConversationError


class TestConversationError:
    """Tests for ConversationError."""

    def test_is_exception(self) -> None:
        assert issubclass(ConversationError, Exception)

    def test_can_be_raised(self) -> None:
        with pytest.raises(ConversationError):
            raise ConversationError("test error")
