"""Tests for service-layer exception classes."""

import pytest

from yapa.services.exceptions import ChatError


class TestChatError:
    def test_is_exception(self):
        assert issubclass(ChatError, Exception)

    def test_can_be_raised(self):
        with pytest.raises(ChatError):
            raise ChatError("test error")

    def test_message_preserved(self):
        try:
            raise ChatError("something broke")
        except ChatError as e:
            assert str(e) == "something broke"
