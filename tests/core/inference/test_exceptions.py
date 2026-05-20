"""Tests for inference exception classes."""

from yapa.core.inference import InferenceError, ModelsFetchError
from yapa.core.inference.exceptions import ModelNotFoundError


class TestInferenceError:
    """InferenceError — raised when a model invocation fails."""

    def test_stores_model_id(self):
        """The model_id passed to the constructor is stored."""
        err = InferenceError("msg", model_id="gpt-4")
        assert err.model_id == "gpt-4"

    def test_stores_cause(self):
        """The cause passed to the constructor is stored and set as __cause__."""
        cause = ValueError("original")
        err = InferenceError("msg", cause=cause)
        assert err.cause is cause
        assert err.__cause__ is cause

    def test_no_model_id_defaults_to_none(self):
        """When model_id is omitted it defaults to None."""
        err = InferenceError("msg")
        assert err.model_id is None

    def test_no_cause_defaults_to_none(self):
        """When cause is omitted it defaults to None and __cause__ is not set."""
        err = InferenceError("msg")
        assert err.cause is None
        assert err.__cause__ is None


class TestModelNotFoundError:
    """ModelNotFoundError — raised when a model has no matching provider."""

    def test_stores_model_id(self):
        """The model_id passed to the constructor is stored."""
        err = ModelNotFoundError("gpt-4")
        assert err.model_id == "gpt-4"

    def test_message_includes_model_id(self):
        """The error message contains the model ID."""
        err = ModelNotFoundError("gpt-4")
        assert "gpt-4" in str(err)


class TestModelsFetchError:
    """ModelsFetchError — raised when fetching models from a provider fails."""

    def test_stores_provider(self):
        """The provider name passed to the constructor is stored."""
        err = ModelsFetchError("OpenRouter")
        assert err.provider == "OpenRouter"

    def test_message_includes_provider(self):
        """The error message contains the provider name."""
        err = ModelsFetchError("OpenRouter")
        assert "OpenRouter" in str(err)

    def test_stores_cause(self):
        """The cause passed to the constructor is stored and set as __cause__."""
        cause = ConnectionError("timeout")
        err = ModelsFetchError("LM Studio", cause=cause)
        assert err.cause is cause
        assert err.__cause__ is cause
