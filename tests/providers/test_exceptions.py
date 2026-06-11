"""Tests for provider exception classes."""

import pytest

from yapa.providers.exceptions import (
    InferenceProviderError,
    ModelInvocationError,
    ModelsFetchError,
)


class TestInferenceProviderError:
    """Tests for InferenceProviderError."""

    def test_is_base_exception(self) -> None:
        assert issubclass(InferenceProviderError, Exception)

    def test_can_be_raised(self) -> None:
        with pytest.raises(InferenceProviderError):
            raise InferenceProviderError("test error")


class TestModelsFetchError:
    """Tests for ModelsFetchError."""

    def test_inherits_from_inference_provider_error(self) -> None:
        assert issubclass(ModelsFetchError, InferenceProviderError)

    def test_can_be_raised(self) -> None:
        with pytest.raises(ModelsFetchError):
            raise ModelsFetchError("fetch failed")


class TestModelInvocationError:
    """Tests for ModelInvocationError."""

    def test_inherits_from_inference_provider_error(self) -> None:
        assert issubclass(ModelInvocationError, InferenceProviderError)

    def test_can_be_raised(self) -> None:
        with pytest.raises(ModelInvocationError):
            raise ModelInvocationError("invoke failed")
