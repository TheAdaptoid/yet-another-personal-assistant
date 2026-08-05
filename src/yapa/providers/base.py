"""Inference provider base class and utilities."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from yapa.logging import get_logger
from yapa.models import (
    AssistantMessage,
    EmbeddingResult,
    EmbedModel,
    InferenceParams,
    LanguageModel,
    Message,
    ModelDataUnion,
    ModelType,
    ReasoningEffort,
    StreamEvent,
)
from yapa.tools import Tool

from .exceptions import ModelInvocationError, ModelsFetchError, ModelTypeError


def _require_language_model(model, operation: str) -> None:
    if not isinstance(model, LanguageModel):
        raise ModelTypeError(
            f"Model '{getattr(model, 'id', '?')}' is not an LLM; "
            f"'{operation}' requires a LanguageModel."
        )


def _require_embed_model(model, operation: str) -> None:
    from yapa.models import EmbedModel

    if not isinstance(model, EmbedModel):
        raise ModelTypeError(
            f"Model '{getattr(model, 'id', '?')}' is not an embedding model; "
            f"'{operation}' requires an EmbedModel."
        )


def _require_provider_id(model, provider_id: str, operation: str) -> None:
    if getattr(model, "provider_id", None) != provider_id:
        raise ModelTypeError(
            f"Model '{model.full_id}' does not belong to provider '{provider_id}'; "
            f"cannot '{operation}'."
        )


class InferenceProvider(ABC):
    """Abstract base class for inference providers."""

    def __init__(self, identifier: str, name: str) -> None:
        """Initialize the provider with an identifier and display name."""
        self._id = identifier
        self._name = name
        self._logger = get_logger(f"inference_provider.{identifier}")

    @property
    def id(self) -> str:
        """Return the unique identifier for this provider."""
        return self._id

    @property
    def name(self) -> str:
        """Return the human-readable name of this provider."""
        return self._name

    # ── Public methods (logging + error wrapping) ──

    async def list_models(
        self, model_type: ModelType | None = None
    ) -> list[ModelDataUnion]:
        """Retrieve available models, optionally filtered by type."""
        self._logger.info("Fetching models...")
        try:
            return await self._list_models_impl(model_type)
        except ModelsFetchError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to fetch models: {e}")
            raise ModelsFetchError(
                f"Failed to fetch models from provider '{self.id}': {e}"
            ) from e

    async def get_model(self, model_id: str) -> ModelDataUnion:
        """Retrieve detailed information about a specific model."""
        self._logger.info(f"Fetching model '{model_id}'...")
        try:
            return await self._get_model_impl(model_id)
        except ModelsFetchError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to fetch model '{model_id}': {e}")
            raise ModelsFetchError(
                f"Failed to fetch model '{model_id}' from provider '{self.id}': {e}"
            ) from e

    async def stream_chat(
        self,
        model: LanguageModel,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Invoke the model and stream the response as events."""
        _require_language_model(model, "stream_chat")
        _require_provider_id(model, self.id, "stream_chat")
        try:
            async for event in self._stream_chat_impl(
                model_id=model.id,
                messages=messages,
                tools=tools,
                params=params,
                reasoning=reasoning,
            ):
                yield event
        except ModelInvocationError:
            raise
        except Exception as e:
            self._logger.error(
                f"Streaming model invocation failed for '{model.id}': {e}"
            )
            raise ModelInvocationError(
                f"Streaming model invocation from provider '{self.id}' "
                f"failed for '{model.id}': {e}"
            ) from e

    async def static_chat(
        self,
        model: LanguageModel,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AssistantMessage:
        """Invoke the model and return the complete response."""
        _require_language_model(model, "static_chat")
        _require_provider_id(model, self.id, "static_chat")
        try:
            return await self._static_chat_impl(
                model_id=model.id,
                messages=messages,
                tools=tools,
                params=params,
                reasoning=reasoning,
            )
        except ModelInvocationError:
            raise
        except Exception as e:
            self._logger.error(f"Model invocation failed for '{model.id}': {e}")
            raise ModelInvocationError(
                f"Model invocation from provider '{self.id}' "
                f"failed for '{model.id}': {e}"
            ) from e

    async def embed(self, model: EmbedModel, input: str | list[str]) -> EmbeddingResult:
        """Embed the given input and return the resulting vectors."""
        _require_embed_model(model, "embed")
        _require_provider_id(model, self.id, "embed")
        try:
            return await self._embed_impl(model_id=model.id, input=input)
        except ModelInvocationError:
            raise
        except Exception as e:
            self._logger.error(f"Embedding failed for '{model.id}': {e}")
            raise ModelInvocationError(
                f"Embedding from provider '{self.id}' failed for '{model.id}': {e}"
            ) from e

    # ── Private implementation methods ──

    @abstractmethod
    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelDataUnion]: ...

    @abstractmethod
    async def _get_model_impl(self, model_id: str) -> ModelDataUnion: ...

    @abstractmethod
    def _stream_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AsyncGenerator[StreamEvent, None]: ...

    @abstractmethod
    async def _static_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AssistantMessage: ...

    @abstractmethod
    async def _embed_impl(
        self, model_id: str, input: str | list[str]
    ) -> EmbeddingResult: ...
