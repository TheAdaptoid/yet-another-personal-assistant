"""OpenAI inference provider implementation."""

from yapa.config import UNSET, Config
from yapa.models import ModelData, ModelType

from ..openai_compat import OpenAICompatibleProvider

_MODEL_METADATA: dict[str, dict[str, object]] = {
    "gpt-4o": {
        "context_length": 128000,
        "max_output": 16384,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-4o-mini": {
        "context_length": 128000,
        "max_output": 16384,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-4-turbo": {
        "context_length": 128000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-4": {
        "context_length": 8192,
        "max_output": 4096,
        "supports_tools": True,
        "supports_vision": False,
    },
    "gpt-3.5-turbo": {
        "context_length": 16385,
        "max_output": 4096,
        "supports_tools": True,
        "supports_vision": False,
    },
    "o1": {
        "context_length": 200000,
        "max_output": 100000,
        "supports_tools": True,
        "supports_vision": True,
    },
    "o1-mini": {
        "context_length": 128000,
        "max_output": 65536,
        "supports_tools": True,
        "supports_vision": True,
    },
    "o3-mini": {
        "context_length": 200000,
        "max_output": 100000,
        "supports_tools": True,
        "supports_vision": True,
    },
}


class OpenAIIP(OpenAICompatibleProvider):
    """Inference provider for OpenAI."""

    def __init__(self, config: Config):
        """
        Initialize the OpenAI provider.

        Args:
            config: Application config containing the OpenAI API key and base URL.
        """
        if config.openai_api_key in (None, UNSET):
            raise ValueError("OpenAI API key is not set.")
        super().__init__(
            identifier="openai",
            name="OpenAI",
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            timeout=config.provider_timeout,
        )

    def _format_model(self, model_id: str) -> ModelData:
        model = super()._format_model(model_id)
        meta = _MODEL_METADATA.get(model_id)
        if meta is not None:
            return ModelData(
                id=model.id,
                provider_id=model.provider_id,
                type=model.type,
                context_length=meta["context_length"],  # type: ignore[arg-type]
                max_output=meta["max_output"],  # type: ignore[arg-type]
                supports_tools=meta["supports_tools"],  # type: ignore[arg-type]
                supports_vision=meta["supports_vision"],  # type: ignore[arg-type]
            )
        return model
