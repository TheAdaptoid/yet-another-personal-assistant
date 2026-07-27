"""OpenAI inference provider implementation."""

from typing import cast

from yapa.models import ModelData
from yapa.services.config import Config, ProviderConfig

from ..openai_compat import OpenAICompatibleProvider

_MODEL_METADATA: dict[str, dict[str, object]] = {
    "gpt-5.6-sol": {
        "context_length": 1_050_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.6": {
        "context_length": 1_050_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.6-terra": {
        "context_length": 1_050_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.6-luna": {
        "context_length": 1_050_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.5": {
        "context_length": 1_000_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.4": {
        "context_length": 400_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.4-mini": {
        "context_length": 400_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.4-nano": {
        "context_length": 400_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
}


class OpenAIIP(OpenAICompatibleProvider):
    """Inference provider for OpenAI."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: Config):
        """Initialize the OpenAI provider."""
        pc = config.provider_configs.get("openai", ProviderConfig())
        if pc.api_key is None:
            raise ValueError("OpenAI API key is not set.")
        super().__init__(
            identifier="openai",
            name="OpenAI",
            api_key=pc.api_key,
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )

    def _format_model(self, model_id: str) -> ModelData:
        model = super()._format_model(model_id)
        meta = _MODEL_METADATA.get(model_id)
        if meta is not None:
            return ModelData(
                id=model.id,
                provider_id=model.provider_id,
                type=model.type,
                context_length=cast(int | None, meta["context_length"]),
                max_output=cast(int | None, meta["max_output"]),
                supports_tools=cast(bool, meta["supports_tools"]),
                supports_vision=cast(bool, meta["supports_vision"]),
            )
        return model
