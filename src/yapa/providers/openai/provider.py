"""OpenAI inference provider implementation."""

from typing import cast

from yapa.models import LanguageModel, ModelPricing
from yapa.services.config import Config, ProviderConfig

from .openai_compat import OpenAICompatibleProvider


class OpenAIIP(OpenAICompatibleProvider):
    """Inference provider for OpenAI."""

    PROVIDER_ID = "openai"

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    _MODEL_METADATA: dict[str, dict[str, object]] = {
        "gpt-5.6-sol": {
            "name": "GPT-5.6 Sol",
            "context_length": 1_050_000,
            "max_output": 131072,
            "supports_tools": True,
            "supports_vision": True,
            "supports_reasoning": True,
            "pricing": ModelPricing(input=15.0, output=60.0, request=0.0),
        },
        "gpt-5.6": {
            "name": "GPT-5.6",
            "context_length": 1_050_000,
            "max_output": 131072,
            "supports_tools": True,
            "supports_vision": True,
            "supports_reasoning": True,
            "pricing": ModelPricing(input=1.25, output=10.0),
        },
        "gpt-5.6-terra": {
            "name": "GPT-5.6 Terra",
            "context_length": 1_050_000,
            "max_output": 131072,
            "supports_tools": True,
            "supports_vision": True,
            "supports_reasoning": True,
            "pricing": ModelPricing(input=2.5, output=15.0),
        },
        "gpt-5.6-luna": {
            "name": "GPT-5.6 Luna",
            "context_length": 1_050_000,
            "max_output": 131072,
            "supports_tools": True,
            "supports_vision": True,
            "supports_reasoning": True,
            "pricing": ModelPricing(input=5.0, output=25.0),
        },
        "gpt-5.5": {
            "name": "GPT-5.5",
            "context_length": 1_000_000,
            "max_output": 131072,
            "supports_tools": True,
            "supports_vision": True,
            "supports_reasoning": True,
            "pricing": ModelPricing(input=1.25, output=10.0),
        },
        "gpt-5.4": {
            "name": "GPT-5.4",
            "context_length": 400_000,
            "max_output": 131072,
            "supports_tools": True,
            "supports_vision": True,
            "supports_reasoning": False,
            "pricing": ModelPricing(input=0.5, output=1.5),
        },
        "gpt-5.4-mini": {
            "name": "GPT-5.4 Mini",
            "context_length": 400_000,
            "max_output": 131072,
            "supports_tools": True,
            "supports_vision": True,
            "supports_reasoning": False,
            "pricing": ModelPricing(input=0.15, output=0.6),
        },
        "gpt-5.4-nano": {
            "name": "GPT-5.4 Nano",
            "context_length": 400_000,
            "max_output": 131072,
            "supports_tools": True,
            "supports_vision": True,
            "supports_reasoning": False,
            "pricing": ModelPricing(input=0.1, output=0.4),
        },
    }

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

    def _format_model(
        self,
        model_id: str,
        native_type: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        context_length: int | None = None,
        max_output: int | None = None,
        supports_tools: bool | None = None,
        supports_vision: bool | None = None,
        supports_reasoning: bool | None = None,
        pricing=None,
    ):
        meta = self._MODEL_METADATA.get(model_id, {})
        return LanguageModel(
            id=model_id,
            provider_id=self.id,
            name=cast(str | None, meta.get("name")),
            description=cast(str | None, meta.get("description")),
            context_length=cast(int | None, meta.get("context_length")),
            max_output=cast(int | None, meta.get("max_output")),
            supports_tools=bool(meta.get("supports_tools", False)),
            supports_vision=bool(meta.get("supports_vision", False)),
            supports_reasoning=bool(meta.get("supports_reasoning", False)),
            pricing=cast(ModelPricing | None, meta.get("pricing")),
        )
