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
        base = super()._format_model(
            model_id,
            native_type,
            name=name,
            description=description,
            context_length=context_length,
            max_output=max_output,
            supports_tools=supports_tools,
            supports_vision=supports_vision,
            supports_reasoning=supports_reasoning,
            pricing=pricing,
        )
        if not isinstance(base, LanguageModel):
            return base

        meta = self._MODEL_METADATA.get(model_id)
        if not meta:
            return base

        return base.model_copy(
            update={
                "name": cast(
                    str | None,
                    name if name is not None else meta.get("name", base.name),
                ),
                "description": cast(
                    str | None,
                    description
                    if description is not None
                    else meta.get("description", base.description),
                ),
                "context_length": cast(
                    int | None,
                    context_length
                    if context_length is not None
                    else meta.get("context_length", base.context_length),
                ),
                "max_output": cast(
                    int | None,
                    max_output
                    if max_output is not None
                    else meta.get("max_output", base.max_output),
                ),
                "supports_tools": bool(
                    supports_tools
                    if supports_tools is not None
                    else meta.get("supports_tools", base.supports_tools)
                ),
                "supports_vision": bool(
                    supports_vision
                    if supports_vision is not None
                    else meta.get("supports_vision", base.supports_vision)
                ),
                "supports_reasoning": bool(
                    supports_reasoning
                    if supports_reasoning is not None
                    else meta.get("supports_reasoning", base.supports_reasoning)
                ),
                "pricing": cast(
                    ModelPricing | None,
                    pricing
                    if pricing is not None
                    else meta.get("pricing", base.pricing),
                ),
            }
        )
