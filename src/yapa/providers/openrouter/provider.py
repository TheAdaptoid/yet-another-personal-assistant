"""OpenRouter inference provider implementation."""

import httpx

from yapa.config import UNSET, Config
from yapa.models import ModelData, ModelType

from ..openai_compat import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """Inference provider for OpenRouter."""

    def __init__(self, config: Config):
        """
        Initialize the OpenRouter provider.

        Args:
            config: Application config containing the OpenRouter API key.
        """
        if config.openrouter_api_key in (None, UNSET):
            raise ValueError("OpenRouter API key is not set.")
        super().__init__(
            identifier="openrouter",
            name="OpenRouter",
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )

    def _format_model_from_openrouter(self, raw: dict) -> ModelData:
        model_id = raw["id"]
        model = self._format_model(model_id)
        context_length = raw.get("context_length")
        max_output = raw.get("max_completion_tokens")
        arch = raw.get("architecture", {})
        modality = arch.get("modality", "")
        supported = raw.get("supported_parameters", [])
        pricing: dict[str, float] | None = None
        if "pricing" in raw:
            p = raw["pricing"]
            try:
                prompt = float(p.get("prompt", 0)) * 1_000_000
                completion = float(p.get("completion", 0)) * 1_000_000
                pricing = {"input": prompt, "output": completion}
            except (ValueError, TypeError):
                pricing = None
        return ModelData(
            id=model.id,
            provider_id=model.provider_id,
            type=model.type,
            context_length=context_length,
            max_output=max_output,
            supports_tools="tools" in supported,
            supports_vision="vision" in modality or "image" in modality,
            pricing=pricing,
        )

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelData]:
        headers = {"Authorization": f"Bearer {self._client.api_key}"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{str(self._client.base_url).rstrip('/')}/models",
                headers=headers,
            )
            resp.raise_for_status()
            raw_models = resp.json().get("data", [])
        formatted = [self._format_model_from_openrouter(m) for m in raw_models]
        if model_type:
            return [m for m in formatted if m.type == model_type]
        return formatted

    async def _get_model_impl(self, model_id: str) -> ModelData:
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                return m
        return self._format_model(model_id)
