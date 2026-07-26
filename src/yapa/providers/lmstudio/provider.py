"""LM Studio inference provider implementation."""

import httpx

from yapa.config import Config
from yapa.models import ModelData, ModelType

from ..openai_compat import OpenAICompatibleProvider


class LMStudioIP(OpenAICompatibleProvider):
    """Inference provider for LM Studio."""

    def __init__(self, config: Config):
        """
        Initialize the LM Studio provider.

        Args:
            config: Application config containing LM Studio credentials.
        """
        super().__init__(
            identifier="lmstudio",
            name="LM Studio",
            api_key=config.lmstudio_api_key,
            base_url=config.lmstudio_base_url,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )

    def _native_base_url(self) -> str:
        base = str(self._client.base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[: -3]
        return base + "/api/v1"

    def _format_model_from_native(self, raw: dict) -> ModelData:
        model_id = raw.get("key", "")
        model = self._format_model(model_id)
        caps = raw.get("capabilities", {})
        instances = raw.get("loaded_instances", [])
        config = instances[0].get("config", {}) if instances else {}
        return ModelData(
            id=model.id,
            provider_id=model.provider_id,
            type=model.type,
            context_length=config.get("context_length"),
            supports_tools=caps.get("trained_for_tool_use", False),
            supports_vision=caps.get("vision", False),
        )

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelData]:
        headers = {}
        if self._client.api_key:
            headers["Authorization"] = f"Bearer {self._client.api_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._native_base_url()}/models", headers=headers
            )
            resp.raise_for_status()
            raw_models = resp.json().get("data", [])
        formatted = [self._format_model_from_native(m) for m in raw_models]
        if model_type:
            return [m for m in formatted if m.type == model_type]
        return formatted

    async def _get_model_impl(self, model_id: str) -> ModelData:
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                return m
        return self._format_model(model_id)
