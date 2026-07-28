"""LM Studio inference provider implementation."""

import httpx

from yapa.models import ModelData, ModelType
from yapa.services.config import Config, ProviderConfig

from ..openai_compat import OpenAICompatibleProvider


class LMStudioIP(OpenAICompatibleProvider):
    """Inference provider for LM Studio."""

    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(self, config: Config):
        """Initialize the LM Studio provider."""
        pc = config.provider_configs.get("lmstudio", ProviderConfig())
        super().__init__(
            identifier="lmstudio",
            name="LM Studio",
            api_key=pc.api_key or "",
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
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
        model_type = raw.get("type", "")
        if model_type == "llm":
            model_type_enum = ModelType.LLM
        else:
            model_type_enum = ModelType.OTHER
        caps = raw.get("capabilities", {})
        context_length = raw.get("max_context_length")
        return ModelData(
            id=model_id,
            provider_id=self.id,
            type=model_type_enum,
            context_length=context_length,
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
            raw_models = resp.json().get("models", [])
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
