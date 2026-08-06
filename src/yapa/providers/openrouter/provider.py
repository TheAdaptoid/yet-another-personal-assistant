"""OpenRouter inference provider - AsyncOpenAI + native model listing."""

from urllib.parse import urljoin

import httpx

from yapa.models import ModelData, ModelDataUnion, ModelPricing, ModelType
from yapa.services.config import Config, ProviderConfig

from ..exceptions import ModelsFetchError
from ..openai.openai_compat import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """Inference provider for OpenRouter."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: Config):
        """Initialize the OpenRouter provider."""
        pc = config.provider_configs.get("openrouter", ProviderConfig())
        if pc.api_key is None or pc.api_key.strip() == "":
            raise ValueError("OpenRouter API key is not set.")
        base_url = pc.base_url or self.DEFAULT_BASE_URL
        super().__init__(
            identifier="openrouter",
            name="OpenRouter",
            api_key=pc.api_key,
            base_url=base_url,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )
        self._base_url = base_url
        self._timeout = config.provider_timeout

    def _models_endpoint(self) -> str:
        base = str(self._base_url).rstrip("/")
        return urljoin(base + "/", "models")

    def _format_model_from_openrouter(self, raw: dict) -> ModelData:
        model_id = raw["id"]
        native_type = self._native_type(raw)
        pricing = self._normalize_pricing(raw.get("pricing"))
        supported = raw.get("supported_parameters", [])
        modality = raw.get("architecture", {}).get("modality", "")
        return self._format_model(
            model_id,
            native_type=native_type,
            name=raw.get("name"),
            description=raw.get("description"),
            context_length=raw.get("context_length"),
            max_output=raw.get("max_completion_tokens"),
            supports_tools="tools" in supported,
            supports_vision=("image" in modality),
            supports_reasoning="reasoning" in supported,
            pricing=pricing,
        )

    def _native_type(self, raw: dict) -> str | None:
        modality = raw.get("architecture", {}).get("modality", "")
        mid = raw.get("id", "").lower()
        if "embed" in mid or "embedding" in modality:
            return "embedding"
        if "image" in modality or "audio" in modality:
            return "other"
        return "llm"

    def _normalize_pricing(self, p: dict | None) -> ModelPricing | None:
        """Normalize OpenRouter pricing (native USD per-1K tokens) to per-1M."""
        if not p:
            return None
        try:
            return ModelPricing(
                input=(
                    float(p["prompt"]) * 1000
                    if p.get("prompt") is not None
                    else (
                        float(p["input"]) * 1000 if p.get("input") is not None else None
                    )
                ),
                output=(
                    float(p["completion"]) * 1000
                    if p.get("completion") is not None
                    else (
                        float(p["output"]) * 1000
                        if p.get("output") is not None
                        else None
                    )
                ),
                request=(float(p["request"]) if p.get("request") is not None else None),
            )
        except (ValueError, TypeError):
            return None

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelDataUnion]:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._models_endpoint(), headers=headers)
            resp.raise_for_status()
            raw_models = resp.json().get("data", [])
        formatted = [self._format_model_from_openrouter(m) for m in raw_models]
        if model_type:
            target = model_type.value
            return [m for m in formatted if getattr(m.type, "value", m.type) == target]
        return formatted

    async def _get_model_impl(self, model_id: str) -> ModelDataUnion:
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                return m
        raise ModelsFetchError(
            f"Model '{model_id}' not found in OpenRouter listing (no fabrication)."
        )
