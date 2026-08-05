"""LM Studio inference provider — AsyncOpenAI + native model listing."""

from urllib.parse import urljoin

import httpx

from yapa.models import ModelData, ModelDataUnion, ModelType, ReasoningEffort
from yapa.services.config import Config, ProviderConfig

from ..exceptions import ModelsFetchError
from ..openai.openai_compat import OpenAICompatibleProvider


class LMStudioIP(OpenAICompatibleProvider):
    """Inference provider for LM Studio."""

    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(self, config: Config):
        """Initialize the LM Studio provider."""
        pc = config.provider_configs.get("lmstudio", ProviderConfig())
        super().__init__(
            identifier="lmstudio",
            name="LM Studio",
            api_key=pc.api_key,
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )
        self._timeout = config.provider_timeout

    def _models_endpoint(self) -> str:
        """Derive the native /models endpoint from the client base URL (REQ-PROV-13)."""
        base = str(self._client.base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return urljoin(base + "/", "api/v1/models")

    def _map_reasoning(self, reasoning):
        """Map a ReasoningEffort to the LM Studio string form (REQ-MODEL-07)."""
        if reasoning is None or reasoning == ReasoningEffort.OFF:
            return {"reasoning": "off"}
        return {"reasoning": reasoning.value}

    def _format_model_from_native(self, raw: dict) -> ModelData:
        """Format a native LM Studio model entry into a ModelData (REQ-PROV-09)."""
        model_id = raw.get("key", "")
        native_type = raw.get("type")
        caps = raw.get("capabilities", [])
        if isinstance(caps, (list, tuple)):
            caps_set = set(caps)
        elif isinstance(caps, dict):
            caps_set = set(caps.keys())
        else:
            caps_set = set()
        return self._format_model(
            model_id,
            native_type=native_type,
            name=raw.get("display_name"),
            context_length=raw.get("max_context_length"),
            supports_tools="trained_for_tool_use" in caps_set,
            supports_vision=("vision" in caps_set) or ("image-completion" in caps_set),
            supports_reasoning="reasoning" in caps_set,
        )

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelDataUnion]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._models_endpoint())
            resp.raise_for_status()
            raw_models = resp.json().get("models", [])
        formatted = [self._format_model_from_native(m) for m in raw_models]
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
            f"Model '{model_id}' not found in LM Studio listing (no fabrication)."
        )
