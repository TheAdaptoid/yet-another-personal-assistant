"""OpenRouter inference provider implementation."""

import logging

from openai import AsyncOpenAI

from yapa.core.inference.exceptions import ModelsFetchError
from yapa.core.inference.provider import InferenceProvider
from yapa.shared.config import get_config
from yapa.shared.models import ModelData


class OpenRouterIP(InferenceProvider):
    """Inference provider for OpenRouter."""

    def __init__(
        self,
        logger: logging.Logger,
    ):
        """
        Initialize a new OpenRouter inference provider.

        Args:
            logger (logging.Logger): The logger to use for this provider.
        """
        config = get_config()
        client = AsyncOpenAI(
            api_key=config.openrouter_api_key, base_url=config.openrouter_base_url
        )
        super().__init__(
            logger=logger, identifier="openrouter", name="OpenRouter", client=client
        )

    async def get_models(self) -> list[ModelData]:
        """Retrieve a list of available models for this provider."""
        try:
            response = await self._client.models.list()
            return [
                ModelData(
                    id=model.id,
                    name=model.id,  # TODO: Convert model ID to human-readable name
                    provider_id=self.id,
                    provider_name=self.name,
                )
                for model in response.data
                if any(
                    kw in model.id.lower()
                    for kw in ["kimi", "gpt-5", "qwen3", "deepseek"]
                )  # TODO: Add more sophisticated filtering logic
            ]
        except Exception as e:
            self._logger.error(f"Error retrieving models from OpenRouter: {e}")
            raise ModelsFetchError(self.name, cause=e) from e
