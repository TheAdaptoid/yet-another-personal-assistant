"""LM Studio inference provider implementation."""

import logging

from openai import AsyncOpenAI

from yapa.core.inference.exceptions import ModelsFetchError
from yapa.core.inference.provider import InferenceProvider
from yapa.shared.config import get_config
from yapa.shared.models import ModelData


def convert_model_id_to_name(model_id: str) -> str:
    """
    Convert a model ID to a human-readable name.

    Args:
        model_id (str): The ID of the model (e.g., "qwen/qwen3-4b-2507").

    Returns:
        str: The human-readable name of the model.

    Example:
        >>> convert_model_id_to_name("qwen/qwen3-4b-2507")
        'Qwen3 4b 2507'
    """

    name_right = model_id.split("/")[-1]
    name_capitalized = name_right.capitalize()
    name_no_dash = name_capitalized.replace("-", " ")
    return name_no_dash


class LMStudioIP(InferenceProvider):
    """Inference provider for LM Studio."""

    def __init__(
        self,
        logger: logging.Logger,
    ):
        """
        Initialize a new LM Studio inference provider.

        Args:
            logger (logging.Logger): The logger to use for this provider.
        """
        config = get_config()
        client = AsyncOpenAI(
            api_key=config.lmstudio_api_key, base_url=config.lmstudio_base_url
        )
        super().__init__(
            logger=logger, identifier="lmstudio", name="LM Studio", client=client
        )

    async def get_models(self) -> list[ModelData]:
        """Retrieve a list of available models for this provider."""
        try:
            response = await self._client.models.list()
            return [
                ModelData(
                    id=model.id,
                    name=convert_model_id_to_name(model.id),
                    provider_id=self.id,
                    provider_name=self.name,
                )
                for model in response.data
            ]
        except Exception as e:
            self._logger.error(f"Error retrieving models from LM Studio: {e}")
            raise ModelsFetchError(self.name, cause=e) from e
