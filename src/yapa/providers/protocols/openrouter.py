"""Protocols for OpenRouter."""

from openrouter import OpenRouter
from openrouter.components import Model, ModelsListResponse

from yapa.config import Config
from yapa.models import ModelData, ModelType

from ..base import ModelFetchProtocol
from ..exceptions import ModelsFetchError


class OpenRouterFetchProtocol(ModelFetchProtocol):
    """Protocol for fetching models from OpenRouter."""

    def __init__(self, config: Config, provider_id: str):
        """Initialize the OpenRouter model fetch protocol."""
        self._provider_id = provider_id
        self._config = config

    def _format_model(self, model_info: Model) -> ModelData:
        """
        Format raw model data from OpenRouter into ModelData.

        Args:
            model_info (Model): The raw model information from OpenRouter.

        Returns:
            ModelData: The formatted model data.
        """
        model_id = model_info.id

        # Infer model type based on output modalities
        if "text" in model_info.architecture.output_modalities:
            inferred_type = ModelType.LLM
        else:
            inferred_type = ModelType.OTHER

        return ModelData(id=model_id, provider_id=self._provider_id, type=inferred_type)

    async def list_models(self, model_type: ModelType | None = None) -> list[ModelData]:
        """Fetch available models from OpenRouter."""

        # Determine filter type for API request
        filter_type = None
        if model_type == ModelType.LLM:
            filter_type = "text"

        async with OpenRouter(
            api_key=self._config.openrouter_api_key,
            url_params={"output_modalities": filter_type} if filter_type else None,
        ) as client:
            response: ModelsListResponse = client.models.list()
            unformatted_models: list[Model] = response.data
            formatted_models = [
                self._format_model(model_info) for model_info in unformatted_models
            ]
            return formatted_models

    async def get_model(self, model_id: str) -> ModelData:
        """
        Fetch detailed information about a specific model from OpenRouter.

        The OpenRouter API does not support fetching detailed model information by ID,
        so this method retrieves the list of models and searches for the specified ID.
        """
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        raise ModelsFetchError(f"Model with ID '{model_id}' not found in OpenRouter.")
