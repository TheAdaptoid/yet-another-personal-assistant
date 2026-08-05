"""Model type classification shared by providers."""

from yapa.models import ModelType

_EMBED_KEYWORDS = ("embed",)
_OTHER_KEYWORDS = ("audio", "image")


def classify_model_type(model_id: str, native_type: str | None = None) -> ModelType:
    """Classify a model type preferring provider-native information."""
    if native_type is not None:
        lowered = native_type.lower()
        if lowered in {"llm", "text-generation", "chat-completion"}:
            return ModelType.LLM
        if lowered in {"embedding", "embeddings"}:
            return ModelType.EMBED
        if lowered in {
            "image",
            "audio",
            "image-generation",
            "text-to-image",
            "speech-to-text",
            "text-to-speech",
        }:
            return ModelType.OTHER
    lower_id = model_id.lower()
    if any(kw in lower_id for kw in _EMBED_KEYWORDS):
        return ModelType.EMBED
    if any(kw in lower_id for kw in _OTHER_KEYWORDS):
        return ModelType.OTHER
    return ModelType.LLM
