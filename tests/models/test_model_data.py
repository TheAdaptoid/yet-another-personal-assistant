from pydantic import TypeAdapter

from yapa.models.inference import (
    EmbedModel,
    LanguageModel,
    ModelData,
    ModelDataUnion,
    ModelType,
)

adapter = TypeAdapter(ModelDataUnion)


def test_model_type_has_three_values() -> None:
    assert ModelType.LLM.value == "llm"
    assert ModelType.EMBED.value == "embedding"
    assert ModelType.OTHER.value == "other"
    assert {m.value for m in ModelType} == {"llm", "embedding", "other"}


def test_llm_record_parses_as_language_model() -> None:
    m = adapter.validate_python({"id": "gpt-4", "provider_id": "openai", "type": "llm"})
    assert type(m) is LanguageModel


def test_embedding_record_parses_as_embed_model() -> None:
    m = adapter.validate_python(
        {"id": "embed", "provider_id": "openai", "type": "embedding"}
    )
    assert type(m) is EmbedModel


def test_other_record_parses_as_bare_model_data() -> None:
    m = adapter.validate_python({"id": "x", "provider_id": "openai", "type": "other"})
    assert type(m) is ModelData


def test_name_and_description_default_to_none() -> None:
    m = adapter.validate_python({"id": "gpt-4", "provider_id": "openai", "type": "llm"})
    assert m.name is None
    assert m.description is None
    m2 = adapter.validate_python({"id": "e", "provider_id": "p", "type": "embedding"})
    assert m2.name is None


def test_full_id() -> None:
    m = adapter.validate_python({"id": "gpt-4", "provider_id": "openai", "type": "llm"})
    assert m.full_id == "openai:gpt-4"


def test_language_model_unknown_type_not_a_bare_model_data() -> None:
    lm = LanguageModel(id="gpt-4", provider_id="openai")
    assert not isinstance(lm, EmbedModel)
    assert isinstance(lm, ModelData)
