from yapa.models.inference import LanguageModel


def test_defaults() -> None:
    m = LanguageModel(id="gpt-4", provider_id="openai")
    assert m.context_length is None
    assert m.max_output is None
    assert m.supports_tools is False
    assert m.supports_vision is False
    assert m.supports_reasoning is False
    assert m.pricing is None




