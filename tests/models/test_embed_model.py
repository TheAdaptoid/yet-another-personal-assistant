from yapa.models.inference import EmbedModel


def test_defaults() -> None:
    m = EmbedModel(id="embed", provider_id="openai")
    assert m.embedding_dimensions is None
    assert m.normalized is False
    assert m.pricing is None


def test_native_dimensions() -> None:
    m = EmbedModel(id="embed", provider_id="openai", embedding_dimensions=1536)
    assert m.embedding_dimensions == 1536
    assert m.normalized is False
