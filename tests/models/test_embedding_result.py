from yapa.models.embedding import EmbeddingResult


def test_one_vector_per_input_in_order() -> None:
    r = EmbeddingResult(vectors=[[0.1, 0.2], [0.3, 0.4]], model_id="embed")
    assert r.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert r.model_id == "embed"


def test_usage_defaults_none() -> None:
    r = EmbeddingResult(vectors=[[1.0]], model_id="embed")
    assert r.usage is None
