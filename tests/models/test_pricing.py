from yapa.models.inference import ModelPricing


def test_pricing_is_structured_object() -> None:
    p = ModelPricing(input=2.5, output=10.0, request=0.01)
    assert isinstance(p, ModelPricing)
    assert p.input == 2.5
    assert p.output == 10.0
    assert p.request == 0.01


def test_pricing_defaults_to_none() -> None:
    p = ModelPricing()
    assert p.input is None
    assert p.output is None
    assert p.request is None


def test_pricing_serializes_as_object() -> None:
    p = ModelPricing(input=1.0)
    assert p.model_dump() == {"input": 1.0, "output": None, "request": None}
