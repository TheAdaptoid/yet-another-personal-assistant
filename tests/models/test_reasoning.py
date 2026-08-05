from yapa.models.inference import ReasoningEffort


def test_has_four_values() -> None:
    assert ReasoningEffort.OFF.value == "off"
    assert ReasoningEffort.LOW.value == "low"
    assert ReasoningEffort.MEDIUM.value == "medium"
    assert ReasoningEffort.HIGH.value == "high"
    assert {e.value for e in ReasoningEffort} == {"off", "low", "medium", "high"}
