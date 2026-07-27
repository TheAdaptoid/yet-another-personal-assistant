"""Tests for Session model — serialization and message discrimination."""

from yapa.models import (
    AssistantMessage,
    ModelData,
    ModelType,
    Session,
    SystemMessage,
    UserMessage,
)


class TestJsonRoundTrip:
    """JSON round-trip preserves Session fields and embedded messages."""

    def test_messages_serialize_as_array(self):
        session = Session(title="test")
        session.messages.append(UserMessage(content="hello"))
        data = session.model_dump(mode="json")
        assert data["title"] == "test"
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "hello"

    def test_round_trip(self):
        session = Session(title="chat")
        session.messages.append(UserMessage(content="hi"))
        session.messages.append(AssistantMessage(content="hello", model="m"))
        data = session.model_dump(mode="json")
        restored = Session(**data)
        assert restored.title == "chat"
        assert len(restored.messages) == 2
        assert isinstance(restored.messages[0], UserMessage)
        assert isinstance(restored.messages[1], AssistantMessage)

    def test_discriminated_messages(self):
        session = Session()
        session.messages.append(SystemMessage(content="system prompt"))
        session.messages.append(UserMessage(content="user text"))
        session.messages.append(AssistantMessage(content="response", model="m"))
        data = session.model_dump(mode="json")
        restored = Session(**data)
        assert isinstance(restored.messages[0], SystemMessage)
        assert isinstance(restored.messages[1], UserMessage)
        assert isinstance(restored.messages[2], AssistantMessage)


class TestModelField:
    """Session.model field serializes and deserializes correctly."""

    def test_defaults_to_none(self):
        session = Session()
        assert session.model is None

    def test_round_trip(self):
        model = ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM)
        session = Session(model=model)
        data = session.model_dump(mode="json")
        restored = Session(**data)
        assert restored.model is not None
        assert restored.model.id == "gpt-4o"
        assert restored.model.provider_id == "openai"
        assert restored.model.type == ModelType.LLM

    def test_none_omits_from_json(self):
        session = Session()
        data = session.model_dump(mode="json")
        assert data.get("model") is None

    def test_serializes_as_object(self):
        model = ModelData(id="claude-3", provider_id="anthropic", type=ModelType.LLM)
        session = Session(model=model)
        data = session.model_dump(mode="json")
        assert isinstance(data["model"], dict)
        assert data["model"]["id"] == "claude-3"
        assert data["model"]["provider_id"] == "anthropic"


from yapa.models import InferenceParams


class TestSessionNewFields:
    def test_system_prompt_defaults_to_none(self):
        session = Session()
        assert session.system_prompt is None

    def test_system_prompt_round_trip(self):
        session = Session(system_prompt="You are helpful.")
        data = session.model_dump(mode="json")
        restored = Session(**data)
        assert restored.system_prompt == "You are helpful."

    def test_inference_params_defaults_to_none(self):
        session = Session()
        assert session.inference_params is None

    def test_inference_params_round_trip(self):
        params = InferenceParams(temperature=0.7, max_tokens=4096)
        session = Session(inference_params=params)
        data = session.model_dump(mode="json")
        restored = Session(**data)
        assert restored.inference_params is not None
        assert restored.inference_params.temperature == 0.7
        assert restored.inference_params.max_tokens == 4096
