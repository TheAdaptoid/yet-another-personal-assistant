"""Tests for SessionService."""

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    ModelData,
    ModelType,
    UserMessage,
)
from yapa.services.session import SessionService
from yapa.services.store import JsonSessionStore


class TestCreate:
    def test_creates_new_session(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        assert session.title == "Untitled Session"
        assert session.id is not None
        assert session.model is None
        assert session.system_prompt is None
        assert session.inference_params is None

    def test_persists_to_disk(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        loaded = service.get(str(session.id))
        assert loaded.id == session.id
        assert loaded.title == "Untitled Session"


class TestList:
    def test_empty_when_no_sessions(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        assert service.list() == []

    def test_returns_all_sessions(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        service.create()
        service.create()
        sessions = service.list()
        assert len(sessions) == 2

    def test_ordered_newest_first_by_default(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        s1 = service.create()
        s2 = service.create()
        sessions = service.list()
        assert sessions[0].id == s2.id
        assert sessions[1].id == s1.id


class TestGet:
    def test_returns_session(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        created = service.create()
        loaded = service.get(str(created.id))
        assert loaded.id == created.id

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.get("nonexistent")


class TestRename:
    def test_updates_title(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        updated = service.rename(str(session.id), "My Chat")
        assert updated.title == "My Chat"

    def test_persists_rename(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.rename(str(session.id), "Persisted")
        loaded = service.get(str(session.id))
        assert loaded.title == "Persisted"

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.rename("nonexistent", "new title")


class TestUpdateSystemPrompt:
    def test_sets_system_prompt(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        updated = service.update_system_prompt(str(session.id), "Be helpful.")
        assert updated.system_prompt == "Be helpful."

    def test_clears_system_prompt(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.update_system_prompt(str(session.id), "Be helpful.")
        service.update_system_prompt(str(session.id), None)
        loaded = service.get(str(session.id))
        assert loaded.system_prompt is None

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.update_system_prompt("nonexistent", "prompt")


class TestUpdateInferenceParams:
    def test_sets_params(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        params = InferenceParams(temperature=0.7, max_tokens=4096)
        updated = service.update_inference_params(str(session.id), params)
        assert updated.inference_params is not None
        assert updated.inference_params.temperature == 0.7

    def test_clears_params(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.update_inference_params(
            str(session.id), InferenceParams(temperature=0.5)
        )
        service.update_inference_params(str(session.id), None)
        loaded = service.get(str(session.id))
        assert loaded.inference_params is None

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.update_inference_params("nonexistent", None)


class TestAddMessages:
    def test_adds_single_message(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        msg = UserMessage(content="hello")
        updated = service.add_messages(str(session.id), [msg])
        assert len(updated.messages) == 1
        assert updated.messages[0].content == "hello"

    def test_adds_multiple_atomically(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        msgs = [
            UserMessage(content="q1"),
            AssistantMessage(content="a1", model="m"),
        ]
        updated = service.add_messages(str(session.id), msgs)
        assert len(updated.messages) == 2

    def test_persists_messages(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.add_messages(
            str(session.id),
            [UserMessage(content="persist-me")],
        )
        loaded = service.get(str(session.id))
        assert len(loaded.messages) == 1

    def test_updates_model(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        updated = service.add_messages(
            str(session.id),
            [UserMessage(content="hi")],
            model=model,
        )
        assert updated.model is not None
        assert updated.model.id == "gpt-4"
        assert updated.model.provider_id == "openai"
        loaded = service.get(str(session.id))
        assert loaded.model is not None
        assert loaded.model.id == "gpt-4"

    def test_updates_model_without_messages(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        model = ModelData(id="claude", provider_id="anthropic", type=ModelType.LLM)
        updated = service.add_messages(str(session.id), [], model=model)
        assert updated.model is not None
        assert updated.model.id == "claude"

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.add_messages("nonexistent", [UserMessage(content="hi")])


class TestDelete:
    def test_removes_session(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.delete(str(session.id))
        assert service.list() == []

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.delete("nonexistent")
