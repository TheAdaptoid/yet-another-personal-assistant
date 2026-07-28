"""Tests for SessionStore protocol and JsonSessionStore."""

import pytest

from yapa.models import AssistantMessage, InferenceParams, Session, UserMessage
from yapa.services.store import JsonSessionStore, SessionStore


class TestSessionStoreProtocol:
    def test_json_session_store_conforms(self):
        assert isinstance(JsonSessionStore, object)
        # Protocol check: JsonSessionStore should implement SessionStore
        store = JsonSessionStore("/tmp")
        assert isinstance(store, SessionStore)


class TestJsonSessionStore:
    def test_save_and_load(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="test")
        store.save(session)
        loaded = store.load(str(session.id))
        assert loaded.id == session.id
        assert loaded.title == "test"

    def test_load_missing_raises(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            store.load("nonexistent")

    def test_save_with_overwrite(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="original")
        store.save(session)

        session.title = "updated"
        # Should not raise — overwrite=True
        store.save(session, overwrite=True)
        loaded = store.load(str(session.id))
        assert loaded.title == "updated"

    def test_save_without_overwrite_raises(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="original")
        store.save(session)

        session.title = "updated"
        with pytest.raises(FileExistsError):
            store.save(session, overwrite=False)

    def test_list_empty(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        assert store.list() == []

    def test_list_returns_all(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        s1 = Session(title="one")
        s2 = Session(title="two")
        store.save(s1)
        store.save(s2)
        sessions = store.list()
        ids = {str(s.id) for s in sessions}
        assert ids == {str(s1.id), str(s2.id)}

    def test_delete(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="delete-me")
        store.save(session)
        store.delete(str(session.id))
        with pytest.raises(FileNotFoundError):
            store.load(str(session.id))

    def test_delete_missing_raises(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.delete("nonexistent")

    def test_preserves_messages(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="chat")
        session.messages = [
            UserMessage(content="hi"),
            AssistantMessage(content="hello", model="m"),
        ]
        store.save(session)
        loaded = store.load(str(session.id))
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "hi"

    def test_preserves_inference_params(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        params = InferenceParams(temperature=0.5, max_tokens=2048)
        session = Session(title="configured", inference_params=params)
        store.save(session)
        loaded = store.load(str(session.id))
        assert loaded.inference_params is not None
        assert loaded.inference_params.temperature == 0.5

    def test_creates_storage_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "sessions"
        store = JsonSessionStore(storage_dir=nested)
        session = Session()
        store.save(session)
        assert nested.exists()
        assert (nested / f"{session.id}.json").exists()
