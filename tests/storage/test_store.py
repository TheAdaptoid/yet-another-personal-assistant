"""Tests for GenericStore."""

from __future__ import annotations

import json
import unittest.mock

import pytest

from yapa.models import AssistantMessage, Session, SystemMessage, UserMessage
from yapa.storage import GenericStore, StorageReadError, StorageWriteError


class TestInit:
    """Tests for GenericStore.__init__()."""

    def test_creates_directory(self, store_dir):
        """Should create the storage directory if it does not exist."""
        target = store_dir / "sub"
        GenericStore(target, Session)
        assert target.is_dir()

    def test_creates_nested_directory(self, store_dir):
        """Should create intermediate directories with parents=True."""
        target = store_dir / "a" / "b" / "c"
        GenericStore(target, Session)
        assert target.is_dir()

    def test_existing_directory_not_removed(self, store_dir):
        """Should not remove pre-existing files in the directory."""
        store_dir.mkdir(parents=True, exist_ok=True)
        marker = store_dir / "keep_me.txt"
        marker.write_text("hello")
        GenericStore(store_dir, Session)
        assert marker.exists()


class TestSave:
    """Tests for GenericStore.save()."""

    def test_save_creates_json_file(self, store, store_dir):
        """Should create a .json file named after the entity ID."""
        session = Session(title="test")
        store.save(session)
        expected = store_dir / f"{session.id}.json"
        assert expected.exists()

    def test_save_file_contents_match_entity(self, store, store_dir):
        """Should write valid JSON matching entity.model_dump()."""
        session = Session(title="check contents")
        store.save(session)
        data = json.loads((store_dir / f"{session.id}.json").read_text())
        assert data["title"] == "check contents"
        assert data["id"] == str(session.id)

    def test_save_calls_touch(self, store):
        """Should update the entity's updated_at timestamp."""
        session = Session(title="touch test")
        original_updated = session.updated_at
        store.save(session)
        assert session.updated_at >= original_updated

    def test_save_no_overwrite_raises_file_exists(self, store):
        """Should raise FileExistsError when file exists and overwrite is False."""
        session = Session(title="dup")
        store.save(session)
        with pytest.raises(FileExistsError):
            store.save(session)

    def test_save_with_overwrite_replaces_file(self, store):
        """Should succeed when overwrite=True on an existing file."""
        session = Session(title="original")
        store.save(session)
        session.title = "updated"
        store.save(session, overwrite=True)
        loaded = store.load(session.id)
        assert loaded.title == "updated"

    def test_save_invalid_path_raises_storage_write_error(self, store):
        """Should raise StorageWriteError when write fails."""
        session = Session(title="fail")
        with pytest.raises(StorageWriteError):
            patch = unittest.mock.patch(
                "pathlib.Path.rename", side_effect=OSError("read-only")
            )
            with patch:
                store.save(session)


class TestLoad:
    """Tests for GenericStore.load()."""

    def test_load_returns_saved_entity(self, store, make_session):
        """Should return the entity matching the given ID."""
        saved = make_session(title="round trip")
        loaded = store.load(saved.id)
        assert loaded.id == saved.id
        assert loaded.title == "round trip"

    def test_load_missing_id_raises_file_not_found(self, store):
        """Should raise FileNotFoundError when the file does not exist."""
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent-id")

    def test_load_corrupt_json_raises_storage_read_error(self, store, store_dir):
        """Should raise StorageReadError when file contains invalid JSON."""
        bad_file = store_dir / "bad.json"
        bad_file.write_text("not json {{{")
        with pytest.raises(StorageReadError):
            store.load("bad")

    def test_load_wrong_type_raises_storage_read_error(self, store, store_dir):
        """Should raise StorageReadError when JSON type conflicts with schema."""
        bad_file = store_dir / "wrong-type.json"
        payload = json.dumps({"id": 12345, "title": "ok", "messages": "not-a-list"})
        bad_file.write_text(payload)
        with pytest.raises(StorageReadError):
            store.load("wrong-type")

    def test_load_preserves_nested_messages(self, store):
        """Should correctly deserialize nested Message objects."""
        messages = [
            UserMessage(content="hello"),
            AssistantMessage(content="hi", model="test-model"),
            SystemMessage(content="be helpful"),
        ]
        session = Session(title="with messages", messages=messages)
        store.save(session)
        loaded = store.load(session.id)
        assert len(loaded.messages) == 3
        assert loaded.messages[0].role == "user"
        assert loaded.messages[0].content == "hello"
        assert loaded.messages[1].role == "assistant"
        assert loaded.messages[1].model == "test-model"
        assert loaded.messages[2].role == "system"


class TestList:
    """Tests for GenericStore.list()."""

    def test_list_empty_dir_returns_empty(self, store):
        """Should return an empty list when no files exist."""
        assert store.list() == []

    def test_list_returns_all_entities(self, store, make_session):
        """Should return all saved entities."""
        s1 = make_session(title="first")
        s2 = make_session(title="second")
        s3 = make_session(title="third")
        results = store.list()
        assert len(results) == 3
        ids = {s.id for s in results}
        assert ids == {s1.id, s2.id, s3.id}

    def test_list_skips_corrupt_files(self, store, store_dir, make_session):
        """Should skip unreadable files and return valid ones."""
        valid = make_session(title="valid")
        bad = store_dir / "corrupt.json"
        bad.write_text("{bad json")
        results = store.list()
        assert len(results) == 1
        assert results[0].id == valid.id

    def test_list_ignores_non_json_files(self, store, store_dir, make_session):
        """Should only process .json files."""
        make_session(title="real")
        (store_dir / "notes.txt").write_text("ignored")
        (store_dir / "backup.tmp").write_text("also ignored")
        results = store.list()
        assert len(results) == 1

    def test_list_count_matches_saved(self, store, make_session):
        """Should return exactly as many entities as were saved."""
        for i in range(5):
            make_session(title=f"session-{i}")
        assert len(store.list()) == 5


class TestDelete:
    """Tests for GenericStore.delete()."""

    def test_delete_removes_file(self, store, store_dir, make_session):
        """Should remove the .json file from disk."""
        session = make_session(title="doomed")
        store.delete(session.id)
        assert not (store_dir / f"{session.id}.json").exists()

    def test_delete_missing_id_raises_file_not_found(self, store):
        """Should raise FileNotFoundError for a nonexistent entity."""
        with pytest.raises(FileNotFoundError):
            store.delete("nonexistent-id")

    def test_delete_does_not_affect_others(self, store, make_session):
        """Should only delete the targeted entity."""
        s1 = make_session(title="keep-1")
        s2 = make_session(title="delete-me")
        s3 = make_session(title="keep-2")
        store.delete(s2.id)
        remaining = {s.id for s in store.list()}
        assert remaining == {s1.id, s3.id}

    def test_delete_allows_re_save(self, store, make_session):
        """Should allow saving a new entity after deleting a different one."""
        old = make_session(title="old")
        store.delete(old.id)
        new = Session(title="new")
        store.save(new)
        assert store.load(new.id).title == "new"


class TestRoundTrip:
    """Integration tests exercising full save → load → verify cycles."""

    def test_roundtrip_session_with_no_messages(self, store):
        """Should persist and load an empty session."""
        session = Session(title="empty")
        store.save(session)
        loaded = store.load(session.id)
        assert loaded.title == "empty"
        assert loaded.messages == []

    def test_roundtrip_session_with_multiple_messages(self, store):
        """Should persist and load a session with several messages."""
        messages = [
            UserMessage(content="what is 2+2?"),
            AssistantMessage(content="4", model="math-model"),
            SystemMessage(content="you are a math tutor"),
        ]
        session = Session(title="math chat", messages=messages)
        store.save(session)
        loaded = store.load(session.id)
        assert len(loaded.messages) == 3
        assert loaded.messages[0].content == "what is 2+2?"
        assert loaded.messages[1].content == "4"
        assert loaded.messages[2].role == "system"

    def test_roundtrip_preserves_uuid_and_timestamps(self, store):
        """Should preserve id, created_at, and updated_at across save/load."""
        session = Session(title="timestamps")
        original_id = session.id
        original_created = session.created_at
        store.save(session)
        loaded = store.load(session.id)
        assert loaded.id == original_id
        assert loaded.created_at == original_created

    def test_roundtrip_overwrite_updates_timestamps(self, store):
        """Should refresh updated_at on overwrite save."""
        session = Session(title="v1")
        store.save(session)
        first_updated = session.updated_at
        session.title = "v2"
        store.save(session, overwrite=True)
        loaded = store.load(session.id)
        assert loaded.title == "v2"
        assert loaded.updated_at >= first_updated
