"""Tests for TrackedEntity."""

from yapa.models import TrackedEntity


class TestJsonRoundTrip:
    """JSON serialization and deserialization."""

    def test_serializes_to_json(self):
        id_ = "00000000-0000-0000-0000-000000000001"
        entity = TrackedEntity(id=id_)
        data = entity.model_dump(mode="json")
        assert data["id"] == id_
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)

    def test_round_trip(self):
        id_ = "00000000-0000-0000-0000-000000000001"
        entity = TrackedEntity(id=id_)
        data = entity.model_dump(mode="json")
        restored = TrackedEntity(**data)
        assert restored.id == entity.id
        assert restored.created_at == entity.created_at
        assert restored.updated_at == entity.updated_at
