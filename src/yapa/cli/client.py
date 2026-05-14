"""Client-side api logic for the YAPA CLI."""

from abc import ABC, abstractmethod
import requests
import logging

from yapa.shared.models import Session

SERVER_URI = "http://localhost:8000"


class APICLient(ABC):
    def __init__(
        self,
        endpoint: str,
        logger: logging.Logger,
    ):
        self.logger = logger
        self.endpoint = f"{SERVER_URI}/{endpoint}"


class SessionClient(APICLient):
    def __init__(self, logger: logging.Logger):
        super().__init__(endpoint="sessions", logger=logger)

    def create_session(self, title: str | None = None) -> str:
        response = requests.post(url=self.endpoint, json={"title": title})
        return response.json()["id"]

    def get_session(self, session_id: str) -> Session:
        session = Session.model_validate(self._get(f"/sessions/{session_id}"))
        return session
