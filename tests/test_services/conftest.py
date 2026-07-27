"""Test fixtures for services-layer tests."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_logger():
    with patch("yapa.services.config.get_logger") as mock:
        yield mock
