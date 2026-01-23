"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
from typing import Generator

import pytest
import respx

from apertis import Apertis, AsyncApertis


@pytest.fixture(autouse=True)
def mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a mock API key for all tests."""
    monkeypatch.setenv("APERTIS_API_KEY", "test-api-key")


@pytest.fixture
def client() -> Generator[Apertis, None, None]:
    """Create a sync client for testing."""
    with Apertis() as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncApertis:
    """Create an async client for testing."""
    client = AsyncApertis()
    yield client
    await client.close()


@pytest.fixture
def mock_api() -> Generator[respx.MockRouter, None, None]:
    """Create a mock API for testing."""
    with respx.mock(base_url="https://api.apertis.ai/v1") as mock:
        yield mock
