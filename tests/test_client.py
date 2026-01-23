"""Tests for the Apertis client."""

from __future__ import annotations

import os
import pytest

from apertis import Apertis, AsyncApertis


class TestClientInitialization:
    """Tests for client initialization."""

    def test_init_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test client initialization with explicit API key."""
        monkeypatch.delenv("APERTIS_API_KEY", raising=False)
        client = Apertis(api_key="test-key")
        assert client._client.api_key == "test-key"

    def test_init_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test client initialization with environment variable."""
        monkeypatch.setenv("APERTIS_API_KEY", "env-key")
        client = Apertis()
        assert client._client.api_key == "env-key"

    def test_init_without_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that initialization without API key raises an error."""
        monkeypatch.delenv("APERTIS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key is required"):
            Apertis()

    def test_init_with_custom_base_url(self) -> None:
        """Test client initialization with custom base URL."""
        client = Apertis(api_key="test-key", base_url="https://custom.api.com/v2/")
        assert client._client.base_url == "https://custom.api.com/v2"

    def test_init_with_custom_timeout(self) -> None:
        """Test client initialization with custom timeout."""
        client = Apertis(api_key="test-key", timeout=30.0)
        assert client._client.timeout == 30.0

    def test_init_with_custom_max_retries(self) -> None:
        """Test client initialization with custom max retries."""
        client = Apertis(api_key="test-key", max_retries=5)
        assert client._client.max_retries == 5

    def test_context_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test client as context manager."""
        monkeypatch.setenv("APERTIS_API_KEY", "test-key")
        with Apertis() as client:
            assert client._client.api_key == "test-key"


class TestAsyncClientInitialization:
    """Tests for async client initialization."""

    def test_init_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test async client initialization with explicit API key."""
        monkeypatch.delenv("APERTIS_API_KEY", raising=False)
        client = AsyncApertis(api_key="test-key")
        assert client._client.api_key == "test-key"

    def test_init_without_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that initialization without API key raises an error."""
        monkeypatch.delenv("APERTIS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key is required"):
            AsyncApertis()


class TestClientResources:
    """Tests for client resources."""

    def test_chat_resource_exists(self, client: Apertis) -> None:
        """Test that chat resource exists."""
        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")

    def test_embeddings_resource_exists(self, client: Apertis) -> None:
        """Test that embeddings resource exists."""
        assert hasattr(client, "embeddings")
