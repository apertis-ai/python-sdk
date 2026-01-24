"""Tests for Models API."""

from __future__ import annotations

import httpx
import pytest
import respx

from apertis import Apertis


class TestModelsList:
    """Tests for listing models."""

    @respx.mock
    def test_list_models(self, client: Apertis) -> None:
        """Test listing all models."""
        respx.get("https://api.apertis.ai/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "gpt-5.2",
                            "object": "model",
                            "created": 1234567890,
                            "owned_by": "openai",
                        },
                        {
                            "id": "claude-sonnet-4.5",
                            "object": "model",
                            "created": 1234567890,
                            "owned_by": "anthropic",
                        },
                        {
                            "id": "gemini-3-pro-preview",
                            "object": "model",
                            "created": 1234567890,
                            "owned_by": "google",
                        },
                    ],
                },
            )
        )

        model_list = client.models.list()

        assert model_list.object == "list"
        assert len(model_list.data) == 3
        assert model_list.data[0].id == "gpt-5.2"
        assert model_list.data[1].id == "claude-sonnet-4.5"

    @respx.mock
    def test_list_models_empty(self, client: Apertis) -> None:
        """Test listing models when empty."""
        respx.get("https://api.apertis.ai/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [],
                },
            )
        )

        model_list = client.models.list()

        assert model_list.object == "list"
        assert len(model_list.data) == 0


class TestModelsRetrieve:
    """Tests for retrieving a specific model."""

    @respx.mock
    def test_retrieve_model(self, client: Apertis) -> None:
        """Test retrieving a specific model."""
        respx.get("https://api.apertis.ai/v1/models/gpt-5.2").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "gpt-5.2",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "openai",
                },
            )
        )

        model = client.models.retrieve("gpt-5.2")

        assert model.id == "gpt-5.2"
        assert model.object == "model"
        assert model.owned_by == "openai"

    @respx.mock
    def test_retrieve_model_not_found(self, client: Apertis) -> None:
        """Test retrieving a non-existent model."""
        from apertis import NotFoundError

        respx.get("https://api.apertis.ai/v1/models/nonexistent").mock(
            return_value=httpx.Response(
                404,
                json={
                    "error": {
                        "message": "Model not found",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )
        )

        with pytest.raises(NotFoundError):
            client.models.retrieve("nonexistent")
