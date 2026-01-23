"""Tests for embeddings."""

from __future__ import annotations

import httpx
import pytest
import respx

from apertis import Apertis, EmbeddingResponse


class TestEmbeddings:
    """Tests for embeddings."""

    def test_create_single_embedding(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test creating a single embedding."""
        mock_api.post("/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                            "index": 0,
                        }
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )
        )

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="Hello, world!",
        )

        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert response.data[0].index == 0
        assert response.usage.prompt_tokens == 2

    def test_create_batch_embeddings(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test creating batch embeddings."""
        mock_api.post("/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                        {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
                        {"object": "embedding", "embedding": [0.5, 0.6], "index": 2},
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 6, "total_tokens": 6},
                },
            )
        )

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=["Hello", "World", "Test"],
        )

        assert len(response.data) == 3
        assert response.data[0].index == 0
        assert response.data[1].index == 1
        assert response.data[2].index == 2

    def test_create_embedding_with_dimensions(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test creating an embedding with custom dimensions."""
        mock_api.post("/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )
        )

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="Hello",
            dimensions=256,
        )

        # Verify dimensions was sent in request
        import json

        request = mock_api.calls[0].request
        body = json.loads(request.content)
        assert body["dimensions"] == 256
