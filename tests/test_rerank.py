"""Tests for Rerank API."""

from __future__ import annotations

import httpx
import pytest
import respx

from apertis import Apertis


class TestRerankCreate:
    """Tests for reranking documents."""

    @respx.mock
    def test_rerank_documents(self, client: Apertis) -> None:
        """Test basic document reranking."""
        respx.post("https://api.apertis.ai/v1/rerank").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "results": [
                        {"index": 2, "relevance_score": 0.95},
                        {"index": 0, "relevance_score": 0.75},
                        {"index": 1, "relevance_score": 0.25},
                    ],
                    "usage": {"total_tokens": 100},
                },
            )
        )

        response = client.rerank.create(
            model="BAAI/bge-reranker-v2-m3",
            query="What is machine learning?",
            documents=[
                "The weather is nice today.",
                "I like pizza.",
                "Machine learning is a subset of AI that enables systems to learn.",
            ],
        )

        assert response.object == "list"
        assert len(response.results) == 3
        # First result should be most relevant (highest score)
        assert response.results[0].relevance_score == 0.95
        assert response.results[0].index == 2

    @respx.mock
    def test_rerank_with_top_n(self, client: Apertis) -> None:
        """Test reranking with top_n parameter."""
        respx.post("https://api.apertis.ai/v1/rerank").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "results": [
                        {"index": 2, "relevance_score": 0.95},
                        {"index": 0, "relevance_score": 0.75},
                    ],
                },
            )
        )

        response = client.rerank.create(
            model="BAAI/bge-reranker-v2-m3",
            query="What is AI?",
            documents=[
                "Artificial intelligence overview.",
                "Cooking recipes.",
                "Deep learning and neural networks.",
            ],
            top_n=2,
        )

        assert len(response.results) == 2

    @respx.mock
    def test_rerank_with_return_documents(self, client: Apertis) -> None:
        """Test reranking with document text in results."""
        respx.post("https://api.apertis.ai/v1/rerank").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "results": [
                        {
                            "index": 0,
                            "relevance_score": 0.90,
                            "document": "Python is a programming language.",
                        },
                    ],
                },
            )
        )

        response = client.rerank.create(
            model="BAAI/bge-reranker-v2-m3",
            query="What is Python?",
            documents=["Python is a programming language."],
            return_documents=True,
        )

        assert response.results[0].document == "Python is a programming language."

    @respx.mock
    def test_rerank_empty_documents(self, client: Apertis) -> None:
        """Test reranking with empty document list."""
        respx.post("https://api.apertis.ai/v1/rerank").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "results": [],
                },
            )
        )

        response = client.rerank.create(
            model="BAAI/bge-reranker-v2-m3",
            query="Test query",
            documents=[],
        )

        assert len(response.results) == 0

    @respx.mock
    def test_rerank_usage_tracking(self, client: Apertis) -> None:
        """Test that usage is tracked in response."""
        respx.post("https://api.apertis.ai/v1/rerank").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "results": [
                        {"index": 0, "relevance_score": 0.85},
                    ],
                    "usage": {"total_tokens": 50},
                },
            )
        )

        response = client.rerank.create(
            model="BAAI/bge-reranker-v2-m3",
            query="Test",
            documents=["Test document."],
        )

        assert response.usage is not None
        assert response.usage.total_tokens == 50
