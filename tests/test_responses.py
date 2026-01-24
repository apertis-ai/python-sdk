"""Tests for Responses API."""

from __future__ import annotations

import httpx
import pytest
import respx

from apertis import Apertis


class TestResponsesCreate:
    """Tests for creating responses."""

    @respx.mock
    def test_create_response_with_string_input(self, client: Apertis) -> None:
        """Test creating a response with string input."""
        respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp-123",
                    "object": "response",
                    "created_at": 1234567890,
                    "status": "completed",
                    "model": "gpt-5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg-123",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Hello! How can I help?"}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 8,
                        "total_tokens": 18,
                    },
                },
            )
        )

        response = client.responses.create(
            model="gpt-5-pro",
            input="Hello!",
        )

        assert response.id == "resp-123"
        assert response.status == "completed"
        assert len(response.output) == 1
        assert response.output[0].role == "assistant"

    @respx.mock
    def test_create_response_with_instructions(self, client: Apertis) -> None:
        """Test creating a response with system instructions."""
        route = respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp-123",
                    "object": "response",
                    "created_at": 1234567890,
                    "status": "completed",
                    "model": "gpt-5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg-123",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Brief response."}],
                        }
                    ],
                },
            )
        )

        response = client.responses.create(
            model="gpt-5-pro",
            input="Tell me about Python",
            instructions="Be brief and concise.",
            max_output_tokens=100,
        )

        assert response.status == "completed"

    @respx.mock
    def test_create_response_with_structured_input(self, client: Apertis) -> None:
        """Test creating a response with structured input items."""
        respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp-123",
                    "object": "response",
                    "created_at": 1234567890,
                    "status": "completed",
                    "model": "gpt-5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg-123",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Continuing..."}],
                        }
                    ],
                },
            )
        )

        response = client.responses.create(
            model="gpt-5-pro",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "Hello",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "Hi there!",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "Continue our conversation",
                },
            ],
        )

        assert response.status == "completed"

    @respx.mock
    def test_create_response_with_reasoning(self, client: Apertis) -> None:
        """Test creating a response with reasoning mode."""
        respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp-123",
                    "object": "response",
                    "created_at": 1234567890,
                    "status": "completed",
                    "model": "o1-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg-123",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "reasoning",
                                    "summary": [{"type": "text", "text": "Thinking..."}],
                                },
                                {"type": "text", "text": "The answer is 42."},
                            ],
                        }
                    ],
                },
            )
        )

        response = client.responses.create(
            model="o1-pro",
            input="What is the meaning of life?",
            reasoning={"effort": "high"},
        )

        assert response.status == "completed"
        assert len(response.output[0].content) == 2
