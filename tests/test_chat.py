"""Tests for chat completions."""

from __future__ import annotations

import httpx
import pytest
import respx

from apertis import Apertis, ChatCompletion


class TestChatCompletions:
    """Tests for chat completions."""

    def test_create_completion(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test creating a chat completion."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-5.2",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! How can I help you?",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18,
                    },
                },
            )
        )

        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert isinstance(response, ChatCompletion)
        assert response.id == "chatcmpl-123"
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 18

    def test_create_completion_with_tool_calls(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test creating a chat completion with tool calls."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-5.2",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"location": "Tokyo"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 20,
                        "total_tokens": 70,
                    },
                },
            )
        )

        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
        )

        assert response.choices[0].finish_reason == "tool_calls"
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        assert response.choices[0].message.tool_calls[0].function.name == "get_weather"

    def test_create_completion_with_parameters(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test creating a chat completion with various parameters."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-5.2",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Test"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            seed=42,
            user="test-user",
        )

        assert response.id == "chatcmpl-123"

        # Verify the request was made with correct parameters
        request = mock_api.calls[0].request
        import json

        body = json.loads(request.content)
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 100
        assert body["top_p"] == 0.9
        assert body["seed"] == 42
        assert body["user"] == "test-user"
