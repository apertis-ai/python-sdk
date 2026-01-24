"""Tests for Messages API (Anthropic native format)."""

from __future__ import annotations

import httpx
import pytest
import respx

from apertis import Apertis


class TestMessagesCreate:
    """Tests for creating messages."""

    @respx.mock
    def test_create_message_basic(self, client: Apertis) -> None:
        """Test creating a basic message."""
        respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg-123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello! How can I help?"}],
                    "model": "claude-sonnet-4.5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 8},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4.5",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024,
        )

        assert message.id == "msg-123"
        assert message.role == "assistant"
        assert len(message.content) == 1
        assert message.content[0].text == "Hello! How can I help?"

    @respx.mock
    def test_create_message_with_system(self, client: Apertis) -> None:
        """Test creating a message with system prompt."""
        route = respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg-123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Brief response."}],
                    "model": "claude-sonnet-4.5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 15, "output_tokens": 5},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4.5",
            messages=[{"role": "user", "content": "Tell me about Python"}],
            max_tokens=1024,
            system="Be brief and concise.",
        )

        assert message.stop_reason == "end_turn"

    @respx.mock
    def test_create_message_with_tools(self, client: Apertis) -> None:
        """Test creating a message with tool use."""
        respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg-123",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-123",
                            "name": "get_weather",
                            "input": {"location": "Tokyo"},
                        }
                    ],
                    "model": "claude-sonnet-4.5",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 50, "output_tokens": 20},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4.5",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            max_tokens=1024,
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }
            ],
        )

        assert message.stop_reason == "tool_use"
        assert len(message.content) == 1
        assert message.content[0].type == "tool_use"
        assert message.content[0].name == "get_weather"

    @respx.mock
    def test_create_message_multi_turn(self, client: Apertis) -> None:
        """Test multi-turn conversation."""
        respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg-123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "It is a programming language."}],
                    "model": "claude-sonnet-4.5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 30, "output_tokens": 10},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4.5",
            messages=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Tell me more."},
            ],
            max_tokens=1024,
        )

        assert message.content[0].text is not None

    @respx.mock
    def test_create_message_with_image(self, client: Apertis) -> None:
        """Test message with image content."""
        respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg-123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I see a cat in the image."}],
                    "model": "claude-sonnet-4.5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "base64imagedata",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )

        assert "cat" in message.content[0].text.lower()
