"""Tests for Messages API (Anthropic native format)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from apertis import Apertis, APIError, AsyncApertis, InternalServerError, RateLimitError
from apertis.types.messages import RedactedThinkingBlock, TextBlock, ThinkingBlock


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
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 8},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4-6",
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
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 15, "output_tokens": 5},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4-6",
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
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 50, "output_tokens": 20},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4-6",
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
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 30, "output_tokens": 10},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4-6",
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
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4-6",
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


def _sse(*events: dict) -> bytes:
    """Encode events the way the Anthropic-native endpoint streams them."""
    return b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )


# Event shapes follow the Anthropic Messages streaming format (anthropic-python 1.11 types).
_THINKING_STREAM = _sse(
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-sonnet-4-6",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 25, "output_tokens": 1},
        },
    },
    {"type": "ping"},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "2+2 is 4."}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig=="}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "4"}},
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 15},
    },
    {"type": "message_stop"},
)


class TestMessagesStream:
    """stream=True yields typed Anthropic stream events."""

    @respx.mock
    def test_sync_stream_with_thinking(self, client: Apertis) -> None:
        route = respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(200, content=_THINKING_STREAM)
        )

        stream = client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "2+2?"}],
            max_tokens=2048,
            stream=True,
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        events = list(stream)

        body = json.loads(route.calls[0].request.content)
        assert body["stream"] is True
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert [e.type for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]  # ping skipped
        assert isinstance(events[1].content_block, ThinkingBlock)
        assert events[2].delta.thinking == "2+2 is 4."
        assert events[3].delta.signature == "sig=="
        assert events[6].delta.text == "4"
        assert events[8].delta.stop_reason == "end_turn"
        assert events[8].usage.output_tokens == 15

    @respx.mock
    async def test_async_stream(self, async_client: AsyncApertis) -> None:
        respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(200, content=_THINKING_STREAM)
        )

        stream = await async_client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "2+2?"}],
            max_tokens=2048,
            stream=True,
        )
        text = "".join(
            [e.delta.text async for e in stream if e.type == "content_block_delta" and e.delta.text]
        )

        assert text == "4"

    @respx.mock
    def test_error_event_raises(self, client: Apertis) -> None:
        respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(
                200,
                content=_sse(
                    {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
                ),
            )
        )

        stream = client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=16,
            stream=True,
        )
        with pytest.raises(APIError, match="Overloaded") as exc:
            list(stream)
        assert exc.value.body == {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


class TestMessagesThinkingAndExtraBody:
    @respx.mock
    def test_thinking_blocks_and_extra_body(self, client: Apertis) -> None:
        route = respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Add them.", "signature": "sig=="},
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "4"},
                    ],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            )
        )

        message = client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "2+2?"}],
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
            extra_body={"service_tier": "auto"},
        )

        body = json.loads(route.calls[0].request.content)
        assert body["service_tier"] == "auto"
        assert "stream" not in body
        thinking, redacted, text = message.content
        assert isinstance(thinking, ThinkingBlock)
        assert (thinking.thinking, thinking.signature) == ("Add them.", "sig==")
        assert isinstance(redacted, RedactedThinkingBlock)
        assert isinstance(text, TextBlock)


class TestMessagesStreamErrors:
    @respx.mock
    def test_error_event_maps_type_and_closes_response(self, client: Apertis) -> None:
        respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(
                200,
                content=_sse(
                    {"type": "error", "error": {"type": "rate_limit_error", "message": "Slow down"}}
                ),
            )
        )

        stream = client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=16,
            stream=True,
        )
        with pytest.raises(RateLimitError, match="Slow down"):
            next(stream)
        assert stream._response.is_closed

    @respx.mock
    async def test_async_overloaded_is_server_error(self, async_client: AsyncApertis) -> None:
        respx.post("https://api.apertis.ai/v1/messages").mock(
            return_value=httpx.Response(
                200,
                content=_sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
            )
        )

        stream = await async_client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=16,
            stream=True,
        )
        with pytest.raises(InternalServerError, match="Overloaded"):
            await stream.__anext__()
        assert stream._response.is_closed
