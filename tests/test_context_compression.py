"""Tests for context compression across all endpoints."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from apertis import Apertis, ChatCompletion


class TestChatCompletionsCompression:
    """Tests for context compression in chat completions."""

    def test_compression_in_request_body(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test that compression config is included in request body."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello!"}],
            compression={
                "enabled": True,
                "strategy": "on",
                "threshold": 8000,
                "keep_turns": 6,
                "model": "auto",
            },
        )

        assert isinstance(response, ChatCompletion)

        request = mock_api.calls[0].request
        body = json.loads(request.content)
        assert body["compression"]["enabled"] is True
        assert body["compression"]["strategy"] == "on"
        assert body["compression"]["threshold"] == 8000
        assert body["compression"]["keep_turns"] == 6
        assert body["compression"]["model"] == "auto"

    def test_compression_minimal_config(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test compression with only required field (enabled)."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello!"}],
            compression={"enabled": True},
        )

        request = mock_api.calls[0].request
        body = json.loads(request.content)
        assert body["compression"] == {"enabled": True}

    def test_compression_aggressive_strategy(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test compression with aggressive strategy."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello!"}],
            compression={
                "enabled": True,
                "strategy": "aggressive",
                "model": "gpt-4.1-mini",
            },
        )

        request = mock_api.calls[0].request
        body = json.loads(request.content)
        assert body["compression"]["strategy"] == "aggressive"

    def test_no_compression_by_default(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test that compression is not included when not specified."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        request = mock_api.calls[0].request
        body = json.loads(request.content)
        assert "compression" not in body


class TestResponsesCompression:
    """Tests for context compression in responses endpoint."""

    def test_compression_in_responses(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test that compression config works with responses endpoint."""
        mock_api.post("/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp-123",
                    "object": "response",
                    "created_at": 1234567890,
                    "status": "completed",
                    "model": "o4-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg-123",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Hello!"}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
        )

        client.responses.create(
            model="o4-mini",
            input=[{"role": "user", "content": "Hello!"}],
            compression={
                "enabled": True,
                "strategy": "on",
                "model": "gpt-4.1-mini",
            },
        )

        request = mock_api.calls[0].request
        body = json.loads(request.content)
        assert body["compression"]["enabled"] is True
        assert body["compression"]["strategy"] == "on"


class TestMessagesCompression:
    """Tests for context compression in messages endpoint."""

    def test_compression_in_messages(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test that compression config works with messages endpoint."""
        mock_api.post("/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg-123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                    },
                },
            )
        )

        client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024,
            compression={
                "enabled": True,
                "strategy": "conservative",
            },
        )

        request = mock_api.calls[0].request
        body = json.loads(request.content)
        assert body["compression"]["enabled"] is True
        assert body["compression"]["strategy"] == "conservative"
