"""Tests for advanced chat completion features."""

from __future__ import annotations

import httpx
import pytest
import respx

from apertis import Apertis


class TestWebSearch:
    """Tests for web search functionality."""

    @respx.mock
    def test_web_search_options(self, client: Apertis) -> None:
        """Test web search options in request."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-5-search-api",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "According to recent news...",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "start_index": 0,
                                        "end_index": 27,
                                        "url": "https://example.com/news",
                                        "title": "News Article",
                                    }
                                ],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="gpt-5-search-api",
            messages=[{"role": "user", "content": "What's the latest news?"}],
            web_search_options={
                "search_context_size": "high",
                "filters": ["reuters.com", "bbc.com"],
            },
        )

        assert response.choices[0].message.annotations is not None
        assert len(response.choices[0].message.annotations) == 1
        assert response.choices[0].message.annotations[0].url == "https://example.com/news"

    @respx.mock
    def test_create_with_web_search(self, client: Apertis) -> None:
        """Test create_with_web_search convenience method."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-5-search-api",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The weather is sunny.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create_with_web_search(
            prompt="What's the weather in Tokyo?",
            context_size="medium",
            country="JP",
            city="Tokyo",
        )

        assert "weather" in response.choices[0].message.content.lower()

    @respx.mock
    def test_create_with_web_search_allowed_domains(self, client: Apertis) -> None:
        """Test web search with domain allow-list."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-5-search-api",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Found results.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create_with_web_search(
            prompt="Latest Python updates",
            allowed_domains=["python.org", "github.com"],
        )

        assert response.choices[0].message.content is not None


class TestReasoningMode:
    """Tests for reasoning mode functionality."""

    @respx.mock
    def test_reasoning_config(self, client: Apertis) -> None:
        """Test reasoning mode configuration."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "glm-4.7",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "There are 3 r's in strawberry.",
                                "reasoning_details": [
                                    {"type": "thinking", "content": "Let me count..."}
                                ],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="glm-4.7",
            messages=[{"role": "user", "content": "How many r's in strawberry?"}],
            reasoning={"enabled": True, "effort": "high"},
        )

        assert response.choices[0].message.reasoning_details is not None

    @respx.mock
    def test_reasoning_effort(self, client: Apertis) -> None:
        """Test reasoning effort parameter."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "glm-4.7",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The answer is 42.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="glm-4.7",
            messages=[{"role": "user", "content": "Solve this problem"}],
            reasoning_effort="high",
        )

        assert response.choices[0].message.content is not None


class TestExtendedThinking:
    """Tests for extended thinking (Gemini) functionality."""

    @respx.mock
    def test_thinking_config(self, client: Apertis) -> None:
        """Test thinking configuration."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gemini-3-pro-preview",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Here's my analysis...",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[{"role": "user", "content": "Analyze this complex problem"}],
            thinking={"type": "enabled"},
        )

        assert response.choices[0].message.content is not None

    @respx.mock
    def test_extra_body_for_thinking(self, client: Apertis) -> None:
        """Test extra_body for Gemini thinking config."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gemini-3-pro-preview",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Detailed analysis...",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[{"role": "user", "content": "Think deeply about this"}],
            extra_body={
                "google": {
                    "thinking_config": {"thinking_budget": 10240}
                }
            },
        )

        assert response.choices[0].message.content is not None


class TestStreamOptions:
    """Tests for stream options."""

    @respx.mock
    def test_stream_options_include_usage(self, client: Apertis) -> None:
        """Test stream options with include_usage."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello!",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream_options={"include_usage": True},
        )

        assert response.usage is not None
        assert response.usage.total_tokens == 15
