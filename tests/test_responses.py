"""Tests for Responses API."""

from __future__ import annotations

import json
from typing import List

import httpx
import pytest
import respx
from pydantic import ValidationError

from apertis import Apertis, AsyncApertis
from apertis.types.chat import ThinkingConfig
from apertis.types.responses import (
    Response,
    ResponseInputContent,
    ResponseInputItem,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUnknownOutputItem,
)


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
                    "model": "gpt-5.4",
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
            model="gpt-5.4",
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
                    "model": "gpt-5.4",
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
            model="gpt-5.4",
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
                    "model": "gpt-5.4",
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
            model="gpt-5.4",
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
                    "model": "gpt-5.4",
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
            model="gpt-5.4",
            input="What is the meaning of life?",
            reasoning={"effort": "high"},
        )

        assert response.status == "completed"
        assert len(response.output[0].content) == 2


# Shapes follow the OpenAI Responses API object (openai-python 3.26 types):
# message items carry `output_text` parts, and reasoning / function_call /
# other tool items sit beside them in `output`.
_REAL_SHAPE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1741476542,
    "status": "completed",
    "model": "gpt-5.5",
    "output": [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Considered the greeting."}],
        },
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Hi", "annotations": []},
                {"type": "output_text", "text": " there.", "annotations": []},
            ],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"city\": \"Paris\"}",
            "status": "completed",
        },
        {"type": "web_search_call", "id": "ws_1", "status": "completed"},
    ],
    "usage": {"input_tokens": 12, "output_tokens": 30, "total_tokens": 42},
}


class TestResponsesOutputItems:
    """Real Responses API output parses into typed items."""

    @respx.mock
    def test_create_parses_real_output(self, client: Apertis) -> None:
        respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(200, json=_REAL_SHAPE)
        )

        response = client.responses.create(model="gpt-5.5", input="Hi")

        assert response.output_text == "Hi there."
        reasoning, message, call, unknown = response.output
        assert isinstance(reasoning, ResponseReasoningItem)
        assert reasoning.summary[0]["text"] == "Considered the greeting."
        assert isinstance(message, ResponseOutputMessage)
        assert isinstance(message.content[0], ResponseOutputText)
        assert message.content[0].annotations == []
        assert isinstance(call, ResponseFunctionToolCall)
        assert (call.call_id, call.name, call.arguments) == (
            "call_1",
            "get_weather",
            "{\"city\": \"Paris\"}",
        )
        # Item types this SDK does not model are kept, not rejected.
        assert isinstance(unknown, ResponseUnknownOutputItem)
        assert unknown.type == "web_search_call"
        assert unknown.model_dump()["status"] == "completed"

    @respx.mock
    async def test_async_create_parses_real_output(self, async_client: AsyncApertis) -> None:
        respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(200, json=_REAL_SHAPE)
        )

        response = await async_client.responses.create(model="gpt-5.5", input="Hi")

        assert response.output_text == "Hi there."

    def test_in_progress_and_refusal(self) -> None:
        response = Response.model_validate(
            {
                **_REAL_SHAPE,
                "status": "in_progress",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "I can't help with that."}],
                    }
                ],
            }
        )

        assert response.status == "in_progress"
        assert isinstance(response.output[0].content[0], ResponseOutputRefusal)
        assert response.output_text == ""

    def test_message_with_bad_content_still_fails(self) -> None:
        """A known item type with a malformed body is an error, not an unknown item."""
        with pytest.raises(ValidationError):
            Response.model_validate(
                {
                    **_REAL_SHAPE,
                    "output": [{"type": "message", "id": "m", "role": "assistant"}],
                }
            )


class TestTypedInputParts:
    """Spec part types type-check and are sent unchanged (checked by mypy too)."""

    @respx.mock
    def test_input_parts_sent_unchanged(self, client: Apertis) -> None:
        route = respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(200, json=_REAL_SHAPE)
        )
        content: List[ResponseInputContent] = [
            {"type": "input_text", "text": "Describe both."},
            {"type": "input_image", "image_url": "https://example.com/cat.png", "detail": "low"},
            {"type": "input_file", "file_data": "data:application/pdf;base64,JVBERi0=", "filename": "a.pdf"},
            {"type": "text", "text": "legacy part"},
        ]
        item: ResponseInputItem = {"type": "message", "role": "user", "content": content}

        client.responses.create(model="gpt-5.4", input=[item])

        assert json.loads(route.calls[0].request.content)["input"] == [item]

    def test_thinking_budget_tokens_type_checks(self) -> None:
        thinking: ThinkingConfig = {"type": "enabled", "budget_tokens": 2048}
        assert thinking["budget_tokens"] == 2048
