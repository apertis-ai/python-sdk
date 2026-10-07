"""Tests for Responses API."""

from __future__ import annotations

import json
from typing import List

import httpx
import pytest
import respx
from pydantic import ValidationError

from apertis import Apertis, AsyncApertis, APIError, RateLimitError
from apertis.types.chat import ThinkingConfig
from apertis.types.responses import (
    Response,
    ResponseInputContent,
    ResponseInputItem,
    ResponseStreamEvent,
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
    """Spec part types are accepted and sent unchanged."""

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
def _sse(*events: dict, done: bool = True) -> bytes:
    """Encode events as the gateway streams /v1/responses."""
    body = b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)
    return body + (b"data: [DONE]\n\n" if done else b"")


# What the gateway builds for a non-OpenAI channel: no sequence_number or item_id.
_GATEWAY_STREAM = _sse(
    {"type": "response.created", "response": {"id": "resp_1", "object": "response", "status": "in_progress"}},
    {"type": "response.output_item.added", "output_index": 0, "item": {"id": "msg_1", "type": "message"}},
    {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hel"},
    {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "lo"},
    {"type": "response.output_text.done", "output_index": 0, "content_index": 0, "text": "Hello"},
    {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output_text": "Hello"}},
)


class TestResponsesStream:
    """stream=True yields Responses API stream events."""

    @respx.mock
    def test_gateway_stream_ends_at_done(self, client: Apertis) -> None:
        route = respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(200, content=_GATEWAY_STREAM + _sse({"type": "after.done"}, done=False))
        )

        stream = client.responses.create(model="claude-sonnet-4.5", input="Hi", stream=True)
        events = list(stream)

        assert json.loads(route.calls[0].request.content)["stream"] is True
        assert all(isinstance(e, ResponseStreamEvent) for e in events)
        assert [e.type for e in events] == [
            "response.created",
            "response.output_item.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.completed",
        ]
        assert "".join(e.delta or "" for e in events if e.type == "response.output_text.delta") == "Hello"
        assert events[1].item == {"id": "msg_1", "type": "message"}
        assert events[-1].response is not None and events[-1].response["output_text"] == "Hello"

    @respx.mock
    def test_passthrough_stream_keeps_all_fields(self, client: Apertis) -> None:
        respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(
                200,
                content=_sse(
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": 4,
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "Hi",
                        "logprobs": [],
                    },
                    {"type": "response.completed", "sequence_number": 5, "response": {"id": "resp_1"}},
                    done=False,
                ),
            )
        )

        events = list(client.responses.create(model="gpt-5.4", input="Hi", stream=True))

        assert [e.sequence_number for e in events] == [4, 5]
        assert events[0].item_id == "msg_1"
        assert events[0].model_dump()["logprobs"] == []

    @respx.mock
    def test_error_event_raises_and_closes(self, client: Apertis) -> None:
        error = {"type": "error", "code": "rate_limit_exceeded", "message": "Slow down", "param": None}
        respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(200, content=_sse({"type": "response.created", "response": {}}, error))
        )

        stream = client.responses.create(model="gpt-5.4", input="Hi", stream=True)
        assert next(stream).type == "response.created"
        with pytest.raises(RateLimitError, match="Slow down") as exc:
            next(stream)
        assert exc.value.body == error
        assert stream._response.is_closed

    @respx.mock
    def test_nested_error_object(self, client: Apertis) -> None:
        error = {"error": {"type": "server_error", "message": "upstream failed"}}
        respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(200, content=b"data: " + json.dumps(error).encode() + b"\n\n")
        )

        with pytest.raises(APIError, match="upstream failed"):
            list(client.responses.create(model="gpt-5.4", input="Hi", stream=True))

    @respx.mock
    def test_non_streaming_body_has_no_stream_key(self, client: Apertis) -> None:
        route = respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(200, json=_REAL_SHAPE)
        )

        client.responses.create(model="gpt-5.4", input="Hi")

        assert "stream" not in json.loads(route.calls[0].request.content)

    @respx.mock
    async def test_async_stream(self, async_client: AsyncApertis) -> None:
        route = respx.post("https://api.apertis.ai/v1/responses").mock(
            return_value=httpx.Response(200, content=_GATEWAY_STREAM)
        )

        stream = await async_client.responses.create(model="claude-sonnet-4.5", input="Hi", stream=True)
        text = "".join([e.delta async for e in stream if e.type == "response.output_text.delta" and e.delta])

        assert text == "Hello"
        assert json.loads(route.calls[0].request.content)["stream"] is True
