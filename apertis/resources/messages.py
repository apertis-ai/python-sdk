"""Messages resource (Anthropic native format)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Union, overload

from apertis._exceptions import _make_api_error
from apertis._streaming import AsyncStream, Stream
from apertis.types.chat import CompressionConfig
from apertis.types.messages import Message, MessageParam, MessageStreamEvent, ToolDefinition

if TYPE_CHECKING:
    import httpx

    from apertis._base_client import AsyncClient, SyncClient


# Anthropic error types to the HTTP status the same error has outside a stream.
_ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "overloaded_error": 529,
}


def _parse_event(data: Dict[str, Any], response: "httpx.Response") -> Optional[MessageStreamEvent]:
    """Map one SSE payload to an event: skip pings, raise on error events."""
    event_type = data.get("type")
    if event_type == "ping":
        return None
    if event_type == "error":
        error = data.get("error")
        error = error if isinstance(error, dict) else {"message": str(error)}
        raise _make_api_error(
            error.get("message") or str(data),
            # The stream itself is HTTP 200; the error type says what failed.
            status_code=_ERROR_STATUS.get(str(error.get("type")), 500),
            response=response,
            body=data,
        )
    return MessageStreamEvent.model_validate(data)


class Messages:
    """Synchronous messages resource (Anthropic native format)."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        stream: Literal[True],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        compression: Optional[CompressionConfig] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Stream[MessageStreamEvent]: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        stream: Literal[False] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        compression: Optional[CompressionConfig] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Message: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        stream: bool = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        compression: Optional[CompressionConfig] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[Message, Stream[MessageStreamEvent]]: ...

    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        stream: bool = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        compression: Optional[CompressionConfig] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[Message, Stream[MessageStreamEvent]]:
        """Create a message using Anthropic's native format.

        Args:
            model: ID of the model to use (e.g., "claude-sonnet-4-6").
            messages: List of messages in the conversation.
            max_tokens: Maximum tokens to generate.
            stream: If True, returns an iterator of MessageStreamEvent.
            system: System prompt.
            temperature: Sampling temperature (0-1).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            stop_sequences: Sequences that stop generation.
            tools: List of tools the model can use.
            tool_choice: Controls tool selection.
            metadata: Metadata to attach to the message.
            thinking: Extended thinking, e.g. {"type": "enabled", "budget_tokens": 1024}.
            compression: Context compression configuration.
            extra_body: Extra fields merged into the top level of the request body.

        Returns:
            Message, or a Stream of MessageStreamEvent when stream is True.
        """
        body = _build_request_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
            thinking=thinking,
            compression=compression,
            extra_body=extra_body,
        )

        if stream:
            return Stream(self._client.stream("POST", "/messages", json=body), _parse_event)

        response = self._client.request("POST", "/messages", json=body)
        return Message.model_validate(response.json())


class AsyncMessages:
    """Asynchronous messages resource (Anthropic native format)."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        stream: Literal[True],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        compression: Optional[CompressionConfig] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AsyncStream[MessageStreamEvent]: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        stream: Literal[False] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        compression: Optional[CompressionConfig] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Message: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        stream: bool = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        compression: Optional[CompressionConfig] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[Message, AsyncStream[MessageStreamEvent]]: ...

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        stream: bool = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        compression: Optional[CompressionConfig] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[Message, AsyncStream[MessageStreamEvent]]:
        """Create a message asynchronously.

        See Messages.create() for parameter documentation.
        """
        body = _build_request_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
            thinking=thinking,
            compression=compression,
            extra_body=extra_body,
        )

        if stream:
            response = await self._client.stream("POST", "/messages", json=body)
            return AsyncStream(response, _parse_event)

        response = await self._client.request("POST", "/messages", json=body)
        return Message.model_validate(response.json())


def _build_request_body(
    *,
    model: str,
    messages: Sequence[MessageParam],
    max_tokens: int,
    stream: bool,
    system: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    stop_sequences: Optional[List[str]],
    tools: Optional[List[ToolDefinition]],
    tool_choice: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, str]],
    thinking: Optional[Dict[str, Any]],
    compression: Optional[CompressionConfig],
    extra_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build request body for messages."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
    }

    if stream:
        body["stream"] = True
    if system is not None:
        body["system"] = system
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if stop_sequences is not None:
        body["stop_sequences"] = stop_sequences
    if tools is not None:
        body["tools"] = list(tools)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if metadata is not None:
        body["metadata"] = metadata
    if thinking is not None:
        body["thinking"] = thinking
    if compression is not None:
        body["compression"] = compression
    if extra_body is not None:
        body.update(extra_body)

    return body
