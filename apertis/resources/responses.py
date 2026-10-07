"""Responses resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Union, overload

from apertis._exceptions import _make_api_error
from apertis._streaming import AsyncStream, Stream
from apertis.types.chat import CompressionConfig
from apertis.types.responses import Response, ResponseInputItem, ResponseStreamEvent

if TYPE_CHECKING:
    import httpx

    from apertis._base_client import AsyncClient, SyncClient


# Error codes and types to the HTTP status the same error has outside a stream.
_ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "rate_limit_exceeded": 429,
    "rate_limit_error": 429,
    "insufficient_quota": 429,
    "server_error": 500,
    "api_error": 500,
    "overloaded_error": 529,
}


def _parse_event(data: Dict[str, Any], response: "httpx.Response") -> ResponseStreamEvent:
    """Map one SSE payload to an event; raise on an error event."""
    # OpenAI sends {"type": "error", "code", "message"}; the gateway may send {"error": {...}}.
    if data.get("type") == "error" or ("type" not in data and "error" in data):
        error = data.get("error")
        error = error if isinstance(error, dict) else data
        # The stream itself is HTTP 200; the error code or type says what failed.
        status = _ERROR_STATUS.get(str(error.get("code"))) or _ERROR_STATUS.get(str(error.get("type")), 500)
        raise _make_api_error(
            str(error.get("message") or data),
            status_code=status,
            response=response,
            body=data,
        )
    return ResponseStreamEvent.model_validate(data)


class Responses:
    """Synchronous responses resource.

    This endpoint is for models that support /v1/responses.
    """

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        stream: Literal[True],
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        compression: Optional[CompressionConfig] = None,
    ) -> Stream[ResponseStreamEvent]: ...

    @overload
    def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        stream: Literal[False] = False,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        compression: Optional[CompressionConfig] = None,
    ) -> Response: ...

    @overload
    def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        stream: bool = False,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        compression: Optional[CompressionConfig] = None,
    ) -> Union[Response, Stream[ResponseStreamEvent]]: ...

    def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        stream: bool = False,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        compression: Optional[CompressionConfig] = None,
    ) -> Union[Response, Stream[ResponseStreamEvent]]:
        """Create a response.

        Args:
            model: ID of the model to use (e.g., "gpt-5.4").
            input: Input text or structured input items.
            stream: If True, returns an iterator of ResponseStreamEvent.
            instructions: System instructions for the model.
            max_output_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0-2).
            top_p: Nucleus sampling parameter.
            reasoning: Reasoning configuration for thinking models.
            tools: List of tools the model can use.
            tool_choice: Controls tool selection.
            metadata: Metadata to attach to the response.
            compression: Context compression configuration.

        Returns:
            Response, or a Stream of ResponseStreamEvent when stream is True.
        """
        body = _build_request_body(
            model=model,
            input=input,
            stream=stream,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning=reasoning,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
            compression=compression,
        )

        if stream:
            return Stream(self._client.stream("POST", "/responses", json=body), _parse_event)

        response = self._client.request("POST", "/responses", json=body)
        return Response.model_validate(response.json())


class AsyncResponses:
    """Asynchronous responses resource."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        stream: Literal[True],
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        compression: Optional[CompressionConfig] = None,
    ) -> AsyncStream[ResponseStreamEvent]: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        stream: Literal[False] = False,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        compression: Optional[CompressionConfig] = None,
    ) -> Response: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        stream: bool = False,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        compression: Optional[CompressionConfig] = None,
    ) -> Union[Response, AsyncStream[ResponseStreamEvent]]: ...

    async def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        stream: bool = False,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        compression: Optional[CompressionConfig] = None,
    ) -> Union[Response, AsyncStream[ResponseStreamEvent]]:
        """Create a response asynchronously.

        See Responses.create() for parameter documentation.
        """
        body = _build_request_body(
            model=model,
            input=input,
            stream=stream,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning=reasoning,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
            compression=compression,
        )

        if stream:
            return AsyncStream(await self._client.stream("POST", "/responses", json=body), _parse_event)

        response = await self._client.request("POST", "/responses", json=body)
        return Response.model_validate(response.json())


def _build_request_body(
    *,
    model: str,
    input: Union[str, Sequence[ResponseInputItem]],
    stream: bool,
    instructions: Optional[str],
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    reasoning: Optional[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    metadata: Optional[Dict[str, str]],
    compression: Optional[CompressionConfig],
) -> Dict[str, Any]:
    """Build request body for responses."""
    body: Dict[str, Any] = {"model": model}

    if isinstance(input, str):
        body["input"] = input
    else:
        body["input"] = list(input)

    if stream:
        body["stream"] = True
    if instructions is not None:
        body["instructions"] = instructions
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if reasoning is not None:
        body["reasoning"] = reasoning
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if metadata is not None:
        body["metadata"] = metadata
    if compression is not None:
        body["compression"] = compression

    return body
