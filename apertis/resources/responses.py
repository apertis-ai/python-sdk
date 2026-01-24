"""Responses resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Union

from apertis.types.responses import Response, ResponseInputItem

if TYPE_CHECKING:
    from apertis._base_client import AsyncClient, SyncClient


class Responses:
    """Synchronous responses resource.

    This endpoint is for models that only support /v1/responses (e.g., gpt-5-pro, o1-pro).
    """

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Response:
        """Create a response.

        Args:
            model: ID of the model to use (e.g., "gpt-5-pro", "o1-pro").
            input: Input text or structured input items.
            instructions: System instructions for the model.
            max_output_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0-2).
            top_p: Nucleus sampling parameter.
            reasoning: Reasoning configuration for thinking models.
            tools: List of tools the model can use.
            tool_choice: Controls tool selection.
            metadata: Metadata to attach to the response.

        Returns:
            Response object with generated content.
        """
        body = self._build_request_body(
            model=model,
            input=input,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning=reasoning,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
        )

        response = self._client.request("POST", "/responses", json=body)
        return Response.model_validate(response.json())

    def _build_request_body(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        instructions: Optional[str],
        max_output_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        reasoning: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        metadata: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Build request body for responses."""
        body: Dict[str, Any] = {"model": model}

        # Handle input - can be string or list of input items
        if isinstance(input, str):
            body["input"] = input
        else:
            body["input"] = list(input)

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

        return body


class AsyncResponses:
    """Asynchronous responses resource."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[ResponseInputItem]],
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Response:
        """Create a response asynchronously.

        See Responses.create() for parameter documentation.
        """
        body = _build_request_body(
            model=model,
            input=input,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning=reasoning,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
        )

        response = await self._client.request("POST", "/responses", json=body)
        return Response.model_validate(response.json())


def _build_request_body(
    *,
    model: str,
    input: Union[str, Sequence[ResponseInputItem]],
    instructions: Optional[str],
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    reasoning: Optional[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    metadata: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Build request body for responses."""
    body: Dict[str, Any] = {"model": model}

    if isinstance(input, str):
        body["input"] = input
    else:
        body["input"] = list(input)

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

    return body
