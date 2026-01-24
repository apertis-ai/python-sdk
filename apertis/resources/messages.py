"""Messages resource (Anthropic native format)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Union

from apertis.types.messages import Message, MessageParam, ToolDefinition

if TYPE_CHECKING:
    from apertis._base_client import AsyncClient, SyncClient


class Messages:
    """Synchronous messages resource (Anthropic native format)."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Message:
        """Create a message using Anthropic's native format.

        Args:
            model: ID of the model to use (e.g., "claude-sonnet-4.5").
            messages: List of messages in the conversation.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            temperature: Sampling temperature (0-1).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            stop_sequences: Sequences that stop generation.
            tools: List of tools the model can use.
            tool_choice: Controls tool selection.
            metadata: Metadata to attach to the message.

        Returns:
            Message object with generated content.
        """
        body = _build_request_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
        )

        response = self._client.request("POST", "/messages", json=body)
        return Message.model_validate(response.json())


class AsyncMessages:
    """Asynchronous messages resource (Anthropic native format)."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        max_tokens: int,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Message:
        """Create a message asynchronously.

        See Messages.create() for parameter documentation.
        """
        body = _build_request_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
        )

        response = await self._client.request("POST", "/messages", json=body)
        return Message.model_validate(response.json())


def _build_request_body(
    *,
    model: str,
    messages: Sequence[MessageParam],
    max_tokens: int,
    system: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    stop_sequences: Optional[List[str]],
    tools: Optional[List[ToolDefinition]],
    tool_choice: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Build request body for messages."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
    }

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

    return body
