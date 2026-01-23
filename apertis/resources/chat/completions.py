"""Chat completions resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence, overload

from apertis._streaming import AsyncStream, Stream
from apertis.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

if TYPE_CHECKING:
    from apertis._base_client import AsyncClient, SyncClient


class Completions:
    """Synchronous chat completions resource."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: Literal[True],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> Stream: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ChatCompletion | Stream: ...

    def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ChatCompletion | Stream:
        """Create a chat completion.

        Args:
            model: ID of the model to use.
            messages: A list of messages comprising the conversation.
            stream: If True, returns a streaming response.
            temperature: Sampling temperature between 0 and 2.
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.
            frequency_penalty: Penalty for token frequency.
            presence_penalty: Penalty for token presence.
            stop: Stop sequences.
            seed: Random seed for deterministic results.
            tools: A list of tools the model may call.
            tool_choice: Controls which tool is called.
            response_format: Format specification for the response.
            user: A unique identifier for the end-user.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of most likely tokens to return.

        Returns:
            ChatCompletion or Stream depending on the stream parameter.
        """
        body = self._build_request_body(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            user=user,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

        if stream:
            response = self._client.stream("POST", "/chat/completions", json=body)
            return Stream(response)

        response = self._client.request("POST", "/chat/completions", json=body)
        return ChatCompletion.model_validate(response.json())

    def _build_request_body(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        frequency_penalty: float | None,
        presence_penalty: float | None,
        stop: str | Sequence[str] | None,
        seed: int | None,
        tools: Sequence[ChatCompletionToolParam] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: dict[str, Any] | None,
        user: str | None,
        logprobs: bool | None,
        top_logprobs: int | None,
    ) -> dict[str, Any]:
        """Build the request body for chat completions."""
        body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": stream,
        }

        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if stop is not None:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed
        if tools is not None:
            body["tools"] = list(tools)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if user is not None:
            body["user"] = user
        if logprobs is not None:
            body["logprobs"] = logprobs
        if top_logprobs is not None:
            body["top_logprobs"] = top_logprobs

        return body


class AsyncCompletions:
    """Asynchronous chat completions resource."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: Literal[True],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> AsyncStream: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ChatCompletion | AsyncStream: ...

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ChatCompletion | AsyncStream:
        """Create a chat completion.

        Args:
            model: ID of the model to use.
            messages: A list of messages comprising the conversation.
            stream: If True, returns a streaming response.
            temperature: Sampling temperature between 0 and 2.
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.
            frequency_penalty: Penalty for token frequency.
            presence_penalty: Penalty for token presence.
            stop: Stop sequences.
            seed: Random seed for deterministic results.
            tools: A list of tools the model may call.
            tool_choice: Controls which tool is called.
            response_format: Format specification for the response.
            user: A unique identifier for the end-user.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of most likely tokens to return.

        Returns:
            ChatCompletion or AsyncStream depending on the stream parameter.
        """
        body = self._build_request_body(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            user=user,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

        if stream:
            response = await self._client.stream("POST", "/chat/completions", json=body)
            return AsyncStream(response)

        response = await self._client.request("POST", "/chat/completions", json=body)
        return ChatCompletion.model_validate(response.json())

    def _build_request_body(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        frequency_penalty: float | None,
        presence_penalty: float | None,
        stop: str | Sequence[str] | None,
        seed: int | None,
        tools: Sequence[ChatCompletionToolParam] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: dict[str, Any] | None,
        user: str | None,
        logprobs: bool | None,
        top_logprobs: int | None,
    ) -> dict[str, Any]:
        """Build the request body for chat completions."""
        body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": stream,
        }

        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if stop is not None:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed
        if tools is not None:
            body["tools"] = list(tools)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if user is not None:
            body["user"] = user
        if logprobs is not None:
            body["logprobs"] = logprobs
        if top_logprobs is not None:
            body["top_logprobs"] = top_logprobs

        return body
