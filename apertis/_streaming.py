"""SSE streaming handler for the Apertis SDK."""

from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterator,
    Optional,
    TypeVar,
)

from apertis.types.chat import ChatCompletionChunk

if TYPE_CHECKING:
    import httpx

_T = TypeVar("_T")

# Turns one decoded `data:` payload into an item; returning None skips it.
ParseFn = Callable[[Dict[str, Any], "httpx.Response"], Optional[_T]]


def _parse_chat_chunk(data: Dict[str, Any], response: "httpx.Response") -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(data)


class Stream(Generic[_T]):
    """Synchronous streaming response handler."""

    def __init__(
        self, response: "httpx.Response", parse: ParseFn[_T] = _parse_chat_chunk  # type: ignore[assignment]
    ) -> None:
        self._response = response
        self._parse = parse
        self._iterator: Optional[Iterator[str]] = None

    def __iter__(self) -> "Stream[_T]":
        return self

    def __next__(self) -> _T:
        if self._iterator is None:
            self._iterator = self._response.iter_lines()

        for line in self._iterator:
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    raise StopIteration
                try:
                    chunk_data = json.loads(data)
                except json.JSONDecodeError:
                    continue
                item = self._parse(chunk_data, self._response)
                if item is not None:
                    return item

        raise StopIteration

    def __enter__(self) -> "Stream[_T]":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying response."""
        self._response.close()


class AsyncStream(Generic[_T]):
    """Asynchronous streaming response handler."""

    def __init__(
        self, response: "httpx.Response", parse: ParseFn[_T] = _parse_chat_chunk  # type: ignore[assignment]
    ) -> None:
        self._response = response
        self._parse = parse
        self._iterator: Optional[AsyncIterator[str]] = None

    def __aiter__(self) -> "AsyncStream[_T]":
        return self

    async def __anext__(self) -> _T:
        if self._iterator is None:
            self._iterator = self._response.aiter_lines()

        async for line in self._iterator:
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    raise StopAsyncIteration
                try:
                    chunk_data = json.loads(data)
                except json.JSONDecodeError:
                    continue
                item = self._parse(chunk_data, self._response)
                if item is not None:
                    return item

        raise StopAsyncIteration

    async def __aenter__(self) -> "AsyncStream[_T]":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying response."""
        await self._response.aclose()
