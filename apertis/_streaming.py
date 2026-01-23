"""SSE streaming handler for the Apertis SDK."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, AsyncIterator, Iterator, Optional

from apertis.types.chat import ChatCompletionChunk

if TYPE_CHECKING:
    import httpx


class Stream:
    """Synchronous streaming response handler."""

    def __init__(self, response: "httpx.Response") -> None:
        self._response = response
        self._iterator: Optional[Iterator[str]] = None

    def __iter__(self) -> "Stream":
        return self

    def __next__(self) -> ChatCompletionChunk:
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
                    return ChatCompletionChunk.model_validate(chunk_data)
                except json.JSONDecodeError:
                    continue

        raise StopIteration

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying response."""
        self._response.close()


class AsyncStream:
    """Asynchronous streaming response handler."""

    def __init__(self, response: "httpx.Response") -> None:
        self._response = response
        self._iterator: Optional[AsyncIterator[str]] = None

    def __aiter__(self) -> "AsyncStream":
        return self

    async def __anext__(self) -> ChatCompletionChunk:
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
                    return ChatCompletionChunk.model_validate(chunk_data)
                except json.JSONDecodeError:
                    continue

        raise StopAsyncIteration

    async def __aenter__(self) -> "AsyncStream":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying response."""
        await self._response.aclose()
