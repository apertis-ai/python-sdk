"""Chat resource module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apertis.resources.chat.completions import Completions, AsyncCompletions

if TYPE_CHECKING:
    from apertis._base_client import AsyncClient, SyncClient


class Chat:
    """Synchronous chat resource."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client
        self.completions = Completions(client)


class AsyncChat:
    """Asynchronous chat resource."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client
        self.completions = AsyncCompletions(client)
